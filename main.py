import os, sys, logging, argparse, yaml, easydict
import numpy as np
import torch

from transformers import (
    TrainingArguments,
    Trainer,
)
from transformers.trainer import Trainer
from peft import (
    LoraConfig,
    get_peft_model,
)
from accelerate import Accelerator
from torchdrug.utils import comm, pretty

from llm import *
from collector import *
from preprocess import *


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def default_checkpoint_path(output_dir, name):
    return os.path.join(output_dir, "%s.pt" % name)


def checkpoint_cfg_value(cfg, key, default=None):
    checkpoint_cfg = getattr(cfg, "checkpoint", None)
    if checkpoint_cfg is None:
        return default
    return getattr(checkpoint_cfg, key, default)


def selected_component_state_dict(model, include_diffusion=False):
    state_dict = model.state_dict()
    selected = {}

    for name, value in state_dict.items():
        should_save = is_component_checkpoint_key(name)
        if include_diffusion and "diffusion" in name:
            should_save = True
        if should_save:
            selected[name] = value.detach().cpu()

    return selected


def is_component_checkpoint_key(name):
    selected_names = (
        "lora_",
        "context_retriever",
        "score_retriever",
    )
    return any(part in name for part in selected_names)


def resolve_checkpoint_path(path):
    if os.path.isdir(path):
        candidate_names = (
            "pytorch_model.bin",
            "adapter_model.bin",
        )
        for name in candidate_names:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "Checkpoint directory %s does not contain pytorch_model.bin, "
            "or adapter_model.bin." % path
        )
    return path


def normalize_checkpoint_state_dict(state_dict):
    normalized = {}
    prefixes = (
        "module.",
        "model.",
        "llmodel.",
    )

    for name, value in state_dict.items():
        normalized_name = name
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if normalized_name.startswith(prefix):
                    normalized_name = normalized_name[len(prefix):]
                    stripped = True

        if is_component_checkpoint_key(normalized_name) or "diffusion" in normalized_name:
            normalized[normalized_name] = value

    return normalized


def save_component_checkpoint(model, path, cfg, config_name, include_diffusion=False):
    if comm.get_rank() != 0:
        return

    ensure_parent_dir(path)
    payload = {
        "format": "mkgl_component_checkpoint_v1",
        "config_name": config_name,
        "include_diffusion": include_diffusion,
        "state_dict": selected_component_state_dict(
            model, include_diffusion=include_diffusion),
        "kgl_token_length": int(cfg.kgl_token_length),
    }
    torch.save(payload, path)
    print("Saved checkpoint to %s" % path)


def load_component_checkpoint(model, path, required=True):
    if not path:
        if required:
            raise ValueError("A checkpoint path is required for this training stage.")
        return

    if not os.path.exists(path):
        if required:
            raise FileNotFoundError("Checkpoint not found: %s" % path)
        if comm.get_rank() == 0:
            print("Checkpoint not found, skipping load: %s" % path)
        return

    path = resolve_checkpoint_path(path)
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get(
        "state_dict",
        payload.get("model_state_dict", payload.get("model", payload)),
    )
    state_dict = normalize_checkpoint_state_dict(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if comm.get_rank() == 0:
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        print("Loaded checkpoint from %s" % path)
        print({
            "loaded_tensors": len(state_dict),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        })
        if unexpected:
            print("Unexpected checkpoint keys: %s" % unexpected[:10])


def print_trainable_parameter_summary(model, max_names=20):
    if comm.get_rank() != 0:
        return

    trainable = []
    frozen = 0
    trainable_count = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable_count += count
            trainable.append((name, count))
        else:
            frozen += count

    print({
        "trainable_parameters": trainable_count,
        "frozen_parameters": frozen,
        "num_trainable_tensors": len(trainable),
    })
    print("Trainable parameter names:")
    for name, count in trainable[:max_names]:
        print("  %s (%d)" % (name, count))
    if len(trainable) > max_names:
        print("  ... %d more" % (len(trainable) - max_names))


def build_image_feature_bank(vocab_df, image_cfg):
    if image_cfg is None or not getattr(image_cfg, "path", None):
        return None, None

    image_path = image_cfg.path
    if not os.path.exists(image_path):
        if getattr(image_cfg, "strict", False):
            raise FileNotFoundError(f"Image feature file not found: {image_path}")
        if comm.get_rank() == 0:
            print("Image feature file not found, falling back to text-only KG embeddings: %s" % image_path)
        return None, None

    payload = torch.load(image_path, map_location="cpu")
    entity_ids = payload.get("entity_ids")
    image_features = payload.get("features")
    image_mask = payload.get("has_image")

    if entity_ids is None or image_features is None:
        raise KeyError(
            f"{image_path} must contain `entity_ids` and `features`."
        )

    image_features = torch.as_tensor(image_features, dtype=torch.float32)
    if image_features.ndim != 2:
        raise ValueError(
            f"`features` in {image_path} must be a 2D tensor, got shape {tuple(image_features.shape)}."
        )
    if len(entity_ids) != image_features.shape[0]:
        raise ValueError(
            f"`entity_ids` and `features` length mismatch in {image_path}: "
            f"{len(entity_ids)} vs {image_features.shape[0]}."
        )

    if image_mask is None:
        image_mask = torch.ones(image_features.shape[0], dtype=torch.bool)
    else:
        image_mask = torch.as_tensor(image_mask, dtype=torch.bool)

    entity2index = {raw_name: idx for idx, raw_name in enumerate(entity_ids)}
    kgl_image_features = torch.zeros(
        len(vocab_df), image_features.shape[-1], dtype=torch.float32)
    kgl_image_mask = torch.zeros(len(vocab_df), dtype=torch.bool)

    missing_entities = 0
    matched_entities = 0
    matched_images = 0
    for row_idx, row in enumerate(vocab_df.itertuples()):
        if not getattr(row, "entity", 0):
            continue

        feature_index = entity2index.get(row.raw_name)
        if feature_index is None:
            missing_entities += 1
            continue

        matched_entities += 1
        if image_mask[feature_index]:
            kgl_image_features[row_idx] = image_features[feature_index]
            kgl_image_mask[row_idx] = True
            matched_images += 1

    if comm.get_rank() == 0:
        print("Loaded image features from %s" % image_path)
        print({
            "kgl_vocab_size": len(vocab_df),
            "matched_entities": matched_entities,
            "entities_with_images": matched_images,
            "missing_entities": missing_entities,
            "feature_dim": int(image_features.shape[-1]),
        })

    if getattr(image_cfg, "strict", False) and matched_images == 0:
        raise ValueError(
            f"No entity images from {image_path} matched the current KG vocabulary."
        )

    return kgl_image_features, kgl_image_mask


def freeze_for_denoiser_training(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "diffusion" in name

    if hasattr(model, "diffusion") and model.diffusion is not None:
        model.diffusion.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str,
                        default='config/fb15k237.yaml')
    parser.add_argument("--version", "-v", type=str,
                        default='')
    parser.add_argument("--seed", "-s", type=str,
                        default=42)
    parser.add_argument("--mkgl-checkpoint", type=str, default=None,
                        help="Path to a trained MKGL component checkpoint for diffusion-stage training.")
    parser.add_argument("--save-mkgl-checkpoint", type=str, default=None,
                        help="Path where the trained MKGL component checkpoint will be saved.")
    parser.add_argument("--save-diffusion-checkpoint", type=str, default=None,
                        help="Path where the diffusion-stage component checkpoint will be saved.")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
        if args.version:
            cfg.dataset.version = args.version
    torch.manual_seed(args.seed + comm.get_rank())

    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version'):
        config_name += '_' + cfg.dataset.version
    args.config_name = config_name
    cfg.trainer.output_dir += config_name
    
    
    if comm.get_rank() == 0:
        print("Config file: %s" % args.config)
        print(pretty.format(cfg))
    

    saved_dir = 'data/preprocessed/'
    file_path = saved_dir+args.config_name+'.pkl'
    if 'ind' in args.config_name:
        dataset = InductiveKGCDataset.load(file_path)
    else:
        dataset = KGCDataset.load(file_path)
    tokenizer = dataset.tokenizer
    cfg.context_retriever.kg_encoder.base_layer.num_relation = int(
        dataset.kgdata.num_relation)
    cfg.score_retriever.kg_encoder.base_layer.num_relation = int(
        dataset.kgdata.num_relation)
    
    torch.nn.Module = torch.nn._Module
    config = MKGLConfig.from_pretrained(**cfg.mkglconfig)
    model = MKGL.from_pretrained(
        **cfg.mkgl, device_map={"": Accelerator().process_index}, config=config)

    lora_config = LoraConfig(**cfg.loraconfig)
    model = get_peft_model(model, lora_config)

    vocab_df = dataset.vocab_df.sort_index()
    kgl2token = torch.tensor(np.stack(vocab_df.text_token_ids)[:, :cfg.kgl_token_length])
    kgl_image_features, kgl_image_mask = build_image_feature_bank(
        vocab_df, getattr(cfg, "image_features", None))
    model.init_kg_specs(
        kgl2token,
        tokenizer.vocab_size,
        cfg,
        image_features=kgl_image_features,
        image_feature_mask=kgl_image_mask,
    ) 

    diffusion_cfg = getattr(cfg, "diffusion", easydict.EasyDict())
    diffusion_enabled = getattr(diffusion_cfg, "enabled", False)
    diffusion_mode = str(getattr(diffusion_cfg, "mode", "joint"))
    default_mkgl_checkpoint = default_checkpoint_path(
        cfg.trainer.output_dir, "mkgl_checkpoint")
    default_diffusion_checkpoint = default_checkpoint_path(
        cfg.trainer.output_dir, "diffusion_checkpoint")
    mkgl_checkpoint_path = (
        args.mkgl_checkpoint
        or checkpoint_cfg_value(cfg, "mkgl_path", None)
        or default_mkgl_checkpoint
    )
    save_mkgl_checkpoint_path = (
        args.save_mkgl_checkpoint
        or checkpoint_cfg_value(cfg, "save_mkgl_path", None)
        or default_mkgl_checkpoint
    )
    save_diffusion_checkpoint_path = (
        args.save_diffusion_checkpoint
        or checkpoint_cfg_value(cfg, "save_diffusion_path", None)
        or default_diffusion_checkpoint
    )

    if diffusion_enabled and diffusion_mode == "denoiser":
        if comm.get_rank() == 0:
            print("Stage 2 diffusion denoiser training enabled.")
            print("Loading frozen MKGL checkpoint from %s" % mkgl_checkpoint_path)
        load_component_checkpoint(model, mkgl_checkpoint_path, required=True)

    if diffusion_enabled:
        if hasattr(dataset.kgdata, "inductive_vocab"):
            num_entities = max(
                len(dataset.kgdata.transductive_vocab),
                len(dataset.kgdata.inductive_vocab),
            )
        else:
            num_entities = int(dataset.kgdata.num_entity)

        model.init_diffusion(
            num_entities=num_entities,
            hidden_dim=int(getattr(diffusion_cfg, "hidden_dim", 2048)),
            num_steps=int(getattr(diffusion_cfg, "num_steps", 40)),
            num_blocks=int(getattr(diffusion_cfg, "num_blocks", 1)),
            mode=diffusion_mode,
        )

        if diffusion_mode == "denoiser":
            freeze_for_denoiser_training(model)
            if comm.get_rank() == 0:
                print("Frozen MKGL/LoRA/retriever parameters; only diffusion parameters remain trainable.")
    
    if comm.get_rank() == 0:
        print(model.print_trainable_parameters())
        print_trainable_parameter_summary(model)
        print(model)

    
    if 'ind' in args.config:
        task = KGL4IndKGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    else:
        task = KGL4KGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    

    data_loader = MKGLDataCollector(dataset)
    
    training_args = TrainingArguments(**cfg.trainer)
    if comm.get_rank() == 0:
        print(training_args)


    def compute_metrics(predictions):
        ranking = predictions[0].astype(float)
        metric = ("mr", "mrr", "hits@1", "hits@3", "hits@10")
        results = {}
        for _metric in metric:
            if _metric == "mr":
                score = ranking.mean()
            elif _metric == "mrr":
                score = (1 / ranking).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            results[_metric] = score
        if comm.get_rank() == 0:
            print(results)
        return results

    removed_columns = ['h_raw', 't_raw', 'r_raw', 'h_fine', 't_fine', 'r_fine', 'inv_r_fine']

    trainer = Trainer(
        model=task,
        args=training_args,
        eval_dataset=dataset.test_data.remove_columns(
            removed_columns),  
        train_dataset=dataset.train_data.remove_columns(removed_columns),
        data_collator=data_loader,
        compute_metrics=compute_metrics
    )
    if not (diffusion_enabled and diffusion_mode == "denoiser"):
        trainer.evaluate()
    trainer.train()

    if diffusion_enabled:
        save_component_checkpoint(
            model,
            save_diffusion_checkpoint_path,
            cfg,
            args.config_name,
            include_diffusion=True,
        )
    else:
        save_component_checkpoint(
            model,
            save_mkgl_checkpoint_path,
            cfg,
            args.config_name,
            include_diffusion=False,
        )
