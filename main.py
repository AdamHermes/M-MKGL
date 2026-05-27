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
    if getattr(diffusion_cfg, "enabled", False):
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
            mode=str(getattr(diffusion_cfg, "mode", "joint")),
        )

        if getattr(diffusion_cfg, "mode", "joint") == "denoiser":
            freeze_for_denoiser_training(model)
    
    if comm.get_rank() == 0:
        print(model.print_trainable_parameters())
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
    trainer.evaluate()
    trainer.train()

