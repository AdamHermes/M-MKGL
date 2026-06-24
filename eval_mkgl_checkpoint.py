import argparse
import sys
import yaml

import easydict
import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torchdrug.utils import comm, pretty
from transformers import Trainer, TrainingArguments

from collector import MKGLDataCollector
from llm import KGL4IndKGC, KGL4KGC, MKGL, MKGLConfig
from main import (
    build_image_feature_bank,
    load_component_checkpoint,
    print_trainable_parameter_summary,
)
from preprocess import InductiveKGCDataset, KGCDataset, Prompter


# Older preprocessed dataset pickles may reference __main__.Prompter because
# they were created by running preprocess.py as a script.
setattr(sys.modules["__main__"], "Prompter", Prompter)


def build_config_name(config_path, dataset_cfg):
    config_name = config_path.split("/")[-1].split("\\")[-1].split(".")[0]
    if hasattr(dataset_cfg, "version"):
        config_name += "_" + dataset_cfg.version
    return config_name


def compute_metrics(predictions):
    ranking = predictions[0].astype(float)
    metrics = ("mr", "mrr", "hits@1", "hits@3", "hits@10")
    results = {}
    for metric in metrics:
        if metric == "mr":
            score = ranking.mean()
        elif metric == "mrr":
            score = (1 / ranking).mean()
        elif metric.startswith("hits@"):
            threshold = int(metric[5:])
            score = (ranking <= threshold).mean()
        else:
            raise ValueError("Unknown metric `%s`" % metric)
        results[metric] = float(score)

    if comm.get_rank() == 0:
        print("Checkpoint evaluation metrics:")
        print(results)
    return results


def freeze_all_parameters(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MKGL component checkpoint, optionally with diffusion."
    )
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint directory or pytorch_model.bin path.")
    parser.add_argument("--version", "-v", type=str, default="")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--eval-split", choices=("valid", "test"), default="test")
    parser.add_argument("--disable-image-features", action="store_true",
                        help="Ignore config.image_features for original text-only MKGL checks.")
    parser.add_argument("--with-diffusion", action="store_true",
                        help="Initialize diffusion from the config and evaluate refined scores.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
        if args.version:
            cfg.dataset.version = args.version

    torch.manual_seed(args.seed + comm.get_rank())
    config_name = build_config_name(args.config, cfg.dataset)
    cfg.trainer.output_dir += config_name + "_checkpoint_eval"

    if comm.get_rank() == 0:
        print("Config file: %s" % args.config)
        print("Checkpoint: %s" % args.checkpoint)
        print("Eval split: %s" % args.eval_split)
        if args.with_diffusion:
            print("Diffusion evaluation is enabled.")
        else:
            print("Diffusion is disabled for this eval run.")
        if args.disable_image_features:
            print("Image features are disabled for this eval run.")
        print(pretty.format(cfg))

    file_path = "data/preprocessed/" + config_name + ".pkl"
    if "ind" in config_name:
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
        **cfg.mkgl,
        device_map={"": Accelerator().process_index},
        config=config,
    )

    lora_config = LoraConfig(**cfg.loraconfig)
    model = get_peft_model(model, lora_config)

    vocab_df = dataset.vocab_df.sort_index()
    kgl2token = torch.tensor(
        np.stack(vocab_df.text_token_ids)[:, :cfg.kgl_token_length]
    )
    image_cfg = None if args.disable_image_features else getattr(cfg, "image_features", None)
    kgl_image_features, kgl_image_mask = build_image_feature_bank(vocab_df, image_cfg)
    model.init_kg_specs(
        kgl2token,
        tokenizer.vocab_size,
        cfg,
        image_features=kgl_image_features,
        image_feature_mask=kgl_image_mask,
    )

    if args.with_diffusion:
        diffusion_cfg = getattr(cfg, "diffusion", easydict.EasyDict())
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
            mode=str(getattr(diffusion_cfg, "mode", "denoiser")),
            score_weight=float(getattr(diffusion_cfg, "score_weight", 1.0)),
            eval_score_weight=float(getattr(diffusion_cfg, "eval_score_weight", 1.0)),
            loss_weight=float(getattr(diffusion_cfg, "loss_weight", 1.0)),
        )

    load_component_checkpoint(model, args.checkpoint, required=True)
    freeze_all_parameters(model)
    model.eval()

    if comm.get_rank() == 0:
        print_trainable_parameter_summary(model)

    if "ind" in args.config:
        task = KGL4IndKGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    else:
        task = KGL4KGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    task.eval()

    data_loader = MKGLDataCollector(dataset)
    training_args = TrainingArguments(**cfg.trainer)
    removed_columns = [
        "h_raw",
        "t_raw",
        "r_raw",
        "h_fine",
        "t_fine",
        "r_fine",
        "inv_r_fine",
    ]
    eval_data = dataset.test_data if args.eval_split == "test" else dataset.valid_data

    trainer = Trainer(
        model=task,
        args=training_args,
        eval_dataset=eval_data.remove_columns(removed_columns),
        data_collator=data_loader,
        compute_metrics=compute_metrics,
    )
    results = trainer.evaluate()
    if comm.get_rank() == 0:
        print("Trainer evaluate output:")
        print(results)


if __name__ == "__main__":
    main()
