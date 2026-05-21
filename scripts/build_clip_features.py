import argparse
import io
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def bootstrap_hf_cache() -> None:
    default_cache = Path("data/hf_cache")
    default_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(default_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(default_cache / "hub"))


bootstrap_hf_cache()

from transformers import CLIPModel


DEFAULT_URL_FILES = (
    "image-graph_urls/URLS_google.txt",
    "image-graph_urls/URLS_bing.txt",
    "image-graph_urls/URLS_yahoo.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-entity CLIP image features for FB15K entities."
    )
    parser.add_argument(
        "--entity-file",
        type=str,
        default="data/names/fb15k237/entity.txt",
        help="TSV file whose first column is the raw entity id (for example /m/010016).",
    )
    parser.add_argument(
        "--url-files",
        type=str,
        nargs="+",
        default=list(DEFAULT_URL_FILES),
        help="URL files in the existing image-graph_urls format.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/image_features/fb15k237_clip.pt",
        help="Path to the output torch checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CLIP encoding.",
    )
    parser.add_argument(
        "--num-urls-per-entity",
        type=int,
        default=1,
        help="Maximum number of URLs to try per entity after deduplication.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=10.0,
        help="Per-image download timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries per URL after the first failed attempt.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/hf_cache",
        help="Writable Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for CLIP inference.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run CLIP inference in fp16 when CUDA is available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate entity / URL alignment and print summary stats.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def raw_entity_to_url_key(raw_name: str) -> str:
    return raw_name.strip().lstrip("/").replace("/", ".")


def read_entity_ids(entity_file: Path, max_entities: Optional[int] = None) -> List[str]:
    entity_ids: List[str] = []
    with entity_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            raw_name = line.split("\t", 1)[0]
            entity_ids.append(raw_name)
            if max_entities is not None and len(entity_ids) >= max_entities:
                break
    if not entity_ids:
        raise ValueError(f"No entities found in {entity_file}")
    return entity_ids


def read_urls(
    url_files: Sequence[Path],
    num_urls_per_entity: int,
) -> Dict[str, List[str]]:
    urls_by_entity: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)

    for url_file in url_files:
        with url_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                url, entity_with_rank = parts
                entity_key = entity_with_rank.rsplit("/", 1)[0]
                if url in seen[entity_key]:
                    continue
                if len(urls_by_entity[entity_key]) >= num_urls_per_entity:
                    continue
                seen[entity_key].add(url)
                urls_by_entity[entity_key].append(url)
    return urls_by_entity


def pick_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def set_hf_cache(cache_dir: Path) -> None:
    ensure_parent(cache_dir / "placeholder")
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def load_clip_model(model_name: str, device: str, fp16: bool) -> CLIPModel:
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    if fp16 and device == "cuda":
        model = model.half()

    return model


def fetch_image(url: str, timeout: float, max_retries: int) -> Optional[Image.Image]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    request = urllib.request.Request(url, headers=headers)
    context = ssl.create_default_context()

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                payload = response.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            return image
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            OSError,
        ):
            if attempt >= max_retries:
                return None
            time.sleep(0.2)
    return None


def encode_images(
    model: CLIPModel,
    pixel_batches: List[Tuple[int, torch.Tensor]],
    batch_size: int,
    device: str,
    fp16: bool,
    progress_desc: Optional[str] = None,
) -> List[Tuple[int, torch.Tensor]]:
    features: List[Tuple[int, torch.Tensor]] = []

    starts = range(0, len(pixel_batches), batch_size)
    if progress_desc is not None:
        starts = tqdm(
            starts,
            total=(len(pixel_batches) + batch_size - 1) // batch_size,
            desc=progress_desc,
            leave=False,
        )

    for start in starts:
        chunk = pixel_batches[start : start + batch_size]
        entity_indices = [item[0] for item in chunk]
        pixel_values = torch.stack([item[1] for item in chunk], dim=0).to(device)
        if fp16 and device == "cuda":
            pixel_values = pixel_values.half()

        with torch.no_grad():
            batch_features = model.get_image_features(pixel_values=pixel_values)
            batch_features = F.normalize(batch_features.float(), p=2, dim=-1)

        for entity_index, feature in zip(entity_indices, batch_features.cpu()):
            features.append((entity_index, feature))

    return features


def main() -> None:
    args = parse_args()

    entity_file = Path(args.entity_file)
    url_files = [Path(path) for path in args.url_files]
    output_file = Path(args.output_file)
    cache_dir = Path(args.cache_dir)

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_file} already exists. Pass --overwrite to replace it."
        )

    for path in [entity_file, *url_files]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file does not exist: {path}")

    entity_ids = read_entity_ids(entity_file, max_entities=args.max_entities)
    urls_by_entity = read_urls(url_files, args.num_urls_per_entity)

    matched_entities = 0
    urls_found = 0
    entity_to_urls: List[List[str]] = []
    for raw_name in entity_ids:
        url_key = raw_entity_to_url_key(raw_name)
        urls = urls_by_entity.get(url_key, [])
        entity_to_urls.append(urls)
        if urls:
            matched_entities += 1
            urls_found += len(urls)

    summary = {
        "num_entities": len(entity_ids),
        "entities_with_urls": matched_entities,
        "entities_without_urls": len(entity_ids) - matched_entities,
        "selected_urls": urls_found,
        "num_urls_per_entity": args.num_urls_per_entity,
    }

    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return

    device = pick_device(args.device)
    use_fp16 = bool(args.fp16 and device == "cuda")
    set_hf_cache(cache_dir)

    preprocess = build_preprocess()
    model = load_clip_model(args.model_name, device=device, fp16=use_fp16)

    pending_pixels: List[Tuple[int, torch.Tensor]] = []
    accumulated_features: Dict[int, List[torch.Tensor]] = defaultdict(list)
    num_download_failures = 0

    entity_progress = tqdm(
        enumerate(entity_to_urls),
        total=len(entity_to_urls),
        desc="Fetching images",
    )
    num_downloaded_images = 0
    for entity_index, urls in entity_progress:
        for url in urls:
            image = fetch_image(
                url=url,
                timeout=args.download_timeout,
                max_retries=args.max_retries,
            )
            if image is None:
                num_download_failures += 1
                continue
            pending_pixels.append((entity_index, preprocess(image)))
            num_downloaded_images += 1
            entity_progress.set_postfix(
                downloaded=num_downloaded_images,
                queued=len(pending_pixels),
                failed=num_download_failures,
            )

            if len(pending_pixels) >= args.batch_size:
                encoded = encode_images(
                    model=model,
                    pixel_batches=pending_pixels,
                    batch_size=args.batch_size,
                    device=device,
                    fp16=use_fp16,
                    progress_desc="Encoding images",
                )
                for idx, feat in encoded:
                    accumulated_features[idx].append(feat)
                entity_progress.set_postfix(
                    downloaded=num_downloaded_images,
                    queued=0,
                    encoded=len(accumulated_features),
                    failed=num_download_failures,
                )
                pending_pixels.clear()

    if pending_pixels:
        encoded = encode_images(
            model=model,
            pixel_batches=pending_pixels,
            batch_size=args.batch_size,
            device=device,
            fp16=use_fp16,
            progress_desc="Encoding final batch",
        )
        for idx, feat in encoded:
            accumulated_features[idx].append(feat)

    feature_dim = int(model.visual_projection.out_features)
    features = torch.zeros(len(entity_ids), feature_dim, dtype=torch.float32)
    has_image = torch.zeros(len(entity_ids), dtype=torch.bool)
    used_urls = torch.zeros(len(entity_ids), dtype=torch.int32)

    for entity_index, feats in accumulated_features.items():
        stacked = torch.stack(feats, dim=0)
        pooled = F.normalize(stacked.mean(dim=0, keepdim=False), p=2, dim=-1)
        features[entity_index] = pooled
        has_image[entity_index] = True
        used_urls[entity_index] = len(feats)

    payload = {
        "model_name": args.model_name,
        "feature_dim": feature_dim,
        "entity_ids": entity_ids,
        "features": features,
        "has_image": has_image,
        "used_urls": used_urls,
        "requested_num_urls_per_entity": args.num_urls_per_entity,
        "num_download_failures": num_download_failures,
        "source_url_files": [str(path) for path in url_files],
        "entity_file": str(entity_file),
    }

    ensure_parent(output_file)
    torch.save(payload, output_file)

    final_stats = {
        **summary,
        "device": device,
        "fp16": use_fp16,
        "feature_dim": feature_dim,
        "entities_with_features": int(has_image.sum().item()),
        "download_failures": num_download_failures,
        "output_file": str(output_file),
    }
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    main()
