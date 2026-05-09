#!/usr/bin/env python3
"""Extract frozen VLM vision features and cache them for training.

Usage:
    python scripts/extract_vlm_features.py \
        --qa outputs/ego3dqa/nusc_train_ego3dqa.jsonl \
        --index outputs/ego3dqa/nusc_train_index.jsonl \
        --model_name openai/clip-vit-large-patch14 \
        --nusc_root /data/nuscenes \
        --out outputs/features/nusc_train_clip_features/ \
        --batch_size 4 \
        --device cuda

For each (sample, camera), loads the image, runs through the frozen
vision encoder, and saves patch features + calibration to disk.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Set, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.nusc_utils import load_qa, load_index
from egotriplane.feature_cache import save_camera_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and cache frozen VLM vision features"
    )
    parser.add_argument("--qa", type=str, required=True,
                        help="QA JSONL with sample tokens")
    parser.add_argument("--index", type=str, required=True,
                        help="Sample index JSONL with calibrations")
    parser.add_argument("--model_name", type=str,
                        default="openai/clip-vit-large-patch14",
                        help="HuggingFace vision model name")
    parser.add_argument("--nusc_root", type=str, required=True,
                        help="Path to nuScenes data")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for cached features")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Resize images to this square size")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading vision model: {args.model_name}")
    model, processor, patch_grid = load_vision_model(args.model_name, args.image_size)
    model = model.to(device).eval()

    # Load data
    qa_list = load_qa(args.qa)
    samples = load_index(args.index)
    samples_by_token = {s["sample_token"]: s for s in samples}

    # Collect unique (sample_token, camera) pairs
    pairs: Set[tuple] = set()
    for qa in qa_list:
        token = qa["sample_token"]
        for cam_name in qa["camera_subset"]:
            # Check if already cached
            out_path = os.path.join(args.out, f"{token}_{cam_name}.pt")
            if not os.path.exists(out_path):
                pairs.add((token, cam_name))

    pairs = sorted(pairs)
    if args.limit_samples:
        pairs = pairs[:args.limit_samples]

    print(f"Extracting features for {len(pairs)} (sample, camera) pairs")

    # Process in batches
    batch = []
    batch_info = []

    for token, cam_name in tqdm(pairs, desc="Extracting features"):
        sample = samples_by_token.get(token)
        if sample is None:
            continue

        cam = next((c for c in sample["cameras"] if c["name"] == cam_name), None)
        if cam is None:
            continue

        img_path = os.path.join(args.nusc_root, cam["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"  Image not found: {img_path}")
            continue

        batch.append(img)
        batch_info.append((token, cam_name, cam))

        if len(batch) >= args.batch_size:
            _process_batch(batch, batch_info, model, processor,
                           patch_grid, args.image_size, args.out, device)
            batch = []
            batch_info = []

    # Process remaining
    if batch:
        _process_batch(batch, batch_info, model, processor,
                       patch_grid, args.image_size, args.out, device)

    print(f"Done. Features saved to {args.out}")


def load_vision_model(model_name: str, image_size: int = 224):
    """Load vision model and return (model, processor, patch_grid).

    Supported: CLIP, SigLIP.
    """
    if "clip" in model_name.lower():
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model = CLIPVisionModel.from_pretrained(model_name)
        processor = CLIPImageProcessor.from_pretrained(model_name)

        # Compute patch grid size
        # CLIP ViT-L/14: image_size=224, patch_size=14 -> 16x16 grid
        # + 1 CLS token
        patch_size = model.config.patch_size
        grid_size = image_size // patch_size
        patch_grid = (grid_size, grid_size)

    elif "siglip" in model_name.lower():
        from transformers import SiglipVisionModel, SiglipImageProcessor
        model = SiglipVisionModel.from_pretrained(model_name)
        processor = SiglipImageProcessor.from_pretrained(model_name)

        patch_size = model.config.patch_size
        grid_size = image_size // patch_size
        patch_grid = (grid_size, grid_size)

    else:
        raise ValueError(f"Unsupported model: {model_name}. Use CLIP or SigLIP.")

    return model, processor, patch_grid


def _process_batch(images, batch_info, model, processor,
                    patch_grid, image_size, out_dir, device):
    """Extract features for one batch and save."""
    # Preprocess
    inputs = processor(
        images=images,
        return_tensors="pt",
    )
    # Resize if needed
    if "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"].to(device)
    else:
        # Fallback: manual preprocessing
        pixel_values = _manual_preprocess(images, image_size).to(device)

    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)

        # Get patch features (last hidden state without CLS token)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state  # [B, N+1, D]
        else:
            hidden = outputs.hidden_states[-1]

        # Remove CLS token
        patch_features = hidden[:, 1:, :]  # [B, N_patches, D]

    # Save each sample
    for idx, (token, cam_name, cam) in enumerate(batch_info):
        features = patch_features[idx].cpu()
        K = torch.tensor(cam["K"], dtype=torch.float32)
        T_ego_cam = torch.tensor(cam["T_ego_cam"], dtype=torch.float32)

        out_path = os.path.join(out_dir, f"{token}_{cam_name}.pt")
        save_camera_features(
            features=features,
            patch_grid=patch_grid,
            K=K,
            T_ego_cam=T_ego_cam,
            image_size=(image_size, image_size),
            output_path=out_path,
        )


def _manual_preprocess(images: list, image_size: int) -> torch.Tensor:
    """Manual image preprocessing for CLIP/SigLIP."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    batch = torch.stack([transform(img) for img in images])
    return batch


if __name__ == "__main__":
    main()
