#!/usr/bin/env python3
"""Extract multi-camera CLIP features for Stage 1 training.

Reads image paths from index file, extracts CLIP ViT-L features for each camera,
and saves them as .pt files organized by sample_token.

Usage:
  python scripts/extract_features.py \
    --index outputs/ego3dqa/nusc_train_index.jsonl \
    --data_root data/v1.0-mini \
    --output_dir outputs/features/nusc_train_clip_features \
    --batch_size 8 \
    --num_workers 4
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)


class ImageDataset(Dataset):
    """Load images from index file."""
    def __init__(self, index_file, data_root, transform=None):
        self.data_root = Path(data_root)
        self.samples = []
        self.transform = transform

        with open(index_file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample_token = sample['sample_token']
                for cam in sample.get('cameras', []):
                    img_path = cam['image_path']
                    self.samples.append({
                        'sample_token': sample_token,
                        'camera_name': cam['name'],
                        'image_path': img_path,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = self.data_root / item['image_path']

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {
                'sample_token': item['sample_token'],
                'camera_name': item['camera_name'],
                'image': image,
            }
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None


def collate_fn(batch):
    """Filter out None items."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    return {
        'sample_token': [x['sample_token'] for x in batch],
        'camera_name': [x['camera_name'] for x in batch],
        'image': torch.stack([x['image'] for x in batch]),
    }


def extract_features(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load CLIP model
    print("Loading CLIP ViT-L model...")
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    # Create dataset
    print(f"Loading index from {args.index}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    dataset = ImageDataset(args.index, args.data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Extract features
    features_by_sample = defaultdict(dict)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            if batch is None:
                continue

            images = batch['image'].to(device)
            sample_tokens = batch['sample_token']
            camera_names = batch['camera_name']

            # Extract image features
            outputs = model.vision_model(
                pixel_values=images,
                output_hidden_states=True,
            )
            # Use pooled output (CLS token)
            image_features = outputs.pooler_output  # [B, 1024]

            for i, (sample_token, camera_name) in enumerate(zip(sample_tokens, camera_names)):
                features_by_sample[sample_token][camera_name] = image_features[i].cpu()

    # Save features
    print(f"\nSaving {len(features_by_sample)} samples...")
    for sample_token, cameras in tqdm(features_by_sample.items(), desc="Saving"):
        sample_dir = output_dir / sample_token
        sample_dir.mkdir(exist_ok=True)

        for camera_name, features in cameras.items():
            # Sanitize camera name (remove spaces, special chars)
            safe_name = camera_name.replace(' ', '_').lower()
            feature_path = sample_dir / f"{safe_name}.pt"
            torch.save(features, feature_path)

    print(f"\nDone! Extracted features for {len(features_by_sample)} samples")
    print(f"Features saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP features for training")
    parser.add_argument("--index", type=str, required=True,
                        help="Path to index JSONL file")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to nuScenes data root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for features")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    args = parser.parse_args()
    extract_features(args)
