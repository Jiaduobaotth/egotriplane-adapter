#!/usr/bin/env python3
"""Quick feature extraction for nuScenes mini (test/debug)."""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
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
    def __init__(self, index_file, data_root, max_samples=None, transform=None):
        self.data_root = Path(data_root)
        self.samples = []
        self.transform = transform

        with open(index_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
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
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return {
        'sample_token': [x['sample_token'] for x in batch],
        'camera_name': [x['camera_name'] for x in batch],
        'image': torch.stack([x['image'] for x in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="outputs/ego3dqa/nusc_train_index.jsonl")
    parser.add_argument("--data_root", type=str, default="data/v1.0-mini")
    parser.add_argument("--output_dir", type=str, default="outputs/features/nusc_train_clip_features")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to extract (for testing)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load CLIP
    print("Loading CLIP ViT-L...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    model.eval()

    # Dataset
    print(f"Loading {args.max_samples} samples from {args.index}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    dataset = ImageDataset(args.index, args.data_root, max_samples=args.max_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn)

    # Extract
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_by_sample = defaultdict(dict)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            if batch is None:
                continue
            images = batch['image'].to(device)
            outputs = model.vision_model(pixel_values=images, output_hidden_states=True)
            image_features = outputs.pooler_output

            for i, (token, cam_name) in enumerate(zip(batch['sample_token'], batch['camera_name'])):
                features_by_sample[token][cam_name] = image_features[i].cpu()

    # Save
    print(f"Saving {len(features_by_sample)} samples...")
    for token, cameras in tqdm(features_by_sample.items(), desc="Saving"):
        sample_dir = output_dir / token
        sample_dir.mkdir(exist_ok=True)
        for cam_name, feat in cameras.items():
            safe_name = cam_name.replace(' ', '_').lower()
            torch.save(feat, sample_dir / f"{safe_name}.pt")

    print(f"Done! Saved to {output_dir}")
    print(f"Sample structure: {list(features_by_sample.keys())[0] if features_by_sample else 'N/A'}")


if __name__ == "__main__":
    main()
