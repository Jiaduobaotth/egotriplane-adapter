#!/usr/bin/env python3
"""Diagnose whether vision features vary across samples or if adapter
has collapsed to a static prior."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch.utils.data import DataLoader
from egotriplane.vision_encoder import VisionEncoderWrapper
from egotriplane.triplane_adapter import EgoTriPlaneAdapter
from egotriplane.heads import CenterDetHead
from egotriplane.image_dataset import NuscImageDataset, image_collate_fn

CKPT = "outputs/stage1_phase1/best.pt"
NUSC_ROOT = "/home/ubuntu/qhr/nuscence/"
NUSC_VERSION = "v1.0-trainval"
IMAGE_SIZE = 448
BACKBONE = "qwen3vl_4b"
DEVICE = "cuda"

print("Loading checkpoint...")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)

vision_encoder = VisionEncoderWrapper(
    backbone=BACKBONE, image_size=IMAGE_SIZE, freeze=True,
    output_hidden_states=False, local_files_only=True, device=DEVICE,
)
vision_encoder.load_state_dict(ckpt["vision_encoder"])
vision_encoder.to(DEVICE).eval()

block_size = vision_encoder.patch_size * vision_encoder.temporal_patch_size
hidden_dim = vision_encoder.get_hidden_dim()

adapter = EgoTriPlaneAdapter(feature_dim=hidden_dim, hidden_dim=512, aggregation="attention")
adapter.load_state_dict(ckpt["adapter"])
adapter.to(DEVICE).eval()

det_head = CenterDetHead(hidden_dim=512, num_classes=5)
det_head.load_state_dict(ckpt["det_head"])
det_head.to(DEVICE).eval()

print("Loading data...")
ds = NuscImageDataset(
    nusc_root=NUSC_ROOT, nusc_version=NUSC_VERSION, split="val",
    image_size=IMAGE_SIZE, camera_dropout=False,
    min_cameras=3, max_cameras=6, num_classes=5, max_objects=50,
    augment=False, patch_size=block_size,
)

loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    pin_memory=True, collate_fn=image_collate_fn)

# Collect vision features and adapter outputs for 5 samples
vision_feats_list = []
triplane_tokens_list = []
plane_xy_list = []

for idx, sample in enumerate(loader):
    if idx >= 5:
        break

    images = sample["images"]
    intrinsics = sample["intrinsics"]
    extrinsics = sample["extrinsics"]
    cam_names = sample["camera_names"]
    image_sizes = sample["image_sizes"]

    imgs_batch = torch.stack([img.to(DEVICE) for img in images], dim=0)
    with torch.no_grad():
        enc_out = vision_encoder(imgs_batch)
        all_feats = enc_out["last_hidden_state"]
        patch_grid = enc_out["patch_grid"]

    vision_feats_list.append(all_feats.clone())

    features_by_camera = {}
    for i, cn in enumerate(cam_names):
        features_by_camera[cn] = {
            "features": all_feats[i],
            "K": intrinsics[i].to(DEVICE),
            "T_ego_cam": extrinsics[i].to(DEVICE),
            "image_size": image_sizes[i].tolist(),
            "patch_grid": list(patch_grid),
        }

    with torch.no_grad():
        adapter_out = adapter(features_by_camera, cam_names)
        triplane_tokens_list.append(adapter_out["triplane_tokens"].clone())
        plane_xy_list.append(adapter_out["plane_xy"].clone())

# Compare
print("\n=== Vision Encoder Features ===")
for i in range(len(vision_feats_list)):
    f = vision_feats_list[i]
    print(f"  Sample {i}: shape={list(f.shape)}, mean={f.mean().item():.4f}, std={f.std().item():.4f}, "
          f"min={f.min().item():.3f}, max={f.max().item():.3f}")

print("\n=== Pairwise Cosine Similarity (vision features) ===")
for i in range(len(vision_feats_list)):
    for j in range(i + 1, len(vision_feats_list)):
        # Compare first camera's features
        fi = vision_feats_list[i][0].flatten()
        fj = vision_feats_list[j][0].flatten()
        cos = torch.nn.functional.cosine_similarity(fi.unsqueeze(0), fj.unsqueeze(0)).item()
        print(f"  Sample {i} vs {j}: cosine_sim = {cos:.4f}")

print("\n=== Triplane Tokens ===")
for i in range(len(triplane_tokens_list)):
    t = triplane_tokens_list[i]
    print(f"  Sample {i}: shape={list(t.shape)}, mean={t.mean().item():.4f}, std={t.std().item():.4f}")

print("\n=== Pairwise Cosine Similarity (triplane tokens) ===")
for i in range(len(triplane_tokens_list)):
    for j in range(i + 1, len(triplane_tokens_list)):
        ti = triplane_tokens_list[i].flatten()
        tj = triplane_tokens_list[j].flatten()
        cos = torch.nn.functional.cosine_similarity(ti.unsqueeze(0), tj.unsqueeze(0)).item()
        print(f"  Sample {i} vs {j}: cosine_sim = {cos:.4f}")

print("\n=== Plane XY ===")
for i in range(len(plane_xy_list)):
    p = plane_xy_list[i]
    print(f"  Sample {i}: mean={p.mean().item():.4f}, std={p.std().item():.4f}")

print("\n=== Pairwise Cosine Similarity (plane XY) ===")
for i in range(len(plane_xy_list)):
    for j in range(i + 1, len(plane_xy_list)):
        pi = plane_xy_list[i].flatten()
        pj = plane_xy_list[j].flatten()
        cos = torch.nn.functional.cosine_similarity(pi.unsqueeze(0), pj.unsqueeze(0)).item()
        print(f"  Sample {i} vs {j}: cosine_sim = {cos:.4f}")

# Check learnable embedding vs actual plane
print("\n=== Learnable Embedding vs Output Plane ===")
xy_embed = adapter.xy_embed  # [sy, sx, hidden_dim]
for i in range(len(plane_xy_list)):
    p = plane_xy_list[i]  # [1, C, sy, sx]
    p_flat = p.squeeze(0).permute(1, 2, 0)  # [sy, sx, C]
    cos = torch.nn.functional.cosine_similarity(
        xy_embed.reshape(-1), p_flat.reshape(-1), dim=0
    ).item()
    print(f"  Sample {i}: cos_sim between learnable embedding and output plane = {cos:.4f}")
