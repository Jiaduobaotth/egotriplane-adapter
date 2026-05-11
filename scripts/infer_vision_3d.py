#!/usr/bin/env python3
"""Run inference with a trained Stage 1 model and save detection results.

Usage:
  python scripts/infer_vision_3d.py \
      --ckpt outputs/stage1_phase1/best.pt \
      --nusc_root ./data --nusc_version v1.0-mini \
      --out_dir outputs/inference/
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.vision_encoder import VisionEncoderWrapper
from egotriplane.triplane_adapter import EgoTriPlaneAdapter
from egotriplane.heads import CenterDetHead
from egotriplane.image_dataset import NuscImageDataset, image_collate_fn

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist", "barrier", "traffic_cone"]


def parse_args():
    p = argparse.ArgumentParser(description="Inference for Stage 1 3D detection")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--nusc_root", type=str, default="./data")
    p.add_argument("--nusc_version", type=str, default="v1.0-mini")
    p.add_argument("--split", type=str, default="val", help="train or val")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--backbone", type=str, default="qwen3vl_4b")
    p.add_argument("--max_cams", type=int, default=6)
    p.add_argument("--min_cams", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--aggregation", type=str, default="attention")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=0,
                   help="Max val samples (0=all)")
    p.add_argument("--score_thresh", type=float, default=0.1)
    p.add_argument("--max_dets", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="outputs/inference/")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--online", action="store_true", default=False)
    return p.parse_args()


def build_model(args, ckpt_state):
    """Rebuild model from checkpoint config."""
    # Read config from checkpoint
    vision_cfg = ckpt_state.get("config", {})
    backbone = vision_cfg.get("backbone", args.backbone)
    image_size = vision_cfg.get("image_size", args.image_size)
    hidden_dim = vision_cfg.get("hidden_dim", args.hidden_dim)
    aggregation = vision_cfg.get("aggregation", args.aggregation)

    vision_encoder = VisionEncoderWrapper(
        backbone=backbone,
        image_size=image_size,
        freeze=True,
        output_hidden_states=False,
        local_files_only=not args.online,
        device=args.device,
    )
    vision_encoder.load_state_dict(ckpt_state["vision_encoder"])
    vision_encoder.to(args.device)
    vision_encoder.eval()

    vision_hidden_dim = vision_encoder.get_hidden_dim()
    block_size = vision_encoder.patch_size * vision_encoder.temporal_patch_size

    adapter = EgoTriPlaneAdapter(
        feature_dim=vision_hidden_dim,
        hidden_dim=hidden_dim,
        aggregation=aggregation,
    )
    adapter.load_state_dict(ckpt_state["adapter"])
    adapter.to(args.device)
    adapter.eval()

    det_head = CenterDetHead(
        hidden_dim=hidden_dim,
        num_classes=args.num_classes,
    )
    det_head.load_state_dict(ckpt_state["det_head"])
    det_head.to(args.device)
    det_head.eval()

    return vision_encoder, adapter, det_head, block_size


def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    vision_encoder, adapter, det_head, block_size = build_model(args, ckpt)
    print(f"  Hidden dim: {vision_encoder.get_hidden_dim()}, block_size: {block_size}")

    # Dataset (val split, no augment, no dropout)
    print("Loading dataset...")
    ds = NuscImageDataset(
        nusc_root=args.nusc_root,
        nusc_version=args.nusc_version,
        split=args.split,
        image_size=args.image_size,
        camera_dropout=False,
        min_cameras=args.min_cams,
        max_cameras=args.max_cams,
        num_classes=args.num_classes,
        max_objects=args.max_objects,
        augment=False,
        patch_size=block_size,
    )
    if args.max_samples > 0:
        ds.samples = ds.samples[:args.max_samples]
    print(f"  {args.split} samples: {len(ds)}")

    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=image_collate_fn,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    all_predictions = []
    all_groundtruths = []

    for idx, sample in enumerate(tqdm(loader, desc="Inference")):
        images = sample["images"]
        intrinsics = sample["intrinsics"]
        extrinsics = sample["extrinsics"]
        cam_names = sample["camera_names"]
        image_sizes = sample["image_sizes"]
        sample_token = sample["sample_token"]

        # Stack images → vision encoder
        imgs_batch = torch.stack([img.to(device) for img in images], dim=0)
        with torch.no_grad():
            enc_out = vision_encoder(imgs_batch)
            all_feats = enc_out["last_hidden_state"]
            patch_grid = enc_out["patch_grid"]

        # Per-camera feature dicts
        features_by_camera = {}
        for i, cam_name in enumerate(cam_names):
            features_by_camera[cam_name] = {
                "features": all_feats[i],
                "K": intrinsics[i].to(device),
                "T_ego_cam": extrinsics[i].to(device),
                "image_size": image_sizes[i].tolist(),
                "patch_grid": list(patch_grid),
            }

        # Adapter → detection head
        with torch.no_grad():
            adapter_out = adapter(features_by_camera, cam_names)
            det_preds = det_head(adapter_out)
            decoded = det_head.decode_detections(
                det_preds,
                score_thresh=args.score_thresh,
                max_dets=args.max_dets,
            )

        # Collect predictions
        for b_idx, det in enumerate(decoded):
            pred = {
                "sample_token": sample_token if isinstance(sample_token, str) else sample_token[0],
                "boxes": det["boxes"].cpu().tolist(),
                "scores": det["scores"].cpu().tolist(),
                "classes": det["classes"].cpu().tolist(),
            }
            all_predictions.append(pred)

        # Collect GT
        gt_boxes = sample.get("gt_boxes_3d")
        gt_labels = sample.get("gt_labels")
        gt_mask = sample.get("gt_mask")
        if gt_boxes is not None and gt_mask is not None:
            mask = gt_mask[0] if gt_mask.dim() > 1 else gt_mask
            boxes = gt_boxes[0] if gt_boxes.dim() > 2 else gt_boxes
            labels = gt_labels[0] if gt_labels.dim() > 1 else gt_labels
            valid_boxes = boxes[mask].tolist() if mask.any() else []
            valid_labels = labels[mask].tolist() if mask.any() else []
            all_groundtruths.append({
                "sample_token": sample_token if isinstance(sample_token, str) else sample_token[0],
                "boxes": valid_boxes,
                "labels": valid_labels,
            })

    # Save results
    pred_path = os.path.join(args.out_dir, "predictions.json")
    gt_path = os.path.join(args.out_dir, "groundtruths.json")
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    with open(gt_path, "w") as f:
        json.dump(all_groundtruths, f, indent=2)
    print(f"Saved: {pred_path}")
    print(f"Saved: {gt_path}")
    print(f"Total predictions: {len(all_predictions)}")

    # Quick stats
    total_dets = sum(len(p["scores"]) for p in all_predictions)
    print(f"Avg detections per sample: {total_dets / max(len(all_predictions), 1):.1f}")

    return all_predictions, all_groundtruths


if __name__ == "__main__":
    run_inference(parse_args())
