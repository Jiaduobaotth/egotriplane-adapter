#!/usr/bin/env python3
"""Train vision encoder for 3D perception (Stage 1: Adapter 3D Pretrain).

End-to-end training: vision encoder → multi-view adapter → detection head → loss.

Key features:
  - Pluggable vision backbone (torchvision ViT local, CLIP/Qwen VL when available)
  - Multi-view fusion via EgoTriPlaneAdapter
  - 3D center-based detection (heatmap + offset + size + yaw)
  - Optional BEV semantic segmentation
  - Memory-efficient: processes cameras sequentially, fp16, grad accumulation
  - Fits 12GB VRAM with ViT-B/16 backbone

Usage (local debug):
  python scripts/train_vision_3d.py \
      --backbone tv_vit_b_16 \
      --image_size 224 \
      --batch_size 1 --grad_accum 4 \
      --epochs 5 \
      --out_dir outputs/vision3d_debug/

Usage (production with Qwen VL, when model is downloaded):
  python scripts/train_vision_3d.py \
      --backbone qwen25vl_3b \
      --image_size 448 \
      --freeze_vision_until 18 \
      --batch_size 1 --grad_accum 8 \
      --epochs 20
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.vision_encoder import VisionEncoderWrapper
from egotriplane.triplane_adapter import EgoTriPlaneAdapter
from egotriplane.heads import CenterDetHead, BEVSegHead
from egotriplane.losses import Adapter3DPretrainLoss
from egotriplane.image_dataset import NuscImageDataset, image_collate_fn


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train vision encoder for 3D perception (Stage 1)"
    )
    # Model
    p.add_argument("--backbone", type=str, default="tv_vit_b_16",
                   choices=["tv_vit_b_16", "tv_vit_l_16", "clip_vit_base",
                            "clip_vit_large", "qwen25vl_3b", "qwen25vl_7b",
                            "qwen3vl_4b", "qwen3vl_8b"],
                   help="Vision backbone")
    p.add_argument("--image_size", type=int, default=224,
                   help="Input image resolution")
    p.add_argument("--aggregation", type=str, default="attention",
                   choices=["mean", "max", "attention"],
                   help="Multi-camera triplane fusion: mean, max, or attention")
    p.add_argument("--hidden_dim", type=int, default=512,
                   help="Adapter hidden dim")
    p.add_argument("--freeze_vision", action="store_true", default=False,
                   help="Freeze vision backbone completely")
    p.add_argument("--freeze_vision_until", type=int, default=0,
                   help="Freeze first N ViT layers, train the rest")
    p.add_argument("--online", action="store_true", default=False,
                   help="Allow downloading models from internet (default: offline, use local cache)")

    # Data
    p.add_argument("--nusc_root", type=str, default="./data")
    p.add_argument("--nusc_version", type=str, default="v1.0-mini")
    p.add_argument("--min_cams", type=int, default=3)
    p.add_argument("--max_cams", type=int, default=6)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--max_objects", type=int, default=50)

    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--no_augment", action="store_true", default=False)

    # Loss weights
    p.add_argument("--lambda_det", type=float, default=1.0)
    p.add_argument("--lambda_bev", type=float, default=0.0,
                   help="BEV seg loss weight (0=disabled)")
    p.add_argument("--w_heatmap", type=float, default=1.0)
    p.add_argument("--w_offset", type=float, default=0.1)
    p.add_argument("--w_size", type=float, default=0.1)
    p.add_argument("--w_yaw", type=float, default=0.1)
    p.add_argument("--w_z", type=float, default=0.1)

    # Checkpointing & logging
    p.add_argument("--out_dir", type=str, default="outputs/vision3d_debug/")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ============================================================
    # Sanity print
    # ============================================================
    print("\n" + "=" * 65)
    print(f"  [Stage] vision_encoder_3d_pretrain")
    print(f"  Backbone:        {args.backbone}")
    print(f"  Image size:      {args.image_size}")
    print(f"  Freeze vision:   {args.freeze_vision}")
    print(f"  Freeze until:    {args.freeze_vision_until}")
    print(f"  Batch/accum:     {args.batch_size}/{args.grad_accum}")
    print(f"  Loss:            det={args.lambda_det}, bev_seg={args.lambda_bev}")
    print(f"  Detection:       heatmap + offset + size + yaw")
    print("=" * 65 + "\n")

    # ============================================================
    # Build model
    # ============================================================
    print("Building vision encoder...")
    vision_encoder = VisionEncoderWrapper(
        backbone=args.backbone,
        image_size=args.image_size,
        freeze=args.freeze_vision,
        freeze_until_layer=args.freeze_vision_until,
        output_hidden_states=False,
        local_files_only=not args.online,
    )
    vision_hidden_dim = vision_encoder.get_hidden_dim()
    patch_grid = vision_encoder.get_grid_size()
    print(f"  Hidden dim: {vision_hidden_dim}, patch grid: {patch_grid}")

    print("Building EgoTriPlaneAdapter...")
    adapter = EgoTriPlaneAdapter(
        feature_dim=vision_hidden_dim,
        hidden_dim=args.hidden_dim,
        x_range=(-20.0, 80.0),
        y_range=(-40.0, 40.0),
        z_range=(-3.0, 8.0),
        sx=96, sy=96, sz=48,
        patch_size=8,
        use_ray_embedding=True,
        aggregation=args.aggregation,
    )

    print("Building detection head...")
    det_head = CenterDetHead(
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        grid_sx=96, grid_sy=96,
        patch_size=8,
        x_range=(-20.0, 80.0),
        y_range=(-40.0, 40.0),
        use_objectness=True,
    )

    # Move to device
    vision_encoder.to(device)
    adapter.to(device)
    det_head.to(device)

    # ============================================================
    # Print trainable params
    # ============================================================
    total_p = 0
    trainable_p = 0
    print("\n--- Model Parameters ---")
    for mod_name, mod in [("vision_encoder", vision_encoder),
                           ("adapter", adapter),
                           ("det_head", det_head)]:
        mod_total = sum(p.numel() for p in mod.parameters())
        mod_train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        total_p += mod_total
        trainable_p += mod_train
        print(f"  {mod_name:<25s}: {mod_total:>10,} total, {mod_train:>10,} trainable")
    print(f"  {'TOTAL':<25s}: {total_p:>10,} total, {trainable_p:>10,} trainable "
          f"({trainable_p/max(total_p,1):.1%})")
    print()

    # ============================================================
    # Loss
    # ============================================================
    criterion = Adapter3DPretrainLoss(
        lambda_det=args.lambda_det,
        lambda_bev=args.lambda_bev,
        lambda_occ=0.0,
        det_loss_cfg={
            "w_heatmap": args.w_heatmap,
            "w_offset": args.w_offset,
            "w_size": args.w_size,
            "w_yaw": args.w_yaw,
            "w_z": args.w_z,
            "num_classes": args.num_classes,
            "use_focal": True,
            "grid_sx": 96, "grid_sy": 96,
            "bev_x_range": (-20.0, 80.0),
            "bev_y_range": (-40.0, 40.0),
        },
    )

    # ============================================================
    # Optimizer
    # ============================================================
    trainable = []
    for mod in [vision_encoder, adapter, det_head]:
        for p in mod.parameters():
            if p.requires_grad:
                trainable.append(p)

    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    # ============================================================
    # Dataset
    # ============================================================
    print("Loading dataset...")
    train_ds = NuscImageDataset(
        nusc_root=args.nusc_root,
        nusc_version=args.nusc_version,
        split="train",
        image_size=args.image_size,
        camera_dropout=True,
        min_cameras=args.min_cams,
        max_cameras=args.max_cams,
        num_classes=args.num_classes,
        max_objects=args.max_objects,
        augment=not args.no_augment,
    )
    print(f"  Train samples: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=image_collate_fn,
    )

    # Scheduler: use actual optimizer steps per epoch
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup / max(total_steps, 1),
    )
    print(f"  Scheduler: total_steps={total_steps}, warmup={args.warmup}")

    # ============================================================
    # Checkpoint dir
    # ============================================================
    os.makedirs(args.out_dir, exist_ok=True)

    # ============================================================
    # Training loop
    # ============================================================
    best_loss = float("inf")
    global_step = 0

    print(f"\nStarting training: {args.epochs} epochs, "
          f"batch={args.batch_size}, accum={args.grad_accum}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}")
    print(f"Device: {device}\n")

    for epoch in range(args.epochs):
        vision_encoder.train()
        adapter.train()
        det_head.train()

        epoch_losses = defaultdict(float)
        epoch_total = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        batches_since_log = 0
        for batch_idx, batch in enumerate(pbar):
            # Handle collated batch (single sample or list)
            if isinstance(batch, list):
                sample = batch[0]
            else:
                sample = batch

            # ====================================================
            # Step 1: Extract vision features for ALL cameras in one batched forward
            # ====================================================
            images = sample["images"]           # list of [3, H, W]
            intrinsics = sample["intrinsics"]   # list of [3, 3]
            extrinsics = sample["extrinsics"]   # list of [4, 4]
            cam_names = sample["camera_names"]
            image_sizes = sample["image_sizes"]

            # Stack all camera images into one batch: [N_cam, 3, H, W]
            imgs_batch = torch.stack([img.to(device) for img in images], dim=0)
            enc_out = vision_encoder(imgs_batch)
            all_feats = enc_out["last_hidden_state"]  # [N_cam, N_patches, D]

            # Split back into per-camera dict (no detach — grad flows through)
            features_by_camera = {}
            for i, cam_name in enumerate(cam_names):
                features_by_camera[cam_name] = {
                    "features": all_feats[i].detach() if args.freeze_vision else all_feats[i],
                    "K": intrinsics[i].to(device),
                    "T_ego_cam": extrinsics[i].to(device),
                    "image_size": image_sizes[i].tolist(),
                    "patch_grid": list(vision_encoder.get_grid_size()),
                }

            # ====================================================
            # Step 2: Adapter forward (fp32 - small module, NaN-safe)
            # ====================================================
            adapter_out = adapter(features_by_camera, cam_names)

            # ====================================================
            # Step 3: Detection head (fp32)
            # ====================================================
            det_preds = det_head(adapter_out)

            # ====================================================
            # Step 4: Loss (fp32)
            # ====================================================
            det_targets = {
                "gt_hm": sample["gt_hm"].to(device),
                "gt_offset": sample["gt_offset"].to(device),
                "gt_size": sample["gt_size"].to(device),
                "gt_yaw": sample["gt_yaw"].to(device),
                "gt_z": sample["gt_z"].to(device),
                "gt_mask": sample["gt_mask"].to(device),
                "gt_obj_idx": sample["gt_obj_idx"].to(device),
            }

            losses = criterion(det_preds=det_preds, det_targets=det_targets)
            loss = losses["total"] / args.grad_accum

            # ====================================================
            # Step 5: Backward
            # ====================================================
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_total += losses["total"].item() if isinstance(losses["total"], torch.Tensor) else float(losses["total"])
                epoch_steps += 1

            # ====================================================
            # Logging
            # ====================================================
            for k, v in losses.items():
                epoch_losses[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
            batches_since_log += 1

            if global_step > 0 and global_step % args.log_every == 0:
                n = max(batches_since_log, 1)
                avg = {k: v / n for k, v in epoch_losses.items()}
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg.get('total', 0):.3f}",
                    "det": f"{avg.get('det_total', 0):.3f}",
                    "lr": f"{lr:.1e}",
                })

                # First step sanity check
                if global_step == args.log_every:
                    _sanity_print(sample, losses, cam_names, args)
                epoch_losses = defaultdict(float)
                batches_since_log = 0

        # --- End of epoch ---
        avg_loss = epoch_total / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            _save_ckpt(args.out_dir, epoch, vision_encoder, adapter, det_head,
                       optimizer, scheduler, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_ckpt(args.out_dir, "best", vision_encoder, adapter, det_head,
                       optimizer, scheduler, avg_loss)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Output: {args.out_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanity_print(sample, losses, cam_names, args):
    """Print batch details for verification."""
    print("\n--- First Step Sanity Check ---")
    imgs = sample.get("images", [])
    if imgs:
        print(f"  Images per camera: {imgs[0].shape}")
        print(f"  Image value range: [{imgs[0].min().item():.2f}, {imgs[0].max().item():.2f}]")
    print(f"  Cameras ({len(cam_names)}): {cam_names}")
    n_obj = sample["gt_mask"].sum().item() if sample.get("gt_mask") is not None else 0
    print(f"  GT objects: {int(n_obj)}")
    hm = sample.get("gt_hm")
    if hm is not None:
        print(f"  GT heatmap: shape={list(hm.shape)}, nonzero={hm.sum().item():.1f}")
    print("  Losses:")
    for k, v in sorted(losses.items()):
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"    {k}: {val:.4f}")
    print(f"  GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB allocated")

    # Check coordinate system
    boxes = sample.get("gt_boxes_3d")
    if boxes is not None and n_obj > 0:
        mask = sample["gt_mask"]
        valid_boxes = boxes[mask][:5]
        print(f"  Sample GT boxes (cx,cy,cz,w,l,h,yaw) - first {len(valid_boxes)}:")
        for b in valid_boxes:
            print(f"    [{b[0]:6.1f}, {b[1]:6.1f}, {b[2]:5.1f}, "
                  f"{b[3]:4.1f}, {b[4]:4.1f}, {b[5]:4.1f}, {b[6]:5.2f}]")
    print("-------------------------------\n")


def _save_ckpt(out_dir, epoch, vision_encoder, adapter, det_head,
               optimizer, scheduler, loss):
    path = os.path.join(out_dir,
                         f"ckpt_epoch{epoch+1}.pt" if isinstance(epoch, int) else f"{epoch}.pt")
    torch.save({
        "epoch": epoch if isinstance(epoch, int) else -1,
        "vision_encoder": vision_encoder.state_dict(),
        "adapter": adapter.state_dict(),
        "det_head": det_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
    }, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
