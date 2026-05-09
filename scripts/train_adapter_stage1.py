#!/usr/bin/env python3
"""Train EgoTriPlane-Adapter Stage 1: 3D Perception Pretraining.

Trains the multi-view visual adapter with 3D perception supervision ONLY
(no VQA / language generation loss).

Training objectives:
  total_loss = lambda_det * loss_3d_det + lambda_bev * loss_bev_seg

Trained modules:
  - EgoTriPlaneAdapter (triplane projection + ray embedding)
  - CenterDetHead (3D detection)
  - BEVSegHead (BEV semantic segmentation, optional)

Frozen:
  - Vision backbone (configurable: can unfreeze last N layers)
  - LLM backbone / tokenizer / language decoder

Usage:
  python scripts/train_adapter_stage1.py \
      --config configs/train_adapter_stage1.yaml \
      --index outputs/ego3dqa/nusc_train_index.jsonl \
      --features_dir outputs/features/nusc_train_clip_features/ \
      --out_dir outputs/adapter_3d_pretrain \
      --training_stage adapter_3d_pretrain \
      --use_3d_det_loss \
      --use_bev_seg_loss \
      --num_cams 6 \
      --batch_size 2 \
      --epochs 10
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.triplane_adapter import EgoTriPlaneAdapter
from egotriplane.heads import CenterDetHead, BEVSegHead, BEVHeatmapHead, VisibilityHead
from egotriplane.losses import Adapter3DPretrainLoss, DetectionLoss, BEVSegLoss
from egotriplane.dataset import Adapter3DPretrainDataset, detection_collate_fn


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stage 1: Adapter 3D Pretrain (perception-supervised)"
    )
    # Config file (base defaults)
    parser.add_argument("--config", type=str, default="configs/train_adapter_stage1.yaml",
                        help="YAML config file")

    # Override paths
    parser.add_argument("--index", type=str, default=None,
                        help="Path to nuScenes sample index JSONL")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="Path to pre-extracted features directory")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override checkpoint save dir")

    # Stage / training objectives
    parser.add_argument("--training_stage", type=str, default="adapter_3d_pretrain",
                        choices=["adapter_3d_pretrain", "vlm_qa"],
                        help="Training stage")
    parser.add_argument("--use_vqa_loss", action="store_true", default=False,
                        help="Enable VQA / language generation loss (disabled in Stage 1)")
    parser.add_argument("--no_vqa_loss", action="store_true", default=True,
                        help="Disable VQA loss (default for Stage 1)")
    parser.add_argument("--use_3d_det_loss", action="store_true", default=True,
                        help="Enable 3D detection loss")
    parser.add_argument("--use_bev_seg_loss", action="store_true", default=True,
                        help="Enable BEV segmentation loss")
    parser.add_argument("--use_occupancy_loss", action="store_true", default=False,
                        help="Enable occupancy loss")

    # Model overrides
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--multi_scale", action="store_true", default=False,
                        help="Enable multi-scale features")
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--unfreeze_last_n", type=int, default=0,
                        help="Unfreeze last N vision backbone layers")

    # Data
    parser.add_argument("--num_cams", type=int, default=None,
                        help="Number of cameras (min=max for fixed)")
    parser.add_argument("--min_cams", type=int, default=3)
    parser.add_argument("--max_cams", type=int, default=6)
    parser.add_argument("--build_bev_mask", action="store_true", default=False)

    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lambda_det", type=float, default=None)
    parser.add_argument("--lambda_bev", type=float, default=None)

    # System
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Config ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Merge CLI overrides into cfg
    _merge_overrides(cfg, args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ============================================================
    # Sanity Check: print configuration
    # ============================================================
    stage = cfg.get("training_stage", "adapter_3d_pretrain")
    print("\n" + "=" * 60)
    print(f"  [Stage] {stage}")
    print(f"  use_vqa_loss          = {cfg.get('use_vqa_loss', False)}")
    print(f"  use_language_loss     = {cfg.get('use_language_loss', False)}")
    print(f"  use_3d_detection_loss = {cfg.get('use_3d_detection_loss', True)}")
    print(f"  use_bev_seg_loss      = {cfg.get('use_bev_seg_loss', True)}")
    print(f"  use_occupancy_loss    = {cfg.get('use_occupancy_loss', False)}")
    print("=" * 60 + "\n")

    # ============================================================
    # Build Model
    # ============================================================
    model_cfg = cfg["model"]
    hidden_dim = model_cfg["hidden_dim"]

    # Adapter
    adapter = EgoTriPlaneAdapter(
        feature_dim=1024,  # CLIP ViT-L default
        hidden_dim=hidden_dim,
        x_range=tuple(model_cfg["triplane"]["x_range"]),
        y_range=tuple(model_cfg["triplane"]["y_range"]),
        z_range=tuple(model_cfg["triplane"]["z_range"]),
        sx=model_cfg["triplane"]["sx"],
        sy=model_cfg["triplane"]["sy"],
        sz=model_cfg["triplane"]["sz"],
        patch_size=model_cfg["triplane"]["patch_size"],
        use_ray_embedding=model_cfg.get("use_ray_embedding", True),
        aggregation=model_cfg.get("aggregation", "mean"),
        multi_scale_mode=model_cfg.get("multi_scale_mode", "concat"),
    )

    # Detection head
    det_cfg = cfg.get("detection", {})
    det_head = CenterDetHead(
        hidden_dim=hidden_dim,
        num_classes=det_cfg.get("num_classes", 5),
        grid_sx=model_cfg["triplane"]["sx"],
        grid_sy=model_cfg["triplane"]["sy"],
        patch_size=model_cfg["triplane"]["patch_size"],
        x_range=tuple(model_cfg["triplane"]["x_range"]),
        y_range=tuple(model_cfg["triplane"]["y_range"]),
        use_objectness=True,
    )

    # BEV segmentation head
    bev_cfg = cfg.get("bev_seg", {})
    bev_seg_head = None
    if cfg.get("use_bev_seg_loss", True) and bev_cfg.get("enabled", True):
        bev_seg_head = BEVSegHead(
            hidden_dim=hidden_dim,
            num_classes=bev_cfg.get("num_classes", 1),
            grid_sx=model_cfg["triplane"]["sx"],
            grid_sy=model_cfg["triplane"]["sy"],
            patch_size=model_cfg["triplane"]["patch_size"],
        )

    # Move to device
    adapter.to(device)
    det_head.to(device)
    if bev_seg_head is not None:
        bev_seg_head.to(device)

    # ============================================================
    # Freezing Logic
    # ============================================================
    # Freeze: all modules by default, then selectively unfreeze
    for p in adapter.parameters():
        p.requires_grad = True  # Adapter is always trainable
    for p in det_head.parameters():
        p.requires_grad = True
    if bev_seg_head is not None:
        for p in bev_seg_head.parameters():
            p.requires_grad = True

    # Print trainable params
    total_params = 0
    trainable_params = 0
    print("\n--- Trainable Parameters ---")
    for name, param in adapter.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  [TRAIN] adapter.{name:<50s} {param.numel():>10,}")
    for name, param in det_head.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  [TRAIN] det_head.{name:<50s} {param.numel():>10,}")
    if bev_seg_head is not None:
        for name, param in bev_seg_head.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  [TRAIN] bev_seg.{name:<50s} {param.numel():>10,}")

    print(f"\n  Total params:       {total_params:>10,}")
    print(f"  Trainable params:   {trainable_params:>10,}")
    print(f"  Trainable ratio:    {trainable_params / max(total_params, 1):.2%}\n")

    # ============================================================
    # Loss
    # ============================================================
    loss_cfg = cfg["loss"]
    criterion = Adapter3DPretrainLoss(
        lambda_det=loss_cfg.get("lambda_det", 1.0),
        lambda_bev=loss_cfg.get("lambda_bev", 1.0),
        lambda_occ=loss_cfg.get("lambda_occ", 0.0),
        det_loss_cfg={
            **loss_cfg.get("det_loss", {}),
            "num_classes": det_cfg.get("num_classes", 5),
            "grid_sx": model_cfg["triplane"]["sx"],
            "grid_sy": model_cfg["triplane"]["sy"],
            "bev_x_range": model_cfg["triplane"]["x_range"],
            "bev_y_range": model_cfg["triplane"]["y_range"],
        },
        bev_seg_cfg={
            **(loss_cfg.get("bev_seg_loss", {})),
            "num_classes": bev_cfg.get("num_classes", 1),
        },
    )

    # ============================================================
    # Optimizer & Scheduler
    # ============================================================
    trainable = list(adapter.parameters()) + list(det_head.parameters())
    if bev_seg_head is not None:
        trainable += list(bev_seg_head.parameters())

    optimizer = optim.AdamW(
        trainable,
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    total_steps = cfg["train"]["num_epochs"] * 1000
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["train"]["learning_rate"],
        total_steps=total_steps,
        pct_start=cfg["train"]["warmup_steps"] / max(total_steps, 1),
    )

    scaler = GradScaler(enabled=cfg["train"].get("mixed_precision") == "fp16")

    # ============================================================
    # Dataset
    # ============================================================
    data_cfg = cfg["data"]
    train_dataset = Adapter3DPretrainDataset(
        index_file=data_cfg["index_file"],
        features_dir=data_cfg["features_dir"],
        split="train",
        x_range=tuple(model_cfg["triplane"]["x_range"]),
        y_range=tuple(model_cfg["triplane"]["y_range"]),
        grid_sx=model_cfg["triplane"]["sx"],
        grid_sy=model_cfg["triplane"]["sy"],
        heatmap_sigma=det_cfg.get("heatmap_sigma", 1.5),
        camera_dropout=data_cfg.get("camera_dropout", True),
        min_cameras=data_cfg.get("min_cameras", 3),
        max_cameras=data_cfg.get("max_cameras", 6),
        num_classes=det_cfg.get("num_classes", 5),
        class_list=det_cfg.get("class_list", None),
        max_objects=det_cfg.get("max_objects", 50),
        build_bev_mask=data_cfg.get("build_bev_mask", False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )

    # ============================================================
    # Checkpointing
    # ============================================================
    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Save merged config for reproducibility
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ============================================================
    # Training Loop
    # ============================================================
    best_loss = float("inf")
    global_step = 0
    log_every = cfg["logging"]["log_every"]
    grad_accum = cfg["train"]["gradient_accumulation_steps"]

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {cfg['train']['batch_size']}")
    print(f"Grad accum steps: {grad_accum}")
    print(f"Effective batch size: {cfg['train']['batch_size'] * grad_accum}")
    print(f"Starting training...\n")

    for epoch in range(cfg["train"]["num_epochs"]):
        adapter.train()
        det_head.train()
        if bev_seg_head is not None:
            bev_seg_head.train()

        epoch_losses = defaultdict(float)
        accum_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['num_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            # Handle collate: single sample dict or list of dicts
            if isinstance(batch, list):
                # Process one sample at a time for simplicity
                sample = batch[0]
            else:
                sample = batch

            sample = _to_device(sample, device)

            with autocast(enabled=scaler.is_enabled()):
                # Forward adapter
                adapter_out = adapter(
                    sample["features_by_camera"],
                    sample["camera_subset"],
                )

                # Forward detection head
                det_preds = det_head(adapter_out)

                # Build detection targets on device
                det_targets = {
                    "gt_hm": sample["gt_hm"].to(device),
                    "gt_offset": sample["gt_offset"].to(device),
                    "gt_size": sample["gt_size"].to(device),
                    "gt_yaw": sample["gt_yaw"].to(device),
                    "gt_mask": sample["gt_mask"].to(device),
                    "gt_obj_idx": sample["gt_obj_idx"].to(device),
                }

                # Forward BEV segmentation head
                bev_pred = None
                bev_target = None
                bev_mask = None
                if bev_seg_head is not None and cfg.get("use_bev_seg_loss", True):
                    bev_pred = bev_seg_head(adapter_out["tokens_xy"])
                    if sample.get("bev_semantic_mask") is not None:
                        bev_target = sample["bev_semantic_mask"].to(device)
                        bev_mask = sample.get("bev_valid_mask")
                        if bev_mask is not None:
                            bev_mask = bev_mask.to(device)

                # Compute loss
                losses = criterion(
                    det_preds=det_preds,
                    det_targets=det_targets,
                    bev_pred=bev_pred,
                    bev_target=bev_target,
                    bev_mask=bev_mask,
                )

                loss = losses["total"] / grad_accum

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum == 0:
                if cfg["train"].get("max_grad_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, cfg["train"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # Logging
            for k, v in losses.items():
                epoch_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

            if global_step > 0 and global_step % log_every == 0:
                avg = {k: v / log_every for k, v in epoch_losses.items()}
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else cfg["train"]["learning_rate"]
                pbar.set_postfix({
                    "total": f"{avg.get('total', 0):.4f}",
                    "det": f"{avg.get('det_total', 0):.4f}",
                    "bev": f"{avg.get('bev_seg_total', 0):.4f}",
                    "lr": f"{lr:.2e}",
                })
                # Sanity check: print batch info on first step
                if global_step == log_every:
                    _sanity_check_batch(sample, losses)
                epoch_losses = defaultdict(float)

        # --- End of epoch ---
        n_batches = max(1, len(train_loader))
        avg_total = sum(
            v / n_batches for k, v in epoch_losses.items() if "total" in k
        ) or epoch_losses.get("total", 0) / n_batches

        print(f"Epoch {epoch+1} avg total loss: {avg_total:.4f}")

        # Save checkpoint
        if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:
            _save_checkpoint(save_dir, epoch, adapter, det_head,
                             bev_seg_head, optimizer, scheduler, avg_total)

        if avg_total < best_loss:
            best_loss = avg_total
            _save_checkpoint(save_dir, "best", adapter, det_head,
                             bev_seg_head, optimizer, scheduler, avg_total)
            print(f"  New best: loss={best_loss:.4f}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {save_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_overrides(cfg: dict, args):
    """Merge CLI args into config dict."""
    if args.index:
        cfg["data"]["index_file"] = args.index
    if args.features_dir:
        cfg["data"]["features_dir"] = args.features_dir
    if args.out_dir:
        cfg["checkpoint"]["save_dir"] = args.out_dir
    if args.training_stage:
        cfg["training_stage"] = args.training_stage
    if args.hidden_dim:
        cfg["model"]["hidden_dim"] = args.hidden_dim
    if args.multi_scale:
        cfg["model"]["multi_scale"] = True
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["train"]["num_epochs"] = args.epochs
    if args.lr:
        cfg["train"]["learning_rate"] = args.lr
    if args.grad_accum:
        cfg["train"]["gradient_accumulation_steps"] = args.grad_accum
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.lambda_det is not None:
        cfg["loss"]["lambda_det"] = args.lambda_det
    if args.lambda_bev is not None:
        cfg["loss"]["lambda_bev"] = args.lambda_bev
    if args.num_cams is not None:
        cfg["data"]["min_cameras"] = args.num_cams
        cfg["data"]["max_cameras"] = args.num_cams
    cfg["data"]["min_cameras"] = min(args.min_cams, args.max_cams)
    cfg["data"]["max_cameras"] = max(args.min_cams, args.max_cams)
    cfg["data"]["build_bev_mask"] = args.build_bev_mask

    # Ensure CLI overrides for loss flags
    cfg["use_vqa_loss"] = args.use_vqa_loss
    cfg["use_language_loss"] = False  # always off in stage 1
    cfg["use_3d_detection_loss"] = args.use_3d_det_loss
    cfg["use_bev_seg_loss"] = args.use_bev_seg_loss
    cfg["use_occupancy_loss"] = args.use_occupancy_loss


def _to_device(batch, device):
    """Move batch items to device (nested dicts)."""
    if isinstance(batch, dict):
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            elif isinstance(v, dict):
                result[k] = _to_device(v, device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                result[k] = [t.to(device) for t in v]
            else:
                result[k] = v
        return result
    return batch


def _sanity_check_batch(batch: dict, losses: dict):
    """Print batch statistics for sanity checking."""
    print("\n--- Batch Sanity Check ---")
    feats = batch.get("features_by_camera", {})
    cam_names = list(feats.keys())
    print(f"  Camera subset: {cam_names}")
    first_feat = None
    for k, v in feats.items():
        if "features" in v:
            f = v["features"]
            if isinstance(f, torch.Tensor):
                first_feat = f
                break
    if first_feat is not None:
        print(f"  Feature shape (per cam): {first_feat.shape}")
    print(f"  Num cameras: {len(cam_names)}")

    if batch.get("gt_mask") is not None:
        n_obj = batch["gt_mask"].sum().item()
        print(f"  GT objects: {int(n_obj)}")
    if batch.get("gt_hm") is not None:
        hm = batch["gt_hm"]
        if isinstance(hm, torch.Tensor):
            print(f"  GT heatmap shape: {list(hm.shape)}, nonzero: {hm.sum().item():.1f}")
    if batch.get("bev_semantic_mask") is not None:
        bm = batch["bev_semantic_mask"]
        if isinstance(bm, torch.Tensor):
            print(f"  BEV mask shape: {list(bm.shape)}")
    else:
        print(f"  BEV semantic mask: None (TODO: implement BEV label generation)")

    print("  Losses:")
    for k, v in losses.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"    {k}: {val:.4f}")
    print("---------------------------\n")


def _save_checkpoint(save_dir, epoch, adapter, det_head,
                     bev_seg_head, optimizer, scheduler, loss_val):
    """Save model checkpoint."""
    if isinstance(epoch, int):
        path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
    else:
        path = os.path.join(save_dir, f"{epoch}.pt")

    ckpt = {
        "epoch": epoch if isinstance(epoch, int) else -1,
        "adapter": adapter.state_dict(),
        "det_head": det_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss_val,
    }
    if bev_seg_head is not None:
        ckpt["bev_seg_head"] = bev_seg_head.state_dict()

    torch.save(ckpt, path)


if __name__ == "__main__":
    main()
