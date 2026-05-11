#!/usr/bin/env python3
"""Visualize Stage 1 3D detection results.

Supports:
  - BEV heatmap overlay with predicted and GT boxes
  - Side-by-side comparison (pred vs GT)
  - Per-class color coding
  - Score filtering

Usage:
  # From saved inference JSON
  python scripts/visualize_detections.py \
      --pred_json outputs/inference/predictions.json \
      --gt_json outputs/inference/groundtruths.json \
      --out_dir outputs/vis/

  # Or run inference + visualize in one go
  python scripts/visualize_detections.py \
      --ckpt outputs/stage1_phase1/best.pt \
      --nusc_root ./data --nusc_version v1.0-mini \
      --out_dir outputs/vis/ --num_samples 10
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist", "barrier", "traffic_cone"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
GT_COLOR = "#555555"
GT_ALPHA = 0.35
PRED_ALPHA = 0.8

# BEV grid config (matches training defaults)
X_RANGE = (-20.0, 80.0)
Y_RANGE = (-40.0, 40.0)
GRID_SX = 96
GRID_SY = 96


def parse_args():
    p = argparse.ArgumentParser(description="Visualize 3D detection results")
    # Input: either JSON files or checkpoint for live inference
    p.add_argument("--pred_json", type=str, default=None)
    p.add_argument("--gt_json", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None,
                   help="If given, runs inference first")
    p.add_argument("--nusc_root", type=str, default="./data")
    p.add_argument("--nusc_version", type=str, default="v1.0-mini")
    p.add_argument("--split", type=str, default="val")

    # Filtering
    p.add_argument("--num_samples", type=int, default=20,
                   help="Number of samples to visualize")
    p.add_argument("--score_thresh", type=float, default=0.15,
                   help="Min score for predicted boxes")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Start visualizing from this sample index")

    # Output
    p.add_argument("--out_dir", type=str, default="outputs/vis/")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no_gt", action="store_true", default=False,
                   help="Don't show GT boxes")

    # Model config (only needed if --ckpt is given)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--backbone", type=str, default="qwen3vl_4b")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--aggregation", type=str, default="attention")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--online", action="store_true", default=False)
    return p.parse_args()


def draw_bev_boxes(ax, boxes, scores=None, classes=None, color=None, alpha=0.7, linewidth=1.5):
    """Draw 3D boxes in BEV (top-down view).

    Args:
        ax: matplotlib axis
        boxes: [N, 7] (cx, cy, cz, w, l, h, yaw) in ego frame
        scores: [N] optional scores for annotation
        classes: [N] optional class indices for color coding
        color: override color (if classes is None)
    """
    patches = []
    edgecolors = []
    for i, box in enumerate(boxes):
        cx, cy, cz, w, l, h, yaw = box
        # Rotate box corners
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = np.array([[-l / 2, -w / 2],
                            [l / 2, -w / 2],
                            [l / 2, w / 2],
                            [-l / 2, w / 2]])
        corners_rot = corners @ R.T + np.array([cx, cy])

        rect = mpatches.Polygon(corners_rot, closed=True, fill=True,
                                alpha=alpha, linewidth=linewidth)
        patches.append(rect)
        if color is not None:
            edgecolors.append(color)
        elif classes is not None:
            edgecolors.append(CLASS_COLORS[classes[i] % len(CLASS_COLORS)])
        else:
            edgecolors.append("#1f77b4")

    pc = PatchCollection(patches, edgecolors=edgecolors, facecolors=edgecolors,
                         alpha=alpha, linewidths=linewidth)
    ax.add_collection(pc)

    # Direction indicator (short line from center)
    for i, box in enumerate(boxes):
        cx, cy, cz, w, l, h, yaw = box
        dx = l / 2 * np.cos(yaw)
        dy = l / 2 * np.sin(yaw)
        ec = edgecolors[i] if i < len(edgecolors) else "#1f77b4"
        ax.arrow(cx, cy, dx * 0.3, dy * 0.3, head_width=0.8, head_length=1.0,
                 fc=ec, ec=ec, alpha=alpha, linewidth=0.5)

    # Score labels
    if scores is not None:
        for i, (box, sc) in enumerate(zip(boxes, scores)):
            cx, cy = box[0], box[1]
            ax.text(cx, cy + 1, f"{sc:.2f}", fontsize=4, ha="center",
                    color=edgecolors[i] if i < len(edgecolors) else "#1f77b4",
                    alpha=0.9)


def plot_bev_comparison(pred_boxes, pred_scores, pred_classes,
                        gt_boxes, gt_labels,
                        sample_token, out_path, score_thresh=0.15):
    """Create BEV comparison figure: prediction (left) vs GT (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Sample: {sample_token[:16]}...", fontsize=9, family="monospace")

    for ax, title in zip(axes, ["Predictions", "Ground Truth"]):
        ax.set_xlim(X_RANGE[0], X_RANGE[1])
        ax.set_ylim(Y_RANGE[0], Y_RANGE[1])
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Draw ego vehicle
        ax.add_patch(mpatches.Rectangle((-2, -1), 4, 2, fill=True,
                                         color="black", alpha=0.6))
        ax.text(0, 0, "EGO", fontsize=6, ha="center", va="center", color="white")

    # Left: predictions
    mask = pred_scores >= score_thresh
    pred_boxes_filt = pred_boxes[mask]
    pred_scores_filt = pred_scores[mask]
    pred_classes_filt = pred_classes[mask]
    draw_bev_boxes(axes[0], pred_boxes_filt, pred_scores_filt, pred_classes_filt,
                   alpha=PRED_ALPHA, linewidth=1.5)

    # Right: GT
    if len(gt_boxes) > 0:
        gt_boxes_np = np.array(gt_boxes)
        draw_bev_boxes(axes[1], gt_boxes_np, classes=gt_labels,
                       color=GT_COLOR, alpha=GT_ALPHA, linewidth=1.0)

    # Legend
    legend_patches = []
    for i, name in enumerate(CLASS_NAMES):
        legend_patches.append(mpatches.Patch(color=CLASS_COLORS[i], label=name, alpha=0.8))
    axes[0].legend(handles=legend_patches, fontsize=6, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bev_overlay(pred_boxes, pred_scores, pred_classes,
                     gt_boxes, gt_labels,
                     sample_token, out_path, score_thresh=0.15):
    """Create single BEV view with pred + GT overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(X_RANGE[0], X_RANGE[1])
    ax.set_ylim(Y_RANGE[0], Y_RANGE[1])
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"BEV Detection — {sample_token[:16]}...")
    ax.grid(True, alpha=0.3)

    # Ego vehicle
    ax.add_patch(mpatches.Rectangle((-2, -1), 4, 2, fill=True,
                                     color="black", alpha=0.6))
    ax.text(0, 0, "EGO", fontsize=7, ha="center", va="center", color="white")

    # GT boxes (background)
    if len(gt_boxes) > 0:
        gt_boxes_np = np.array(gt_boxes)
        draw_bev_boxes(ax, gt_boxes_np, color=GT_COLOR, alpha=GT_ALPHA, linewidth=1.8)

    # Pred boxes (foreground)
    mask = pred_scores >= score_thresh
    pred_boxes_filt = pred_boxes[mask]
    pred_scores_filt = pred_scores[mask]
    pred_classes_filt = pred_classes[mask]
    draw_bev_boxes(ax, pred_boxes_filt, pred_scores_filt, pred_classes_filt,
                   alpha=PRED_ALPHA, linewidth=1.2)

    # Legend
    legend_patches = [mpatches.Patch(color=GT_COLOR, label="GT", alpha=GT_ALPHA)]
    for i, name in enumerate(CLASS_NAMES):
        legend_patches.append(mpatches.Patch(color=CLASS_COLORS[i], label=name, alpha=0.8))
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_grid(preds_list, gts_list, out_dir, score_thresh=0.15):
    """Create a grid summary of multiple samples."""
    n = min(len(preds_list), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        ax = axes[i]
        ax.set_xlim(X_RANGE[0], X_RANGE[1])
        ax.set_ylim(Y_RANGE[0], Y_RANGE[1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.add_patch(mpatches.Rectangle((-2, -1), 4, 2, fill=True,
                                         color="black", alpha=0.5))

        pred = preds_list[i]
        gt = gts_list[i] if i < len(gts_list) else None

        # GT
        if gt and len(gt.get("boxes", [])) > 0:
            gt_boxes_np = np.array(gt["boxes"])
            draw_bev_boxes(ax, gt_boxes_np, color=GT_COLOR, alpha=GT_ALPHA, linewidth=1.5)

        # Pred
        pred_boxes_np = np.array(pred["boxes"])
        pred_scores_np = np.array(pred["scores"])
        pred_classes_np = np.array(pred["classes"])
        mask = pred_scores_np >= score_thresh
        if mask.any():
            draw_bev_boxes(ax, pred_boxes_np[mask], pred_scores_np[mask],
                           pred_classes_np[mask], alpha=PRED_ALPHA, linewidth=1.0)

        n_pred = mask.sum()
        n_gt = len(gt["boxes"]) if gt else 0
        token_short = pred.get("sample_token", "?")[:12]
        ax.set_title(f"{token_short} | pred:{n_pred} gt:{n_gt}", fontsize=7)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "summary_grid.png")
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary grid: {grid_path}")


def main():
    args = parse_args()

    # Get predictions
    if args.ckpt:
        # Run inference first
        from scripts.infer_vision_3d import build_model, NuscImageDataset, image_collate_fn
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Running inference with checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

        from egotriplane.vision_encoder import VisionEncoderWrapper
        from egotriplane.triplane_adapter import EgoTriPlaneAdapter
        from egotriplane.heads import CenterDetHead

        vision_encoder, adapter, det_head, block_size = build_model(args, ckpt)

        ds = NuscImageDataset(
            nusc_root=args.nusc_root,
            nusc_version=args.nusc_version,
            split=args.split,
            image_size=args.image_size,
            camera_dropout=False,
            min_cameras=3, max_cameras=6,
            num_classes=5, max_objects=50,
            augment=False, patch_size=block_size,
        )
        if args.num_samples > 0:
            ds.samples = ds.samples[args.start_idx:args.start_idx + args.num_samples]

        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True,
                            collate_fn=image_collate_fn)

        all_predictions = []
        all_groundtruths = []
        for sample in tqdm(loader, desc="Inference"):
            images = sample["images"]
            intrinsics = sample["intrinsics"]
            extrinsics = sample["extrinsics"]
            cam_names = sample["camera_names"]
            image_sizes = sample["image_sizes"]
            sample_token = sample["sample_token"]
            if isinstance(sample_token, list):
                sample_token = sample_token[0]

            imgs_batch = torch.stack([img.to(device) for img in images], dim=0)
            with torch.no_grad():
                enc_out = vision_encoder(imgs_batch)
                all_feats = enc_out["last_hidden_state"]
                patch_grid = enc_out["patch_grid"]

            features_by_camera = {}
            for i, cn in enumerate(cam_names):
                features_by_camera[cn] = {
                    "features": all_feats[i],
                    "K": intrinsics[i].to(device),
                    "T_ego_cam": extrinsics[i].to(device),
                    "image_size": image_sizes[i].tolist(),
                    "patch_grid": list(patch_grid),
                }

            with torch.no_grad():
                adapter_out = adapter(features_by_camera, cam_names)
                det_preds = det_head(adapter_out)
                decoded = det_head.decode_detections(
                    det_preds, score_thresh=0.05, max_dets=100,
                )

            det = decoded[0]
            all_predictions.append({
                "sample_token": sample_token,
                "boxes": det["boxes"].cpu().tolist(),
                "scores": det["scores"].cpu().tolist(),
                "classes": det["classes"].cpu().tolist(),
            })

            gt_boxes = sample.get("gt_boxes_3d")
            gt_labels = sample.get("gt_labels")
            gt_mask = sample.get("gt_mask")
            if gt_boxes is not None:
                mask = gt_mask[0] if hasattr(gt_mask, 'dim') and gt_mask.dim() > 1 else gt_mask
                boxes = gt_boxes[0] if hasattr(gt_boxes, 'dim') and gt_boxes.dim() > 2 else gt_boxes
                labels = gt_labels[0] if hasattr(gt_labels, 'dim') and gt_labels.dim() > 1 else gt_labels
                valid = mask.bool() if hasattr(mask, 'bool') else mask
                all_groundtruths.append({
                    "sample_token": sample_token,
                    "boxes": boxes[valid].tolist() if valid.any() else [],
                    "labels": labels[valid].tolist() if valid.any() else [],
                })
    else:
        # Load from JSON
        if not args.pred_json:
            print("ERROR: Provide --ckpt or --pred_json")
            sys.exit(1)
        with open(args.pred_json) as f:
            all_predictions = json.load(f)
        all_groundtruths = []
        if args.gt_json and os.path.exists(args.gt_json):
            with open(args.gt_json) as f:
                all_groundtruths = json.load(f)

    print(f"Loaded {len(all_predictions)} predictions, {len(all_groundtruths)} GTs")

    # Convert to numpy for convenience
    os.makedirs(args.out_dir, exist_ok=True)

    limit = min(args.num_samples, len(all_predictions))
    offset = args.start_idx
    preds_subset = all_predictions[offset:offset + limit]
    gts_subset = all_groundtruths[offset:offset + limit] if all_groundtruths else []

    # Individual sample visualizations
    for i, pred in enumerate(preds_subset):
        token = pred["sample_token"]
        pred_boxes = np.array(pred["boxes"])
        pred_scores = np.array(pred["scores"])
        pred_classes = np.array(pred["classes"])

        gt = gts_subset[i] if i < len(gts_subset) else {}
        gt_boxes = gt.get("boxes", [])
        gt_labels = gt.get("labels", [])

        # Overlay view
        out_overlay = os.path.join(args.out_dir, f"bev_{i:03d}_{token[:12]}.png")
        plot_bev_overlay(pred_boxes, pred_scores, pred_classes,
                         gt_boxes, gt_labels, token, out_overlay,
                         score_thresh=args.score_thresh)

        # Comparison view
        out_comp = os.path.join(args.out_dir, f"compare_{i:03d}_{token[:12]}.png")
        plot_bev_comparison(pred_boxes, pred_scores, pred_classes,
                            gt_boxes, gt_labels, token, out_comp,
                            score_thresh=args.score_thresh)

    print(f"Saved {limit} individual visualizations to {args.out_dir}")

    # Summary grid
    if len(preds_subset) >= 2:
        plot_summary_grid(preds_subset, gts_subset, args.out_dir,
                          score_thresh=args.score_thresh)

    print("Done!")


if __name__ == "__main__":
    main()
