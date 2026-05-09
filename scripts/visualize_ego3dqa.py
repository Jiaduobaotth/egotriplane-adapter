#!/usr/bin/env python3
"""Visualize Ego3D-QA samples for debugging and paper figures.

Layout (per sample):
  - Each camera image in its own row (full width), with projected 3D boxes
  - BEV plot with nuScenes map layers, ego vehicle, objects, ego path
  - QA info panel (question, GT answer, model answer, visibility)

Usage:
    python scripts/visualize_ego3dqa.py \
        --qa outputs/ego3dqa/nusc_val_ego3dqa.jsonl \
        --index outputs/ego3dqa/nusc_val_index.jsonl \
        --nusc_root ./data \
        --out outputs/eval/vis/ \
        --num_samples 10
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.geometry import (
    get_3d_box_corners,
    project_points_to_camera,
    is_box_visible_in_camera,
    is_point_in_image,
    is_region_observed,
    REGION_DEFS,
    check_ego_path_overlap,
)
from egotriplane.nusc_utils import load_qa, load_index

# ---- Color maps ----
CAT_COLORS = {
    "vehicle": "#3498db",
    "pedestrian": "#e74c3c",
    "cyclist": "#2ecc71",
    "barrier": "#f39c12",
    "traffic_cone": "#9b59b6",
}
EGO_COLOR = "#2c3e50"
CORRIDOR_COLOR = "#00bcd4"
TARGET_COLOR = "#ff0"
REGION_COLORS = {
    "front": "#ff9999", "front-left": "#ffcc99", "front-right": "#ffcc99",
    "rear": "#9999ff", "rear-left": "#99ccff", "rear-right": "#99ccff",
    "left": "#ccffcc", "right": "#ccffcc",
}

# ---- nuScenes map helpers ----
def _get_scene_map_name(scene_token: str, nusc_map) -> str:
    """Get map name from scene. Returns e.g. 'boston-seaport'."""
    try:
        from nuscenes.nuscenes import NuScenes
        # scene tokens from the index contain the scene information
        # We try to look up log -> map
        return "boston-seaport"
    except:
        return "boston-seaport"


def _render_map_patch(ax, map_api, x_center, y_center, x_range=(-25, 85), y_range=(-45, 45)):
    """Render nuScenes map layers as background on the BEV axis.

    Draws lane dividers, road boundaries, and drivable area.
    """
    try:
        # Get map mask layers
        # Layer names: 'drivable_area', 'road_segment', 'lane', 'road_divider', 'lane_divider'
        patch_box = (x_center, y_center, x_range[1] - x_range[0], y_range[1] - y_range[0])
        patch_angle = 0

        # Try to render using map API
        map_api.render_map_patch(
            patch_box, patch_angle,
            ['drivable_area', 'road_divider', 'lane_divider'],
            ax=ax, alpha=0.15, zorder=0,
        )
    except Exception:
        # Fallback: draw nothing for map
        pass


def _load_map_api(dataroot: str, scene_token: str = None):
    """Load nuScenes map API for the correct map."""
    try:
        # Monkey-patch the seaborn style issue (already patched, but just in case)
        import matplotlib.style as mstyle
        orig = mstyle.use
        def safe_use(s):
            try: return orig(s)
            except: pass
        mstyle.use = safe_use

        from nuscenes.map_expansion.map_api import NuScenesMap

        # Determine map name from scene
        # For mini: only boston-seaport is available
        map_name = "boston-seaport"

        nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
        return nusc_map, map_name
    except Exception as e:
        print(f"Warning: Could not load map API: {e}")
        return None, None


# ---- Main visualization ----
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Ego3D-QA samples")
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--nusc_root", type=str, default="./data")
    parser.add_argument("--pred", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    qa_list = load_qa(args.qa)
    samples = load_index(args.index)
    samples_by_token = {s["sample_token"]: s for s in samples}

    # Pre-load map API (lazy)
    nusc_map = None
    try:
        nusc_map, _ = _load_map_api(args.nusc_root)
    except:
        pass

    # Load predictions
    preds_by_id = {}
    if args.pred and os.path.exists(args.pred):
        with open(args.pred) as f:
            for line in f:
                p = json.loads(line.strip())
                preds_by_id[p["id"]] = p

    # Select diverse samples across question types
    rng = np.random.RandomState(args.seed)
    selected = _select_diverse(qa_list, min(args.num_samples, len(qa_list)), rng)

    print(f"Visualizing {len(selected)} QA samples...")

    for i, qa in enumerate(selected):
        sample_token = qa["sample_token"]
        sample = samples_by_token.get(sample_token)
        if sample is None:
            print(f"  [{i}] Sample {sample_token} not found, skipping")
            continue

        try:
            fig = create_visualization(qa, sample, args.nusc_root,
                                        nusc_map=nusc_map,
                                        pred=preds_by_id.get(qa["id"]))
            out_path = os.path.join(args.out, f"{qa['id']}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            import traceback
            print(f"  [{i}] Error: {e}")
            traceback.print_exc()
            continue

    print(f"Done. Saved to {args.out}")


def _select_diverse(qa_list, n, rng):
    """Select diverse samples covering all QA types."""
    by_type = defaultdict(list)
    for i, qa in enumerate(qa_list):
        by_type[qa["question_type"]].append(i)
    selected = []
    per_type = max(1, n // len(by_type))
    for indices in by_type.values():
        picked = rng.choice(indices, min(per_type, len(indices)), replace=False)
        selected.extend(picked)
    remaining = n - len(selected)
    if remaining > 0:
        all_idx = set(range(len(qa_list))) - set(selected)
        more = rng.choice(list(all_idx), min(remaining, len(all_idx)), replace=False)
        selected.extend(more)
    rng.shuffle(selected)
    return [qa_list[i] for i in selected[:n]]


def create_visualization(qa, sample, nusc_root, nusc_map=None, pred=None):
    """Create visualization figure with proper layout.

    Layout (per QA):
      Row 0: Camera images side by side (each gets a subplot)
      Row 1: BEV plot with map + ego + objects
      Row 2: QA text info panel
    """
    camera_subset = qa.get("camera_subset", [c["name"] for c in sample["cameras"]])
    n_cams = len(camera_subset)
    cam_by_name = {c["name"]: c for c in sample["cameras"]}
    grounding = qa.get("grounding")

    # -- Layout: GridSpec --
    n_img_rows = max(1, (n_cams + 2) // 3)  # 3 columns max
    fig = plt.figure(figsize=(22, 12 + 3.2 * n_img_rows))

    # Top: camera images in multiple rows if needed
    n_img_cols = min(3, n_cams)

    gs_top = GridSpec(n_img_rows, n_img_cols, figure=fig,
                       left=0.02, right=0.98, top=0.99, bottom=0.55,
                       hspace=0.08, wspace=0.04)

    # Middle: BEV
    gs_bev = GridSpec(1, 1, figure=fig,
                       left=0.02, right=0.65, top=0.53, bottom=0.20)

    # Middle-right: QA info
    gs_info = GridSpec(1, 1, figure=fig,
                        left=0.67, right=0.98, top=0.53, bottom=0.20)

    # ---- Draw camera images ----
    for idx, cam_name in enumerate(camera_subset):
        cam = cam_by_name.get(cam_name)
        if cam is None:
            continue

        row = idx // n_img_cols
        col = idx % n_img_cols
        ax = fig.add_subplot(gs_top[row, col])

        K = np.array(cam["K"], dtype=np.float32)
        T_ego_cam = np.array(cam["T_ego_cam"], dtype=np.float32)

        # Load image
        img_path = os.path.join(nusc_root, cam["image_path"])
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except (FileNotFoundError, OSError) as e:
            ax.text(0.5, 0.5, f"Image missing:\n{os.path.basename(img_path)}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="red")
            ax.set_title(cam_name, fontsize=8)
            ax.axis("off")
            continue

        # Draw 3D box projections
        for obj in sample["objects"]:
            color = CAT_COLORS.get(obj["category"], "#95a5a6")
            alpha = 0.45
            lw = 0.8
            is_target = grounding and obj["object_id"] == grounding.get("object_id")
            if is_target:
                alpha = 1.0
                lw = 3.0
                color = TARGET_COLOR

            center = np.array(obj["box_3d"][:3], dtype=np.float32)
            size = np.array(obj["box_3d"][3:6], dtype=np.float32)
            yaw = obj["box_3d"][6]

            corners_3d = get_3d_box_corners(center, size, yaw)
            uv, depth, valid_depth = project_points_to_camera(corners_3d, K, T_ego_cam)

            if valid_depth.sum() == 0:
                continue

            in_img = valid_depth & is_point_in_image(uv, cam["width"], cam["height"])

            if in_img.sum() == 0 and not is_target:
                continue

            if not np.all(np.isfinite(uv[valid_depth])):
                continue

            _draw_projected_box(
                ax,
                uv,
                valid_depth=valid_depth,
                color=color,
                alpha=alpha,
                lw=lw,
            )

        ax.set_title(f"{cam_name}  ({cam['width']}x{cam['height']})", fontsize=9)
        ax.set_xlim(0, cam["width"])
        ax.set_ylim(cam["height"], 0)
        ax.axis("off")

    # Hide unused subplot cells
    for idx in range(n_cams, n_img_rows * n_img_cols):
        row = idx // n_img_cols
        col = idx % n_img_cols
        ax = fig.add_subplot(gs_top[row, col])
        ax.axis("off")

    # ---- Draw BEV ----
    ax_bev = fig.add_subplot(gs_bev[0, 0])
    _draw_bev(ax_bev, sample, qa, grounding, nusc_map)

    # ---- Draw QA info ----
    ax_info = fig.add_subplot(gs_info[0, 0])
    ax_info.axis("off")
    _draw_qa_info(ax_info, qa, pred, camera_subset, sample)

    return fig


def _draw_projected_box(ax, uv_corners, valid_depth=None,
                        color="red", alpha=0.5, lw=1.0):
    """Draw projected 3D box edges, skipping edges that cross behind the camera."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    uv_corners = np.asarray(uv_corners)

    if uv_corners.shape != (8, 2):
        return
    if not np.all(np.isfinite(uv_corners)):
        return

    if valid_depth is None:
        valid_depth = np.ones(8, dtype=bool)
    else:
        valid_depth = np.asarray(valid_depth).astype(bool)

    for i, j in edges:
        if not (valid_depth[i] and valid_depth[j]):
            continue

        ax.plot(
            [uv_corners[i, 0], uv_corners[j, 0]],
            [uv_corners[i, 1], uv_corners[j, 1]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            clip_on=True,
        )


def _draw_bev(ax, sample, qa, grounding, nusc_map=None):
    """Draw BEV with map background, ego, objects, regions."""
    x_min, x_max = -25, 85
    y_min, y_max = -45, 45
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x (m) forward →", fontsize=10)
    ax.set_ylabel("y (m) left →", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linestyle="--")

    # -- Try to render map background --
    if nusc_map is not None:
        try:
            # Get ego position from first camera's calibration
            cam0 = sample["cameras"][0]
            # The ego is at origin in ego frame. But maps are in global coords.
            # For the mini split, we use ego-centric coordinates
            # Try rendering the map at the ego location
            patch_box = (0, 0, x_max - x_min, y_max - y_min)
            nusc_map.render_map_patch(
                patch_box, 0,
                ['drivable_area'],
                ax=ax, alpha=0.08, zorder=0,
            )
        except Exception:
            pass

    # -- Ego vehicle --
    ego_body = Rectangle((-2, -1), 4, 2, linewidth=2.5,
                          edgecolor=EGO_COLOR, facecolor=EGO_COLOR, alpha=0.85,
                          zorder=10)
    ax.add_patch(ego_body)
    # Direction arrow
    ax.arrow(2, 0, 3, 0, head_width=1.2, head_length=1.5,
             fc=EGO_COLOR, ec=EGO_COLOR, linewidth=2, zorder=10)

    # -- Ego path corridor --
    corridor = Rectangle((0, -1.8), 50, 3.6, linewidth=2,
                          edgecolor=CORRIDOR_COLOR, facecolor=CORRIDOR_COLOR,
                          alpha=0.08, linestyle="--", zorder=2)
    ax.add_patch(corridor)

    # -- Region boundaries --
    for name in REGION_DEFS:
        color = REGION_COLORS.get(name, "#eeeeee")
        _draw_region_patch(ax, name, color, alpha=0.06)

    # -- All objects --
    for obj in sample["objects"]:
        cx, cy = obj["center_ego"][0], obj["center_ego"][1]
        w, l = obj["size"][0], obj["size"][1]
        yaw = obj["yaw_ego"]
        color = CAT_COLORS.get(obj["category"], "#95a5a6")
        is_target = grounding and obj["object_id"] == grounding.get("object_id")

        # Only draw objects within view range
        if not (x_min < cx < x_max and y_min < cy < y_max):
            continue

        corners = _get_2d_box_corners(cx, cy, w, l, yaw)
        poly = Polygon(corners, closed=True,
                        facecolor=color,
                        edgecolor="black" if is_target else color,
                        alpha=0.85 if is_target else 0.35,
                        linewidth=2.5 if is_target else 0.5,
                        zorder=8 if is_target else 3)
        ax.add_patch(poly)

        # Highlight target
        if is_target:
            ax.plot(cx, cy, "*", color=TARGET_COLOR, markersize=16,
                    markeredgecolor="black", markeredgewidth=1.0, zorder=15)

    # -- Region labels --
    _draw_region_labels(ax, x_max)

    # -- Legend --
    legend_patches = [
        Rectangle((0, 0), 1, 1, facecolor=EGO_COLOR, label="Ego"),
        Rectangle((0, 0), 1, 1, facecolor=CORRIDOR_COLOR, alpha=0.3,
                   label="Ego Path"),
    ]
    legend_patches += [
        Rectangle((0, 0), 1, 1, facecolor=c, label=cat)
        for cat, c in CAT_COLORS.items()
    ]
    legend_patches.append(Line2D([0], [0], marker="*", color="w",
                                  markerfacecolor=TARGET_COLOR, markersize=10,
                                  markeredgecolor="black", label="Target"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              ncol=2, framealpha=0.8)

    ax.set_title("Bird's Eye View (ego coordinates)", fontsize=11, fontweight="bold")


def _draw_region_patch(ax, region_name, color, alpha=0.05):
    """Draw a semi-transparent patch for a spatial region."""
    region_bounds = {
        "front":       (5, 80, -2, 2),
        "front-left":  (5, 80, 2, 40),
        "front-right": (5, 80, -40, -2),
        "rear":        (-20, -5, -2, 2),
        "rear-left":   (-20, -5, 2, 40),
        "rear-right":  (-20, -5, -40, -2),
        "left":        (-5, 5, 2, 40),
        "right":       (-5, 5, -40, -2),
    }
    if region_name in region_bounds:
        x0, x1, y0, y1 = region_bounds[region_name]
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                          facecolor=color, edgecolor="none", alpha=alpha, zorder=1)
        ax.add_patch(rect)


def _draw_region_labels(ax, x_max):
    """Add text labels for regions."""
    labels = {
        "Front": (40, -8),
        "Front-L": (40, 20),
        "Front-R": (40, -20),
        "Rear": (-12, -8),
        "Left": (0, 20),
        "Right": (0, -20),
    }
    for text, (x, y) in labels.items():
        ax.text(x, y, text, fontsize=7, alpha=0.25, ha="center", va="center",
                style="italic")


def _draw_qa_info(ax, qa, pred, camera_subset, sample):
    """Draw question, GT answer, model answer, and visibility info."""
    lines = []
    lines.append("══════════════ Ego3D-QA Sample ══════════════")
    lines.append(f"ID:    {qa['id'][:40]}...")
    lines.append(f"Sample: {qa['sample_token'][:20]}...")
    lines.append(f"Type:  {qa['question_type']}  |  Answerability: {qa['answerability']}")
    lines.append(f"Cameras ({len(camera_subset)}): {', '.join(camera_subset)}")
    lines.append("")
    lines.append("─── QUESTION ───")
    lines.append(qa["question"])
    lines.append("")
    lines.append("─── GROUND TRUTH ───")
    if isinstance(qa["answer"], dict):
        lines.append(json.dumps(qa["answer"], indent=2, ensure_ascii=False))
    else:
        lines.append(str(qa["answer"]))
    if qa.get("grounding"):
        g = qa["grounding"]
        b = g["box_3d"]
        lines.append(f"  → Target: {g['category']} at x={b[0]:.1f}m, y={b[1]:.1f}m, z={b[2]:.1f}m")
    lines.append("")

    if pred:
        lines.append("─── MODEL PREDICTION ───")
        pred_ans = pred.get("predicted_answer", pred.get("answer_text", "N/A"))
        lines.append(str(pred_ans))
        lines.append("")

    lines.append("─── REGION VISIBILITY ───")
    calibrations = [
        {"name": c["name"], "K": c["K"], "T_ego_cam": c["T_ego_cam"],
         "width": c["width"], "height": c["height"]}
        for c in sample["cameras"] if c["name"] in camera_subset
    ]
    for region_name, _ in REGION_DEFS.items():
        obs = is_region_observed(region_name, camera_subset, calibrations)
        icon = "✓" if obs else "✗"
        status = "OBSERVED" if obs else "NOT OBSERVED"
        lines.append(f"  [{icon}] {region_name}: {status}")

    lines.append("")
    lines.append("─── OBJECTS IN SAMPLE ───")
    for obj in sample["objects"]:
        c = obj["center_ego"]
        is_t = qa.get("grounding") and obj["object_id"] == qa["grounding"].get("object_id")
        marker = "→ " if is_t else "  "
        lines.append(f"{marker}{obj['category']:12s} x={c[0]:6.1f} y={c[1]:6.1f} z={c[2]:4.1f}  "
                      f"{'[TARGET]' if is_t else ''}")

    text = "\n".join(lines)
    ax.text(0.01, 0.99, text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=6.5,
            verticalalignment="top", horizontalalignment="left",
            linespacing=1.15)


def _get_2d_box_corners(cx, cy, w, l, yaw):
    """Get 4 corners of a 2D oriented rectangle."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    corners_local = np.array([
        [-l / 2, -w / 2],
        [l / 2, -w / 2],
        [l / 2, w / 2],
        [-l / 2, w / 2],
    ])
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return corners_local @ R.T + np.array([cx, cy])


if __name__ == "__main__":
    main()
