"""PyTorch Dataset classes for EgoTriPlane-Adapter training and evaluation.

Stage 1 (adapter_3d_pretrain): Adapter3DPretrainDataset
  - Loads nuScenes index directly (no QA required)
  - Outputs: images/features, calibrations, 3D detection targets,
    optional BEV segmentation / occupancy targets

Stage 2 (vlm_qa): Ego3DQADataset, AdapterStage1Dataset (legacy)
  - Loads QA JSONL + features for VQA training/eval
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .nusc_utils import load_qa, load_index, normalize_category
from .camera_dropout import (
    FULL_NUSC_CAMERAS,
    sample_camera_subset,
    sample_camera_subset_excluding,
)
from .geometry import (
    gaussian_heatmap,
    is_region_observed,
    REGION_DEFS,
    world_to_grid,
)


# Category -> class index mapping for detection
DEFAULT_DET_CLASSES = ["vehicle", "pedestrian", "cyclist", "barrier", "traffic_cone"]
CAT_TO_IDX = {cat: i for i, cat in enumerate(DEFAULT_DET_CLASSES)}


# ---------------------------------------------------------------------------
# Stage 1: Adapter 3D Pretrain Dataset (perception-supervised, NO VQA)
# ---------------------------------------------------------------------------

class Adapter3DPretrainDataset(Dataset):
    """Dataset for Stage 1: adapter 3D perception pretraining.

    Loads nuScenes sample index directly — no QA JSONL required.
    Outputs perception supervision:
      - Multi-camera features + calibrations
      - 3D detection targets (boxes, labels, heatmaps)
      - Optional BEV segmentation / occupancy targets

    Does NOT output question/answer fields.
    """

    def __init__(self,
                 index_file: str,
                 features_dir: str,
                 split: str = "train",
                 x_range: Tuple[float, float] = (-20.0, 80.0),
                 y_range: Tuple[float, float] = (-40.0, 40.0),
                 grid_sx: int = 96,
                 grid_sy: int = 96,
                 heatmap_sigma: float = 1.5,
                 camera_dropout: bool = True,
                 min_cameras: int = 3,
                 max_cameras: int = 6,
                 num_classes: int = 5,
                 class_list: Optional[List[str]] = None,
                 max_objects: int = 50,
                 build_bev_mask: bool = False,
                 bev_mask_resolution: float = 0.5):
        self.samples = load_index(index_file)
        self.features_dir = Path(features_dir)
        self.split = split
        self.x_range = x_range
        self.y_range = y_range
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy
        self.heatmap_sigma = heatmap_sigma
        self.camera_dropout = camera_dropout
        self.min_cameras = min_cameras
        self.max_cameras = max_cameras
        self.num_classes = num_classes
        self.class_list = class_list or DEFAULT_DET_CLASSES
        self.max_objects = max_objects
        self.build_bev_mask = build_bev_mask

        self.cell_x = (x_range[1] - x_range[0]) / grid_sx
        self.cell_y = (y_range[1] - y_range[0]) / grid_sy

        # BEV mask parameters (TODO: connect to actual BEV label generation)
        self.bev_mask_resolution = bev_mask_resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        sample_token = sample["sample_token"]

        # --- Determine available cameras ---
        all_camera_names = [c for c in FULL_NUSC_CAMERAS
                            if (self.features_dir / f"{sample_token}_{c}.pt").exists()]
        if not all_camera_names:
            all_camera_names = [c["name"] for c in sample.get("cameras", [])]

        # --- Camera subset ---
        if self.camera_dropout and self.split == "train":
            camera_subset = sample_camera_subset(
                all_camera_names, self.min_cameras, self.max_cameras
            )
        else:
            camera_subset = all_camera_names

        # --- Load per-camera features ---
        features_by_camera = {}
        camera_intrinsics = []
        camera_extrinsics = []
        for cam_name in camera_subset:
            # Try cached feature file first
            feat_path = self.features_dir / f"{sample_token}_{cam_name}.pt"
            if feat_path.exists():
                data = torch.load(feat_path, map_location="cpu", weights_only=False)
                features_by_camera[cam_name] = data
                K = data["K"]
                T_ego_cam = data["T_ego_cam"]
            else:
                # Try from sample dict (raw camera info)
                cam_info = _find_camera(sample, cam_name)
                if cam_info is None:
                    continue
                K = torch.tensor(cam_info["K"], dtype=torch.float32)
                T_ego_cam = torch.tensor(cam_info["T_ego_cam"], dtype=torch.float32)
                # Placeholder empty features
                features_by_camera[cam_name] = {
                    "features": torch.zeros(256, 1024),
                    "K": K,
                    "T_ego_cam": T_ego_cam,
                    "image_size": [cam_info["height"], cam_info["width"]],
                    "patch_grid": [16, 16],
                }

            camera_intrinsics.append(K)
            camera_extrinsics.append(T_ego_cam)

        # --- Build 3D detection targets ---
        det_targets = self._build_detection_targets(sample)

        # --- Build BEV segmentation target (TODO: placeholder) ---
        bev_target = None
        bev_mask = None
        if self.build_bev_mask:
            bev_target, bev_mask = self._build_bev_seg_target(sample)

        return {
            "sample_token": sample_token,
            "features_by_camera": features_by_camera,
            "camera_subset": camera_subset,
            "camera_intrinsics": camera_intrinsics,
            "camera_extrinsics": camera_extrinsics,
            # 3D detection targets
            "gt_boxes_3d": det_targets["boxes"],          # [max_obj, 7] (cx,cy,cz,w,l,h,yaw)
            "gt_labels_3d": det_targets["labels"],         # [max_obj]
            "gt_mask": det_targets["mask"],                # [max_obj] bool
            "gt_hm": det_targets["heatmap"],               # [num_classes, H, W]
            "gt_offset": det_targets["offset"],            # [max_obj, 2]
            "gt_size": det_targets["size"],                # [max_obj, 3]
            "gt_yaw": det_targets["yaw"],                  # [max_obj, 2]
            "gt_obj_idx": det_targets["obj_idx"],          # [max_obj, 2]
            # BEV segmentation targets (optional)
            "bev_semantic_mask": bev_target,
            "bev_valid_mask": bev_mask,
        }

    def _build_detection_targets(self, sample: dict) -> dict:
        """Build center-based detection targets from sample objects.

        Returns dict with keys:
          - boxes: [max_obj, 7]
          - labels: [max_obj]
          - mask: [max_obj]
          - heatmap: [num_classes, grid_sy, grid_sx]
          - offset: [max_obj, 2]
          - size: [max_obj, 3]
          - yaw: [max_obj, 2]
          - obj_idx: [max_obj, 2]
        """
        M = self.max_objects
        C = self.num_classes
        H, W = self.grid_sy, self.grid_sx

        boxes = torch.zeros(M, 7)
        labels = torch.zeros(M, dtype=torch.long)
        obj_mask = torch.zeros(M, dtype=torch.bool)
        offset = torch.zeros(M, 2)
        size = torch.zeros(M, 3)
        yaw = torch.zeros(M, 2)
        obj_idx = torch.zeros(M, 2, dtype=torch.long)
        heatmap = torch.zeros(C, H, W)

        objs = sample.get("objects", [])
        obj_count = 0

        for obj in objs:
            if obj_count >= M:
                break
            cat = obj.get("category", "ignore")
            if cat not in self.class_list:
                continue
            cls_idx = self.class_list.index(cat)

            box = obj.get("box_3d")
            if not box or len(box) < 7:
                continue
            cx, cy, cz, w, l, h, yaw_val = box

            # Convert to grid
            gx, gy = world_to_grid(cx, cy, self.x_range, self.y_range, W, H)

            # Sub-cell offset (in cell units)
            x_min = self.x_range[0]
            y_min = self.y_range[0]
            cx_cell = (cx - x_min) / self.cell_x
            cy_cell = (cy - y_min) / self.cell_y
            dx = cx_cell - gx
            dy = cy_cell - gy

            # Fill targets
            boxes[obj_count] = torch.tensor([cx, cy, cz, w, l, h, yaw_val])
            labels[obj_count] = cls_idx
            obj_mask[obj_count] = True
            offset[obj_count] = torch.tensor([dx, dy])
            size[obj_count] = torch.tensor([w, l, h])
            yaw[obj_count] = torch.tensor([np.sin(yaw_val), np.cos(yaw_val)])
            obj_idx[obj_count] = torch.tensor([gx, gy])

            # Gaussian heatmap
            sigma_cells = self.heatmap_sigma
            radius = max(1, int(3 * sigma_cells))
            for dgx in range(-radius, radius + 1):
                for dgy in range(-radius, radius + 1):
                    ngx, ngy = gx + dgx, gy + dgy
                    if 0 <= ngx < W and 0 <= ngy < H:
                        dist2 = dgx**2 + dgy**2
                        val = np.exp(-dist2 / (2 * sigma_cells**2))
                        heatmap[cls_idx, ngy, ngx] = max(
                            heatmap[cls_idx, ngy, ngx].item(), val
                        )

            obj_count += 1

        return {
            "boxes": boxes,
            "labels": labels,
            "mask": obj_mask,
            "heatmap": heatmap,
            "offset": offset,
            "size": size,
            "yaw": yaw,
            "obj_idx": obj_idx,
        }

    def _build_bev_seg_target(self, sample: dict):
        """Build BEV semantic segmentation target.

        TODO: Implement proper BEV label generation from:
          - 3D box rasterization into BEV grid
          - LiDAR point projection
          - Map-based lane/road rendering

        Returns (bev_target, bev_mask) tuples or (None, None).
        """
        H, W = self.grid_sy, self.grid_sx
        # Placeholder: binary occupancy from 3D boxes
        occ = torch.zeros(H, W)
        mask = torch.ones(H, W)  # all cells valid by default

        for obj in sample.get("objects", []):
            box = obj.get("box_3d")
            if not box or len(box) < 7:
                continue
            cx, cy, cz, w, l, h, yaw_val = box
            gx, gy = world_to_grid(cx, cy, self.x_range, self.y_range, W, H)
            # Mark occupied cell
            if 0 <= gx < W and 0 <= gy < H:
                occ[gy, gx] = 1.0

        return occ, mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_camera(sample: dict, cam_name: str) -> Optional[dict]:
    """Find camera info in a sample dict."""
    for cam in sample.get("cameras", []):
        if cam.get("name") == cam_name:
            return cam
    return None


def detection_collate_fn(batch: List[dict]) -> dict:
    """Collate function for Adapter3DPretrainDataset.

    Since the adapter processes one sample at a time (features_by_camera
    per sample), we return a list and the training loop handles batching
    by iterating over samples and accumulating gradients.

    For true batching with per-sample forward passes, set batch_size=1
    and use gradient accumulation.
    """
    if len(batch) == 1:
        return batch[0]
    # Multi-sample: return list for sequential processing
    return batch


class Ego3DQADataset(Dataset):
    """Dataset for Stage 2 training: (features, question) -> JSON answer."""

    def __init__(
        self,
        qa_file: str,
        features_dir: str,
        split: str = "train",
        camera_dropout: bool = True,
        min_cameras: int = 3,
        max_cameras: int = 6,
        exclude_unseen: bool = True,
        max_seq_length: int = 512,
        tokenizer = None,
    ):
        self.qa_list = load_qa(qa_file)
        self.features_dir = Path(features_dir)
        self.split = split
        self.camera_dropout = camera_dropout
        self.min_cameras = min_cameras
        self.max_cameras = max_cameras
        self.exclude_unseen = exclude_unseen
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.qa_list)

    def __getitem__(self, idx: int) -> dict:
        qa = self.qa_list[idx]
        sample_token = qa["sample_token"]

        # Load cached features
        all_camera_names = [c for c in FULL_NUSC_CAMERAS
                            if (self.features_dir / f"{sample_token}_{c}.pt").exists()]

        if self.camera_dropout and self.split == "train":
            if self.exclude_unseen:
                camera_subset = sample_camera_subset_excluding(
                    all_camera_names, self.min_cameras, self.max_cameras
                )
            else:
                camera_subset = sample_camera_subset(
                    all_camera_names, self.min_cameras, self.max_cameras
                )
        else:
            camera_subset = qa.get("camera_subset", all_camera_names)

        # Load per-camera features
        features_by_camera = {}
        for cam_name in camera_subset:
            feat_path = self.features_dir / f"{sample_token}_{cam_name}.pt"
            if feat_path.exists():
                data = torch.load(feat_path, map_location="cpu", weights_only=False)
                features_by_camera[cam_name] = data

        # Tokenize question if tokenizer provided
        question_tokens = None
        if self.tokenizer is not None:
            question_tokens = self.tokenizer(
                qa["question"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
            )

        return {
            "qa_id": qa["id"],
            "sample_token": sample_token,
            "features_by_camera": features_by_camera,
            "question": qa["question"],
            "question_type": qa["question_type"],
            "answer": qa["answer"],
            "answerability": qa["answerability"],
            "question_tokens": question_tokens,
            "camera_subset": camera_subset,
            "grounding": qa.get("grounding"),
        }


class AdapterStage1Dataset(Dataset):
    """Dataset for Stage 1: train adapter with BEV heatmap + visibility."""

    def __init__(
        self,
        qa_file: str,
        features_dir: str,
        x_range: Tuple[float, float] = (-20.0, 80.0),
        y_range: Tuple[float, float] = (-40.0, 40.0),
        grid_sx: int = 96,
        grid_sy: int = 96,
        heatmap_sigma: float = 1.5,
        camera_dropout: bool = True,
        min_cameras: int = 3,
        max_cameras: int = 6,
    ):
        self.qa_list = load_qa(qa_file)
        self.features_dir = Path(features_dir)
        self.x_range = x_range
        self.y_range = y_range
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy
        self.heatmap_sigma = heatmap_sigma
        self.camera_dropout = camera_dropout
        self.min_cameras = min_cameras
        self.max_cameras = max_cameras

    def __len__(self) -> int:
        return len(self.qa_list)

    def __getitem__(self, idx: int) -> dict:
        qa = self.qa_list[idx]
        sample_token = qa["sample_token"]

        all_camera_names = [c for c in FULL_NUSC_CAMERAS
                            if (self.features_dir / f"{sample_token}_{c}.pt").exists()]

        if self.camera_dropout:
            camera_subset = sample_camera_subset(
                all_camera_names, self.min_cameras, self.max_cameras
            )
        else:
            camera_subset = qa.get("camera_subset", all_camera_names)

        # Load features
        features_by_camera = {}
        calibrations = []
        for cam_name in camera_subset:
            feat_path = self.features_dir / f"{sample_token}_{cam_name}.pt"
            if feat_path.exists():
                data = torch.load(feat_path, map_location="cpu", weights_only=False)
                features_by_camera[cam_name] = data
                calibrations.append({
                    "name": cam_name,
                    "K": data["K"].tolist() if hasattr(data["K"], "tolist") else data["K"],
                    "T_ego_cam": data["T_ego_cam"].tolist() if hasattr(data["T_ego_cam"], "tolist") else data["T_ego_cam"],
                    "width": data["image_size"][1],
                    "height": data["image_size"][0],
                })

        # Build BEV heatmap target
        gt_heatmap = np.zeros((self.grid_sy, self.grid_sx), dtype=np.float32)
        grounding = qa.get("grounding")
        if grounding and "box_3d" in grounding:
            cx = grounding["box_3d"][0]
            cy = grounding["box_3d"][1]
            gt_heatmap = gaussian_heatmap(
                (cx, cy), self.x_range, self.y_range,
                self.grid_sx, self.grid_sy, self.heatmap_sigma
            )

        # Build visibility map
        gt_visibility = self._build_visibility_map(camera_subset, calibrations)

        return {
            "qa_id": qa["id"],
            "sample_token": sample_token,
            "features_by_camera": features_by_camera,
            "camera_subset": camera_subset,
            "gt_heatmap": torch.from_numpy(gt_heatmap),
            "gt_visibility": torch.from_numpy(gt_visibility),
            "question_type": qa["question_type"],
        }

    def _build_visibility_map(self, camera_subset: List[str],
                               calibrations: List[dict]) -> np.ndarray:
        """Build a binary visibility grid over the BEV plane."""
        vis = np.zeros((self.grid_sy, self.grid_sx), dtype=np.float32)

        cell_x = (self.x_range[1] - self.x_range[0]) / self.grid_sx
        cell_y = (self.y_range[1] - self.y_range[0]) / self.grid_sy

        for gy in range(0, self.grid_sy, 4):  # subsample for speed
            for gx in range(0, self.grid_sx, 4):
                wx = self.x_range[0] + (gx + 0.5) * cell_x
                wy = self.y_range[0] + (gy + 0.5) * cell_y
                point_ego = np.array([[wx, wy, 1.0]], dtype=np.float32)

                visible = False
                for cam in calibrations:
                    from .geometry import project_points_to_camera
                    uv, depth, valid = project_points_to_camera(
                        point_ego,
                        np.array(cam["K"], dtype=np.float32),
                        np.array(cam["T_ego_cam"], dtype=np.float32),
                    )
                    if valid[0] and depth[0] > 0:
                        u, v = uv[0, 0], uv[0, 1]
                        if 0 <= u < cam["width"] and 0 <= v < cam["height"]:
                            visible = True
                            break

                if visible:
                    vis[gy:gy+4, gx:gx+4] = 1.0

        return vis
