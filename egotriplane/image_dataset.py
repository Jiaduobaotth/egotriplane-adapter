"""End-to-end image dataset for vision encoder 3D pretraining.

Loads raw nuScenes images (no pre-extracted features), applies transforms,
and provides 3D detection targets for end-to-end training.

Stage 1 output per sample:
  {
    "images": [N_cam, 3, H, W] tensors,
    "intrinsics": [N_cam, 3, 3],
    "extrinsics": [N_cam, 4, 4],
    "camera_names": [N_cam],
    "image_sizes": [N_cam, 2],
    # 3D detection targets
    "gt_boxes_3d": [max_obj, 7],
    "gt_labels": [max_obj],
    "gt_mask": [max_obj] bool,
    "gt_hm": [num_classes, H_bev, W_bev],
    "gt_offset": [max_obj, 2],
    "gt_size": [max_obj, 3],
    "gt_yaw": [max_obj, 2],
    "gt_obj_idx": [max_obj, 2],
    "sample_token": str,
  }

No question/answer fields — pure perception supervision.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

from .nusc_utils import normalize_category
from .camera_dropout import (
    FULL_NUSC_CAMERAS,
    sample_camera_subset,
)
from .geometry import world_to_grid

# Category mapping for detection
DEFAULT_CLASSES = ["vehicle", "pedestrian", "cyclist", "barrier", "traffic_cone"]
CAT_TO_IDX = {cat: i for i, cat in enumerate(DEFAULT_CLASSES)}

# nuScenes image mean/std (approx, from CLIP)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NuscImageDataset(Dataset):
    """Load raw nuScenes images for end-to-end 3D pretraining.

    Uses nuScenes devkit to load annotations and PIL for images.
    No pre-extracted features or QA JSONL required.
    """

    def __init__(self,
                 nusc_root: str = "./data",
                 nusc_version: str = "v1.0-mini",
                 split: str = "train",
                 image_size: int = 224,
                 cameras: Optional[List[str]] = None,
                 camera_dropout: bool = True,
                 min_cameras: int = 3,
                 max_cameras: int = 6,
                 x_range: Tuple[float, float] = (-20.0, 80.0),
                 y_range: Tuple[float, float] = (-40.0, 40.0),
                 grid_sx: int = 96,
                 grid_sy: int = 96,
                 heatmap_sigma: float = 1.5,
                 num_classes: int = 5,
                 max_objects: int = 50,
                 max_distance: float = 60.0,
                 augment: bool = True):
        super().__init__()
        self.nusc_root = Path(nusc_root)
        self.image_size = image_size
        self.cameras = cameras or FULL_NUSC_CAMERAS
        self.camera_dropout = camera_dropout and split == "train"
        self.min_cameras = min_cameras
        self.max_cameras = max_cameras
        self.x_range = x_range
        self.y_range = y_range
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy
        self.heatmap_sigma = heatmap_sigma
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.max_distance = max_distance
        self.augment = augment and split == "train"

        self.cell_x = (x_range[1] - x_range[0]) / grid_sx
        self.cell_y = (y_range[1] - y_range[0]) / grid_sy

        # Load nuScenes
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(version=nusc_version, dataroot=str(self.nusc_root),
                             verbose=False)

        # Build sample list
        self.samples = self._build_sample_list(split)

        # Image transforms
        if augment:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def _build_sample_list(self, split: str) -> List[dict]:
        """Build flat list of (sample_token, scene_token) pairs."""
        samples = []
        for scene in self.nusc.scene:
            token = scene["first_sample_token"]
            while token:
                sample = self.nusc.get("sample", token)
                samples.append({
                    "sample_token": token,
                    "scene_token": sample["scene_token"],
                })
                token = sample["next"]

        # Train/val split (80/20 since mini has 10 scenes)
        if "val" in split:
            samples = samples[-80:]  # last ~80 samples for val
        else:
            samples = samples[:-80]  # first ~324 samples for train

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        sample_token = sample_info["sample_token"]
        sample = self.nusc.get("sample", sample_token)

        # --- Get available cameras ---
        available_cams = [c for c in self.cameras if c in sample["data"]]
        if not available_cams:
            available_cams = [c for c in FULL_NUSC_CAMERAS if c in sample["data"]]

        # --- Camera subset (with dropout for training) ---
        if self.camera_dropout:
            cam_subset = sample_camera_subset(available_cams,
                                              self.min_cameras, self.max_cameras)
        else:
            cam_subset = available_cams

        # --- Load images and calibrations ---
        images = []
        intrinsics = []
        extrinsics = []
        image_sizes = []
        valid_cam_names = []

        for cam_name in cam_subset:
            sd_token = sample["data"][cam_name]
            sd = self.nusc.get("sample_data", sd_token)
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

            # Load image
            img_path = self.nusc_root / sd["filename"]
            try:
                img = Image.open(str(img_path)).convert("RGB")
            except (FileNotFoundError, OSError):
                continue

            # Transform
            img_tensor = self.transform(img)  # [3, H, W]

            # Intrinsics
            K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)  # [3, 3]

            # Extrinsics: ego -> cam
            T_ego_cam = _build_ego_to_cam(cs)

            images.append(img_tensor)
            intrinsics.append(K)
            extrinsics.append(T_ego_cam)
            image_sizes.append(torch.tensor([sd["height"], sd["width"]], dtype=torch.long))
            valid_cam_names.append(cam_name)

        # --- Get ego pose and annotations ---
        ego_pose_data = self.nusc.get("ego_pose", sample["data"]["LIDAR_TOP"])
        T_ego_global = _build_ego_pose_matrix(ego_pose_data)

        # --- Build 3D detection targets ---
        det_targets = self._build_detection_targets(sample, T_ego_global)

        return {
            "images": images,                     # list of [3, H, W]
            "intrinsics": intrinsics,             # list of [3, 3]
            "extrinsics": extrinsics,             # list of [4, 4]
            "camera_names": valid_cam_names,
            "image_sizes": image_sizes,
            "sample_token": sample_token,
            # Detection targets
            "gt_boxes_3d": det_targets["boxes"],
            "gt_labels": det_targets["labels"],
            "gt_mask": det_targets["mask"],
            "gt_hm": det_targets["heatmap"],
            "gt_offset": det_targets["offset"],
            "gt_size": det_targets["size"],
            "gt_yaw": det_targets["yaw"],
            "gt_obj_idx": det_targets["obj_idx"],
            "num_objects": det_targets["num_objects"],
        }

    def _build_detection_targets(self, sample: dict, T_ego_global: np.ndarray) -> dict:
        """Build center-based 3D detection targets from nuScenes annotations.

        Converts global-frame annotations to ego-frame.
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

        obj_count = 0

        for ann_token in sample["anns"]:
            if obj_count >= M:
                break

            ann = self.nusc.get("sample_annotation", ann_token)
            cat_name = ann["category_name"]
            norm_cat = normalize_category(cat_name)

            if norm_cat not in CAT_TO_IDX:
                continue
            cls_idx = CAT_TO_IDX[norm_cat]

            # Convert global → ego frame
            center_global = np.array(ann["translation"], dtype=np.float32)
            rot_global = np.array(ann["rotation"], dtype=np.float32)
            anno_size = np.array(ann["size"], dtype=np.float32)  # [w, l, h]

            # Transform center to ego frame
            T_global_ego = np.linalg.inv(T_ego_global)
            center_h = np.append(center_global, 1.0)
            center_ego_h = T_global_ego @ center_h
            cx, cy, cz = center_ego_h[0], center_ego_h[1], center_ego_h[2]

            # Yaw in ego frame
            yaw_global = _quaternion_to_yaw(rot_global)
            ego_yaw_global = _quaternion_to_yaw(
                np.array(self.nusc.get("ego_pose",
                         sample["data"]["LIDAR_TOP"])["rotation"],
                         dtype=np.float32)
            )
            yaw_ego = _normalize_angle(yaw_global - ego_yaw_global)

            # Check distance
            dist = np.sqrt(cx**2 + cy**2)
            if dist > self.max_distance:
                continue

            # Check BEV range bounds (exclude objects whose center is outside)
            if not (self.x_range[0] <= cx <= self.x_range[1] and
                    self.y_range[0] <= cy <= self.y_range[1]):
                continue

            w, l, h = float(anno_size[0]), float(anno_size[1]), float(anno_size[2])

            # Convert to grid
            gx, gy = world_to_grid(cx, cy, self.x_range, self.y_range, W, H)

            # Sub-cell offset
            cx_cell = (cx - self.x_range[0]) / self.cell_x
            cy_cell = (cy - self.y_range[0]) / self.cell_y
            dx = cx_cell - gx
            dy = cy_cell - gy

            # Fill
            boxes[obj_count] = torch.tensor([cx, cy, cz, w, l, h, yaw_ego])
            labels[obj_count] = cls_idx
            obj_mask[obj_count] = True
            offset[obj_count] = torch.tensor([dx, dy])
            size[obj_count] = torch.tensor([w, l, h])
            yaw[obj_count] = torch.tensor([np.sin(yaw_ego), np.cos(yaw_ego)])
            obj_idx[obj_count] = torch.tensor([gx, gy])

            # Gaussian heatmap
            sigma = self.heatmap_sigma
            radius = max(1, int(3 * sigma))
            for dgx in range(-radius, radius + 1):
                for dgy in range(-radius, radius + 1):
                    ngx, ngy = gx + dgx, gy + dgy
                    if 0 <= ngx < W and 0 <= ngy < H:
                        d2 = dgx**2 + dgy**2
                        val = np.exp(-d2 / (2 * sigma**2))
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
            "num_objects": obj_count,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def image_collate_fn(batch: List[dict]) -> dict:
    """Collate for NuscImageDataset.

    Since multi-camera data has variable N_cam per sample,
    we return a list for sequential processing.
    For true batching, set batch_size=1 with gradient accumulation.
    """
    if len(batch) == 1:
        return batch[0]
    return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ego_to_cam(calibrated_sensor: dict) -> torch.Tensor:
    """Build T_ego_cam (4x4) from nuScenes calibrated_sensor.

    nuScenes stores T_sensor<-ego. We want T_cam<-ego.
    """
    from scipy.spatial.transform import Rotation

    trans = np.array(calibrated_sensor["translation"], dtype=np.float32)
    rot_q = np.array(calibrated_sensor["rotation"], dtype=np.float32)
    R = Rotation.from_quat([rot_q[1], rot_q[2], rot_q[3], rot_q[0]]).as_matrix()

    # T_sensor_ego
    T_se = np.eye(4, dtype=np.float32)
    T_se[:3, :3] = R
    T_se[:3, 3] = trans

    # T_ego_sensor = inv(T_sensor_ego)
    T_es = np.eye(4, dtype=np.float32)
    T_es[:3, :3] = R.T
    T_es[:3, 3] = -R.T @ trans

    return torch.from_numpy(T_es)


def _build_ego_pose_matrix(ego_pose: dict) -> np.ndarray:
    """Build 4x4 ego pose matrix from nuScenes ego_pose record."""
    from scipy.spatial.transform import Rotation

    trans = np.array(ego_pose["translation"], dtype=np.float32)
    rot_q = np.array(ego_pose["rotation"], dtype=np.float32)
    R = Rotation.from_quat([rot_q[1], rot_q[2], rot_q[3], rot_q[0]]).as_matrix()

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


def _quaternion_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from [w, x, y, z] quaternion."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny, cosy))


def _normalize_angle(a: float) -> float:
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return a
