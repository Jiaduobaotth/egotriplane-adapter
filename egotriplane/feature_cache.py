"""Feature caching utilities for pre-computed vision encoder features.

Stores per-camera features with calibration metadata so training
does not re-run the frozen vision encoder on every epoch.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_camera_features(features: torch.Tensor,
                          patch_grid: tuple,
                          K: torch.Tensor,
                          T_ego_cam: torch.Tensor,
                          image_size: tuple,
                          output_path: str):
    """Save per-camera features to disk.

    Args:
        features: [Hf, Wf, D] patch features
        patch_grid: (Hf, Wf)
        K: [3, 3] intrinsics
        T_ego_cam: [4, 4] extrinsics
        image_size: (H, W) original image dimensions
        output_path: .pt file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        "features": features.detach().cpu(),
        "patch_grid": list(patch_grid),
        "K": K.detach().cpu(),
        "T_ego_cam": T_ego_cam.detach().cpu(),
        "image_size": list(image_size),
    }

    torch.save(data, output_path)


def load_camera_features(path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load cached camera features.

    Returns dict with keys: features, patch_grid, K, T_ego_cam, image_size
    """
    data = torch.load(path, map_location=device, weights_only=False)
    return data


def build_feature_cache_path(sample_token: str,
                              cam_name: str,
                              feature_dir: str) -> str:
    """Build standardized feature cache file path."""
    return os.path.join(feature_dir, f"{sample_token}_{cam_name}.pt")


def check_cache_exists(sample_token: str,
                        cam_name: str,
                        feature_dir: str) -> bool:
    """Check if cached features exist for a sample+camera."""
    return os.path.exists(build_feature_cache_path(sample_token, cam_name, feature_dir))


def get_cache_paths_for_sample(sample_token: str,
                                 camera_names: list,
                                 feature_dir: str) -> Dict[str, str]:
    """Get all cache paths for one sample's cameras.

    Returns dict mapping camera name to file path.
    """
    return {
        name: build_feature_cache_path(sample_token, name, feature_dir)
        for name in camera_names
    }
