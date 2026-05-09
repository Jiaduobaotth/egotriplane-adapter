"""nuScenes-specific utility functions for data loading and processing."""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from .geometry import (
    nusc_ego_pose_to_matrix,
    nusc_ego_to_cam_transform,
    global_to_ego,
    get_box_centroid,
    get_box_yaw,
    check_ego_path_overlap,
    is_box_visible_in_camera,
    classify_region,
    compute_direction,
    bin_distance_x,
    bin_distance_y,
)


# Category mapping from nuScenes to normalized types
CATEGORY_MAP = {
    "vehicle.car": "vehicle",
    "vehicle.truck": "vehicle",
    "vehicle.bus.rigid": "vehicle",
    "vehicle.bus.bendy": "vehicle",
    "vehicle.trailer": "vehicle",
    "vehicle.construction": "vehicle",
    "vehicle.emergency.police": "vehicle",
    "vehicle.emergency.ambulance": "vehicle",
    "vehicle.motorcycle": "cyclist",
    "vehicle.bicycle": "cyclist",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}

IGNORE_CATEGORIES = {
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "animal",
    "static_object.bicycle_rack",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
}


def normalize_category(nusc_category: str) -> str:
    """Map nuScenes category string to normalized type."""
    if nusc_category in CATEGORY_MAP:
        return CATEGORY_MAP[nusc_category]
    if nusc_category in IGNORE_CATEGORIES:
        return "ignore"
    # Fallback heuristics
    if "vehicle" in nusc_category:
        return "vehicle"
    if "human.pedestrian" in nusc_category:
        return "pedestrian"
    if "cycle" in nusc_category:
        return "cyclist"
    if "movable_object.barrier" in nusc_category:
        return "barrier"
    if "movable_object.trafficcone" in nusc_category:
        return "traffic_cone"
    return "ignore"


def load_nusc_sample(nusc, sample_token: str) -> dict:
    """Load a single nuScenes keyframe sample and return our unified schema.

    Args:
        nusc: NuScenes instance
        sample_token: sample token string

    Returns:
        dict matching the Sample schema
    """
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])

    # Ego pose
    ego_pose_data = nusc.get("ego_pose", sample["data"]["LIDAR_TOP"])
    T_ego_global = nusc_ego_pose_to_matrix(ego_pose_data)

    # Cameras
    cameras = []
    cam_channels = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]
    for cam_name in cam_channels:
        if cam_name not in sample["data"]:
            continue
        sd_token = sample["data"][cam_name]
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        K = np.array(cs["camera_intrinsic"], dtype=np.float32).tolist()
        T_ego_cam = nusc_ego_to_cam_transform(cs).tolist()

        cameras.append({
            "name": cam_name,
            "image_path": sd["filename"],
            "K": K,
            "T_ego_cam": T_ego_cam,
            "width": sd["width"],
            "height": sd["height"],
        })

    # Annotations (objects)
    objects = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        category_name = ann["category_name"]
        normalized = normalize_category(category_name)

        if normalized == "ignore":
            continue

        # Convert global annotation to ego frame
        center_global = np.array(ann["translation"], dtype=np.float32)
        size = np.array(ann["size"], dtype=np.float32)

        # nuScenes box convention: (w, l, h)
        # Rotation from quaternion
        rot = np.array(ann["rotation"], dtype=np.float32)
        yaw_global = _quaternion_to_yaw(rot)

        center_ego = global_to_ego(
            center_global.reshape(1, 3), T_ego_global
        ).reshape(3)

        # Yaw in ego frame
        # nuScenes rotation is global. We need the relative yaw.
        ego_yaw_global = _quaternion_to_yaw(np.array(ego_pose_data["rotation"],
                                                      dtype=np.float32))
        yaw_ego = _normalize_angle(yaw_global - ego_yaw_global)

        dist = float(np.linalg.norm(center_ego[:2]))
        if dist > 60.0:
            continue

        box_3d = [
            float(center_ego[0]), float(center_ego[1]), float(center_ego[2]),
            float(size[0]), float(size[1]), float(size[2]),
            float(yaw_ego),
        ]

        in_path = check_ego_path_overlap(center_ego, size, yaw=float(yaw_ego))

        objects.append({
            "object_id": ann["instance_token"],
            "category": normalized,
            "center_ego": center_ego.tolist(),
            "size": size.tolist(),
            "yaw_ego": float(yaw_ego),
            "box_3d": box_3d,
            "distance": dist,
            "direction": compute_direction(center_ego),
            "region": classify_region(center_ego),
            "x_bin": bin_distance_x(float(center_ego[0])),
            "y_bin": bin_distance_y(float(center_ego[1])),
            "in_ego_path": in_path,
            "answerable": True,
        })

    return {
        "sample_token": sample_token,
        "timestamp": sample["timestamp"],
        "scene_token": sample["scene_token"],
        "cameras": cameras,
        "objects": objects,
    }


def compute_object_answerability(sample: dict,
                                  camera_subset: Optional[List[str]] = None,
                                  min_area: float = 50.0) -> List[dict]:
    """Recompute answerability for all objects given a camera subset.

    Modifies sample["objects"] in place and returns it.
    """
    if camera_subset is None:
        camera_subset = [c["name"] for c in sample["cameras"]]

    cam_by_name = {c["name"]: c for c in sample["cameras"]}

    for obj in sample["objects"]:
        visible = False
        for cam_name in camera_subset:
            if cam_name not in cam_by_name:
                continue
            cam = cam_by_name[cam_name]
            if is_box_visible_in_camera(
                obj["box_3d"],
                np.array(cam["K"], dtype=np.float32),
                np.array(cam["T_ego_cam"], dtype=np.float32),
                cam["width"],
                cam["height"],
                min_area=min_area,
            ):
                visible = True
                break
        obj["answerable"] = visible
        obj["visibility"] = "visible" if visible else "not_observed"

    return sample["objects"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quaternion_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (rotation around z) from [w,x,y,z] quaternion."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


# ---------------------------------------------------------------------------
# Index loading/saving
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_index(samples: List[dict], path: str):
    """Save sample index as JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, cls=_NumpyEncoder) + "\n")


def load_index(path: str) -> List[dict]:
    """Load sample index from JSONL file."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_qa(path: str) -> List[dict]:
    """Load QA instances from JSONL file."""
    return load_index(path)


def save_qa(qa_list: List[dict], path: str):
    """Save QA instances as JSONL file."""
    save_index(qa_list, path)
