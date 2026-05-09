"""Geometry utilities for EgoTriPlane-Adapter.

Coordinate system:
  - Ego frame: x=forward, y=left, z=up
  - Camera frame: standard pinhole, z forward

All transforms are 4x4 homogeneous matrices.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def make_transform_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from 3x3 rotation and 3x1 translation."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    T_inv = np.eye(4, dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform Nx3 points by 4x4 homogeneous matrix.

    Args:
        points: [N, 3] or [..., 3]
        T: [4, 4]

    Returns:
        transformed_points: same shape as points
    """
    shape = points.shape
    pts = points.reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)  # [N, 4]
    transformed = (T @ pts_h.T).T  # [N, 4]
    return transformed[:, :3].reshape(shape)


def global_to_ego(points_global: np.ndarray,
                  ego_pose_global: np.ndarray) -> np.ndarray:
    """Convert points from nuScenes global frame to ego frame.

    Args:
        points_global: [N, 3] in global coordinates
        ego_pose_global: [4, 4] ego pose in global frame

    Returns:
        points_ego: [N, 3]
    """
    T_ego_global = invert_transform(ego_pose_global)
    return transform_points(points_global, T_ego_global)


# ---------------------------------------------------------------------------
# 3D box operations
# ---------------------------------------------------------------------------

def get_3d_box_corners(center: np.ndarray,
                       size: np.ndarray,
                       yaw: float) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box.

    Args:
        center: [3] (x, y, z) in ego frame
        size: [3] (w, l, h) width, length, height
        yaw: rotation around z-axis (radians)

    Returns:
        corners: [8, 3] ordered as a cuboid
    """
    w, l, h = float(size[0]), float(size[1]), float(size[2])
    # corners in object-local frame (centered at origin)
    # x=forward (length l), y=left (width w), z=up (height h)
    x_corners = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2.0
    y_corners = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2.0
    z_corners = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * h / 2.0
    corners_local = np.stack([x_corners, y_corners, z_corners], axis=1)  # [8, 3]

    # rotation around z
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    Rz = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    corners = corners_local @ Rz.T + center.reshape(1, 3)
    return corners


def get_box_centroid(box_3d: List[float]) -> np.ndarray:
    """Extract centroid [x, y, z] from box_3d [cx, cy, cz, w, l, h, yaw]."""
    return np.array(box_3d[:3], dtype=np.float32)


def get_box_yaw(box_3d: List[float]) -> float:
    """Extract yaw from box_3d."""
    return float(box_3d[6])


# ---------------------------------------------------------------------------
# Camera projection
# ---------------------------------------------------------------------------

def project_points_to_camera(points_ego: np.ndarray,
                              K: np.ndarray,
                              T_ego_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D ego-frame points into camera pixel coordinates.

    Args:
        points_ego: [N, 3] in ego frame
        K: [3, 3] camera intrinsic matrix
        T_ego_cam: [4, 4] ego-to-camera transform (T_cam <- ego)

    Returns:
        uv: [N, 2] pixel coordinates (u, v)
        depth: [N] depth in camera frame
        valid_depth: [N] boolean, True if depth > 0
    """
    if isinstance(K, list):
        K = np.array(K, dtype=np.float32)
    if isinstance(T_ego_cam, list):
        T_ego_cam = np.array(T_ego_cam, dtype=np.float32)

    # Transform to camera frame
    points_cam = transform_points(points_ego, T_ego_cam)  # [N, 3]

    depth = points_cam[:, 2]
    valid_depth = depth > 1e-3

    # Perspective projection
    uv_h = (K @ points_cam.T).T  # [N, 3]
    uv = np.zeros((points_ego.shape[0], 2), dtype=np.float32)
    valid = valid_depth
    uv[valid, 0] = uv_h[valid, 0] / uv_h[valid, 2]
    uv[valid, 1] = uv_h[valid, 1] / uv_h[valid, 2]

    return uv, depth, valid_depth


def project_box_to_image(box_3d: List[float],
                          K: np.ndarray,
                          T_ego_cam: np.ndarray) -> np.ndarray:
    """Project 3D box to 2D image corners.

    Args:
        box_3d: [cx, cy, cz, w, l, h, yaw]
        K, T_ego_cam: camera params

    Returns:
        corners_2d: [8, 2] image coordinates
    """
    center = np.array(box_3d[:3], dtype=np.float32)
    size = np.array(box_3d[3:6], dtype=np.float32)
    yaw = box_3d[6]
    corners_3d = get_3d_box_corners(center, size, yaw)
    uv, _, _ = project_points_to_camera(corners_3d, K, T_ego_cam)
    return uv


def project_box_to_image_with_depth(box_3d: List[float],
                                     K: np.ndarray,
                                     T_ego_cam: np.ndarray):
    """Project 3D box to image and keep depth validity.

    Args:
        box_3d: [cx, cy, cz, w, l, h, yaw]
        K, T_ego_cam: camera params

    Returns:
        uv: [8, 2] pixel coordinates
        depth: [8] depth in camera frame
        valid_depth: [8] boolean, True if depth > 0
    """
    center = np.array(box_3d[:3], dtype=np.float32)
    size = np.array(box_3d[3:6], dtype=np.float32)
    yaw = box_3d[6]
    corners_3d = get_3d_box_corners(center, size, yaw)
    uv, depth, valid_depth = project_points_to_camera(corners_3d, K, T_ego_cam)
    return uv, depth, valid_depth


# ---------------------------------------------------------------------------
# Visibility checks
# ---------------------------------------------------------------------------

def is_point_in_image(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """Check if pixel coordinates are within image bounds."""
    return (uv[:, 0] >= 0) & (uv[:, 0] < width) & \
           (uv[:, 1] >= 0) & (uv[:, 1] < height)


def compute_projected_bbox_area(uv_corners: np.ndarray,
                                 width: int, height: int) -> float:
    """Compute area of 2D bounding box from projected 3D corners."""
    valid = is_point_in_image(uv_corners, width, height)
    if valid.sum() < 2:
        return 0.0
    u_min = np.min(uv_corners[valid, 0])
    u_max = np.max(uv_corners[valid, 0])
    v_min = np.min(uv_corners[valid, 1])
    v_max = np.max(uv_corners[valid, 1])
    # clip to image bounds
    u_min = max(0, u_min)
    u_max = min(width, u_max)
    v_min = max(0, v_min)
    v_max = min(height, v_max)
    if u_max <= u_min or v_max <= v_min:
        return 0.0
    return float((u_max - u_min) * (v_max - v_min))


def is_box_visible_in_camera(box_3d: List[float],
                              K: np.ndarray,
                              T_ego_cam: np.ndarray,
                              width: int,
                              height: int,
                              min_corners: int = 1,
                              min_area: float = 50.0) -> bool:
    """Check if a 3D box is visible in camera view.

    Args:
        box_3d: [cx, cy, cz, w, l, h, yaw]
        K, T_ego_cam: camera calibration
        width, height: image dimensions
        min_corners: minimum corners inside image
        min_area: minimum projected area in pixels

    Returns:
        True if visible
    """
    uv, depth, valid_depth = project_box_to_image_with_depth(box_3d, K, T_ego_cam)

    if valid_depth.sum() == 0:
        return False

    in_img = valid_depth & is_point_in_image(uv, width, height)

    if in_img.sum() >= min_corners:
        return True

    if valid_depth.sum() < 2:
        return False

    area = compute_projected_bbox_area(uv[valid_depth], width, height)
    return area > min_area


def is_box_visible_in_any_camera(box_3d: List[float],
                                  cameras: List[dict],
                                  min_cameras: int = 1) -> bool:
    """Check if a box is visible in at least min_cameras cameras."""
    visible_count = 0
    for cam in cameras:
        if is_box_visible_in_camera(
            box_3d,
            np.array(cam["K"], dtype=np.float32),
            np.array(cam["T_ego_cam"], dtype=np.float32),
            cam["width"],
            cam["height"]
        ):
            visible_count += 1
            if visible_count >= min_cameras:
                return True
    return False


# ---------------------------------------------------------------------------
# Direction and distance labeling
# ---------------------------------------------------------------------------

def compute_direction(center_ego: np.ndarray) -> str:
    """Classify object direction relative to ego vehicle.

    Args:
        center_ego: [3] array (x=forward, y=left, z=up)

    Returns:
        direction label
    """
    x, y = center_ego[0], center_ego[1]

    if x > 5:
        if y > 2:
            return "front-left"
        elif y < -2:
            return "front-right"
        else:
            return "front"
    elif x < -5:
        if y > 2:
            return "rear-left"
        elif y < -2:
            return "rear-right"
        else:
            return "rear"
    else:
        if y > 2:
            return "left"
        elif y < -2:
            return "right"
        else:
            return "near-ego"


def bin_distance_x(x: float) -> str:
    """Bin longitudinal distance into human-readable levels."""
    if x < 0:
        return "behind"
    if x < 5:
        return "0-5m"
    if x < 10:
        return "5-10m"
    if x < 15:
        return "10-15m"
    if x < 20:
        return "15-20m"
    if x < 30:
        return "20-30m"
    if x < 40:
        return "30-40m"
    if x < 60:
        return "40-60m"
    return "60m+"


def bin_distance_y(y: float) -> str:
    """Bin lateral offset into human-readable levels."""
    if y < -10:
        return "right_10m+"
    if y < -5:
        return "right_5_10m"
    if y < -2:
        return "right_0_5m"
    if y <= 2:
        return "center"
    if y <= 5:
        return "left_0_5m"
    if y <= 10:
        return "left_5_10m"
    return "left_10m+"


def bin_distance_x_numeric(x: float) -> int:
    """Return numeric bin for distance x (for metric computation)."""
    boundaries = [-float("inf"), 0, 5, 10, 15, 20, 30, 40, 60, float("inf")]
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if lo <= x < hi:
            return i
    return len(boundaries) - 1


def bin_distance_y_numeric(y: float) -> int:
    """Return numeric bin for lateral y (for metric computation)."""
    boundaries = [-float("inf"), -10, -5, -2, 2, 5, 10, float("inf")]
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if lo < y <= hi:
            return i
    return len(boundaries) - 1


# ---------------------------------------------------------------------------
# Ego path overlap
# ---------------------------------------------------------------------------

def check_ego_path_overlap(center_ego: np.ndarray,
                            box_size: np.ndarray,
                            x_min: float = 0.0,
                            x_max: float = 50.0,
                            half_width: float = 1.8,
                            yaw: float = None) -> bool:
    """Check if a 3D box overlaps with the ego lane corridor.

    When yaw is provided, computes the 4 BEV corners and checks whether
    any corner lies inside the corridor or any edge crosses a boundary.
    When yaw is None, falls back to a center-based check.

    Args:
        center_ego: [3] box center in ego frame
        box_size: [3] (w, l, h) — width, length, height
        x_min, x_max: longitudinal corridor range
        half_width: half-width of ego corridor in meters
        yaw: optional rotation around z (radians)

    Returns:
        True if the box overlaps the corridor
    """
    x, y = float(center_ego[0]), float(center_ego[1])
    w, l = float(box_size[0]), float(box_size[1])

    # Quick reject: box entirely behind or too far ahead
    if x + l / 2.0 < x_min or x - l / 2.0 > x_max:
        return False

    if yaw is not None:
        # Full 4-corner BEV overlap check
        hl, hw = l / 2.0, w / 2.0
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        corners_local = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw],
        ], dtype=np.float32)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
        corners = corners_local @ R.T + np.array([x, y], dtype=np.float32)

        # Check if any corner is inside the corridor
        inside_x = (corners[:, 0] >= x_min) & (corners[:, 0] <= x_max)
        inside_y = abs(corners[:, 1]) <= half_width
        if np.any(inside_x & inside_y):
            return True

        # Check if the box spans across the corridor laterally
        # (both y_min < -half_width AND y_max > half_width, with x overlap)
        y_vals = corners[:, 1]
        if np.min(y_vals) <= -half_width and np.max(y_vals) >= half_width:
            if np.any(inside_x):
                return True

        # Check if the box spans across the corridor longitudinally
        x_vals = corners[:, 0]
        if np.min(x_vals) <= x_min and np.max(x_vals) >= x_max:
            if np.any(inside_y):
                return True

        return False

    # Fallback: center-based check
    return x_min < x < x_max and abs(y) < half_width + w / 2.0


# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------

REGION_DEFS = {
    "front":       lambda x, y: x > 5 and abs(y) <= 2,
    "front-left":  lambda x, y: x > 5 and y > 2,
    "front-right": lambda x, y: x > 5 and y < -2,
    "rear":        lambda x, y: x < -5 and abs(y) <= 2,
    "rear-left":   lambda x, y: x < -5 and y > 2,
    "rear-right":  lambda x, y: x < -5 and y < -2,
    "left":        lambda x, y: abs(x) <= 5 and y > 2,
    "right":       lambda x, y: abs(x) <= 5 and y < -2,
}


def classify_region(center_ego: np.ndarray) -> Optional[str]:
    """Classify which region a point falls into."""
    x, y = center_ego[0], center_ego[1]
    for name, pred in REGION_DEFS.items():
        if pred(x, y):
            return name
    return "near-ego"


# ---------------------------------------------------------------------------
# Region visibility
# ---------------------------------------------------------------------------

def get_region_anchor_points(region_name: str,
                              num_anchors: int = 9) -> np.ndarray:
    """Generate anchor points distributed within a region.

    Returns [num_anchors, 3] in ego frame.
    """
    bounds = {
        "front":       (np.array([10, 30, 50]), np.array([-1, 0, 1])),
        "front-left":  (np.array([10, 30, 50]), np.array([3, 10, 20])),
        "front-right": (np.array([10, 30, 50]), np.array([-20, -10, -3])),
        "rear":        (np.array([-10, -25, -40]), np.array([-1, 0, 1])),
        "rear-left":   (np.array([-10, -25, -40]), np.array([3, 10, 20])),
        "rear-right":  (np.array([-10, -25, -40]), np.array([-20, -10, -3])),
        "left":        (np.array([-3, 0, 3]), np.array([3, 10, 20])),
        "right":       (np.array([-3, 0, 3]), np.array([-20, -10, -3])),
    }
    if region_name not in bounds:
        return np.zeros((0, 3), dtype=np.float32)

    xs, ys = bounds[region_name]
    anchors = []
    for x in xs:
        for y in ys:
            anchors.append([x, y, 1.0])  # z=1m for rough object height
    return np.array(anchors, dtype=np.float32)


def is_region_observed(region_name: str,
                        camera_subset: List[str],
                        calibrations: List[dict],
                        min_visible_anchors: int = 1) -> bool:
    """Check if a spatial region is observed by any camera in the subset.

    Samples anchor points in the region and projects to all cameras.

    Args:
        region_name: one of REGION_DEFS keys
        camera_subset: list of camera names
        calibrations: list of per-camera dicts with K, T_ego_cam, width, height
        min_visible_anchors: minimum anchors that must be visible

    Returns:
        True if the region is visible
    """
    anchors = get_region_anchor_points(region_name)
    if len(anchors) == 0:
        return False

    cam_by_name = {c["name"]: c for c in calibrations}

    for anchor in anchors:
        for cam_name in camera_subset:
            if cam_name not in cam_by_name:
                continue
            cam = cam_by_name[cam_name]
            uv, depth, valid = project_points_to_camera(
                anchor.reshape(1, 3),
                np.array(cam["K"], dtype=np.float32),
                np.array(cam["T_ego_cam"], dtype=np.float32)
            )
            if valid[0] and depth[0] > 0:
                u, v = uv[0, 0], uv[0, 1]
                if 0 <= u < cam["width"] and 0 <= v < cam["height"]:
                    min_visible_anchors -= 1
                    if min_visible_anchors <= 0:
                        return True
                    break  # anchor visible in at least one camera

    return min_visible_anchors <= 0


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def gaussian_heatmap(center_xy: Tuple[float, float],
                      x_range: Tuple[float, float],
                      y_range: Tuple[float, float],
                      grid_sx: int,
                      grid_sy: int,
                      sigma: float = 1.5) -> np.ndarray:
    """Generate 2D Gaussian heatmap centered at (cx, cy).

    Args:
        center_xy: (cx, cy) in ego coordinates
        x_range: (x_min, x_max) in meters
        y_range: (y_min, y_max) in meters
        grid_sx: grid resolution in x
        grid_sy: grid resolution in y
        sigma: Gaussian sigma in grid cells

    Returns:
        heatmap: [grid_sy, grid_sx]
    """
    cx, cy = center_xy
    x_min, x_max = x_range
    y_min, y_max = y_range

    xs = np.linspace(x_min, x_max, grid_sx)
    ys = np.linspace(y_min, y_max, grid_sy)
    xx, yy = np.meshgrid(xs, ys)

    # sigma in meters (sigma cells * cell_size)
    cell_size_x = (x_max - x_min) / grid_sx
    cell_size_y = (y_max - y_min) / grid_sy
    sigma_x = sigma * cell_size_x
    sigma_y = sigma * cell_size_y

    heatmap = np.exp(-((xx - cx) ** 2 / (2 * sigma_x ** 2) +
                        (yy - cy) ** 2 / (2 * sigma_y ** 2)))
    # normalize peak to 1
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.astype(np.float32)


def world_to_grid(x: float, y: float,
                   x_range: Tuple[float, float],
                   y_range: Tuple[float, float],
                   grid_sx: int,
                   grid_sy: int) -> Tuple[int, int]:
    """Convert world coordinate to grid index."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    gx = int((x - x_min) / (x_max - x_min) * grid_sx)
    gy = int((y - y_min) / (y_max - y_min) * grid_sy)
    gx = max(0, min(grid_sx - 1, gx))
    gy = max(0, min(grid_sy - 1, gy))
    return gx, gy


def grid_to_world(gx: int, gy: int,
                   x_range: Tuple[float, float],
                   y_range: Tuple[float, float],
                   grid_sx: int,
                   grid_sy: int) -> Tuple[float, float]:
    """Convert grid index to world coordinate (cell center)."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    cell_x = (x_max - x_min) / grid_sx
    cell_y = (y_max - y_min) / grid_sy
    x = x_min + (gx + 0.5) * cell_x
    y = y_min + (gy + 0.5) * cell_y
    return x, y


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # xyzw format for scipy
    return r.as_matrix()


def nusc_ego_to_cam_transform(calibrated_sensor: dict) -> np.ndarray:
    """Build T_ego_cam (T_cam<-ego) from nuScenes calibrated_sensor record.

    nuScenes provides translation + rotation from sensor to ego.
    We need the inverse: T_cam_ego = inv(T_sensor_ego).
    """
    translation = np.array(calibrated_sensor["translation"], dtype=np.float32)
    rotation = np.array(calibrated_sensor["rotation"], dtype=np.float32)
    R = quaternion_to_rotation_matrix(rotation)
    # nuScenes gives T_sensor<-ego: p_ego = R * p_sensor + t
    # We want T_cam<-ego, which is the INVERSE
    T_sensor_ego = make_transform_4x4(R, translation)
    T_ego_sensor = invert_transform(T_sensor_ego)
    return T_ego_sensor


def nusc_ego_pose_to_matrix(ego_pose: dict) -> np.ndarray:
    """Build 4x4 ego pose matrix from nuScenes ego_pose record."""
    translation = np.array(ego_pose["translation"], dtype=np.float32)
    rotation = np.array(ego_pose["rotation"], dtype=np.float32)
    R = quaternion_to_rotation_matrix(rotation)
    return make_transform_4x4(R, translation)
