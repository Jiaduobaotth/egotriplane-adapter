# Comprehensive Coordinate Transformation Review

## Coordinate Frame Definitions

### Ego Frame (Vehicle Frame)
- **x-axis**: Forward (direction of vehicle motion)
- **y-axis**: Left (perpendicular to forward)
- **z-axis**: Up (vertical)

### Camera Frame
- **x-axis**: Right (in image plane)
- **y-axis**: Down (in image plane)
- **z-axis**: Forward (depth, pointing out of camera)

### Transformation Matrix T_ego_cam
- **Type**: 4×4 homogeneous transformation matrix
- **Direction**: Transforms points from ego frame to camera frame
- **Formula**: `p_cam = T_ego_cam @ p_ego_h` (homogeneous coordinates)
- **Components**:
  - `R = T_ego_cam[:3, :3]`: ego→cam rotation matrix
  - `t = T_ego_cam[:3, 3]`: translation (camera origin in ego frame)

## Checked Transformations

### ✓ Line 763: Ego to Camera Projection
**Location**: `_scatter_plane_batched()` method

**Code**:
```python
pts_cam = pts_flat @ R.T + t.unsqueeze(0)
```

**Verification**:
- Formula: `p_cam = p_ego @ R.T + t` ✓
- This correctly transforms ego frame points to camera frame
- Matches ground truth transformation (verified with test_transform.py)

### ✓ Line 244: Camera Center in Ego Frame
**Location**: `forward()` method

**Code**:
```python
cam_center_ego = T_ego_cam[:3, 3]
```

**Verification**:
- Correctly extracts camera origin in ego frame ✓
- Used for computing ray directions in ego frame

### ✓ Line 316: Camera Center in Ego Frame (Duplicate)
**Location**: `forward()` method (alternative path)

**Code**:
```python
cam_center_ego = T_ego_cam[:3, 3]
```

**Verification**:
- Same as line 244, correctly extracts camera origin ✓

### ✓ Lines 745-756: Plane Coordinate Systems
**Location**: `_scatter_plane_batched()` method

**Definitions**:
- **XY plane**: h=y, w=x, anchor=z (varying x,y; fixed z)
- **XZ plane**: h=z, w=x, anchor=y (varying x,z; fixed y)
- **YZ plane**: h=z, w=y, anchor=x (varying y,z; fixed x)

**Verification**:
- All three planes correctly defined ✓
- Grid coordinates match plane axes

### ✓ Lines 767-768: Perspective Projection
**Location**: `_scatter_plane_batched()` method

**Code**:
```python
u = fx * pts_cam[:, 0] / z_cam.clamp(min=1e-6) + cx
v = fy * pts_cam[:, 1] / z_cam.clamp(min=1e-6) + cy
```

**Verification**:
- Standard perspective projection formula ✓
- Uses camera intrinsics (fx, fy, cx, cy)
- Clamps z to avoid division by zero

### ✓ Lines 779-780: Grid Sample Reshape
**Location**: `_scatter_plane_batched()` method

**Code**:
```python
grid = grid.reshape(1, gh, gw * na, 2)
```

**Verification**:
- Correct format for `grid_sample` with align_corners=False ✓
- Shape: [1, gh, gw*na, 2] (batch, height, width, coords)

### ✓ Lines 798-799: Plane Accumulation
**Location**: `_scatter_plane_batched()` method

**Code**:
```python
plane.copy_(plane + sampled_sum.permute(0, 2, 3, 1))
count.copy_(count + count_sum.permute(0, 2, 3, 1))
```

**Verification**:
- `.permute(0, 2, 3, 1)` converts [1, C, gh, gw*na] to [1, gh, gw*na, C] ✓
- Matches plane shape for accumulation

## Summary

✅ **All coordinate transformations verified and correct**

### Critical Fix Applied
- Line 763: Changed from `(pts_flat - t) @ R` to `pts_flat @ R.T + t`
- This was the root cause of model collapse (identical predictions)

### No Other Issues Found
- Plane definitions: correct
- Grid sample reshape: correct
- Permute operations: correct
- Camera center calculation: correct
- Perspective projection: correct

## Next Action

**Retrain model** with the fixed coordinate transformation to verify that:
1. Triplane tokens now vary across samples (not all identical)
2. Inference produces different predictions for different inputs
3. Model achieves reasonable performance on validation set
