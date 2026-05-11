# Coordinate Transformation Fixes - Critical Review

## Summary
Fixed critical coordinate transformation bug in `egotriplane/triplane_adapter.py` line 763 that was preventing proper projection of camera features onto triplane planes.

## The Bug

### Location
`egotriplane/triplane_adapter.py`, line 763 in `_scatter_plane_batched()` method

### What Was Wrong
```python
# WRONG - this is cam->ego transformation, not ego->cam
pts_cam = (pts_flat - t.unsqueeze(0)) @ R
```

### Why It Was Wrong
1. `T_ego_cam` is a 4×4 transformation matrix that transforms points from **ego frame to camera frame**
2. `R = T_ego_cam[:3, :3]` is the ego→cam rotation matrix
3. `t = T_ego_cam[:3, 3]` is the translation component
4. The correct formula for ego→cam transformation is: `p_cam = p_ego @ R.T + t`
5. The old formula `(p_ego - t) @ R` is the inverse transformation (cam→ego)

### The Fix
```python
# CORRECT - ego->cam transformation
pts_cam = pts_flat @ R.T + t.unsqueeze(0)
```

## Verification

Created `test_transform.py` to verify the transformation:
- Correct formula matches ground truth (error ≈ 0)
- Wrong formula produces completely different results

## Impact

This bug caused:
1. Camera features to be projected to wrong locations in triplane planes
2. Features not being scattered into the correct grid cells
3. Triplane tokens becoming identical across all samples (complete model collapse)

## Other Checks Performed

### Plane Definitions ✓
- XY plane: h=y, w=x (correct)
- XZ plane: h=z, w=x (correct)
- YZ plane: h=z, w=y (correct)

### Grid Sample Reshape ✓
- Reshape to `[1, gh, gw*na, 2]` is correct for grid_sample

### Permute Operations ✓
- `.permute(0, 2, 3, 1)` correctly converts `[1, C, gh, gw*na]` to `[1, gh, gw*na, C]`

### Camera Center Calculation ✓
- `cam_center_ego = T_ego_cam[:3, 3]` correctly gets camera origin in ego frame

## Next Steps

1. Retrain model with fixed coordinate transformation
2. Run diagnostic script to verify triplane tokens now vary across samples
3. Verify inference produces different predictions for different samples
