"""EgoTriPlane-Adapter: the core module.

Projects multi-camera 2D features into a unified ego-coordinate
triplane representation (XY, XZ, YZ planes). The triplane tokens
are independent of camera count and resolution, making the
representation robust to sensor configuration changes.

Architecture:
  1. Per-camera feature unprojection + ray embedding
  2. Projection-guided sampling onto triplane grids
  3. Cross-plane feature aggregation
  4. Patchify into fixed token count (~288 for default config)

Supports:
  - Multi-scale features from vision backbone layers
  - Batched multi-camera input [B, N_cam, C, H, W]
  - Camera-count-agnostic output tokens

Reference grid (first version):
  x_range: [-20, 80]  (forward),  sx=96,  patch_size=8
  y_range: [-40, 40]  (left),     sy=96,  patch_size=8
  z_range: [-3, 8]    (up),       sz=48,  patch_size=8

Tokens: (96/8)*(96/8) + (96/8)*(48/8) + (96/8)*(48/8)
       = 12*12 + 12*6 + 12*6 = 144 + 72 + 72 = 288
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from einops import rearrange, repeat


class EgoTriPlaneAdapter(nn.Module):
    """Project multi-camera features to ego-coordinate triplane tokens.

    The adapter is lightweight (a few million params) and works with
    frozen vision encoders. Output token count is fixed regardless
    of number of cameras or image resolution.
    """

    def __init__(
        self,
        feature_dim: Union[int, List[int]] = 1024,  # int or list for multi-scale
        hidden_dim: int = 512,          # internal hidden dimension
        x_range: Tuple[float, float] = (-20.0, 80.0),
        y_range: Tuple[float, float] = (-40.0, 40.0),
        z_range: Tuple[float, float] = (-3.0, 8.0),
        sx: int = 96,                   # grid resolution x
        sy: int = 96,                   # grid resolution y
        sz: int = 48,                   # grid resolution z
        patch_size: int = 8,
        use_ray_embedding: bool = True,
        aggregation: str = "mean",      # "mean" or "attention"
        multi_scale_mode: str = "concat",  # "concat", "sum", "hierarchical"
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.multi_scale_mode = multi_scale_mode
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.patch_size = patch_size
        self.use_ray_embedding = use_ray_embedding
        self.aggregation = aggregation

        # Derived sizes
        self.tokens_xy = (sx // patch_size) * (sy // patch_size)  # 144
        self.tokens_xz = (sx // patch_size) * (sz // patch_size)  # 72
        self.tokens_yz = (sy // patch_size) * (sz // patch_size)  # 72
        self.num_tokens = self.tokens_xy + self.tokens_xz + self.tokens_yz  # 288

        # Grid cell sizes in meters
        self.cell_x = (x_range[1] - x_range[0]) / sx
        self.cell_y = (y_range[1] - y_range[0]) / sy
        self.cell_z = (z_range[1] - z_range[0]) / sz

        # Multi-scale feature handling
        if isinstance(feature_dim, (list, tuple)):
            self._scale_dims = list(feature_dim)
            self._multi_scale = True
        else:
            self._scale_dims = [feature_dim]
            self._multi_scale = False

        # Input projection(s): per-scale linear -> hidden_dim
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for d in self._scale_dims
        ])
        # Keep self.input_proj for backward compat
        self.input_proj = self.input_projs[0]

        # Multi-scale fusion if >1 scale
        if self._multi_scale and multi_scale_mode == "concat":
            self.scale_fusion = nn.Sequential(
                nn.Linear(hidden_dim * len(self._scale_dims), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
        elif self._multi_scale and multi_scale_mode == "hierarchical":
            # FPN-style fusion: smaller scales upsampled and added
            self.scale_fusion = None  # handled in forward
        else:
            self.scale_fusion = None  # "sum" mode or single scale

        # Ray embedding MLP
        if use_ray_embedding:
            self.ray_mlp = nn.Sequential(
                nn.Linear(6, hidden_dim // 2),   # origin(3) + direction(3)
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )

        # Triplane learnable embeddings (initialized as learnable queries)
        self.xy_embed = nn.Parameter(torch.randn(sy, sx, hidden_dim) * 0.02)
        self.xz_embed = nn.Parameter(torch.randn(sz, sx, hidden_dim) * 0.02)
        self.yz_embed = nn.Parameter(torch.randn(sz, sy, hidden_dim) * 0.02)

        # Post-fusion refinement: light conv after populating planes
        self.xy_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.xz_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.yz_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        # Optional attention aggregation: per-cell per-camera learned weights
        if aggregation == "attention":
            # Per-plane score MLPs: input [C] → output scalar confidence
            self.cam_score_xy = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, 1, 1),
            )
            self.cam_score_xz = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, 1, 1),
            )
            self.cam_score_yz = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, 1, 1),
            )

        # Patchify projection
        self.patch_proj = nn.Linear(patch_size * patch_size * hidden_dim, hidden_dim)

        # Output norm
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features_by_camera: Dict[str, Dict],
        camera_subset: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (per-sample, dict-based input).

        Args:
            features_by_camera: dict mapping camera name to cached feature dict.
                Each dict has:
                  - "features": [Hf*Wf, D] or [Hf, Wf, D] patch features.
                    For multi-scale: can also be a list of tensors or dict
                    {"scale0": tensor, "scale1": tensor, ...}
                  - "K": [3, 3] intrinsics
                  - "T_ego_cam": [4, 4] extrinsics
                  - "image_size": [H, W]
            camera_subset: list of camera names to use (default: all)

        Returns:
            dict with:
              - "triplane_tokens": [1, num_tokens, hidden_dim]
              - "tokens_xy": [1, tokens_xy, hidden_dim]
              - "tokens_xz": [1, tokens_xz, hidden_dim]
              - "tokens_yz": [1, tokens_yz, hidden_dim]
              - "plane_xy": [1, hidden_dim, sy, sx]
              - "plane_xz": [1, hidden_dim, sz, sx]
              - "plane_yz": [1, hidden_dim, sz, sy]
              - "multi_scale_features": dict of intermediate features (if multi-scale)
        """
        B = 1  # We process one sample at a time; caller can batch
        device = self.xy_embed.device

        if camera_subset is None:
            camera_subset = list(features_by_camera.keys())
        N_cam = len(camera_subset)

        # --- Step 1: Initialize triplane grids ---
        plane_xy = self.xy_embed.clone().unsqueeze(0)  # [1, sy, sx, C]
        plane_xz = self.xz_embed.clone().unsqueeze(0)  # [1, sz, sx, C]
        plane_yz = self.yz_embed.clone().unsqueeze(0)  # [1, sz, sy, C]

        if self.aggregation == "attention":
            # --- Attention mode: per-camera planes, then learned fusion ---
            planes_xy_cams = []   # list of [1, C, sy, sx]
            planes_xz_cams = []   # list of [1, C, sz, sx]
            planes_yz_cams = []   # list of [1, C, sz, sy]

            for cam_name in camera_subset:
                if cam_name not in features_by_camera:
                    continue
                cam_data = features_by_camera[cam_name]
                feats = cam_data["features"]
                K = cam_data["K"]
                T_ego_cam = cam_data["T_ego_cam"]
                img_size = cam_data.get("image_size", [224, 224])
                if isinstance(feats, torch.Tensor):
                    feats = feats.to(device)
                if isinstance(K, torch.Tensor):
                    K = K.to(device)
                if isinstance(T_ego_cam, torch.Tensor):
                    T_ego_cam = T_ego_cam.to(device)

                # Prepare feature map
                multi_scale_feats = self._extract_multi_scale_features(feats, cam_data)
                feats_2d = self._fuse_multi_scale(multi_scale_feats)
                _, _, Hf, Wf = feats_2d.shape

                cam_center_ego = T_ego_cam[:3, 3]  # camera position in ego frame

                ray_origins, ray_directions = _compute_patch_rays(
                    cam_center_ego, K, T_ego_cam, Hf, Wf, img_size, device,
                )
                if self.use_ray_embedding:
                    ray_emb = self.ray_mlp(
                        torch.cat([ray_origins, ray_directions], dim=-1)
                    )
                    ray_emb = ray_emb.reshape(1, Hf, Wf, self.hidden_dim).permute(0, 3, 1, 2)
                    feats_2d = feats_2d + ray_emb

                # Fresh plane clones for this camera
                cam_xy = self.xy_embed.clone().unsqueeze(0)  # [1, sy, sx, C]
                cam_xz = self.xz_embed.clone().unsqueeze(0)  # [1, sz, sx, C]
                cam_yz = self.yz_embed.clone().unsqueeze(0)  # [1, sz, sy, C]
                c_xy = torch.zeros(1, self.sy, self.sx, 1, device=device)
                c_xz = torch.zeros(1, self.sz, self.sx, 1, device=device)
                c_yz = torch.zeros(1, self.sz, self.sy, 1, device=device)

                _scatter_to_triplane(
                    feats_2d, cam_xy, cam_xz, cam_yz,
                    c_xy, c_xz, c_yz,
                    K, T_ego_cam, cam_center_ego,
                    self.x_range, self.y_range, self.z_range,
                    self.sx, self.sy, self.sz,
                    Hf, Wf, img_size, device,
                )

                # Normalize and convert to [1, C, H, W] (channels-first)
                eps = 1e-8
                planes_xy_cams.append(
                    (cam_xy / (c_xy + eps)).permute(0, 3, 1, 2)  # [1, C, sy, sx]
                )
                planes_xz_cams.append(
                    (cam_xz / (c_xz + eps)).permute(0, 3, 1, 2)  # [1, C, sz, sx]
                )
                planes_yz_cams.append(
                    (cam_yz / (c_yz + eps)).permute(0, 3, 1, 2)  # [1, C, sz, sy]
                )

            # --- Attention fusion over cameras ---
            plane_xy = self._fuse_cameras_attention(planes_xy_cams, self.cam_score_xy)
            plane_xz = self._fuse_cameras_attention(planes_xz_cams, self.cam_score_xz)
            plane_yz = self._fuse_cameras_attention(planes_yz_cams, self.cam_score_yz)

        else:
            # --- Mean mode: accumulate into shared planes ---
            xy_count = torch.zeros(1, self.sy, self.sx, 1, device=device)
            xz_count = torch.zeros(1, self.sz, self.sx, 1, device=device)
            yz_count = torch.zeros(1, self.sz, self.sy, 1, device=device)

            for cam_name in camera_subset:
                if cam_name not in features_by_camera:
                    continue
                cam_data = features_by_camera[cam_name]
                feats = cam_data["features"]
                K = cam_data["K"]
                T_ego_cam = cam_data["T_ego_cam"]
                img_size = cam_data.get("image_size", [224, 224])
                if isinstance(feats, torch.Tensor):
                    feats = feats.to(device)
                if isinstance(K, torch.Tensor):
                    K = K.to(device)
                if isinstance(T_ego_cam, torch.Tensor):
                    T_ego_cam = T_ego_cam.to(device)

                multi_scale_feats = self._extract_multi_scale_features(feats, cam_data)
                feats_2d = self._fuse_multi_scale(multi_scale_feats)
                _, _, Hf, Wf = feats_2d.shape

                cam_center_ego = T_ego_cam[:3, 3]  # camera position in ego frame

                ray_origins, ray_directions = _compute_patch_rays(
                    cam_center_ego, K, T_ego_cam, Hf, Wf, img_size, device,
                )
                if self.use_ray_embedding:
                    ray_emb = self.ray_mlp(
                        torch.cat([ray_origins, ray_directions], dim=-1)
                    )
                    ray_emb = ray_emb.reshape(1, Hf, Wf, self.hidden_dim).permute(0, 3, 1, 2)
                    feats_2d = feats_2d + ray_emb

                _scatter_to_triplane(
                    feats_2d, plane_xy, plane_xz, plane_yz,
                    xy_count, xz_count, yz_count,
                    K, T_ego_cam, cam_center_ego,
                    self.x_range, self.y_range, self.z_range,
                    self.sx, self.sy, self.sz,
                    Hf, Wf, img_size, device,
                )

            eps = 1e-8
            plane_xy = (plane_xy / (xy_count + eps)).permute(0, 3, 1, 2)  # [1, C, sy, sx]
            plane_xz = (plane_xz / (xz_count + eps)).permute(0, 3, 1, 2)  # [1, C, sz, sx]
            plane_yz = (plane_yz / (yz_count + eps)).permute(0, 3, 1, 2)  # [1, C, sz, sy]

        # --- Step 5: Refine (planes are [1, C, H, W]) ---
        plane_xy = self.xy_refine(plane_xy)  # [1, C, sy, sx]
        plane_xz = self.xz_refine(plane_xz)  # [1, C, sz, sx]
        plane_yz = self.yz_refine(plane_yz)  # [1, C, sz, sy]

        # --- Step 6: Patchify ---
        tokens_xy = _patchify(plane_xy, self.patch_size)  # [1, 144, C]
        tokens_xz = _patchify(plane_xz, self.patch_size)  # [1, 72, C]
        tokens_yz = _patchify(plane_yz, self.patch_size)  # [1, 72, C]

        tokens_xy = self.patch_proj(tokens_xy)
        tokens_xz = self.patch_proj(tokens_xz)
        tokens_yz = self.patch_proj(tokens_yz)

        # Concatenate all triplane tokens
        triplane_tokens = torch.cat([tokens_xy, tokens_xz, tokens_yz], dim=1)
        triplane_tokens = self.output_norm(triplane_tokens)

        return {
            "triplane_tokens": triplane_tokens,
            "tokens_xy": tokens_xy,
            "tokens_xz": tokens_xz,
            "tokens_yz": tokens_yz,
            "plane_xy": plane_xy,
            "plane_xz": plane_xz,
            "plane_yz": plane_yz,
        }

    def forward_batched(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        vision_encoder: Optional[nn.Module] = None,
        multi_scale_layers: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Batched forward: [B, N_cam, C, H, W] -> triplane tokens.

        For use when vision features are NOT pre-extracted.
        Optionally runs a vision encoder on flattened images first.

        Args:
            images: [B, N_cam, C, H, W]
            intrinsics: [B, N_cam, 3, 3]
            extrinsics: [B, N_cam, 4, 4]
            image_sizes: list of (H, W) per camera, or None to derive from images
            vision_encoder: optional nn.Module to extract features from images.
                If None, images are treated as pre-extracted feature maps.
            multi_scale_layers: which backbone layers to extract (for multi-scale).

        Returns:
            dict with triplane tokens & planes, each [B, ...].
        """
        B, N_cam, C, H, W = images.shape
        device = images.device

        # Flatten batch+ cameras for encoder
        if vision_encoder is not None:
            images_flat = images.reshape(B * N_cam, C, H, W)
            with torch.no_grad():
                if multi_scale_layers:
                    # Extract multi-scale features (model-dependent)
                    features_flat = vision_encoder(images_flat, output_hidden_states=True)
                    # TODO: actual layer selection logic
                else:
                    features_flat = vision_encoder(images_flat)
            # Reshape back to [B, N_cam, ...]
            # (feature shape depends on encoder)
            # For now: placeholder
            raise NotImplementedError(
                "forward_batched with vision_encoder: implement feature extraction"
            )
        else:
            # Treat images as pre-extracted feature maps [B, N_cam, D, Hf, Wf]
            features_flat = images.reshape(B * N_cam, C, H, W)

        # Process each sample in the batch through the adapter
        # Since the adapter processes one sample at a time, we loop
        all_outputs = []
        for b in range(B):
            features_by_camera = {}
            for n in range(N_cam):
                cam_name = f"cam_{n}"
                K = intrinsics[b, n]
                T = extrinsics[b, n]
                img_size = image_sizes[n] if image_sizes else [H, W]
                # Extract per-camera feature
                feat = features_flat[b * N_cam + n] if vision_encoder is None else features_flat[b * N_cam + n]
                features_by_camera[cam_name] = {
                    "features": feat,
                    "K": K,
                    "T_ego_cam": T,
                    "image_size": list(img_size),
                }
            out = self.forward(features_by_camera)
            all_outputs.append(out)

        # Stack outputs across batch
        batched = {}
        for key in ["triplane_tokens", "tokens_xy", "tokens_xz", "tokens_yz"]:
            if key in all_outputs[0]:
                batched[key] = torch.cat([o[key] for o in all_outputs], dim=0)
        for key in ["plane_xy", "plane_xz", "plane_yz"]:
            if key in all_outputs[0]:
                batched[key] = torch.cat([o[key] for o in all_outputs], dim=0)

        return batched

    def _extract_multi_scale_features(
        self, feats, cam_data: dict
    ) -> List[torch.Tensor]:
        """Extract and project multi-scale features from a camera's data.

        Args:
            feats: single tensor, list of tensors, or dict of tensors
            cam_data: camera data dict with optional "patch_grid"

        Returns:
            list of [1, hidden_dim, Hf_i, Wf_i] tensors, one per scale
        """
        device = self.xy_embed.device

        # Normalize to list of tensors
        if isinstance(feats, dict):
            # {"scale0": tensor, "scale1": tensor, ...}
            feat_list = [feats[k] for k in sorted(feats.keys())]
        elif isinstance(feats, (list, tuple)):
            feat_list = list(feats)
        else:
            feat_list = [feats]

        multi_scale_out = []
        for scale_idx, f in enumerate(feat_list):
            if isinstance(f, torch.Tensor):
                f = f.to(device)
            else:
                continue

            if f.dim() == 2:
                N, D = f.shape
                pg = cam_data.get("patch_grid", [16, 16])
                Hf, Wf = pg[0], pg[1]
                f_2d = f.reshape(1, Hf, Wf, D).permute(0, 3, 1, 2)
            elif f.dim() == 3:
                Hf, Wf, D = f.shape
                f_2d = f.permute(2, 0, 1).unsqueeze(0)
            elif f.dim() == 4:
                f_2d = f  # [1, D, Hf, Wf]
            else:
                continue

            # Project to hidden_dim
            proj_idx = min(scale_idx, len(self.input_projs) - 1)
            f_proj = self.input_projs[proj_idx](
                f_2d.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)  # [1, hidden_dim, Hf, Wf]
            multi_scale_out.append(f_proj)

        return multi_scale_out

    def _fuse_multi_scale(
        self, scale_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """Fuse multi-scale features into a single feature map.

        Args:
            scale_feats: list of [1, hidden_dim, H_i, W_i]

        Returns:
            fused: [1, hidden_dim, H_primary, W_primary]
        """
        if len(scale_feats) == 1:
            return scale_feats[0]

        if self.multi_scale_mode == "sum":
            # Upsample all to max resolution and sum
            max_h = max(f.shape[2] for f in scale_feats)
            max_w = max(f.shape[3] for f in scale_feats)
            fused = torch.zeros(1, self.hidden_dim, max_h, max_w,
                                device=scale_feats[0].device)
            for f in scale_feats:
                if f.shape[2] != max_h or f.shape[3] != max_w:
                    f = F.interpolate(f, size=(max_h, max_w), mode="bilinear",
                                      align_corners=False)
                fused = fused + f
            return fused / len(scale_feats)

        elif self.multi_scale_mode == "concat":
            # Upsample all to primary scale, concat along channel, project
            primary = scale_feats[0]
            ph, pw = primary.shape[2], primary.shape[3]
            upsampled = []
            for f in scale_feats:
                if f.shape[2] != ph or f.shape[3] != pw:
                    f = F.interpolate(f, size=(ph, pw), mode="bilinear",
                                      align_corners=False)
                upsampled.append(f)
            concat = torch.cat(upsampled, dim=1)  # [1, hidden_dim*N, H, W]
            # Project back to hidden_dim
            fused = self.scale_fusion(concat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return fused

        elif self.multi_scale_mode == "hierarchical":
            # FPN-style: top-down with lateral connections
            # For now, simple: use primary scale with learned weights
            primary = scale_feats[0]
            ph, pw = primary.shape[2], primary.shape[3]
            fused = primary
            for f in scale_feats[1:]:
                if f.shape[2] != ph or f.shape[3] != pw:
                    f = F.interpolate(f, size=(ph, pw), mode="bilinear",
                                      align_corners=False)
                fused = fused + f
            return fused / len(scale_feats)

        else:
            return scale_feats[0]

    def _fuse_cameras_attention(
        self, planes: List[torch.Tensor], score_net: nn.Module
    ) -> torch.Tensor:
        """Attention-based camera fusion for a single triplane.

        For each cell, learns per-camera confidence scores and computes
        a weighted sum. A camera that sees a cell gets high weight;
        cameras that don't (background plane embedding) get low weight.

        Args:
            planes: list of [1, C, H, W] per-camera plane features
            score_net: small Conv2d MLP that maps [C] → scalar score

        Returns:
            [1, C, H, W] fused plane
        """
        N = len(planes)
        if N == 1:
            return planes[0]

        # Stack: [N, C, H, W]
        stacked = torch.cat(planes, dim=0)

        # Per-cell per-camera scores: [N, H, W]
        scores = score_net(stacked).squeeze(1)

        # Softmax over cameras
        weights = F.softmax(scores, dim=0)  # [N, H, W]

        # Weighted sum
        fused = (stacked * weights.unsqueeze(1)).sum(dim=0, keepdim=True)  # [1, C, H, W]
        return fused


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_patch_rays(
    cam_center_ego: torch.Tensor,
    K: torch.Tensor,
    T_ego_cam: torch.Tensor,
    Hf: int,
    Wf: int,
    img_size: list,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ray origin and direction for each feature patch.

    Returns:
        ray_origins: [Hf*Wf, 3] all equal to camera center
        ray_directions: [Hf*Wf, 3] unit vectors in ego frame
    """
    H, W = img_size[0], img_size[1]

    # Patch centers in pixel coordinates
    patch_h = H / Hf
    patch_w = W / Wf

    ys = torch.arange(Hf, device=device).float() * patch_h + patch_h / 2
    xs = torch.arange(Wf, device=device).float() * patch_w + patch_w / 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [N, 2]

    # Unproject to camera frame directions
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    directions_cam = torch.stack([
        (pixels[:, 0] - cx) / fx,
        (pixels[:, 1] - cy) / fy,
        torch.ones(pixels.shape[0], device=device),
    ], dim=1)  # [N, 3]

    # Normalize
    directions_cam = F.normalize(directions_cam, dim=1)

    # Transform to ego frame
    R = T_ego_cam[:3, :3]  # rotation from ego to cam
    directions_ego = directions_cam @ R  # inverse rotation: cam to ego
    directions_ego = F.normalize(directions_ego, dim=1)

    # Ray origins
    origins = cam_center_ego.unsqueeze(0).expand(pixels.shape[0], -1)

    return origins, directions_ego


def _scatter_to_triplane(
    feats_2d: torch.Tensor,
    plane_xy: torch.Tensor,
    plane_xz: torch.Tensor,
    plane_yz: torch.Tensor,
    xy_count: torch.Tensor,
    xz_count: torch.Tensor,
    yz_count: torch.Tensor,
    K: torch.Tensor,
    T_ego_cam: torch.Tensor,
    cam_center_ego: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    sx: int,
    sy: int,
    sz: int,
    Hf: int,
    Wf: int,
    img_size: list,
    device: torch.device,
):
    """Vectorized scatter of 2D image features into triplane grids.

    Builds meshes of all anchor points per plane, batch-projects them,
    and uses F.grid_sample for bilinear interpolation.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    cell_x = (x_max - x_min) / sx
    cell_y = (y_max - y_min) / sy
    cell_z = (z_max - z_min) / sz

    H_img, W_img = int(img_size[0]), int(img_size[1])

    # Camera projection params
    R = T_ego_cam[:3, :3]   # [3, 3]  ego -> cam rotation
    t = T_ego_cam[:3, 3]    # [3]     ego -> cam translation
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # --- XY plane: for each (x, y) cell, vary z anchors ---
    wy = y_min + (torch.arange(sy, device=device).float() + 0.5) * cell_y  # [sy]
    wx = x_min + (torch.arange(sx, device=device).float() + 0.5) * cell_x  # [sx]
    z_anchors = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)        # [4]
    _scatter_plane_batched(
        feats_2d, plane_xy, xy_count, wx, wy, z_anchors, 2,
        R, t, fx, fy, cx, cy, H_img, W_img, device,
    )

    # --- XZ plane: for each (x, z) cell, vary y anchors ---
    wz = z_min + (torch.arange(sz, device=device).float() + 0.5) * cell_z  # [sz]
    wx = x_min + (torch.arange(sx, device=device).float() + 0.5) * cell_x  # [sx]
    y_anchors = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], device=device)  # [5]
    _scatter_plane_batched(
        feats_2d, plane_xz, xz_count, wx, wz, y_anchors, 1,
        R, t, fx, fy, cx, cy, H_img, W_img, device,
    )

    # --- YZ plane: for each (y, z) cell, vary x anchors ---
    wz = z_min + (torch.arange(sz, device=device).float() + 0.5) * cell_z  # [sz]
    wy = y_min + (torch.arange(sy, device=device).float() + 0.5) * cell_y  # [sy]
    x_anchors = torch.tensor([0.0, 10.0, 20.0, 40.0, 60.0], device=device)  # [5]
    _scatter_plane_batched(
        feats_2d, plane_yz, yz_count, wx, wz, x_anchors, 0,
        R, t, fx, fy, cx, cy, H_img, W_img, device,
    )


def _scatter_plane_batched(
    feats_2d: torch.Tensor,   # [1, C, Hf, Wf]
    plane: torch.Tensor,       # [1, gh, gw, C]  (in-place updates)
    count: torch.Tensor,       # [1, gh, gw, 1]  (in-place updates)
    cell_w: torch.Tensor,      # [gw] world coords along w-axis
    cell_h: torch.Tensor,      # [gh] world coords along h-axis
    anchor_vals: torch.Tensor, # [na] world coords along anchor dim
    anchor_dim: int,           # which xyz dim the anchors fill (0=x, 1=y, 2=z)
    R: torch.Tensor,           # [3, 3] rotation ego->cam
    t: torch.Tensor,           # [3]    translation ego->cam
    fx, fy, cx, cy: float,
    H_img: int, W_img: int,
    device: torch.device,
):
    """Batch-project all (grid_h, grid_w, anchor) combos and bilinear-sample.

    anchor_dim: 0 -> anchor fills x, grid=(h=z, w=y)
    anchor_dim: 1 -> anchor fills y, grid=(h=z, w=x)
    anchor_dim: 2 -> anchor fills z, grid=(h=y, w=x)
    """
    gh = len(cell_h)
    gw = len(cell_w)
    na = len(anchor_vals)
    # Build 3D point cloud: [gh, gw, na, 3]
    pts = torch.zeros(gh, gw, na, 3, device=device)

    if anchor_dim == 2:   # XY plane: h=y, w=x, anchor=z
        pts[..., 0] = cell_w[None, :, None]   # x
        pts[..., 1] = cell_h[:, None, None]   # y
        pts[..., 2] = anchor_vals[None, None, :]  # z
    elif anchor_dim == 1:  # XZ plane: h=z, w=x, anchor=y
        pts[..., 0] = cell_w[None, :, None]   # x
        pts[..., 1] = anchor_vals[None, None, :]  # y
        pts[..., 2] = cell_h[:, None, None]   # z
    else:                  # YZ plane: h=z, w=y, anchor=x
        pts[..., 0] = anchor_vals[None, None, :]  # x
        pts[..., 1] = cell_w[None, :, None]   # y
        pts[..., 2] = cell_h[:, None, None]   # z

    pts_flat = pts.reshape(-1, 3)  # [gh*gw*na, 3]

    # Transform ego -> camera: p_cam = p_ego @ R.T + t
    # R is ego->cam rotation, t is translation (camera origin in ego frame)
    pts_cam = pts_flat @ R.T + t.unsqueeze(0)
    z_cam = pts_cam[:, 2]

    # Project to pixel
    u = fx * pts_cam[:, 0] / z_cam.clamp(min=1e-6) + cx
    v = fy * pts_cam[:, 1] / z_cam.clamp(min=1e-6) + cy

    # Validity mask
    valid = (z_cam > 1e-3) & (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)

    # Normalize to [-1, 1] for grid_sample
    u_norm = u / W_img * 2.0 - 1.0
    v_norm = v / H_img * 2.0 - 1.0

    # grid_sample wants grid [B, H_out, W_out, 2]
    # We lay out gh rows, gw*na columns
    grid = torch.stack([u_norm, v_norm], dim=1)           # [N, 2]
    grid = grid.reshape(1, gh, gw * na, 2)                 # [1, gh, gw*na, 2]

    # Sample: [1, C, gh, gw*na]
    sampled = F.grid_sample(
        feats_2d, grid, mode='bilinear',
        padding_mode='zeros', align_corners=False,
    )

    # Reshape to [1, C, gh, gw, na] and mask invalid
    sampled = sampled.reshape(1, -1, gh, gw, na)   # [1, C, gh, gw, na]
    valid_3d = valid.reshape(gh, gw, na).float()    # [gh, gw, na]

    # Weighted sum over anchors (zero out invalid)
    valid_bc = valid_3d.unsqueeze(0).unsqueeze(0)    # [1, 1, gh, gw, na]
    sampled_sum = (sampled * valid_bc).sum(dim=-1)    # [1, C, gh, gw]
    count_sum = valid_bc.sum(dim=-1)                  # [1, 1, gh, gw]

    # Accumulate into plane and count
    plane.copy_(plane + sampled_sum.permute(0, 2, 3, 1))  # [1, gh, gw, C]
    count.copy_(count + count_sum.permute(0, 2, 3, 1))    # [1, gh, gw, 1]


def _patchify(plane: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert a triplane grid to patch tokens.

    Args:
        plane: [B, C, H, W]
        patch_size: spatial size of each patch

    Returns:
        tokens: [B, (H/patch_size)*(W/patch_size), C * patch_size^2]
    """
    B, C, H, W = plane.shape
    ph = H // patch_size
    pw = W // patch_size

    # Reshape into patches
    patches = plane.reshape(B, C, ph, patch_size, pw, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)  # [B, ph, pw, C, ps, ps]
    patches = patches.reshape(B, ph * pw, C * patch_size * patch_size)
    return patches
