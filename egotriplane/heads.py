"""Heads for EgoTriPlane-Adapter.

Provides lightweight prediction heads:
  - BEVHeatmapHead: predicts 2D Gaussian heatmap over XOY plane
  - VisibilityHead: predicts per-cell visibility from triplane tokens
  - CenterDetHead: center-based 3D detection head
  - BEVSegHead: BEV semantic segmentation head
  - TextAnswerHead: lightweight transformer decoder for JSON answer generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class BEVHeatmapHead(nn.Module):
    """Predict BEV heatmap from triplane tokens.

    Converts triplane tokens (especially XOY tokens) into a
    2D heatmap for object localization.
    """

    def __init__(self,
                 hidden_dim: int = 512,
                 grid_sx: int = 96,
                 grid_sy: int = 96,
                 grid_sz: int = 48,
                 patch_size: int = 8):
        super().__init__()
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy
        self.patch_size = patch_size

        # XOY tokens -> heatmap
        token_sx = grid_sx // patch_size  # 12
        token_sy = grid_sy // patch_size  # 12

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, 1, kernel_size=1),
        )

        self.token_sx = token_sx
        self.token_sy = token_sy

    def forward(self, tokens_xy: torch.Tensor) -> torch.Tensor:
        """Predict BEV heatmap.

        Args:
            tokens_xy: [B, num_xy_tokens, C] XOY plane tokens

        Returns:
            heatmap: [B, grid_sy, grid_sx]
        """
        B = tokens_xy.shape[0]
        C = tokens_xy.shape[-1]
        T = tokens_xy.shape[1]

        tokens_2d = tokens_xy.reshape(B, self.token_sy, self.token_sx, C)
        tokens_2d = tokens_2d.permute(0, 3, 1, 2)  # [B, C, H, W]

        heatmap = self.upsample(tokens_2d)  # [B, 1, grid_sy, grid_sx]
        return heatmap.squeeze(1)


class VisibilityHead(nn.Module):
    """Predict per-cell visibility from triplane tokens.

    Given the triplane tokens and camera subset info, predicts
    whether each BEV cell is observable.
    """

    def __init__(self,
                 hidden_dim: int = 512,
                 grid_sx: int = 96,
                 grid_sy: int = 96,
                 patch_size: int = 8):
        super().__init__()
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy

        token_sx = grid_sx // patch_size
        token_sy = grid_sy // patch_size

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.token_sx = token_sx
        self.token_sy = token_sy

    def forward(self, tokens_xy: torch.Tensor) -> torch.Tensor:
        """Predict visibility logits per token.

        Args:
            tokens_xy: [B, num_xy_tokens, C]

        Returns:
            visibility: [B, token_sy, token_sx] logits
        """
        B, T, C = tokens_xy.shape
        logits = self.decoder(tokens_xy)  # [B, T, 1]
        logits = logits.reshape(B, self.token_sy, self.token_sx)
        return logits


# ---------------------------------------------------------------------------
# 3D Detection Head (CenterPoint-style)
# ---------------------------------------------------------------------------

class CenterDetHead(nn.Module):
    """Center-based 3D detection head operating on BEV (XY) plane features.

    Predicts per-cell:
      - class heatmap [num_classes]
      - center offset [2] (sub-cell dx, dy)
      - size [3] (w, l, h)
      - yaw [2] (sin, cos)
      - optional centerness / objectness
    """

    def __init__(self,
                 hidden_dim: int = 512,
                 num_classes: int = 5,
                 grid_sx: int = 96,
                 grid_sy: int = 96,
                 patch_size: int = 8,
                 x_range: tuple = (-20.0, 80.0),
                 y_range: tuple = (-40.0, 40.0),
                 use_objectness: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy
        self.patch_size = patch_size
        self.token_sx = grid_sx // patch_size
        self.token_sy = grid_sy // patch_size
        self.x_range = x_range
        self.y_range = y_range
        self.cell_x = (x_range[1] - x_range[0]) / grid_sx
        self.cell_y = (y_range[1] - y_range[0]) / grid_sy
        self.use_objectness = use_objectness

        # Shared backbone: upsample token grid to full BEV resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
        )

        # Heads
        shared_dim = hidden_dim // 4
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, num_classes, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, 2, 1),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, 3, 1),
        )
        self.yaw_head = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, 2, 1),
        )
        if use_objectness:
            self.obj_head = nn.Sequential(
                nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
                nn.BatchNorm2d(shared_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_dim, 1, 1),
            )
        # z head for per-cell center height prediction (on BEV features)
        self.z_head = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, 1, 1),
        )

    def forward(self, adapter_out: dict) -> dict:
        """Predict 3D detections from adapter output.

        Args:
            adapter_out: dict from EgoTriPlaneAdapter with:
                - "tokens_xy": [B, num_xy_tokens, C]
                - "plane_xy": [B, C, sy, sx]
                - "triplane_tokens": [B, num_tokens, C]

        Returns:
            dict with:
              - "heatmap": [B, num_classes, grid_sy, grid_sx]
              - "offset": [B, 2, grid_sy, grid_sx]
              - "size": [B, 3, grid_sy, grid_sx]
              - "yaw": [B, 2, grid_sy, grid_sx]
              - "objectness": [B, 1, grid_sy, grid_sx] (if use_objectness)
              - "z_refined": [B, grid_sy, grid_sx] (optional)
        """
        tokens_xy = adapter_out["tokens_xy"]
        B = tokens_xy.shape[0]
        C = tokens_xy.shape[-1]

        tokens_2d = tokens_xy.reshape(B, self.token_sy, self.token_sx, C)
        tokens_2d = tokens_2d.permute(0, 3, 1, 2)  # [B, C, H, W]

        feat = self.upsample(tokens_2d)  # [B, shared_dim, grid_sy, grid_sx]

        outputs = {
            "heatmap": self.heatmap_head(feat),
            "offset": self.offset_head(feat),
            "size": self.size_head(feat),
            "yaw": self.yaw_head(feat),
            "z": self.z_head(feat),
        }
        if self.use_objectness:
            outputs["objectness"] = self.obj_head(feat)

        return outputs

    @torch.no_grad()
    def decode_detections(
        self,
        outputs: dict,
        score_thresh: float = 0.1,
        max_dets: int = 100,
        nms_kernel: int = 3,
    ) -> list:
        """Decode raw head outputs into list of 3D boxes.

        Args:
            outputs: dict from forward() with heatmap, offset, size, yaw, z, objectness
            score_thresh: minimum score after objectness multiplier
            max_dets: max detections per sample
            nms_kernel: max-pool kernel size for local maxima suppression

        Returns:
            list of dicts per batch element, each with:
              boxes: [N, 7] (cx, cy, cz, w, l, h, yaw)
              scores: [N]
              classes: [N] int
        """
        heatmap = outputs["heatmap"]          # [B, C, H, W]
        offset = outputs["offset"]            # [B, 2, H, W]
        size = outputs["size"]                # [B, 3, H, W]
        yaw = outputs["yaw"]                  # [B, 2, H, W]
        z = outputs["z"]                      # [B, 1, H, W]
        obj = outputs.get("objectness")       # [B, 1, H, W] or None

        B, C, H, W = heatmap.shape
        device = heatmap.device

        # Sigmoid on heatmap
        scores = heatmap.sigmoid()            # [B, C, H, W]
        if obj is not None:
            scores = scores * obj.sigmoid()   # multiply by objectness

        # Max-pool NMS per class
        pool = torch.nn.functional.max_pool2d(
            scores.view(-1, 1, H, W),
            kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2,
        ).view(B, C, H, W)
        keep = (scores == pool) & (scores > score_thresh)

        batch_results = []
        for b in range(B):
            class_ids, ys, xs = torch.where(keep[b])
            sc = scores[b, class_ids, ys, xs]

            # Top-k
            if sc.numel() > max_dets:
                topk = sc.topk(max_dets).indices
                class_ids = class_ids[topk]
                ys = ys[topk]
                xs = xs[topk]
                sc = sc[topk]

            N = sc.numel()
            if N == 0:
                batch_results.append({
                    "boxes": torch.zeros(0, 7, device=device),
                    "scores": torch.zeros(0, device=device),
                    "classes": torch.zeros(0, dtype=torch.long, device=device),
                })
                continue

            # Decode center: world coords from grid cells
            cx = self.x_range[0] + (xs.float() + offset[b, 0, ys, xs]) * self.cell_x
            cy = self.y_range[0] + (ys.float() + offset[b, 1, ys, xs]) * self.cell_y
            cz = z[b, 0, ys, xs]

            # Size
            w_val = size[b, 0, ys, xs]
            l_val = size[b, 1, ys, xs]
            h_val = size[b, 2, ys, xs]

            # Yaw
            sin_val = yaw[b, 0, ys, xs]
            cos_val = yaw[b, 1, ys, xs]
            yaw_val = torch.atan2(sin_val, cos_val)

            boxes = torch.stack([cx, cy, cz, w_val, l_val, h_val, yaw_val], dim=1)

            batch_results.append({
                "boxes": boxes,
                "scores": sc,
                "classes": class_ids,
            })

        return batch_results


# ---------------------------------------------------------------------------
# BEV Semantic Segmentation Head
# ---------------------------------------------------------------------------

class BEVSegHead(nn.Module):
    """BEV semantic segmentation head on XY plane.

    Predicts per-cell semantic class (or binary occupancy).
    Uses the XY plane features from the adapter.
    """

    def __init__(self,
                 hidden_dim: int = 512,
                 num_classes: int = 1,  # 1 for binary occupancy, >1 for multi-class
                 grid_sx: int = 96,
                 grid_sy: int = 96,
                 patch_size: int = 8,
                 use_deconv: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy
        self.patch_size = patch_size
        self.token_sx = grid_sx // patch_size
        self.token_sy = grid_sy // patch_size

        if use_deconv:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_dim // 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 8, num_classes, 1),
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(size=(grid_sy, grid_sx), mode='bilinear', align_corners=False),
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, num_classes, 1),
            )

    def forward(self, tokens_xy: torch.Tensor) -> torch.Tensor:
        """Predict BEV segmentation.

        Args:
            tokens_xy: [B, num_xy_tokens, C]

        Returns:
            seg_logits: [B, num_classes, grid_sy, grid_sx]
        """
        B = tokens_xy.shape[0]
        C = tokens_xy.shape[-1]

        tokens_2d = tokens_xy.reshape(B, self.token_sy, self.token_sx, C)
        tokens_2d = tokens_2d.permute(0, 3, 1, 2)

        return self.upsample(tokens_2d)


class TextAnswerHead(nn.Module):
    """Lightweight transformer decoder for text answer generation.

    Used in Stage 2 to convert triplane tokens + question embedding
    into a structured JSON answer. If a full VLM is attached, this
    can be replaced by the VLM's own language head.
    """

    def __init__(self,
                 hidden_dim: int = 512,
                 num_triplane_tokens: int = 288,
                 vocab_size: int = 5000,
                 num_decoder_layers: int = 4,
                 num_heads: int = 8,
                 max_answer_len: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_answer_len = max_answer_len

        # Project triplane tokens to decoder dim
        self.triplane_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learned answer query tokens
        self.answer_queries = nn.Parameter(
            torch.randn(1, max_answer_len, hidden_dim) * 0.02
        )

        # Positional encoding for answer sequence
        self.pos_encoding = PositionalEncoding(hidden_dim, max_answer_len)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, triplane_tokens: torch.Tensor,
                question_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate answer tokens.

        Args:
            triplane_tokens: [B, num_tokens, C] from EgoTriPlane-Adapter
            question_embedding: [B, seq_len, C] optional question tokens

        Returns:
            answer_logits: [B, max_answer_len, vocab_size]
        """
        B = triplane_tokens.shape[0]

        # Memory: triplane tokens (+ optional question)
        memory = self.triplane_proj(triplane_tokens)
        if question_embedding is not None:
            memory = torch.cat([memory, question_embedding], dim=1)

        memory = memory.permute(1, 0, 2)  # [T, B, C] for nn.TransformerDecoder

        # Queries: learned answer position embeddings
        tgt = self.pos_encoding(self.answer_queries.expand(B, -1, -1))
        tgt = tgt.permute(1, 0, 2)  # [max_len, B, C]

        # Decode
        decoded = self.decoder(tgt, memory)  # [max_len, B, C]
        decoded = decoded.permute(1, 0, 2)  # [B, max_len, C]

        logits = self.output_proj(decoded)  # [B, max_len, vocab_size]
        return logits


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: [B, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]
