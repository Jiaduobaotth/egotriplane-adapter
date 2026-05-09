"""Loss functions for EgoTriPlane-Adapter training.

Stage 1 (adapter_3d_pretrain):
  total = lambda_det * loss_3d_det + lambda_bev * loss_bev_seg [+ lambda_occ * loss_occupancy]

Stage 2 (vlm_qa):
  total = w_ce * loss_ce + w_bev * loss_bev + w_vis * loss_vis + w_cfg * loss_cfg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Focal Loss (shared)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for heatmap supervision.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            pred: [B, ...] predicted probabilities
            target: [B, ...] ground truth (0 or 1)
        """
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        target = target.float()

        pos_loss = -self.alpha * (1 - pred) ** self.gamma * target * torch.log(pred)
        neg_loss = -(1 - self.alpha) * pred ** self.gamma * (1 - target) * torch.log(1 - pred)

        num_pos = target.sum().clamp(min=1)
        return (pos_loss + neg_loss).sum() / num_pos


# ---------------------------------------------------------------------------
# 3D Detection Loss
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """Center-based 3D detection loss.

    Components:
      - heatmap: focal loss on class heatmap
      - offset: L1 loss on sub-cell center offset
      - size: L1 loss on box dimensions (w, l, h)
      - yaw: sin-cos regression loss
      - z: L1 loss on z coordinate (optional)
    """

    def __init__(self,
                 w_heatmap: float = 1.0,
                 w_offset: float = 0.1,
                 w_size: float = 0.1,
                 w_yaw: float = 0.1,
                 w_z: float = 0.05,
                 num_classes: int = 5,
                 use_focal: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 max_objects: int = 50,
                 bev_x_range: tuple = (-20.0, 80.0),
                 bev_y_range: tuple = (-40.0, 40.0),
                 grid_sx: int = 96,
                 grid_sy: int = 96):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_offset = w_offset
        self.w_size = w_size
        self.w_yaw = w_yaw
        self.w_z = w_z
        self.max_objects = max_objects
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.grid_sx = grid_sx
        self.grid_sy = grid_sy

        self.cell_x = (bev_x_range[1] - bev_x_range[0]) / grid_sx
        self.cell_y = (bev_y_range[1] - bev_y_range[0]) / grid_sy

        if use_focal:
            self.heatmap_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.heatmap_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, preds: dict, targets: dict) -> Dict[str, torch.Tensor]:
        """Compute 3D detection loss.

        Args:
            preds: dict from CenterDetHead with:
                "heatmap": [B, num_classes, H, W] logits
                "offset": [B, 2, H, W]
                "size": [B, 3, H, W]
                "yaw": [B, 2, H, W]
                "objectness": [B, 1, H, W] (optional)
            targets: dict with:
                "gt_hm": [B, num_classes, H, W] ground truth gaussian heatmaps
                "gt_offset": [B, max_obj, 2] (dx, dy) in cell units
                "gt_size": [B, max_obj, 3] (w, l, h)
                "gt_yaw": [B, max_obj, 2] (sin, cos)
                "gt_mask": [B, max_obj] boolean mask for valid objects
                "gt_obj_idx": [B, max_obj, 2] (gx, gy) cell indices for each object

        Returns:
            dict of loss components
        """
        # Ensure batch dimension on all targets
        gt_hm = targets["gt_hm"]
        gt_mask = targets["gt_mask"]
        gt_idx = targets["gt_obj_idx"]
        gt_offset = targets["gt_offset"]
        gt_size = targets["gt_size"]
        gt_yaw = targets["gt_yaw"]

        if gt_hm.dim() == 3:
            gt_hm = gt_hm.unsqueeze(0)
        if gt_mask.dim() == 1:
            gt_mask = gt_mask.unsqueeze(0)
        if gt_idx.dim() == 2:
            gt_idx = gt_idx.unsqueeze(0)
        if gt_offset.dim() == 2:
            gt_offset = gt_offset.unsqueeze(0)
        if gt_size.dim() == 2:
            gt_size = gt_size.unsqueeze(0)
        if gt_yaw.dim() == 2:
            gt_yaw = gt_yaw.unsqueeze(0)

        B = preds["heatmap"].shape[0]
        device = preds["heatmap"].device

        # Heatmap loss
        pred_hm = preds["heatmap"]  # [B, C, H, W]
        l_heatmap = self.heatmap_loss(pred_hm.sigmoid(), gt_hm)

        # Per-object regression losses
        l_offset = torch.tensor(0.0, device=device)
        l_size = torch.tensor(0.0, device=device)
        l_yaw = torch.tensor(0.0, device=device)

        for b in range(B):
            n_obj = gt_mask[b].sum().int().item()
            if n_obj == 0:
                continue

            for o in range(n_obj):
                gx, gy = gt_idx[b, o, 0].long(), gt_idx[b, o, 1].long()

                # Offset loss
                pred_off = preds["offset"][b, :, gy, gx]       # [2]
                gt_off = gt_offset[b, o]                         # [2]
                l_offset = l_offset + F.l1_loss(pred_off, gt_off)

                # Size loss
                pred_sz = preds["size"][b, :, gy, gx]          # [3]
                gt_sz = gt_size[b, o]                            # [3]
                l_size = l_size + F.l1_loss(pred_sz, gt_sz)

                # Yaw loss (sin-cos)
                pred_yaw = preds["yaw"][b, :, gy, gx]          # [2]
                gt_yaw_val = gt_yaw[b, o]                        # [2]
                l_yaw = l_yaw + F.l1_loss(pred_yaw, gt_yaw_val)

        total_obj = gt_mask.sum().clamp(min=1)
        l_offset = l_offset / total_obj
        l_size = l_size / total_obj
        l_yaw = l_yaw / total_obj

        losses = {
            "det_heatmap": l_heatmap * self.w_heatmap,
            "det_offset": l_offset * self.w_offset,
            "det_size": l_size * self.w_size,
            "det_yaw": l_yaw * self.w_yaw,
        }
        losses["det_total"] = sum(losses.values())
        return losses


# ---------------------------------------------------------------------------
# BEV Segmentation Loss
# ---------------------------------------------------------------------------

class BEVSegLoss(nn.Module):
    """BEV semantic segmentation / occupancy loss.

    Supports: CrossEntropy (multi-class), BCE (binary), Dice loss.
    """

    def __init__(self,
                 num_classes: int = 1,
                 loss_type: str = "ce+dice",  # "ce", "bce", "dice", "ce+dice", "bce+dice", "focal"
                 dice_weight: float = 0.5,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.dice_weight = dice_weight
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute BEV segmentation loss.

        Args:
            pred: [B, C, H, W] logits (C=1 for binary)
            target: [B, H, W] class indices (CE) or [B, 1, H, W] / [B, H, W] float (BCE)
            mask: [B, H, W] optional valid region mask

        Returns:
            dict of loss components
        """
        losses = {}

        if self.num_classes > 1:
            # Multi-class
            if "ce" in self.loss_type:
                if mask is not None:
                    ce = F.cross_entropy(pred, target.long(), reduction='none',
                                         weight=self.class_weights.to(pred.device)
                                         if self.class_weights is not None else None)
                    ce = (ce * mask).sum() / mask.sum().clamp(min=1)
                else:
                    ce = F.cross_entropy(pred, target.long(),
                                         weight=self.class_weights.to(pred.device)
                                         if self.class_weights is not None else None)
                losses["bev_ce"] = ce

            if "dice" in self.loss_type:
                dice = self._dice_loss_multiclass(pred, target, mask)
                losses["bev_dice"] = dice * self.dice_weight
        else:
            # Binary
            if "bce" in self.loss_type:
                target_f = target.float().unsqueeze(1) if target.dim() == 3 else target.float()
                if mask is not None:
                    bce = F.binary_cross_entropy_with_logits(
                        pred, target_f, reduction='none')
                    bce = (bce * mask.unsqueeze(1)).sum() / mask.sum().clamp(min=1)
                else:
                    bce = F.binary_cross_entropy_with_logits(pred, target_f)
                losses["bev_bce"] = bce

            if "dice" in self.loss_type:
                dice = self._dice_loss_binary(pred, target, mask)
                losses["bev_dice"] = dice * self.dice_weight

            if "focal" in self.loss_type:
                target_f = target.float().unsqueeze(1) if target.dim() == 3 else target.float()
                focal = FocalLoss()(pred.sigmoid(), target_f)
                losses["bev_focal"] = focal

        losses["bev_seg_total"] = sum(losses.values())
        return losses

    @staticmethod
    def _dice_loss_binary(pred: torch.Tensor, target: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dice loss for binary segmentation."""
        pred_s = pred.sigmoid()
        target_f = target.float().unsqueeze(1) if target.dim() == 3 else target.float()
        if mask is not None:
            pred_s = pred_s * mask.unsqueeze(1)
            target_f = target_f * mask.unsqueeze(1)
        intersection = (pred_s * target_f).sum(dim=(2, 3))
        union = pred_s.sum(dim=(2, 3)) + target_f.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return (1.0 - dice).mean()

    @staticmethod
    def _dice_loss_multiclass(pred: torch.Tensor, target: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dice loss for multi-class segmentation (averaged over classes)."""
        pred_s = pred.softmax(dim=1)
        target_onehot = F.one_hot(target.long(), pred.shape[1]).permute(0, 3, 1, 2).float()
        if mask is not None:
            pred_s = pred_s * mask.unsqueeze(1)
            target_onehot = target_onehot * mask.unsqueeze(1)
        intersection = (pred_s * target_onehot).sum(dim=(2, 3))
        union = pred_s.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return (1.0 - dice).mean()


# ---------------------------------------------------------------------------
# Legacy loss modules (kept for reference / Stage 2)
# ---------------------------------------------------------------------------

class BEVLoss(nn.Module):
    """BEV heatmap loss with optional focal loss."""

    def __init__(self, use_focal: bool = True, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        if use_focal:
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, pred_heatmap: torch.Tensor,
                gt_heatmap: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_heatmap, gt_heatmap)


class VisibilityLoss(nn.Module):
    """Binary cross-entropy loss for visibility prediction."""

    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, pred_visibility: torch.Tensor,
                gt_visibility: torch.Tensor) -> torch.Tensor:
        return self.bce(pred_visibility, gt_visibility.float())


class ConfigConsistencyLoss(nn.Module):
    """Cross-configuration consistency loss."""

    def __init__(self, use_l2: bool = True):
        super().__init__()
        self.use_l2 = use_l2

    def forward(self,
                tokens_full: torch.Tensor,
                tokens_drop: torch.Tensor,
                visibility_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if visibility_mask is None:
            visibility_mask = torch.ones(tokens_full.shape[0], tokens_full.shape[1],
                                          device=tokens_full.device)
        diff = tokens_full - tokens_drop
        if self.use_l2:
            loss = (diff ** 2).mean(dim=-1)
        else:
            loss = diff.abs().mean(dim=-1)
        loss = (loss * visibility_mask).sum() / (visibility_mask.sum() + 1e-8)
        return loss


# ---------------------------------------------------------------------------
# Combined Stage 1 Loss (3D perception only, NO VQA)
# ---------------------------------------------------------------------------

class Adapter3DPretrainLoss(nn.Module):
    """Combined loss for Stage 1: adapter 3D perception pretraining.

    L_stage1 = lambda_det * L_det + lambda_bev * L_bev_seg [+ lambda_occ * L_occ]

    Does NOT use VQA / language generation loss.
    """

    def __init__(self,
                 lambda_det: float = 1.0,
                 lambda_bev: float = 1.0,
                 lambda_occ: float = 0.0,
                 det_loss_cfg: Optional[dict] = None,
                 bev_seg_cfg: Optional[dict] = None):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_bev = lambda_bev
        self.lambda_occ = lambda_occ

        det_cfg = det_loss_cfg or {}
        self.det_loss = DetectionLoss(
            w_heatmap=det_cfg.get("w_heatmap", 1.0),
            w_offset=det_cfg.get("w_offset", 0.1),
            w_size=det_cfg.get("w_size", 0.1),
            w_yaw=det_cfg.get("w_yaw", 0.1),
            w_z=det_cfg.get("w_z", 0.05),
            num_classes=det_cfg.get("num_classes", 5),
            use_focal=det_cfg.get("use_focal", True),
            grid_sx=det_cfg.get("grid_sx", 96),
            grid_sy=det_cfg.get("grid_sy", 96),
            bev_x_range=tuple(det_cfg.get("bev_x_range", (-20.0, 80.0))),
            bev_y_range=tuple(det_cfg.get("bev_y_range", (-40.0, 40.0))),
        )

        bev_cfg = bev_seg_cfg or {}
        self.bev_seg_loss = BEVSegLoss(
            num_classes=bev_cfg.get("num_classes", 1),
            loss_type=bev_cfg.get("loss_type", "ce+dice"),
            dice_weight=bev_cfg.get("dice_weight", 0.5),
        )

    def forward(self,
                det_preds: Optional[dict] = None,
                det_targets: Optional[dict] = None,
                bev_pred: Optional[torch.Tensor] = None,
                bev_target: Optional[torch.Tensor] = None,
                bev_mask: Optional[torch.Tensor] = None,
                occ_pred: Optional[torch.Tensor] = None,
                occ_target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute Stage 1 loss.

        Returns:
            dict with per-component losses and "total" key.
        """
        all_losses = {}
        total = torch.tensor(0.0)

        if det_preds is not None and det_targets is not None:
            det_losses = self.det_loss(det_preds, det_targets)
            all_losses.update(det_losses)
            total = total + self.lambda_det * det_losses["det_total"]

        if bev_pred is not None and bev_target is not None:
            bev_losses = self.bev_seg_loss(bev_pred, bev_target, bev_mask)
            all_losses.update(bev_losses)
            total = total + self.lambda_bev * bev_losses.get("bev_seg_total", 0.0)

        if occ_pred is not None and occ_target is not None:
            occ_losses = self.bev_seg_loss(occ_pred, occ_target)
            occ_losses = {f"occ_{k}": v for k, v in occ_losses.items()}
            all_losses.update(occ_losses)
            total = total + self.lambda_occ * occ_losses.get("occ_bev_seg_total", 0.0)

        all_losses["total"] = total
        return all_losses


# ---------------------------------------------------------------------------
# Combined Stage 2 Loss (VQA)
# ---------------------------------------------------------------------------

class EgoTriPlaneLoss(nn.Module):
    """Combined loss for Stage 2: VQA training.

    L_stage2 = w_ce * L_CE + w_bev * L_BEV + w_vis * L_vis + w_cfg * L_cfg
    """

    def __init__(self,
                 w_bev: float = 0.5,
                 w_vis: float = 0.5,
                 w_cfg: float = 0.2,
                 w_ce: float = 1.0,
                 use_focal: bool = True,
                 use_stage2: bool = False):
        super().__init__()
        self.w_bev = w_bev
        self.w_vis = w_vis
        self.w_cfg = w_cfg
        self.w_ce = w_ce
        self.use_stage2 = use_stage2

        self.bev_loss = BEVLoss(use_focal=use_focal)
        self.vis_loss = VisibilityLoss()
        self.cfg_loss = ConfigConsistencyLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                pred_heatmap: Optional[torch.Tensor] = None,
                gt_heatmap: Optional[torch.Tensor] = None,
                pred_visibility: Optional[torch.Tensor] = None,
                gt_visibility: Optional[torch.Tensor] = None,
                tokens_full: Optional[torch.Tensor] = None,
                tokens_drop: Optional[torch.Tensor] = None,
                answer_logits: Optional[torch.Tensor] = None,
                answer_targets: Optional[torch.Tensor] = None,
                visibility_mask: Optional[torch.Tensor] = None) -> dict:
        """Compute combined Stage 2 loss."""
        losses = {}

        if pred_heatmap is not None and gt_heatmap is not None:
            l_bev = self.bev_loss(pred_heatmap, gt_heatmap)
            losses["bev"] = l_bev
        else:
            l_bev = 0.0

        if pred_visibility is not None and gt_visibility is not None:
            l_vis = self.vis_loss(pred_visibility, gt_visibility)
            losses["vis"] = l_vis
        else:
            l_vis = 0.0

        if tokens_full is not None and tokens_drop is not None:
            l_cfg = self.cfg_loss(tokens_full, tokens_drop, visibility_mask)
            losses["cfg"] = l_cfg
        else:
            l_cfg = 0.0

        total = self.w_bev * l_bev + self.w_vis * l_vis + self.w_cfg * l_cfg

        if self.use_stage2 and answer_logits is not None and answer_targets is not None:
            l_ce = self.ce_loss(
                answer_logits.reshape(-1, answer_logits.shape[-1]),
                answer_targets.reshape(-1),
            )
            losses["ce"] = l_ce
            total = total + self.w_ce * l_ce

        losses["total"] = total
        return losses
