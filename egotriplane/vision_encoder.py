"""Pluggable vision encoder wrapper for 3D pretraining.

Supports:
  - CLIP ViT (available offline, from transformers)
  - Qwen2.5-VL / Qwen3-VL vision tower (when model is downloaded)
  - torchvision ViT (always available, no pretrained weights needed)

Unified interface:
  forward(images) -> {"last_hidden_state": [B, N, D], "hidden_states": [B, N, D]*}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class VisionEncoderWrapper(nn.Module):
    """Unified vision encoder that extracts patch features.

    Architecture-agnostic wrapper. Outputs patch features (without CLS token)
    that feed into the multi-view fusion / BEV adapter.
    """

    def __init__(self,
                 backbone: str = "clip_vit_large",
                 image_size: int = 224,
                 freeze: bool = False,
                 freeze_until_layer: int = 0,
                 output_hidden_states: bool = True,
                 output_layers: Optional[List[int]] = None,
                 local_files_only: bool = True,
                 device: str = "cuda"):
        super().__init__()
        self.backbone_name = backbone
        self.image_size = image_size
        self.freeze = freeze
        self.freeze_until_layer = freeze_until_layer
        self.output_hidden_states = output_hidden_states
        self.output_layers = output_layers
        self.local_files_only = local_files_only
        self.device_str = device

        self.encoder, self.hidden_dim, self.patch_size, self.temporal_patch_size, \
            self.spatial_merge_size, self.grid_size = \
            self._build_encoder(backbone, image_size)

        if freeze:
            self._apply_freezing(freeze_until_layer)

    def _build_encoder(self, backbone: str, image_size: int):
        """Build the vision encoder based on backbone name."""
        if backbone.startswith("clip_"):
            return self._build_clip(backbone, image_size)
        elif backbone.startswith("qwen"):
            return self._build_qwen_vl(backbone, image_size)
        elif backbone.startswith("tv_"):
            return self._build_torchvision(backbone, image_size)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _build_clip(self, backbone: str, image_size: int):
        """Build CLIP vision model."""
        from transformers import CLIPVisionModel, CLIPVisionConfig

        model_id = {
            "clip_vit_large": "openai/clip-vit-large-patch14",
            "clip_vit_base": "openai/clip-vit-base-patch16",
            "clip_vit_huge": "openai/clip-vit-large-patch14-336",
        }.get(backbone, "openai/clip-vit-large-patch14")

        try:
            model = CLIPVisionModel.from_pretrained(model_id, local_files_only=self.local_files_only)
        except Exception:
            # Offline fallback: initialize from config with random weights
            cfg = CLIPVisionConfig.from_pretrained(model_id, local_files_only=self.local_files_only)
            model = CLIPVisionModel(cfg)

        patch_size = model.config.patch_size
        hidden_dim = model.config.hidden_size

        return model, hidden_dim, patch_size, 1, 1, None  # grid computed dynamically

    def _build_qwen_vl(self, backbone: str, image_size: int):
        """Build Qwen2.5-VL / Qwen3-VL vision encoder.

        Extracts the vision tower from the full VLM. Loads in fp32 for
        gradient-correct training (fp16 underflows attention/layernorm grads).
        """
        model_specs = {
            "qwen25vl_3b":  ("Qwen/Qwen2.5-VL-3B-Instruct",  "qwen25"),
            "qwen25vl_7b":  ("Qwen/Qwen2.5-VL-7B-Instruct",  "qwen25"),
            "qwen3vl_4b":   ("Qwen/Qwen3-VL-4B-Instruct",     "qwen3"),
            "qwen3vl_8b":   ("Qwen/Qwen3-VL-8B-Instruct",     "qwen3"),
        }
        model_id, family = model_specs.get(
            backbone, ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen25")
        )

        # Import the right model class
        if family == "qwen3":
            try:
                from transformers import Qwen3VLForConditionalGeneration as VLModel
            except ImportError:
                raise RuntimeError(
                    "Qwen3-VL requires transformers >= 4.51. "
                    "Upgrade: pip install transformers>=4.51.0"
                )
        else:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
            except ImportError:
                raise RuntimeError(
                    "Qwen2.5-VL requires transformers >= 4.46. "
                    "Upgrade: pip install transformers>=4.46.0"
                )

        # Load full VLM in fp16 (fits 12GB for 4B), extract vision tower,
        # then convert to fp32 for gradient-correct training.
        try:
            full_model = VLModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.device_str,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_id}.\n"
                f"  Reason: {e}\n"
                f"  Did you download the model? Try:\n"
                f"    huggingface-cli download {model_id}\n"
                f"  Or set --backbone tv_vit_b_16 for local testing."
            )

        # Extract vision tower and convert to fp32 (avoids grad underflow)
        # Qwen2.5-VL: model.visual, Qwen3-VL: model.model.visual
        if hasattr(full_model, 'visual'):
            model = full_model.visual
        elif hasattr(full_model, 'model') and hasattr(full_model.model, 'visual'):
            model = full_model.model.visual
        else:
            # Print available top-level attrs to help debug
            attrs = [a for a in dir(full_model) if not a.startswith('_')]
            raise AttributeError(
                f"Cannot find vision encoder in {type(full_model).__name__}.\n"
                f"  Tried: .visual, .model.visual\n"
                f"  Available attributes: {sorted(attrs)[:30]}"
            )
        model = model.to(dtype=torch.float32)

        if hasattr(model.config, 'patch_size'):
            patch_size = model.config.patch_size
        elif hasattr(model, 'patch_size'):
            patch_size = model.patch_size
        else:
            patch_size = 14

        if hasattr(model.config, 'temporal_patch_size'):
            temporal_patch_size = model.config.temporal_patch_size
        elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'temporal_patch_size'):
            temporal_patch_size = model.patch_embed.temporal_patch_size
        else:
            temporal_patch_size = 1

        spatial_merge_size = getattr(model.config, 'spatial_merge_size', 2)

        hidden_dim = model.config.hidden_size

        # Free LLM memory immediately
        del full_model
        torch.cuda.empty_cache()

        return model, hidden_dim, patch_size, temporal_patch_size, spatial_merge_size, None

    def _build_torchvision(self, backbone: str, image_size: int):
        """Build torchvision ViT (lightweight, no pretrained weights needed)."""
        import torchvision

        vit_cfg = {
            "tv_vit_b_16": ("vit_b_16", 768, 16),
            "tv_vit_l_16": ("vit_l_16", 1024, 16),
            "tv_vit_b_32": ("vit_b_32", 768, 32),
            "tv_vit_l_32": ("vit_l_32", 1024, 32),
            "tv_vit_h_14": ("vit_h_14", 1280, 14),
        }
        vit_name, hidden_dim, patch_size = vit_cfg.get(
            backbone, ("vit_b_16", 768, 16)
        )

        # Use random initialization (no pretrained weights needed)
        model = torchvision.models.__dict__[vit_name](weights=None, image_size=image_size)
        # Adapt: torchvision ViT returns [B, N+1, D] with CLS token
        return model, hidden_dim, patch_size, 1, 1, None  # grid computed dynamically

    def _apply_freezing(self, freeze_until_layer: int):
        """Freeze vision encoder layers."""
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        if freeze_until_layer > 0:
            # Unfreeze later layers
            if hasattr(self.encoder, 'vision_model'):
                encoder_layers = self.encoder.vision_model.encoder.layers
            elif hasattr(self.encoder, 'encoder'):
                encoder_layers = self.encoder.encoder.layers
            elif hasattr(self.encoder, 'blocks'):
                encoder_layers = self.encoder.blocks
            else:
                return

            total_layers = len(encoder_layers)
            for i in range(total_layers - freeze_until_layer, total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract patch features from images.

        Args:
            images: [B, 3, H, W] RGB images, normalized (can be non-square)

        Returns:
            dict with:
              - "last_hidden_state": [B, N_patches, hidden_dim]
              - "hidden_states": list of [B, N_patches, hidden_dim] (if output_hidden_states)
              - "patch_grid": (Hf, Wf) feature map size (computed from actual input)
        """
        B, C, H, W = images.shape
        self._last_grid = (H // self.patch_size, W // self.patch_size)
        B = images.shape[0]

        if self.backbone_name.startswith("clip_"):
            return self._forward_clip(images)
        elif self.backbone_name.startswith("qwen"):
            return self._forward_qwen_vl(images)
        elif self.backbone_name.startswith("tv_"):
            return self._forward_torchvision(images)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

    def _forward_clip(self, images):
        """Forward through CLIP vision model."""
        outputs = self.encoder(images, output_hidden_states=self.output_hidden_states)

        # Remove CLS token: [B, N+1, D] -> [B, N, D]
        last_hidden = outputs.last_hidden_state[:, 1:, :]
        result = {
            "last_hidden_state": last_hidden,
            "patch_grid": self._last_grid,
        }

        if self.output_hidden_states and outputs.hidden_states:
            hs = []
            for h in outputs.hidden_states:
                hs.append(h[:, 1:, :])  # remove CLS
            # Select output layers if specified
            if self.output_layers:
                hs = [hs[i] for i in self.output_layers if i < len(hs)]
            result["hidden_states"] = hs

        return result

    def _forward_qwen_vl(self, images):
        """Forward through Qwen VL vision encoder (Qwen2.5-VL / Qwen3-VL).

        Qwen VL uses 3D patch embedding (Conv3d with kernel [T, P, P]).
        Pixel data must be in patch-major order: for each spatial patch,
        channels then temporal copies then spatial pixels. The image
        processor normally does this rearrangement; we replicate it here.
        """
        B, C, H_img, W_img = images.shape
        P = self.patch_size
        T = self.temporal_patch_size
        h_grid = H_img // P
        w_grid = W_img // P

        # Rearrange [B, C, H, W] to patch-major format expected by 3D conv:
        #   [B, n_patches, C, T, P, P] where n_patches = h_grid * w_grid
        #   then flatten to 1D so view(-1, C, T, P, P) works correctly.
        # Step 1: extract patches → [B, C, h_grid, P, w_grid, P]
        x = images.view(B, C, h_grid, P, w_grid, P)
        # Step 2: permute to [B, h_grid, w_grid, C, T=1, P, P]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Step 3: add temporal dim → [B, h_grid, w_grid, C, T, P, P]
        x = x.unsqueeze(4).expand(-1, -1, -1, -1, T, -1, -1)
        # Step 4: merge batch and spatial patches, flatten temporal-channel-spatial
        #         → [B * h_grid * w_grid * C * T * P * P] 1D tensor
        x = x.reshape(-1)

        h_grid_t = torch.full((B,), h_grid, device=images.device, dtype=torch.long)
        w_grid_t = torch.full((B,), w_grid, device=images.device, dtype=torch.long)
        grid_thw = torch.stack([
            torch.ones(B, device=images.device, dtype=torch.long),
            h_grid_t, w_grid_t,
        ], dim=1)  # [B, 3] with (1, h_grid, w_grid) per image

        try:
            outputs = self.encoder(x, grid_thw=grid_thw,
                                   output_hidden_states=self.output_hidden_states)
        except TypeError:
            try:
                outputs = self.encoder(x, grid_thw=grid_thw)
            except TypeError:
                outputs = self.encoder(x)

        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        elif isinstance(outputs, torch.Tensor):
            last_hidden = outputs
        elif isinstance(outputs, (list, tuple)):
            last_hidden = outputs[-1]
        else:
            raise TypeError(f"Unexpected vision encoder output type: {type(outputs)}")

        # Reshape flat tokens back to [B, n_tokens, D] where n_tokens after merger
        # = h_grid * w_grid / spatial_merge_size^2
        S = self.spatial_merge_size
        n_tokens = (h_grid // S) * (w_grid // S)
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.view(B, n_tokens, -1)

        # Update grid for downstream adapters (after spatial merge)
        merged_grid = (h_grid // S, w_grid // S)
        self._last_grid = merged_grid

        result = {
            "last_hidden_state": last_hidden,
            "patch_grid": merged_grid,
        }
        if self.output_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            result["hidden_states"] = list(outputs.hidden_states)
        return result

    def _forward_torchvision(self, images):
        """Forward through torchvision ViT.

        Manual forward: conv_proj → add CLS + pos embed → encoder blocks.
        Returns patch features WITHOUT CLS token.
        """
        x = self.encoder.conv_proj(images)        # [B, D, Hf, Wf]
        x = x.flatten(2).transpose(1, 2)           # [B, N_patches, D]

        # Add CLS token + position embedding
        B = x.shape[0]
        cls_token = self.encoder.class_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)                    # [B, N+1, D]
        x = x + self.encoder.encoder.pos_embedding

        # Run through transformer blocks
        hidden_states = []
        x_current = x
        blocks = self.encoder.encoder.layers
        for block in blocks:
            x_current = block(x_current)
            if self.output_hidden_states:
                hidden_states.append(x_current[:, 1:, :])  # remove CLS

        # Apply final norm (LayerNorm after encoder)
        if hasattr(self.encoder.encoder, 'ln'):
            x_current = self.encoder.encoder.ln(x_current)

        # Remove CLS token: [B, N+1, D] -> [B, N, D]
        last_hidden = x_current[:, 1:, :]

        result = {
            "last_hidden_state": last_hidden,
            "patch_grid": self._last_grid,
        }
        if self.output_hidden_states:
            if self.output_layers:
                hidden_states = [hidden_states[i] for i in self.output_layers
                                 if i < len(hidden_states)]
            result["hidden_states"] = hidden_states

        return result

    def get_patch_size(self) -> int:
        return self.patch_size

    def get_temporal_patch_size(self) -> int:
        return self.temporal_patch_size

    def get_grid_size(self) -> Tuple[int, int]:
        if hasattr(self, '_last_grid') and self._last_grid is not None:
            return self._last_grid
        # Fallback for square image estimate
        g = self.image_size // self.patch_size
        return (g, g)

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep frozen layers in eval mode
            self._apply_freezing(self.freeze_until_layer)
        return self
