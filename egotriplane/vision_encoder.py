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

        self.encoder, self.hidden_dim, self.patch_size, self.grid_size = \
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
        grid_size = image_size // patch_size
        hidden_dim = model.config.hidden_size

        return model, hidden_dim, patch_size, grid_size

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
        model = full_model.visual
        model = model.to(dtype=torch.float32)

        if hasattr(model.config, 'patch_size'):
            patch_size = model.config.patch_size
        elif hasattr(model, 'patch_size'):
            patch_size = model.patch_size
        else:
            patch_size = 14
        grid_size = image_size // patch_size
        hidden_dim = model.config.hidden_size

        # Free LLM memory immediately
        del full_model
        torch.cuda.empty_cache()

        return model, hidden_dim, patch_size, grid_size

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
        grid_size = image_size // patch_size

        # Adapt: torchvision ViT returns [B, N+1, D] with CLS token
        return model, hidden_dim, patch_size, grid_size

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
            images: [B, 3, H, W] RGB images, normalized

        Returns:
            dict with:
              - "last_hidden_state": [B, N_patches, hidden_dim]
              - "hidden_states": list of [B, N_patches, hidden_dim] (if output_hidden_states)
              - "patch_grid": (Hf, Wf) feature map size
        """
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
            "patch_grid": (self.grid_size, self.grid_size),
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

        Handles both model families' output formats:
          - BaseModelOutputWithPooling (last_hidden_state + hidden_states)
          - Plain tensor (older versions)
          - list/tuple of tensors
        """
        # Qwen3-VL may not support output_hidden_states; try with, fall back without
        try:
            outputs = self.encoder(images, output_hidden_states=self.output_hidden_states)
        except TypeError:
            outputs = self.encoder(images)

        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        elif isinstance(outputs, torch.Tensor):
            last_hidden = outputs
        elif isinstance(outputs, (list, tuple)):
            last_hidden = outputs[-1]
        else:
            raise TypeError(f"Unexpected vision encoder output type: {type(outputs)}")

        # Remove CLS token if present (most ViT-based encoders prepend one)
        expected_tokens = self.grid_size * self.grid_size
        if last_hidden.shape[1] == expected_tokens + 1:
            last_hidden = last_hidden[:, 1:, :]
        elif last_hidden.shape[1] != expected_tokens:
            # Dynamic resolution or merge strategy — trust the encoder
            pass

        result = {
            "last_hidden_state": last_hidden,
            "patch_grid": (self.grid_size, self.grid_size),
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
            "patch_grid": (self.grid_size, self.grid_size),
        }
        if self.output_hidden_states:
            if self.output_layers:
                hidden_states = [hidden_states[i] for i in self.output_layers
                                 if i < len(hidden_states)]
            result["hidden_states"] = hidden_states

        return result

    def get_grid_size(self) -> Tuple[int, int]:
        return (self.grid_size, self.grid_size)

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep frozen layers in eval mode
            self._apply_freezing(self.freeze_until_layer)
        return self
