#!/usr/bin/env python3
"""Train EgoTriPlane-Adapter Stage 2: VLM + triplane for 3D-QA.

Freezes vision encoder and LLM base weights.
Trains:
  - EgoTriPlane-Adapter (from Stage 1 checkpoint)
  - Projector (triplane tokens -> LLM input space)
  - Optional LoRA on LLM
  - Text answer head (or LLM language head)

Usage:
    python scripts/train_vlm_stage2.py \
        --config configs/train_vlm_stage2.yaml
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.triplane_adapter import EgoTriPlaneAdapter
from egotriplane.heads import BEVHeatmapHead, VisibilityHead, TextAnswerHead
from egotriplane.losses import EgoTriPlaneLoss
from egotriplane.dataset import Ego3DQADataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 VLM+Adapter")
    parser.add_argument("--config", type=str, default="configs/train_vlm_stage2.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])

    model_cfg = cfg["model"]

    # --- Adapter ---
    adapter = EgoTriPlaneAdapter(
        feature_dim=1024,
        hidden_dim=model_cfg["hidden_dim"],
        x_range=tuple(model_cfg["triplane"]["x_range"]),
        y_range=tuple(model_cfg["triplane"]["y_range"]),
        z_range=tuple(model_cfg["triplane"]["z_range"]),
        sx=model_cfg["triplane"]["sx"],
        sy=model_cfg["triplane"]["sy"],
        sz=model_cfg["triplane"]["sz"],
        patch_size=model_cfg["triplane"]["patch_size"],
        use_ray_embedding=model_cfg.get("use_ray_embedding", True),
        aggregation=model_cfg.get("aggregation", "attention"),
    )

    bev_head = BEVHeatmapHead(
        hidden_dim=model_cfg["hidden_dim"],
        grid_sx=model_cfg["triplane"]["sx"],
        grid_sy=model_cfg["triplane"]["sy"],
        grid_sz=model_cfg["triplane"]["sz"],
        patch_size=model_cfg["triplane"]["patch_size"],
    )

    vis_head = VisibilityHead(
        hidden_dim=model_cfg["hidden_dim"],
        grid_sx=model_cfg["triplane"]["sx"],
        grid_sy=model_cfg["triplane"]["sy"],
        patch_size=model_cfg["triplane"]["patch_size"],
    )

    # Load Stage 1 checkpoint
    stage1_path = cfg["checkpoint"]["stage1_checkpoint"]
    if os.path.exists(stage1_path):
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
        adapter.load_state_dict(checkpoint["adapter"])
        if "bev_head" in checkpoint:
            bev_head.load_state_dict(checkpoint["bev_head"])
        if "vis_head" in checkpoint:
            vis_head.load_state_dict(checkpoint["vis_head"])
    else:
        print(f"WARNING: Stage 1 checkpoint not found at {stage1_path}")

    # Projector: triplane tokens -> LLM input dim
    projector = nn.Sequential(
        nn.Linear(model_cfg["hidden_dim"], model_cfg["projector_hidden_dim"]),
        nn.GELU(),
        nn.Linear(model_cfg["projector_hidden_dim"], model_cfg["projector_hidden_dim"]),
    )

    # Answer head (lightweight, in lieu of full VLM)
    answer_head = TextAnswerHead(
        hidden_dim=model_cfg["projector_hidden_dim"],
        num_triplane_tokens=model_cfg["num_triplane_tokens"],
        max_answer_len=128,
    )

    # Move to device
    adapter.to(device)
    bev_head.to(device)
    vis_head.to(device)
    projector.to(device)
    answer_head.to(device)

    # Loss
    criterion = EgoTriPlaneLoss(
        w_bev=cfg["loss"].get("bev_weight", 0.5),
        w_vis=cfg["loss"].get("vis_weight", 0.5),
        w_cfg=cfg["loss"].get("cfg_weight", 0.2),
        w_ce=cfg["loss"].get("ce_weight", 1.0),
        use_stage2=True,
    )

    # Optimizer
    trainable = (
        list(adapter.parameters()) +
        list(bev_head.parameters()) +
        list(vis_head.parameters()) +
        list(projector.parameters()) +
        list(answer_head.parameters())
    )

    optimizer = optim.AdamW(
        trainable,
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    total_steps = cfg["train"]["num_epochs"] * 2000
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["train"]["learning_rate"],
        total_steps=total_steps,
        pct_start=cfg["train"]["warmup_steps"] / max(total_steps, 1),
    )

    scaler = GradScaler(enabled=cfg["train"].get("mixed_precision") == "fp16")

    # Dataset
    data_cfg = cfg["data"]
    train_dataset = Ego3DQADataset(
        qa_file=data_cfg["qa_file"],
        features_dir=data_cfg["features_dir"],
        split="train",
        camera_dropout=data_cfg.get("camera_dropout", True),
        min_cameras=data_cfg.get("min_cameras", 3),
        max_cameras=data_cfg.get("max_cameras", 6),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=_collate_stage2,
    )

    # Tokenizer for answer text
    # Simplified: build a small vocabulary from the answer JSON templates
    answer_vocab = _build_answer_vocab()
    answer_head.output_proj = nn.Linear(
        model_cfg["projector_hidden_dim"],
        len(answer_vocab),
    ).to(device)

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    global_step = 0

    for epoch in range(cfg["train"]["num_epochs"]):
        adapter.train()
        bev_head.train()
        vis_head.train()
        projector.train()
        answer_head.train()

        epoch_losses = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['num_epochs']}")
        for batch in pbar:
            batch = _to_device(batch, device)
            optimizer.zero_grad()

            with autocast(enabled=scaler.is_enabled()):
                adapter_out = adapter(
                    batch["features_by_camera"],
                    batch["camera_subset"],
                )

                # Project triplane tokens
                projected = projector(adapter_out["triplane_tokens"])

                # Generate answer
                answer_logits = answer_head(projected)

                # BEV + vis heads
                pred_heatmap = bev_head(adapter_out["tokens_xy"])
                pred_visibility = vis_head(adapter_out["tokens_xy"])

                # Build answer targets
                answer_targets = _encode_answer(
                    json.dumps(batch["answer"]),
                    answer_vocab,
                    answer_head.max_answer_len,
                    device,
                )

                losses = criterion(
                    pred_heatmap=pred_heatmap,
                    gt_heatmap=batch.get("gt_heatmap"),
                    pred_visibility=pred_visibility,
                    gt_visibility=batch.get("gt_visibility"),
                    answer_logits=answer_logits,
                    answer_targets=answer_targets,
                )

            scaler.scale(losses["total"]).backward()
            if cfg["train"].get("max_grad_norm", 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, cfg["train"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            for k, v in losses.items():
                epoch_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

            if global_step % cfg["logging"]["log_every"] == 0:
                avg = {k: v / cfg["logging"]["log_every"] for k, v in epoch_losses.items()}
                pbar.set_postfix(avg)
                epoch_losses = defaultdict(float)

        avg_loss = epoch_losses.get("total", 0) / max(1, len(train_loader))

        if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:
            _save_checkpoint(save_dir, epoch, adapter, projector, answer_head,
                             bev_head, vis_head, optimizer, scheduler, avg_loss,
                             answer_vocab)

    print(f"Training complete. Checkpoints saved to {save_dir}")


# ---------------------------------------------------------------------------
# Answer encoding helpers
# ---------------------------------------------------------------------------

def _build_answer_vocab() -> Dict[str, int]:
    """Build a simple character-level vocabulary for answer JSON strings."""
    chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                "{}[]\":,.-_ /<>()\n")
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for i, c in enumerate(sorted(chars), start=4):
        vocab[c] = i
    return vocab


def _encode_answer(text: str, vocab: Dict[str, int],
                    max_len: int, device: torch.device) -> torch.Tensor:
    """Encode answer text into token indices."""
    ids = [vocab.get("<BOS>", 1)]
    for c in text:
        ids.append(vocab.get(c, vocab.get("<UNK>", 3)))
    ids.append(vocab.get("<EOS>", 2))
    # Pad
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def _collate_stage2(batch):
    return batch[0]


def _to_device(batch, device):
    if isinstance(batch, dict):
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            elif isinstance(v, dict):
                result[k] = _to_device(v, device)
            else:
                result[k] = v
        return result
    return batch


def _save_checkpoint(save_dir, epoch, adapter, projector, answer_head,
                     bev_head, vis_head, optimizer, scheduler, loss_val,
                     vocab):
    path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
    torch.save({
        "epoch": epoch,
        "adapter": adapter.state_dict(),
        "projector": projector.state_dict(),
        "answer_head": answer_head.state_dict(),
        "bev_head": bev_head.state_dict(),
        "vis_head": vis_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss_val,
        "vocab": vocab,
    }, path)


if __name__ == "__main__":
    main()
