#!/usr/bin/env python3
"""Evaluate EgoTriPlane-Adapter on Ego3D-QA benchmarks.

Runs evaluation across all test splits:
  - nusc_val_6cam
  - nusc_val_5cam_random
  - nusc_val_4cam_random
  - nusc_val_3cam_random
  - nusc_val_front3
  - nusc_val_no_rear
  - nusc_val_unseen_subset

Computes metrics:
  - Answer accuracy (by field)
  - Grounding error
  - Unknown/hallucination rates
  - Robustness scores (R_4cam, R_3cam)

Usage:
    python scripts/eval_ego3dqa.py \
        --config configs/eval.yaml \
        --checkpoint outputs/checkpoints/stage2/best.pt \
        --qa outputs/ego3dqa/nusc_val_ego3dqa.jsonl
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.triplane_adapter import EgoTriPlaneAdapter
from egotriplane.heads import BEVHeatmapHead, VisibilityHead, TextAnswerHead
from egotriplane.metrics import (
    compute_metrics,
    compute_grounding_error,
    compute_robustness_score,
    compute_per_type_metrics,
    save_results_csv,
    save_results_markdown,
    parse_json_answer,
)
from egotriplane.nusc_utils import load_qa
from egotriplane.camera_dropout import (
    FULL_NUSC_CAMERAS,
    CAMERA_SUBSETS,
    generate_random_ncam_val,
    UNSEEN_HOLDOUT_CAMERAS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EgoTriPlane-Adapter")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--qa", type=str, required=True, help="QA JSONL file")
    parser.add_argument("--features_dir", type=str,
                        default="outputs/features/nusc_val_clip_features/")
    parser.add_argument("--out_preds", type=str, default="outputs/eval/preds.jsonl")
    parser.add_argument("--out_results", type=str, default="outputs/eval/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of QA samples")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, heads, vocab = load_model(args.checkpoint, device)

    # Load all QAs
    all_qa = load_qa(args.qa)
    if args.limit:
        all_qa = all_qa[:args.limit]
    print(f"Loaded {len(all_qa)} QA instances")

    # Group QAs by camera split
    qa_by_split = _split_qas_by_camera_config(all_qa)

    # Evaluate each split
    all_metrics = {}
    all_preds = []

    for split_name in cfg["eval"]["test_splits"]:
        split_qas = qa_by_split.get(split_name, [])
        if not split_qas:
            print(f"  Split '{split_name}': no QAs, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name} ({len(split_qas)} QAs)")
        print(f"{'='*60}")

        preds, metrics = evaluate_split(
            model, heads, vocab, split_qas,
            args.features_dir, device,
        )

        all_metrics[split_name] = metrics
        all_preds.extend(preds)

        # Print metrics
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")

    # Compute robustness scores
    acc_6cam = all_metrics.get("nusc_val_6cam", {}).get("answer_accuracy", 0)
    acc_4cam = all_metrics.get("nusc_val_4cam_random", {}).get("answer_accuracy", 0)
    acc_3cam = all_metrics.get("nusc_val_3cam_random", {}).get("answer_accuracy", 0)
    robustness = compute_robustness_score(acc_6cam, acc_4cam, acc_3cam)

    print(f"\nRobustness: R_4cam={robustness['r_4cam']:.3f}, R_3cam={robustness['r_3cam']:.3f}")

    for split_name in all_metrics:
        all_metrics[split_name].update(robustness)

    # Save predictions
    os.makedirs(os.path.dirname(args.out_preds), exist_ok=True)
    with open(args.out_preds, "w") as f:
        for pred in all_preds:
            f.write(json.dumps(pred) + "\n")
    print(f"\nPredictions saved to {args.out_preds}")

    # Save results
    save_results_csv(all_metrics["nusc_val_6cam"], args.out_results + ".csv")
    save_results_markdown(all_metrics, args.out_results + ".md")
    print(f"Results saved to {args.out_results}.csv/.md")


def evaluate_split(model, heads, vocab, qa_list, features_dir, device):
    """Run evaluation on a split of QA instances."""
    adapter = model
    bev_head = heads.get("bev_head")
    vis_head = heads.get("vis_head")
    answer_head = heads.get("answer_head")
    projector = heads.get("projector")

    adapter.eval()
    if bev_head:
        bev_head.eval()
    if vis_head:
        vis_head.eval()
    if answer_head:
        answer_head.eval()
    if projector:
        projector.eval()

    preds = []
    with torch.no_grad():
        for qa in tqdm(qa_list, desc="Evaluating"):
            sample_token = qa["sample_token"]
            camera_subset = qa.get("camera_subset", FULL_NUSC_CAMERAS)

            # Load features
            features_by_camera = {}
            for cam_name in camera_subset:
                feat_path = os.path.join(features_dir, f"{sample_token}_{cam_name}.pt")
                if os.path.exists(feat_path):
                    data = torch.load(feat_path, map_location=device, weights_only=False)
                    features_by_camera[cam_name] = data

            if not features_by_camera:
                preds.append({
                    "id": qa["id"],
                    "predicted_answer": '{"answer": "unknown"}',
                    "answerability": "unanswerable",
                })
                continue

            try:
                adapter_out = adapter(features_by_camera, camera_subset)
            except Exception as e:
                preds.append({
                    "id": qa["id"],
                    "predicted_answer": '{"answer": "unknown"}',
                    "error": str(e),
                })
                continue

            # Generate answer
            if projector is not None and answer_head is not None:
                projected = projector(adapter_out["triplane_tokens"])
                answer_logits = answer_head(projected)
                predicted_answer = decode_answer(answer_logits, vocab)
            else:
                predicted_answer = '{"answer": "unknown"}'

            # BEV prediction
            pred_heatmap = None
            if bev_head is not None:
                pred_heatmap = bev_head(adapter_out["tokens_xy"])
                pred_heatmap = pred_heatmap.squeeze().cpu().numpy()

            preds.append({
                "id": qa["id"],
                "predicted_answer": predicted_answer,
                "pred_heatmap": pred_heatmap.tolist() if pred_heatmap is not None else None,
                "question_type": qa["question_type"],
                "answerability": qa["answerability"],
            })

    metrics = compute_metrics(preds, qa_list)
    return preds, metrics


def load_model(checkpoint_path: str, device: torch.device):
    """Load model and heads from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Default config
    hidden_dim = 512
    x_range = (-20.0, 80.0)
    y_range = (-40.0, 40.0)
    z_range = (-3.0, 8.0)
    sx, sy, sz = 96, 96, 48
    patch_size = 8

    adapter = EgoTriPlaneAdapter(
        feature_dim=1024,
        hidden_dim=hidden_dim,
        x_range=x_range, y_range=y_range, z_range=z_range,
        sx=sx, sy=sy, sz=sz,
        patch_size=patch_size,
        use_ray_embedding=True,
        aggregation="attention",
    )

    if "adapter" in checkpoint:
        adapter.load_state_dict(checkpoint["adapter"])

    bev_head = BEVHeatmapHead(hidden_dim, sx, sy, sz, patch_size)
    if "bev_head" in checkpoint:
        bev_head.load_state_dict(checkpoint["bev_head"])

    vis_head = VisibilityHead(hidden_dim, sx, sy, patch_size)
    if "vis_head" in checkpoint:
        vis_head.load_state_dict(checkpoint["vis_head"])

    vocab = checkpoint.get("vocab", _build_default_vocab())

    answer_head = TextAnswerHead(
        hidden_dim=1024,
        num_triplane_tokens=288,
        max_answer_len=128,
    )
    if "answer_head" in checkpoint:
        answer_head.load_state_dict(checkpoint["answer_head"])

    projector = nn.Sequential(
        nn.Linear(hidden_dim, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
    )
    if "projector" in checkpoint:
        projector.load_state_dict(checkpoint["projector"])

    adapter.to(device)
    bev_head.to(device)
    vis_head.to(device)
    answer_head.to(device)
    projector.to(device)

    heads = {
        "bev_head": bev_head,
        "vis_head": vis_head,
        "answer_head": answer_head,
        "projector": projector,
    }

    return adapter, heads, vocab


import torch.nn as nn


def decode_answer(logits: torch.Tensor, vocab: dict) -> str:
    """Decode answer logits to text."""
    if logits is None:
        return '{"answer": "unknown"}'
    token_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
    id_to_char = {v: k for k, v in vocab.items()}

    chars = []
    for tid in token_ids:
        c = id_to_char.get(int(tid), "")
        if c in ("<EOS>", "<PAD>"):
            break
        if c not in ("<BOS>", "<UNK>"):
            chars.append(c)
    return "".join(chars)


def _build_default_vocab():
    chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                "{}[]\":,.-_ /<>()\n")
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for i, c in enumerate(sorted(chars), start=4):
        vocab[c] = i
    return vocab


def _split_qas_by_camera_config(qa_list: List[dict]) -> Dict[str, List[dict]]:
    """Group QAs into test splits based on camera subset."""
    splits = defaultdict(list)

    for qa in qa_list:
        cams = tuple(sorted(qa.get("camera_subset", FULL_NUSC_CAMERAS)))
        n_cams = len(cams)

        # Classify into splits
        if n_cams == 6 and set(cams) == set(FULL_NUSC_CAMERAS):
            splits["nusc_val_6cam"].append(qa)
        elif n_cams == 5:
            splits["nusc_val_5cam_random"].append(qa)
        elif n_cams == 4:
            splits["nusc_val_4cam_random"].append(qa)
        elif n_cams == 3:
            splits["nusc_val_3cam_random"].append(qa)

        if set(cams) == set(CAMERA_SUBSETS["front3"]):
            splits["nusc_val_front3"].append(qa)
        if set(cams) == set(CAMERA_SUBSETS["no_rear"]):
            splits["nusc_val_no_rear"].append(qa)
        if set(cams) == set(UNSEEN_HOLDOUT_CAMERAS):
            splits["nusc_val_unseen_subset"].append(qa)

    return dict(splits)


if __name__ == "__main__":
    main()
