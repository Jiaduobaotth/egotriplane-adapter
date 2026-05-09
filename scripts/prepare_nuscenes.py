#!/usr/bin/env python3
"""Prepare nuScenes samples into a unified JSONL index.

Usage:
    python scripts/prepare_nuscenes.py \
        --nusc_root /data/nuscenes \
        --version v1.0-trainval \
        --split train \
        --out outputs/ego3dqa/nusc_train_index.jsonl

Outputs one JSON object per line, each conforming to the Sample schema.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.nusc_utils import load_nusc_sample, save_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare nuScenes sample index as JSONL"
    )
    parser.add_argument("--nusc_root", type=str, required=True,
                        help="Path to nuScenes root directory")
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        help="nuScenes version")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "mini_train", "mini_val"],
                        help="Data split")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSONL path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy import: only require nuscenes-devkit when running this script
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("ERROR: nuscenes-devkit not installed.")
        print("  pip install nuscenes-devkit")
        sys.exit(1)

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)

    # Determine sample tokens for the requested split
    if args.split == "train":
        sample_tokens = _get_train_tokens(nusc)
    elif args.split == "val":
        sample_tokens = _get_val_tokens(nusc)
    elif args.split == "mini_train":
        sample_tokens = _get_mini_train_tokens(nusc)
    elif args.split == "mini_val":
        sample_tokens = _get_mini_val_tokens(nusc)
    else:
        raise ValueError(f"Unknown split: {args.split}")

    print(f"Processing {len(sample_tokens)} samples for split={args.split}")

    samples = []
    skip_no_box = 0
    for token in tqdm(sample_tokens, desc="Loading samples"):
        sample = load_nusc_sample(nusc, token)
        if not sample["objects"]:
            skip_no_box += 1
            continue
        samples.append(sample)

    print(f"Kept {len(samples)} samples (skipped {skip_no_box} with no boxes)")

    save_index(samples, args.out)
    print(f"Saved to {args.out}")


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _get_train_tokens(nusc) -> list:
    """Return training sample tokens."""
    tokens = []
    for scene in nusc.scene:
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            tokens.append(token)
            token = sample["next"]
    # nuScenes official: train scenes 0-699, val 700-849
    # We filter by scene number
    train_tokens = []
    for scene in nusc.scene:
        scene_num = _get_scene_number(scene["name"])
        if scene_num is not None and scene_num <= 699:
            token = scene["first_sample_token"]
            while token:
                sample = nusc.get("sample", token)
                train_tokens.append(token)
                token = sample["next"]
    return train_tokens


def _get_val_tokens(nusc) -> list:
    """Return validation sample tokens."""
    val_tokens = []
    for scene in nusc.scene:
        scene_num = _get_scene_number(scene["name"])
        if scene_num is not None and scene_num >= 700:
            token = scene["first_sample_token"]
            while token:
                sample = nusc.get("sample", token)
                val_tokens.append(token)
                token = sample["next"]
    return val_tokens


def _get_mini_train_tokens(nusc) -> list:
    """Use mini split scenes for quick testing."""
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    mini_train_scenes = splits["mini_train"]
    tokens = []
    for scene in nusc.scene:
        if scene["name"] in mini_train_scenes:
            token = scene["first_sample_token"]
            while token:
                tokens.append(token)
                sample = nusc.get("sample", token)
                token = sample["next"]
    return tokens


def _get_mini_val_tokens(nusc) -> list:
    """Mini val scenes for quick testing."""
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    mini_val_scenes = splits["mini_val"]
    tokens = []
    for scene in nusc.scene:
        if scene["name"] in mini_val_scenes:
            token = scene["first_sample_token"]
            while token:
                tokens.append(token)
                sample = nusc.get("sample", token)
                token = sample["next"]
    return tokens


def _get_scene_number(scene_name: str):
    """Extract scene number from name like 'scene-0001'."""
    parts = scene_name.split("-")
    if len(parts) == 2 and parts[0] == "scene":
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


if __name__ == "__main__":
    main()
