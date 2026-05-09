#!/usr/bin/env python3
"""Generate Ego3D-QA dataset from nuScenes sample index.

Usage:
    python scripts/generate_ego3dqa.py \
        --index outputs/ego3dqa/nusc_train_index.jsonl \
        --out outputs/ego3dqa/nusc_train_ego3dqa.jsonl \
        --num_dropout_versions 3 \
        --max_qa_per_sample 8

For each sample, generates QA instances under multiple camera subsets
including full 6-camera and random dropout versions.
Controls the answer distribution to target ~40/20/40 yes/no/unknown.
"""

import argparse
import sys
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.nusc_utils import load_index, save_qa
from egotriplane.qa_generator import generate_qa_for_sample
from egotriplane.camera_dropout import (
    FULL_NUSC_CAMERAS,
    CAMERA_SUBSETS,
    generate_dropout_subsets,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Ego3D-QA dataset")
    parser.add_argument("--index", type=str, required=True,
                        help="Input sample index JSONL")
    parser.add_argument("--out", type=str, required=True,
                        help="Output QA JSONL path")
    parser.add_argument("--num_dropout_versions", type=int, default=3,
                        help="Number of random camera dropout versions per sample")
    parser.add_argument("--max_qa_per_sample", type=int, default=8,
                        help="Maximum QA instances per sample (across all subsets)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--target_yes_ratio", type=float, default=0.4)
    parser.add_argument("--target_no_ratio", type=float, default=0.2)
    parser.add_argument("--target_unknown_ratio", type=float, default=0.4)
    parser.add_argument("--debug", action="store_true",
                        help="Print debug diagnostics for closest/ego_path QAs")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    samples = load_index(args.index)
    print(f"Loaded {len(samples)} samples from {args.index}")

    all_qa = []
    qa_per_sample = []

    for sample in tqdm(samples, desc="Generating QA"):
        sample_qas = []

        # Generate dropout camera subsets
        all_cameras = [c["name"] for c in sample["cameras"]]
        subsets = generate_dropout_subsets(
            all_cameras,
            num_versions=args.num_dropout_versions,
            include_full=True,
        )

        # Also add known fixed subsets that are compatible
        for subset_name, subset_cams in CAMERA_SUBSETS.items():
            if all(c in all_cameras for c in subset_cams):
                if subset_cams not in subsets:
                    subsets.append(subset_cams)

        # Generate QAs for each subset
        for camera_subset in subsets:
            qas = generate_qa_for_sample(sample, camera_subset, max_per_type=2,
                                          debug=args.debug)
            sample_qas.extend(qas)

        # Cap and collect
        sample_qas = sample_qas[:args.max_qa_per_sample]
        qa_per_sample.append(len(sample_qas))
        all_qa.extend(sample_qas)

    # Rebalance to target distribution
    all_qa = _rebalance_distribution(
        all_qa,
        args.target_yes_ratio,
        args.target_no_ratio,
        args.target_unknown_ratio,
        args.seed,
    )

    # Statistics
    print(f"\nGenerated {len(all_qa)} total QA instances")
    print(f"  Avg QA per sample: {np.mean(qa_per_sample):.1f}")

    type_counts = Counter(q["question_type"] for q in all_qa)
    print(f"  Types: {dict(type_counts)}")

    answerable_counts = Counter(q["answerability"] for q in all_qa)
    print(f"  Answerability: {dict(answerable_counts)}")

    # Answer distribution
    answers = []
    for q in all_qa:
        ans = q["answer"].get("answer", "unknown")
        if ans in ("yes", "no", "unknown"):
            answers.append(ans)
        elif ans == "visible":
            answers.append("yes")  # closest object => treat as descriptive
        else:
            answers.append("other")
    ans_counts = Counter(answers)
    total = len(answers)
    print(f"  Answer distribution: yes={ans_counts.get('yes',0)/total:.2%}, "
          f"no={ans_counts.get('no',0)/total:.2%}, "
          f"unknown={ans_counts.get('unknown',0)/total:.2%}, "
          f"other={ans_counts.get('other',0)/total:.2%}")

    # Camera subset distribution
    subset_sizes = Counter(len(q["camera_subset"]) for q in all_qa)
    print(f"  Camera subset sizes: {dict(sorted(subset_sizes.items()))}")

    save_qa(all_qa, args.out)
    print(f"\nSaved to {args.out}")


def _rebalance_distribution(qa_list, target_yes, target_no, target_unknown, seed):
    """Downsample QAs to approach target answer distribution."""
    rng = random.Random(seed)

    # Classify
    yes_qa, no_qa, unknown_qa, other_qa = [], [], [], []
    for q in qa_list:
        ans = q["answer"].get("answer", "unknown")
        if ans == "no" or ans == "none":
            no_qa.append(q)
        elif ans == "unknown":
            unknown_qa.append(q)
        elif ans in ("yes", "visible"):
            yes_qa.append(q)
        else:
            other_qa.append(q)

    total = len(qa_list)
    target_yes_n = int(total * target_yes)
    target_no_n = int(total * target_no)
    target_unknown_n = int(total * target_unknown)

    # Downsample to targets
    balanced = []
    balanced.extend(rng.sample(yes_qa, min(len(yes_qa), target_yes_n)))
    balanced.extend(rng.sample(no_qa, min(len(no_qa), target_no_n)))
    balanced.extend(rng.sample(unknown_qa, min(len(unknown_qa), target_unknown_n)))
    balanced.extend(other_qa)  # keep all "other"

    rng.shuffle(balanced)
    return balanced


if __name__ == "__main__":
    main()
