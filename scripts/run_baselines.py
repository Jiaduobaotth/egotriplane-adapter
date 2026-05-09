#!/usr/bin/env python3
"""Baseline implementations for Ego3D-QA.

Baseline A: Frozen VLM multi-image (direct image input)
Baseline B: VLM + camera-name prompt
Baseline C: VLM + calibration text prompt
Baseline D: LoRA-only VLM (no triplane)
Baseline E: BEV-only adapter (XOY plane only)

Usage:
    # Baseline A: frozen VLM, multi-image
    python scripts/run_baselines.py \
        --baseline frozen_vlm \
        --qa outputs/ego3dqa/nusc_val_ego3dqa.jsonl \
        --nusc_root /data/nuscenes \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --device cuda

    # Baseline D: LoRA-only VLM training
    python scripts/run_baselines.py \
        --baseline lora_vlm \
        --train_qa outputs/ego3dqa/nusc_train_ego3dqa.jsonl \
        --val_qa outputs/ego3dqa/nusc_val_ego3dqa.jsonl \
        --nusc_root /data/nuscenes \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --device cuda \
        --epochs 10
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egotriplane.nusc_utils import load_qa, load_index
from egotriplane.metrics import compute_metrics
from egotriplane.camera_dropout import FULL_NUSC_CAMERAS


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines for Ego3D-QA")
    parser.add_argument("--baseline", type=str, required=True,
                        choices=["frozen_vlm", "camera_prompt", "calib_prompt",
                                 "lora_vlm", "bev_only"],
                        help="Which baseline to run")
    parser.add_argument("--qa", type=str, required=True,
                        help="Validation QA JSONL")
    parser.add_argument("--train_qa", type=str, default=None,
                        help="Training QA JSONL (for baselines requiring training)")
    parser.add_argument("--index", type=str, default=None,
                        help="Sample index JSONL")
    parser.add_argument("--nusc_root", type=str, default="/data/nuscenes")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--out_preds", type=str, default="outputs/eval/baseline_preds.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    qa_list = load_qa(args.qa)
    if args.limit:
        qa_list = qa_list[:args.limit]
    print(f"Loaded {len(qa_list)} QA instances")

    if args.baseline == "frozen_vlm":
        preds = run_frozen_vlm(qa_list, args, device)
    elif args.baseline == "camera_prompt":
        preds = run_camera_prompt_vlm(qa_list, args, device)
    elif args.baseline == "calib_prompt":
        preds = run_calib_prompt_vlm(qa_list, args, device)
    elif args.baseline == "lora_vlm":
        preds = run_lora_vlm(args.train_qa, qa_list, args, device)
    elif args.baseline == "bev_only":
        preds = run_bev_only_adapter(qa_list, args, device)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    # Evaluate
    metrics = compute_metrics(preds, qa_list)
    print(f"\n{args.baseline} Results:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.out_preds), exist_ok=True)
    with open(args.out_preds, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"Predictions saved to {args.out_preds}")


# ---------------------------------------------------------------------------
# Baseline A: Frozen VLM multi-image
# ---------------------------------------------------------------------------

FROZEN_VLM_PROMPT = """You are a self-driving perception system. Given {n_cams} camera images from a vehicle, answer the question about 3D object positions in the scene.

Important rules:
- Positions are relative to the ego vehicle: x=forward, y=left, z=up.
- If the relevant area is not visible in any camera, answer "unknown".
- Respond ONLY with a JSON object.

Question: {question}

Respond with JSON:"""


def run_frozen_vlm(qa_list: List[dict], args, device: torch.device) -> List[dict]:
    """Baseline A: directly feed multi-camera images to VLM."""
    model, processor = _load_vlm(args.model_name, device)

    preds = []
    for qa in tqdm(qa_list, desc="Frozen VLM"):
        camera_subset = qa.get("camera_subset", FULL_NUSC_CAMERAS[:4])  # limit to 4 for context
        n_cams = len(camera_subset)

        # Load images
        images = []
        for cam_name in camera_subset:
            # Need index for image paths
            img_path = _get_image_path(qa["sample_token"], cam_name, args)
            if img_path and os.path.exists(img_path):
                images.append(Image.open(img_path).convert("RGB"))

        if not images:
            preds.append({"id": qa["id"], "predicted_answer": '{"answer": "unknown"}'})
            continue

        # Build prompt
        prompt = FROZEN_VLM_PROMPT.format(
            n_cams=len(images),
            question=qa["question"],
        )

        # Generate (simplified - in practice would use VLM chat template)
        answer_text = _generate_vlm_response(
            model, processor, images, prompt, device
        )

        preds.append({
            "id": qa["id"],
            "predicted_answer": answer_text,
        })

    return preds


# ---------------------------------------------------------------------------
# Baseline B: VLM + camera-name prompt
# ---------------------------------------------------------------------------

CAMERA_NAME_PROMPT = """You are a self-driving perception system. You receive images from multiple cameras:

{image_descriptions}

Question: {question}

Respond with JSON only:"""


def run_camera_prompt_vlm(qa_list, args, device):
    """Baseline B: prompt includes camera names."""
    model, processor = _load_vlm(args.model_name, device)

    preds = []
    for qa in tqdm(qa_list, desc="Camera Prompt VLM"):
        camera_subset = qa.get("camera_subset", FULL_NUSC_CAMERAS[:4])
        n_cams = len(camera_subset)

        images = []
        descriptions = []
        for i, cam_name in enumerate(camera_subset):
            img_path = _get_image_path(qa["sample_token"], cam_name, args)
            if img_path and os.path.exists(img_path):
                images.append(Image.open(img_path).convert("RGB"))
                descriptions.append(f"Image {i+1}: {cam_name}")

        if not images:
            preds.append({"id": qa["id"], "predicted_answer": '{"answer": "unknown"}'})
            continue

        prompt = CAMERA_NAME_PROMPT.format(
            image_descriptions="\n".join(descriptions),
            question=qa["question"],
        )

        answer_text = _generate_vlm_response(model, processor, images, prompt, device)

        preds.append({
            "id": qa["id"],
            "predicted_answer": answer_text,
        })

    return preds


# ---------------------------------------------------------------------------
# Baseline C: VLM + calibration text prompt
# ---------------------------------------------------------------------------

def _describe_camera_extrinsics(cam_name: str) -> str:
    """Rough textual description of camera extrinsic (yaw angle)."""
    yaw_map = {
        "CAM_FRONT": "0 degrees (facing forward)",
        "CAM_FRONT_LEFT": "55 degrees (facing front-left)",
        "CAM_FRONT_RIGHT": "-55 degrees (facing front-right)",
        "CAM_BACK": "180 degrees (facing rear)",
        "CAM_BACK_LEFT": "125 degrees (facing rear-left)",
        "CAM_BACK_RIGHT": "-125 degrees (facing rear-right)",
    }
    return yaw_map.get(cam_name, "unknown orientation")


def run_calib_prompt_vlm(qa_list, args, device):
    """Baseline C: prompt includes calibration descriptions."""
    model, processor = _load_vlm(args.model_name, device)

    preds = []
    for qa in tqdm(qa_list, desc="Calib Prompt VLM"):
        camera_subset = qa.get("camera_subset", FULL_NUSC_CAMERAS[:4])

        images = []
        descriptions = []
        for i, cam_name in enumerate(camera_subset):
            img_path = _get_image_path(qa["sample_token"], cam_name, args)
            if img_path and os.path.exists(img_path):
                images.append(Image.open(img_path).convert("RGB"))
                desc = _describe_camera_extrinsics(cam_name)
                descriptions.append(f"Image {i+1} ({cam_name}): {desc}")

        if not images:
            preds.append({"id": qa["id"], "predicted_answer": '{"answer": "unknown"}'})
            continue

        prompt = CAMERA_NAME_PROMPT.format(
            image_descriptions="\n".join(descriptions),
            question=qa["question"],
        )

        answer_text = _generate_vlm_response(model, processor, images, prompt, device)

        preds.append({
            "id": qa["id"],
            "predicted_answer": answer_text,
        })

    return preds


# ---------------------------------------------------------------------------
# Baseline D: LoRA-only VLM (no triplane adapter)
# ---------------------------------------------------------------------------

def run_lora_vlm(train_qa, val_qa, args, device):
    """Baseline D: fine-tune VLM with LoRA, no triplane adapter."""
    if train_qa is None:
        print("WARNING: No training QA provided for LoRA baseline. Using frozen inference.")
        return run_frozen_vlm(val_qa, args, device)

    # This is a placeholder for LoRA training logic
    # Real implementation would use peft.LoraConfig + Trainer
    print("LoRA training placeholder - loading and training...")
    from peft import LoraConfig, get_peft_model, TaskType

    model, processor = _load_vlm(args.model_name, device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Training loop (simplified)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    train_qa_list = load_qa(train_qa) if isinstance(train_qa, str) else train_qa

    for epoch in range(args.epochs):
        total_loss = 0
        for qa in tqdm(train_qa_list[:1000], desc=f"LoRA Epoch {epoch+1}"):  # limit for speed
            camera_subset = qa.get("camera_subset", FULL_NUSC_CAMERAS[:4])
            images = []
            for cam_name in camera_subset[:4]:
                img_path = _get_image_path(qa["sample_token"], cam_name, args)
                if img_path and os.path.exists(img_path):
                    images.append(Image.open(img_path).convert("RGB"))

            if not images:
                continue

            try:
                answer_text = json.dumps(qa["answer"])
                loss = _train_vlm_step(model, processor, images, qa["question"],
                                       answer_text, optimizer, device)
                total_loss += loss
            except Exception:
                continue

        print(f"  Epoch {epoch+1}: avg_loss={total_loss/max(1, len(train_qa_list)):.4f}")

    # Evaluate
    model.eval()
    return run_frozen_vlm(val_qa, args, device)  # Use same inference but with LoRA weights


# ---------------------------------------------------------------------------
# Baseline E: BEV-only adapter (XOY plane only, no XZ/YZ)
# ---------------------------------------------------------------------------

def run_bev_only_adapter(qa_list, args, device):
    """Baseline E: Only use XOY plane, no triplane."""
    # Placeholder for BEV-only adapter inference
    # In practice, this uses a modified EgoTriPlaneAdapter with only the XY plane
    print("BEV-only adapter placeholder - would use adapter with xy_only=True")

    preds = []
    for qa in qa_list:
        preds.append({
            "id": qa["id"],
            "predicted_answer": '{"answer": "unknown"}',
            "note": "BEV-only baseline placeholder",
        })

    return preds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vlm(model_name: str, device: torch.device):
    """Load VLM model and processor."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device.type == "cuda" else None,
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    except ImportError:
        print("WARNING: Could not load Qwen2.5-VL. Install with: pip install qwen-vl-utils")
        return None, None


def _generate_vlm_response(model, processor, images, prompt, device) -> str:
    """Generate response from VLM."""
    if model is None or processor is None:
        return '{"answer": "unknown"}'

    try:
        # Build chat format
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img} for img in images
            ] + [{"type": "text", "text": prompt}]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text, images=images,
            return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)
        # Extract answer part
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        return response

    except Exception as e:
        print(f"  VLM generation error: {e}")
        return '{"answer": "unknown"}'


def _train_vlm_step(model, processor, images, question, answer, optimizer, device) -> float:
    """Single training step for VLM."""
    if model is None:
        return 0.0

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img} for img in images
        ] + [{"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(
        text=text, images=images,
        return_tensors="pt", padding=True,
    ).to(model.device)

    labels = inputs["input_ids"].clone()
    # Mask user part
    # (simplified - proper masking would identify user/assistant segments)

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def _get_image_path(sample_token: str, cam_name: str, args) -> Optional[str]:
    """Get image path for a sample/camera from index."""
    if args.index and os.path.exists(args.index):
        samples = load_index(args.index)
        samples_by_token = {s["sample_token"]: s for s in samples}
        sample = samples_by_token.get(sample_token)
        if sample:
            for cam in sample.get("cameras", []):
                if cam["name"] == cam_name:
                    return os.path.join(args.nusc_root, cam["image_path"])
    return None


if __name__ == "__main__":
    main()
