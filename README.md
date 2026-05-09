# EgoTriPlane-Adapter

**Lightweight Sensor-Configuration Agnostic 3D Adaptation for Driving VLMs**

## Overview

EgoTriPlane-Adapter projects multi-camera 2D features into a unified ego-coordinate triplane representation (XY/XZ/YZ planes). This decouples the token count from camera count and resolution, enabling robust 3D understanding even when cameras are dropped or reconfigured.

### Key Claims

1. **3D QA accuracy** higher than frozen VLM baselines
2. **BEV grounding** more precise (lower center error)
3. **Camera dropout robustness**: smaller accuracy drop when reducing cameras
4. **Unknown/observability**: reliably answers "unknown" for unobserved regions
5. **Cross-configuration generalization**: better than prompt-based or LoRA-only baselines

### Triplane Token Count

With default config (sx=96, sy=96, sz=48, patch_size=8):
- XY plane: 12×12 = 144 tokens
- XZ plane: 12×6  = 72 tokens
- YZ plane: 12×6  = 72 tokens
- **Total: 288 tokens** (fixed, independent of camera count)

## Installation

```bash
cd egotriplane_adapter
pip install -r requirements.txt
```

Requires nuScenes data at `/data/nuscenes/` (or symlink).

## Quick Start

### 1. Prepare nuScenes Index

```bash
python scripts/prepare_nuscenes.py \
    --nusc_root /data/nuscenes \
    --version v1.0-trainval \
    --split train \
    --out outputs/ego3dqa/nusc_train_index.jsonl

python scripts/prepare_nuscenes.py \
    --nusc_root /data/nuscenes \
    --version v1.0-trainval \
    --split val \
    --out outputs/ego3dqa/nusc_val_index.jsonl
```

### 2. Generate Ego3D-QA

```bash
python scripts/generate_ego3dqa.py \
    --index outputs/ego3dqa/nusc_train_index.jsonl \
    --out outputs/ego3dqa/nusc_train_ego3dqa.jsonl \
    --num_dropout_versions 3 \
    --max_qa_per_sample 8

python scripts/generate_ego3dqa.py \
    --index outputs/ego3dqa/nusc_val_index.jsonl \
    --out outputs/ego3dqa/nusc_val_ego3dqa.jsonl \
    --num_dropout_versions 5 \
    --max_qa_per_sample 12
```

### 3. Extract Frozen Vision Features

```bash
python scripts/extract_vlm_features.py \
    --qa outputs/ego3dqa/nusc_train_ego3dqa.jsonl \
    --index outputs/ego3dqa/nusc_train_index.jsonl \
    --model_name openai/clip-vit-large-patch14 \
    --nusc_root /data/nuscenes \
    --out outputs/features/nusc_train_clip_features/
```

### 4. Train Stage 1 (Adapter + BEV/Vis Heads)

```bash
python scripts/train_adapter_stage1.py \
    --config configs/train_adapter_stage1.yaml
```

### 5. Train Stage 2 (VLM + Adapter for QA)

```bash
python scripts/train_vlm_stage2.py \
    --config configs/train_vlm_stage2.yaml
```

### 6. Evaluate

```bash
python scripts/eval_ego3dqa.py \
    --config configs/eval.yaml \
    --checkpoint outputs/checkpoints/stage2/best.pt \
    --qa outputs/ego3dqa/nusc_val_ego3dqa.jsonl
```

### 7. Visualize

```bash
python scripts/visualize_ego3dqa.py \
    --qa outputs/ego3dqa/nusc_val_ego3dqa.jsonl \
    --index outputs/ego3dqa/nusc_val_index.jsonl \
    --nusc_root /data/nuscenes \
    --pred outputs/eval/preds.jsonl \
    --out outputs/eval/vis/ \
    --num_samples 50
```

### 8. Baselines

```bash
python scripts/run_baselines.py \
    --baseline frozen_vlm \
    --qa outputs/ego3dqa/nusc_val_ego3dqa.jsonl \
    --nusc_root /data/nuscenes \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct
```

## Project Structure

```
egotriplane_adapter/
├── configs/              # YAML configuration files
├── data/                 # Data schemas
├── scripts/              # Executable scripts
│   ├── prepare_nuscenes.py
│   ├── generate_ego3dqa.py
│   ├── extract_vlm_features.py
│   ├── train_adapter_stage1.py
│   ├── train_vlm_stage2.py
│   ├── eval_ego3dqa.py
│   ├── visualize_ego3dqa.py
│   └── run_baselines.py
├── egotriplane/          # Core library
│   ├── geometry.py       # Coordinate transforms, projection, visibility
│   ├── nusc_utils.py     # nuScenes data loading
│   ├── qa_generator.py   # Ego3D-QA generation (4 types)
│   ├── camera_dropout.py # Camera subset sampling
│   ├── feature_cache.py  # Feature I/O
│   ├── triplane_adapter.py  # Core EgoTriPlane model
│   ├── heads.py          # BEV, visibility, answer heads
│   ├── losses.py         # Combined loss functions
│   ├── dataset.py        # PyTorch datasets
│   └── metrics.py        # Evaluation metrics
├── outputs/              # Generated data, features, checkpoints
└── requirements.txt
```

## Coordinate System

- **Ego frame**: x=forward, y=left, z=up
- All 3D boxes and positions are in ego coordinate

## Q&A Types

1. **closest_object**: Localize the nearest object in a region
2. **ego_path**: Detect objects blocking the ego lane
3. **spatial_relation**: Classify object direction (left/right, front/back)
4. **unknown**: Determine if a region is observable by current cameras

## Camera Dropout

Training uses random camera subsets (3-6 cameras). Eval tests:
- 6cam (full), 5cam random, 4cam random, 3cam random
- Fixed subsets: front3, no_rear, unseen holdout

## Metrics

- Answer accuracy (by field: object, x-bin, y-bin, direction, visibility)
- BEV grounding error (meters)
- Unknown accuracy / hallucination rate
- Robustness score: R_4cam = Acc_4cam / Acc_6cam

## Baselines

| Baseline | Description |
|----------|-------------|
| A: Frozen VLM | Direct multi-image input |
| B: Camera-name prompt | Add camera names to prompt |
| C: Calibration prompt | Add extrinsic descriptions to prompt |
| D: LoRA-only VLM | Fine-tune VLM with LoRA, no adapter |
| E: BEV-only adapter | XOY plane only (no XZ/YZ) |
| **Ours** | EgoTriPlane-Adapter (3 planes + ray emb + vis loss) |

## Milestones

1. Data generation working (QA JSONL valid)
2. Stage 1 training: BEV heatmap + visibility losses decrease
3. Stage 2 training: QA accuracy above baseline
4. Camera dropout experiments: R_4cam, R_3cam above baselines
5. Ablation: full model outperforms ablations, visibility loss reduces hallucination
