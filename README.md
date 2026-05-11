# EgoTriPlane-Adapter

Multi-camera 2D features → unified ego-coordinate triplane representation (XY/XZ/YZ planes) for 3D perception.

Token count is decoupled from camera count: with default config (sx=96, sy=96, sz=48, patch_size=8), always **288 tokens** regardless of how many cameras are used.

## Server Setup

### 1. Clone & environment

```bash
git clone https://github.com/Jiaduobaotth/egotriplane-adapter.git
cd egotriplane-adapter

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install other dependencies
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Download nuScenes dataset

Download **nuScenes** (full trainval, ~350GB) from https://www.nuscenes.org/download.

After downloading, organize as:

```
./data/
├── maps/
├── samples/
├── sweeps/
└── v1.0-trainval/
```

Or symlink to your data location:

```bash
ln -s /path/to/nuscenes ./data
```

### 3. Download Qwen3-VL model (for vision backbone)

```bash
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# Set HF mirror for model download
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct
```

## Training

### Stage 1: Vision Encoder 3D Perception Pretrain

Train the vision encoder to understand 3D from multi-view images via center-based detection supervision. No VQA/language loss — pure 3D perception.

**Local debug (torchvision ViT, no model download needed, <4GB VRAM):**

```bash
python scripts/train_vision_3d.py \
    --backbone tv_vit_b_16 \
    --image_size 224 \
    --batch_size 1 --grad_accum 4 \
    --epochs 5 \
    --out_dir outputs/vision3d_debug/
```

**Production training (Qwen3-VL 4B, progressive unfreezing):**

```bash
# Phase 1: freeze most ViT layers, train adapter + head + last few layers
python scripts/train_vision_3d.py \
    --backbone qwen3vl_4b \
    --image_size 448 \
    --freeze_vision_until 18 \
    --aggregation attention \
    --batch_size 1 --grad_accum 16 \
    --epochs 10 \
    --lr 1e-4 --wd 0.01 \
    --out_dir outputs/stage1_phase1/

# Phase 2: unfreeze more layers, resume from phase 1
python scripts/train_vision_3d.py \
    --backbone qwen3vl_4b \
    --image_size 448 \
    --freeze_vision_until 8 \
    --aggregation attention \
    --batch_size 1 --grad_accum 16 \
    --epochs 10 \
    --resume outputs/stage1_phase1/best.pt \
    --out_dir outputs/stage1_phase2/
```

## Inference & Visualization

### Run inference on validation set

```bash
python scripts/infer_vision_3d.py --ckpt outputs/stage1_phase1/best.pt --nusc_root ./data --nusc_version v1.0-trainval --image_size 448 --backbone qwen3vl_4b --score_thresh 0.15 --out_dir outputs/inference/
```

### Visualize detection results

```bash
python scripts/visualize_detections.py --pred_json outputs/inference/predictions.json --gt_json outputs/inference/groundtruths.json --num_samples 30 --score_thresh 0.15 --out_dir outputs/vis/
```

Or run inference + visualization in one command:

```bash
python scripts/visualize_detections.py --ckpt outputs/stage1_phase1/best.pt --nusc_root ./data --nusc_version v1.0-trainval --num_samples 20 --score_thresh 0.15 --out_dir outputs/vis/
```

Output files:
- `outputs/inference/predictions.json` — decoded 3D boxes, scores, classes per sample
- `outputs/inference/groundtruths.json` — GT boxes and labels per sample
- `outputs/vis/bev_XXX.png` — BEV overlay: pred (colored) + GT (grey)
- `outputs/vis/compare_XXX.png` — side-by-side: left=pred, right=GT
- `outputs/vis/summary_grid.png` — multi-sample thumbnail grid

## Project Structure

```
egotriplane_adapter/
├── egotriplane/               # Core library
│   ├── triplane_adapter.py    # EgoTriPlaneAdapter (2D→3D projection)
│   ├── vision_encoder.py      # Pluggable vision encoder wrapper
│   ├── heads.py               # CenterDetHead, BEVSegHead
│   ├── losses.py              # Detection loss, BEV segmentation loss
│   ├── image_dataset.py       # nuScenes image dataset
│   ├── geometry.py            # Coordinate transforms & projection
│   └── ...
├── scripts/
│   ├── train_vision_3d.py     # Stage 1: vision encoder 3D pretraining
│   ├── infer_vision_3d.py     # Inference on val set, output JSON detections
│   ├── visualize_detections.py # BEV visualization of predicted 3D boxes
│   └── ...
├── configs/                   # YAML configs
└── requirements.txt
```

## Key Parameters (train_vision_3d.py)

| Parameter | Default | Description |
|---|---|---|
| `--backbone` | `tv_vit_b_16` | Vision backbone: `tv_vit_b_16`, `tv_vit_l_16`, `clip_vit_base`, `clip_vit_large`, `qwen25vl_3b`, `qwen25vl_7b`, `qwen3vl_4b`, `qwen3vl_8b` |
| `--image_size` | 224 | Input resolution (Qwen3-VL: 448 recommended) |
| `--aggregation` | `attention` | Multi-camera fusion: `mean`, `max`, `attention` |
| `--freeze_vision` | False | Completely freeze vision backbone |
| `--freeze_vision_until` | 0 | Freeze first N ViT layers, train the rest |
| `--batch_size` | 1 | Samples per step |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--epochs` | 5 | Training epochs |
| `--lr` | 1e-4 | Learning rate (AdamW) |
| `--wd` | 0.01 | Weight decay |
| `--warmup` | 200 | Warmup steps (OneCycleLR) |
| `--max_grad_norm` | 1.0 | Gradient clipping |
| `--lambda_det` | 1.0 | Detection loss weight |
| `--lambda_bev` | 0.0 | BEV seg loss weight (0=disabled) |
| `--w_heatmap` | 1.0 | Heatmap (focal) loss weight |
| `--w_offset` | 0.1 | Offset regression loss weight |
| `--w_size` | 0.1 | Size regression loss weight |
| `--w_yaw` | 0.1 | Yaw regression loss weight |
| `--nusc_root` | `./data` | nuScenes data root |
| `--nusc_version` | `v1.0-mini` | nuScenes version (`v1.0-trainval` for full) |
| `--min_cams` | 3 | Min cameras per sample (dropout) |
| `--max_cams` | 6 | Max cameras per sample |
| `--out_dir` | `outputs/vision3d_debug/` | Checkpoint output directory |
| `--resume` | None | Resume from checkpoint path |
| `--save_every` | 5 | Save checkpoint every N epochs |
| `--seed` | 42 | Random seed |

## Coordinate System

- **Ego frame**: x=forward, y=left, z=up
- All 3D boxes and positions are in ego coordinate
