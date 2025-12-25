# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains code for **two research papers** on synthetic-to-real domain adaptation for Activities of Daily Living (ADL) recognition using the Sims4Action synthetic dataset.

### Paper 1: MMGen (IROS 2022)
**Multimodal Domain Generation for Synthetic-to-Real Activity Recognition**
- Uses adversarial domain generator (L2A-OT) to create novel modalities
- Improves domain generalization via diversified training
- Domain generator training code: `L2A-OT/` submodule

### Paper 2: ModSelect (ECCV 2022)
**Unsupervised Modality Selection for Cross-Domain Action Recognition**
- Selects beneficial modalities via correlation + MMD thresholds
- Adds YOLO as 5th modality (object detection distances)
- Uses late fusion (Borda Count, Sum, Product, etc.)

## Repository Structure

```
main.py                          # Entry point for action classifier training/testing
├── lib/
│   ├── s3d.py                   # S3D backbone (primary classifier)
│   ├── late_fusion_s3d.py       # Two-stream late fusion
│   ├── YOLO_mlp.py              # MLP for YOLO vectors (ModSelect)
│   └── s3d_yolo_fusion.py       # S3D + YOLO fusion (ModSelect)
├── datasets/
│   ├── sims_dataset_video.py    # Single modality loader
│   ├── sims_dataset_video_multiple_modalities.py  # Multi-modality
│   └── sims_dataset_YOLO_detections.py            # YOLO loader (ModSelect)
├── training/
│   └── train_video_stream.py    # Training loop
├── testing/
│   ├── test_video_stream.py     # Main test script (optional GAN)
│   ├── test_video_stream_GAN_required.py  # Always uses GAN
│   ├── test_video_stream_original.py      # No GAN support
│   └── generator.py             # Domain generator (inference)
├── utils/
│   ├── late_fusion_borda_count.py            # Borda voting
│   ├── late_fusion_borda_count_multimodal.py # Multi-modal fusion
│   └── generate_*.py            # Data preprocessing
└── L2A-OT/                      # Submodule: domain generator training
```

## Common Commands

### MMGen: Train Domain Generator
```bash
# Clone with submodule
git clone --recurse-submodules <repo-url>

# Train domain generator (in L2A-OT)
cd L2A-OT
python main_SIMS_S3D.py --modalities heatmaps limbs --epochs 30000

# Test action classifier with domain generation
cd ..
python main.py --test_only --G_path L2A-OT/checkpoints/model.pth \
    --eval_datasets /path/to/toyota
```

### ModSelect: Train and Fuse Modalities
```bash
# Train unimodal classifiers
python main.py --dataset sims_video --modality heatmaps --n_channels 1 --epochs 200
python main.py --dataset sims_video --modality limbs --n_channels 1 --epochs 200
python main.py --dataset sims_video --modality optical_flow --n_channels 3 --epochs 200
python main.py --dataset sims_video --modality rgb --n_channels 3 --epochs 200
python main.py --dataset YOLO_detections_only --model_vid YOLO_mlp --epochs 200

# Late fusion via Borda Count
python utils/late_fusion_borda_count.py \
    --csv_roots h.csv l.csv of.csv rgb.csv yolo.csv \
    --modalities heatmaps limbs optical_flow rgb yolo
```

### Basic Training
```bash
# Single modality
python main.py --gpu 0 1 --dataset sims_video --modality heatmaps \
    --n_channels 1 --epochs 200 --dataset-video-root /path/to/data

# Early fusion (4 modalities)
python main.py --gpu 0 1 --dataset sims_video_multimodal --n_modalities 4 \
    --modalities heatmaps limbs optical_flow rgb \
    --n_channels_each_modality 1 1 3 3 --n_channels 8
```

### Data Preprocessing
```bash
# Skeleton heatmaps/limbs from AlphaPose
python utils/generate_skeletons_heatmaps.py \
    --root_dir /path/to/alphapose --result_dir /path/to/output

# Optical flow
python utils/generate_optical_flow.py \
    --root_dir /path/to/rgb --result_dir /path/to/output --n_workers 3
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | `sims_video`, `sims_video_multimodal`, `adl`, `YOLO_detections_only` |
| `--model_vid` | `s3d`, `i3d`, `s3d_yolo_fusion`, `YOLO_mlp` |
| `--modality` / `--modalities` | `heatmaps`, `limbs`, `optical_flow`, `rgb`, `yolo` |
| `--G_path` | Path to domain generator checkpoint (MMGen) |
| `--fine_tune_late_fusion` | Enable two-stream late fusion training |

## Dataset Structure

```
dataset_root/
├── <Action>/
│   └── <video_id>/
│       ├── heatmaps.avi
│       ├── limbs.avi
│       ├── optical_flow.avi
│       └── rgb.avi
```

## Modalities

| Modality | Channels | Description |
|----------|----------|-------------|
| Heatmaps (H) | 1 | Gaussian at joint locations from AlphaPose |
| Limbs (L) | 1 | Lines connecting joints |
| Optical Flow (OF) | 3 | HSV encoding of motion |
| RGB | 3 | Original video frames |
| YOLO | 80-dim vector | Object detection distances (ModSelect only) |

## Notes

- DataParallel for multi-GPU training
- Sequence length: 16 frames at 112x112 resolution
- Results saved to `experiments/exp_{num}/`
- TensorBoard logs in `experiments/exp_{num}/logs/`
