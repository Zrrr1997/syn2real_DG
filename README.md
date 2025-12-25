# Synthetic-to-Real Domain Adaptation for Action Recognition

Repository for multimodal action recognition using synthetic data from Sims4Action, with domain adaptation for real-world datasets (Toyota Smarthome, ETRI-Activity3D).

## Papers

This repository contains implementations for two papers:

### MMGen (IROS 2022)
**Multimodal Domain Generation for Synthetic-to-Real Activity Recognition**
- Adversarial domain generation to create novel modalities
- Improves generalization to unseen real-world domains

### ModSelect (ECCV 2022)
**Unsupervised Modality Selection for Cross-Domain Action Recognition**
- Selects beneficial modalities via correlation and MMD thresholds
- Adds YOLO object detections as a 5th modality
- Late fusion strategies (Borda Count, Sum, Product)

## Installation

```bash
# Clone with submodule (required for domain generator)
git clone --recurse-submodules <repo-url>

# Or if already cloned:
git submodule update --init --recursive

# Install dependencies
pip install torch torchvision numpy pandas tqdm tensorboard opencv-python pillow
```

Requirements:
- PyTorch >= 1.7
- numpy, pandas, tensorboard, tqdm
- OpenCV (for video processing)

## Dataset Structure

Expected structure for each modality:
```
dataset_root/
├── <Action>/
│   └── <video_id>/
│       ├── heatmaps.avi
│       ├── limbs.avi
│       ├── optical_flow.avi
│       └── rgb.avi
```

## Usage

### Basic Training
```bash
# Single modality (heatmaps)
python main.py --gpu 0 1 --dataset sims_video --modality heatmaps \
    --n_channels 1 --epochs 200 --dataset-video-root /path/to/heatmaps

# Early fusion (4 modalities)
python main.py --gpu 0 1 --dataset sims_video_multimodal --n_modalities 4 \
    --modalities heatmaps limbs optical_flow rgb \
    --dataset_roots /path/to/h /path/to/l /path/to/of /path/to/rgb \
    --n_channels_each_modality 1 1 3 3 --n_channels 8 --epochs 200
```

### MMGen: Domain Generation
```bash
# 1. Train domain generator (in L2A-OT submodule)
cd L2A-OT
python main_SIMS_S3D.py --modalities heatmaps limbs --iterations 30000

# 2. Test with domain generation
cd ..
python main.py --test_only --G_path L2A-OT/checkpoints/model.pth \
    --pretrained_model_s3d /path/to/classifier.pth --eval_datasets /path/to/toyota
```

### ModSelect: Late Fusion
```bash
# 1. Train unimodal classifiers
python main.py --dataset sims_video --modality heatmaps --n_channels 1 --epochs 200
python main.py --dataset sims_video --modality limbs --n_channels 1 --epochs 200
python main.py --dataset sims_video --modality optical_flow --n_channels 3 --epochs 200
python main.py --dataset sims_video --modality rgb --n_channels 3 --epochs 200
python main.py --dataset YOLO_detections_only --model_vid YOLO_mlp --epochs 200

# 2. Late fusion via Borda Count
python utils/late_fusion_borda_count.py \
    --csv_roots h.csv l.csv of.csv rgb.csv yolo.csv \
    --modalities heatmaps limbs optical_flow rgb yolo
```

### Data Preprocessing
```bash
# Generate skeleton heatmaps/limbs from AlphaPose
python utils/generate_skeletons_heatmaps.py \
    --root_dir /path/to/alphapose_results \
    --result_dir /path/to/output

# Generate optical flow
python utils/generate_optical_flow.py \
    --root_dir /path/to/rgb_videos \
    --result_dir /path/to/output --n_workers 3
```

### Monitoring
```bash
tensorboard --logdir experiments/{exp-folder}/logs
```

## Modalities

| Modality | Channels | Source |
|----------|----------|--------|
| Heatmaps | 1 | Gaussian at AlphaPose joint locations |
| Limbs | 1 | Lines connecting joints |
| Optical Flow | 3 | Farneback algorithm (HSV encoded) |
| RGB | 3 | Original video frames |
| YOLO | 80-dim | Object detection distances (ModSelect) |

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | `sims_video`, `sims_video_multimodal`, `adl`, `YOLO_detections_only` |
| `--model_vid` | `s3d`, `i3d`, `YOLO_mlp`, `s3d_yolo_fusion` |
| `--G_path` | Domain generator checkpoint (MMGen) |
| `--fine_tune_late_fusion` | Enable late fusion training |
| `--split_policy` | `frac`, `cross-subject`, `cross-view-1`, `cross-view-2` |

## Project Structure

```
├── main.py                 # Entry point
├── lib/                    # Model architectures (S3D, I3D, YOLO MLP)
├── datasets/               # Data loaders
├── training/               # Training loops
├── testing/                # Evaluation scripts
├── utils/                  # Preprocessing and fusion utilities
└── L2A-OT/                 # Submodule: domain generator training
```
