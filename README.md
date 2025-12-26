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
git clone <repo-url>
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

See the detailed [L2A-OT Domain Generation Guide](#l2a-ot-domain-generation) below for comprehensive instructions.

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
└── L2A-OT/                 # Domain generator training (L2A-OT)
    ├── main_SIMS_S3D.py    # Main training script
    ├── model.py            # Generator architecture
    ├── resnet.py           # Domain classifier (ResNet18)
    ├── lib/                # S3D backbone for task classifier
    ├── datasets/           # Multi-modality video datasets
    └── utils/              # Augmentation, losses, helpers
```

---

## L2A-OT Domain Generation

The `L2A-OT/` directory contains an adapted implementation of "Learning to Generate Novel Domains for Domain Generalization" (ECCV 2020) for video action recognition with multiple modalities.

### Method Overview

The domain generation approach trains three networks jointly:

1. **Generator (G)**: Conditional image-to-image translator that transforms source domain frames into novel synthetic domains
2. **Domain Classifier (D)**: ResNet18-based classifier that distinguishes between source modalities (trained first to provide gradients for G)
3. **Task Classifier (DGC)**: S3D-based action classifier trained on both source and generated novel domains

The key insight is that by generating diverse novel domains and training the task classifier on them, the model learns domain-invariant features that generalize better to unseen real-world data.

### Architecture Details

**Generator (G)**: StarGAN-style conditional generator
- Input: 3-channel image + domain label (one-hot, 2K dimensions where K = number of modalities)
- Architecture: 7x7 conv → 2 downsampling blocks → 6 residual blocks → 2 upsampling blocks → 7x7 conv
- Output: 3-channel transformed image
- Domain conditioning via channel-wise concatenation

**Domain Classifier (D)**: ResNet18
- Pre-trained on ImageNet, fine-tuned for K-way domain classification
- Provides Wasserstein distance gradients for generator training

**Task Classifier (DGC)**: S3D (Separable 3D CNN)
- Takes concatenated multi-modality video clips as input
- Trained on both original source domains and generated novel domains

### Training Pipeline

#### Step 1: Pre-train Task Classifier on Source Domains

First, train an S3D classifier on the source modalities using early fusion:

```bash
# In main repository
python main.py --gpu 0 1 --dataset sims_video_multimodal \
    --modalities heatmaps limbs optical_flow rgb \
    --dataset_roots /path/to/h /path/to/l /path/to/of /path/to/rgb \
    --n_channels_each_modality 1 1 3 3 --n_channels 8 --epochs 200
```

Save the checkpoint path for use as `--pretrained_model_C` (frozen classifier Y^).

#### Step 2: Train Domain Classifier (D)

```bash
cd L2A-OT
python main_SIMS_S3D.py \
    --gpu 0 1 \
    --modalities heatmaps limbs optical_flow rgb \
    --modality_indices 0 1 2 3 \
    --dataset_roots /path/to/h /path/to/l /path/to/of /path/to/rgb \
    --dataset_roots_test /path/to/adl_h /path/to/adl_l /path/to/adl_of /path/to/adl_rgb \
    --num_iterations_D 1000 \
    --num_iterations_G 0 \
    --exp_tag domain_classifier_training
```

This trains D to classify which modality (domain) each frame belongs to.

#### Step 3: Train Generator (G) and Task Classifier (DGC)

```bash
cd L2A-OT
python main_SIMS_S3D.py \
    --gpu 0 1 \
    --modalities heatmaps limbs optical_flow rgb \
    --modality_indices 0 1 2 3 \
    --dataset_roots /path/to/h /path/to/l /path/to/of /path/to/rgb \
    --dataset_roots_test /path/to/adl_h /path/to/adl_l /path/to/adl_of /path/to/adl_rgb \
    --pretrained_model_C /path/to/pretrained_s3d.tar \
    --num_iterations_D 10 \
    --num_iterations_G 30000 \
    --test_every 100 \
    --save_img_every 1000 \
    --exp_tag heatmaps_limbs_of_rgb_GAN \
    --batch_size 6
```

### Loss Functions

The generator is trained with three losses:

1. **Cycle Reconstruction Loss** (L1): Ensures G(G(x, d_novel), d_source) ≈ x
2. **Domain Distribution Loss** (Sinkhorn/Wasserstein): Maximizes distance between source and novel domain distributions in D's feature space
3. **Classification Loss** (Cross-Entropy): Generated novel domains should be correctly classified by frozen classifier C

```
L_total = λ_cycle * L_cycle - λ_domain * L_novel - λ_domain * L_diversity + λ_CE * L_CE
```

Where:
- `λ_cycle = 10` (reconstruction weight)
- `λ_domain = 1` (domain distribution weight)
- `λ_CE = 1` (classification weight)

### Key Arguments for L2A-OT

| Argument | Description | Default |
|----------|-------------|---------|
| `--modalities` | List of modality names | `heatmaps limbs optical_flow rgb` |
| `--modality_indices` | Indices of modalities to use | `0 1 2 3` |
| `--dataset_roots` | Paths to source domain datasets | - |
| `--dataset_roots_test` | Paths to target domain datasets (for evaluation) | - |
| `--pretrained_model_C` | Path to pre-trained frozen classifier | - |
| `--pretrained_model_G` | Path to pre-trained generator (for resuming) | - |
| `--pretrained_model_DGC` | Path to pre-trained task classifier | - |
| `--num_iterations_D` | Domain classifier training iterations | 10 |
| `--num_iterations_G` | Generator + task classifier training iterations | 10 |
| `--test_every` | Validation frequency (iterations) | 100 |
| `--save_img_every` | Save generated images frequency | 1000 |
| `--exp_tag` | Experiment name for checkpoints/logs | - |
| `--test_classifier_only` | Only evaluate, no training | False |
| `--train_classifier_only` | Train DGC with frozen G | False |
| `--freeze_generator` | Freeze generator weights | False |

### Modality Combinations

You can train with any subset of modalities by adjusting `--modality_indices`:

```bash
# Heatmaps + Limbs only
--modalities heatmaps limbs --modality_indices 0 1

# Optical Flow + RGB only
--modalities optical_flow rgb --modality_indices 2 3

# All four modalities
--modalities heatmaps limbs optical_flow rgb --modality_indices 0 1 2 3
```

Channel mapping:
- Index 0: Heatmaps (1 channel, padded to 3)
- Index 1: Limbs (1 channel, padded to 3)
- Index 2: Optical Flow (3 channels)
- Index 3: RGB (3 channels)

### Evaluation

After training, evaluate the task classifier on the target domain:

```bash
cd L2A-OT
python main_SIMS_S3D.py \
    --gpu 0 1 \
    --modalities heatmaps limbs optical_flow rgb \
    --modality_indices 0 1 2 3 \
    --dataset_roots /path/to/h /path/to/l /path/to/of /path/to/rgb \
    --dataset_roots_test /path/to/adl_h /path/to/adl_l /path/to/adl_of /path/to/adl_rgb \
    --pretrained_model_DGC checkpoints/heatmaps_limbs_of_rgb_GAN/best_val_DGC.tar \
    --pretrained_model_G checkpoints/heatmaps_limbs_of_rgb_GAN/G_iteration_30000.pth \
    --test_classifier_only \
    --exp_tag eval_heatmaps_limbs_of_rgb_GAN
```

This outputs:
- Normal accuracy on source domains
- Normal accuracy on novel (generated) domains
- Balanced accuracy on source domains
- Balanced accuracy on novel domains
- Per-class accuracy breakdowns

### Output Files

Training produces:
- `checkpoints/<exp_tag>/G_iteration_<N>.pth`: Generator weights
- `checkpoints/<exp_tag>/DGC_iteration_<N>.pth`: Task classifier weights
- `checkpoints/<exp_tag>/best_val_DGC.tar`: Best validation task classifier
- `checkpoints/<exp_tag>/best_loss_DGC.tar`: Best loss task classifier
- `results/<exp_tag>/<N>_<modality>_{real,fake,rec}.jpg`: Sample images
- `runs/<exp_tag>/`: TensorBoard logs

### Using Trained Generator for Inference

The trained generator can be loaded in the main repository for inference:

```bash
# In main repository
python main.py --test_only \
    --G_path L2A-OT/checkpoints/<exp_tag>/G_iteration_30000.pth \
    --pretrained_model_s3d L2A-OT/checkpoints/<exp_tag>/best_val_DGC.tar \
    --eval_datasets /path/to/target_dataset
```

See `testing/test_video_stream.py` and `testing/generator.py` for the inference implementation.

### References

The L2A-OT implementation is based on:
- [Learning to Generate Novel Domains for Domain Generalization (ECCV 2020)](https://arxiv.org/abs/2007.03304)
- [StarGAN: Unified Generative Adversarial Networks](https://arxiv.org/abs/1711.09020)
