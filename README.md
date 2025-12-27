# Synthetic-to-Real Domain Adaptation for Action Recognition

Repository for multimodal action recognition using synthetic data from Sims4Action, with domain adaptation for real-world datasets (Toyota Smarthome, ETRI-Activity3D).

## Papers

This repository contains implementations for two papers:

### [MMGen (IROS 2022)](https://arxiv.org/abs/2208.01910)
**Multimodal Domain Generation for Synthetic-to-Real Activity Recognition**
- Trains only on synthetic Sims4Action data
- Generates novel modalities via adversarial domain generation
- Evaluates on real Toyota Smarthome and ETRI-Activity3D datasets
- Domain generator code in `L2A-OT/`

### [ModSelect (ECCV 2022)](https://arxiv.org/abs/2208.09414)
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

## Modalities

| Modality | Model | Input | Description |
|----------|-------|-------|-------------|
| Heatmaps (H) | S3D | 1 ch | Gaussian maps at AlphaPose joint locations |
| Limbs (L) | S3D | 1 ch | Lines connecting skeleton joints |
| Optical Flow (OF) | S3D | 3 ch | Farneback algorithm (HSV encoded) |
| RGB | S3D | 3 ch | Original video frames |
| YOLO | MLP | 80-dim | Reciprocal distance vector to detected objects (ModSelect only) |

---

## MMGen Experiments

The MMGen paper trains action classifiers on synthetic data with generated novel domains to improve generalization to real-world data.

### Source and Novel Modalities

The domain generator transforms source modalities (top row) into novel synthetic domains (bottom row), creating a more diverse training set:

![Source and Novel Modalities](assets/mmgen_modalities.png)

*H: Heatmaps, L: Limbs, OF: Optical Flow. The model trains on all 8 modalities (4 source + 4 novel) but evaluates on real data.*

### Method Overview

![MMGen Architecture](assets/mmgen_architecture.png)

The approach uses three networks:
1. **Frozen Classifier (AC_f)**: Pre-trained S3D action classifier, frozen during domain generation training
2. **Task Classifier (AC)**: S3D trained on both source and novel modalities
3. **Domain Generator (DG)**: Transforms source modalities into novel synthetic domains

Additionally, a **Domain Classifier (DC)** (ResNet18) is trained to distinguish modalities and provides the Sinkhorn distance loss for training DG.

### Training Dynamics

1. **Phase 1 - Train DC:** The domain classifier learns to distinguish source modalities (4-way classification: H, L, OF, RGB). This creates a feature space that separates domains.

2. **Phase 2 - Train DG + AC:** DC is **frozen** (eval mode). DG and AC train jointly:
   - DG generates novel domains from source modalities
   - The Sinkhorn distance (computed using DC's frozen features) measures how different novel domains are from source domains
   - DG maximizes this distance (novelty loss) while AC learns to classify actions on both source and novel domains
   - AC_f (frozen) ensures generated domains preserve action semantics

**Why not alternating?** DC is not an adversary—it's a **fixed domain distance metric**. DG doesn't try to "fool" DC; instead, DG uses DC's frozen features to measure domain novelty. A stable feature space is crucial for meaningful Sinkhorn distances.

### Training Pipeline

The paper evaluates all 15 modality combinations. Below we show the full 4-modality example.

**Prerequisites for L2A-OT:** Download ResNet18 pretrained weights for the domain classifier:
```bash
cd L2A-OT/checkpoints
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
```

#### Step 1: Pre-train Action Classifier on Source Modalities

Train an S3D classifier using early fusion (channel concatenation) on Sims4Action:

```bash
python main.py --gpu 0 1 \
    --dataset sims_video_multimodal \
    --modalities heatmaps limbs optical_flow rgb \
    --dataset_roots /path/to/sims/heatmaps /path/to/sims/limbs /path/to/sims/optical_flow /path/to/sims/rgb \
    --n_modalities 4 \
    --n_channels_each_modality 1 1 3 3 \
    --n_channels 8 \
    --epochs 200 \
    --img_dim 112 \
    --seq_len 16
```

This produces a checkpoint at `experiments/<exp>/model/model_best_val_acc.pth.tar`, used as the frozen action classifier AC_f.

#### Step 2: Train Domain Classifier (DC), Generator (DG), and Task Classifier (AC)

Train all components in a single run. The script first trains DC for `num_iterations_D` iterations, then jointly trains DG and AC for `num_iterations_G` iterations:

```bash
cd L2A-OT
python main_SIMS_S3D.py \
    --gpu 0 1 \
    --modalities heatmaps limbs optical_flow rgb \
    --modality_indices 0 1 2 3 \
    --dataset_roots /path/to/sims/heatmaps /path/to/sims/limbs /path/to/sims/optical_flow /path/to/sims/rgb \
    --dataset_roots_test /path/to/adl/heatmaps /path/to/adl/limbs /path/to/adl/optical_flow /path/to/adl/rgb \
    --pretrained_model_C ../experiments/<exp>/model/model_best_val_acc.pth.tar \
    --num_iterations_D 1000 \
    --num_iterations_G 30000 \
    --test_every 500 \
    --save_img_every 1000 \
    --exp_tag GAN_h_l_of_rgb \
    --batch_size 6
# Parameter mapping:
#   --pretrained_model_C  → AC_f (frozen action classifier)
#   --num_iterations_D    → DC training iterations
#   --num_iterations_G    → DG + AC joint training iterations
```

The frozen action classifier AC_f (`--pretrained_model_C`) provides the classification loss to ensure generated domains preserve action semantics.

#### Step 3: Evaluate on Real Data

Evaluate the trained task classifier on Toyota Smarthome or ETRI:

```bash
cd L2A-OT
python main_SIMS_S3D.py \
    --gpu 0 1 \
    --modalities heatmaps limbs optical_flow rgb \
    --modality_indices 0 1 2 3 \
    --dataset_roots /path/to/sims/heatmaps /path/to/sims/limbs /path/to/sims/optical_flow /path/to/sims/rgb \
    --dataset_roots_test /path/to/adl/heatmaps /path/to/adl/limbs /path/to/adl/optical_flow /path/to/adl/rgb \
    --pretrained_model_DGC checkpoints/GAN_h_l_of_rgb/best_val_DGC.tar \
    --pretrained_model_G checkpoints/GAN_h_l_of_rgb/G_iteration_30000.pth \
    --test_classifier_only \
    --exp_tag eval_GAN_h_l_of_rgb
# Parameter mapping:
#   --pretrained_model_DGC → AC (action classifier / task model)
#   --pretrained_model_G   → DG (domain generator)
```

Outputs balanced and unbalanced accuracy on both source and novel domains.

### Modality Combinations

The paper tests all 15 combinations. Examples:

```bash
# Single modality: Limbs only
--modalities limbs --modality_indices 1 --n_channels 1

# Two modalities: Heatmaps + Limbs
--modalities heatmaps limbs --modality_indices 0 1 --n_channels 2

# Three modalities: H + L + OF
--modalities heatmaps limbs optical_flow --modality_indices 0 1 2 --n_channels 5
```

### Loss Functions

The domain generator DG is trained with:
- **Novelty Loss**: Maximizes Sinkhorn distance between source and novel modality distributions
- **Diversity Loss**: Maximizes Sinkhorn distance between different novel modalities
- **Classification Loss**: Novel modalities should be correctly classified by frozen AC_f
- **Cycle Loss**: Reconstruction consistency (DG(DG(x)) ≈ x)

```
L_DG = λ_c * L_class + λ_r * L_cycle - λ_d * (L_novelty + L_diversity)
```

Default: λ_c = λ_d = 1, λ_r = 10

### Implementation Details (from paper)

- Input size: 112 × 112
- Sequence length: 16 frames
- Video chunks: 90 frames
- Optimizer: Adam (lr=1e-4, β1=0.5, β2=0.999)
- Weight decay: 5e-5
- Pre-training: 200 epochs
- Joint training: ~50 epochs equivalent

### Output Files

Training produces:
- `checkpoints/<exp_tag>/G_iteration_<N>.pth`: Domain generator (DG) weights
- `checkpoints/<exp_tag>/best_val_DGC.tar`: Best action classifier (AC)
- `results/<exp_tag>/*.jpg`: Sample generated images
- `runs/<exp_tag>/`: TensorBoard logs

### Domain Embedding Visualization

The t-SNE visualization below shows how the domain generator learns to produce novel modalities that are distinct from the source modalities, effectively diversifying the training distribution:

![t-SNE Embedding Visualization](assets/mmgen_tsne.png)

*Each color represents a different modality. Source and novel modalities form separate clusters, indicating the generator has learned to produce diverse but semantically consistent domains.*

---

## ModSelect Experiments

The ModSelect paper proposes an **unsupervised modality selection** method that identifies beneficial modalities without requiring target domain labels. It trains unimodal classifiers, computes prediction correlations and embedding MMD between modalities, and uses these metrics to select modalities that improve late fusion performance.

### Method Overview

1. Train unimodal S3D classifiers on each modality (H, L, OF, RGB) and an MLP on YOLO detection vectors
2. Evaluate all 31 modality combinations with late fusion strategies
3. **ModSelect** (unsupervised): Compute prediction correlation ρ and MMD between classifier embeddings to select beneficial modalities via Winsorized Mean thresholds

**YOLO Representation:** For each frame, a 80-dimensional vector **v** is computed where **v[i]** is the reciprocal Euclidean distance between the person's bounding box center and the i-th detected object's center. The vector is normalized: **v ← v/||v||**. Objects closer to the person have larger weights.

### Step 1: Train Unimodal Classifiers

Train S3D classifiers for image-based modalities:

```bash
# Heatmaps
python main.py --gpu 0 1 --dataset sims_video --modality heatmaps \
    --n_channels 1 --epochs 200 --dataset_video_root /path/to/heatmaps

# Limbs
python main.py --gpu 0 1 --dataset sims_video --modality limbs \
    --n_channels 1 --epochs 200 --dataset_video_root /path/to/limbs

# Optical Flow
python main.py --gpu 0 1 --dataset sims_video --modality optical_flow \
    --n_channels 3 --epochs 200 --dataset_video_root /path/to/optical_flow

# RGB
python main.py --gpu 0 1 --dataset sims_video --modality rgb \
    --n_channels 3 --epochs 200 --dataset_video_root /path/to/rgb
```

Train MLP for YOLO detection vectors:

```bash
# YOLO (MLP on detection vectors)
# Requires both RGB videos (for video indexing) and precomputed YOLO detections
python main.py --gpu 0 1 \
    --dataset YOLO_detections_only \
    --model_vid YOLO_mlp \
    --epochs 200 \
    --dataset_video_root /path/to/rgb \
    --detections_root /path/to/yolo_detections
# Detection files expected at: /path/to/yolo_detections/<Action>/<video_id>/detections.csv
```

### Step 2: Test Classifiers and Generate Predictions

Before late fusion, test each trained classifier to generate CSV files with predictions and embeddings:

```bash
# Test each modality (repeat for all 5 modalities)
python main.py --gpu 0 1 --test_only \
    --dataset sims_video --modality heatmaps --n_channels 1 \
    --pretrained_model experiments/<exp_heatmaps>/model/model_best_val_acc.pth.tar \
    --eval_dataset_root /path/to/target/heatmaps

# For YOLO:
python main.py --gpu 0 1 --test_only \
    --dataset YOLO_detections_only --model_vid YOLO_mlp \
    --pretrained_model experiments/<exp_yolo>/model/model_best_val_acc.pth.tar \
    --dataset_video_root /path/to/target/rgb \
    --detections_root /path/to/target/yolo_detections
```

This produces (in `experiments/<exp>/logs/`):
- `results_test_*.csv` — Top-5 predictions per sample (for late fusion)
- `*_embeddings.npy` — Classifier embeddings (for MMD computation)
- `*_scores.npy` — Raw class scores (for correlation computation)

### Step 3: Late Fusion Evaluation

Combine predictions from multiple modalities using voting strategies. The fusion scripts expect CSV files with columns: `vid_id`, `label`, `pred1`, `pred2`, `pred3`, `pred4`, `pred5`.

**Borda Count Voting:**

```bash
python utils/late_fusion_borda_count.py \
    --csv_roots results_h.csv results_l.csv results_of.csv results_rgb.csv \
    --modalities heatmaps limbs optical_flow rgb
```

**With save path (for multimodal version):**

```bash
python utils/late_fusion_borda_count_multimodal.py \
    --csv_roots results_h.csv results_l.csv results_of.csv results_rgb.csv results_yolo.csv \
    --modalities heatmaps limbs optical_flow rgb yolo \
    --save_path results_fused.txt
```

### Late Fusion Strategies

The paper evaluates 6 strategies. This repository implements:

| Strategy | Script | Status |
|----------|--------|--------|
| Borda Count | `utils/late_fusion_borda_count.py` | ✓ Implemented |
| Sum | `utils/late_fusion_sum_square_multimodal.py` | ✓ Implemented |
| Squared Sum | `utils/late_fusion_sum_square_multimodal.py` | ✓ Implemented |
| Product | — | Not implemented |
| Maximum | — | Not implemented |
| Median | — | Not implemented |

### ModSelect: Unsupervised Modality Selection

The paper proposes selecting modalities based on:
1. **Prediction Correlation ρ(m,n):** High correlation between correct predictions is more likely than between wrong ones
2. **MMD between embeddings:** Lower domain discrepancy indicates better agreement

**Thresholds:** The Winsorized Mean (λ=0.2) is used to compute selection thresholds. Modalities are selected if they meet either criterion (high ρ OR low MMD).

### Analysis Scripts

The `utils/modselect_analysis/` directory contains scripts for computing MMD, visualizing embeddings, and analyzing the relationship between domain discrepancy and performance.

#### Step 1: Compute Mean Embeddings

After testing classifiers (Step 2 above), compute mean embedding vectors for each modality:

```bash
python utils/modselect_analysis/mean_embedding.py \
    --embeddings_path experiments/<exp>/logs/results_test_*_embeddings.npy \
    --save_path results/<dataset>/<modality>/mean_vec.npy
```

#### Step 2: Compute MMD Matrix

Compute MMD between all modality pairs using mean embeddings:

```bash
python utils/modselect_analysis/mmd_table.py --datasets Sims Toyota
# Outputs: MMD_tables/mmd_Sims_Toyota.npy, .svg, .pdf
```

The MMD is computed as the Euclidean distance between mean embedding vectors (linear kernel). YOLO is excluded due to different embedding dimensionality.

#### Step 3: Visualize Embeddings (Optional)

Generate t-SNE visualizations of classifier embeddings:

```bash
python utils/modselect_analysis/tsne.py \
    --embeddings_path results/Sims/heatmaps/*_embeddings.npy \
    --labels_path results/Sims/*_labels.npy \
    --save_path tsne_plots/sims_heatmaps.svg
```

#### Step 4: Analyze Discrepancy vs Performance

Plot the relationship between domain discrepancy metrics and late fusion performance:

```bash
# Correlation vs Performance comparison
python utils/modselect_analysis/line_plot_performance_discrepancy.py

# MMD/Energy Distance vs Performance comparison
python utils/modselect_analysis/line_plot_similarity_discrepancy.py
```

These scripts expect precomputed data in `correlations/`, `MMD_tables/`, and `performance/` directories.

#### Analysis Scripts Reference

| Script | Purpose |
|--------|---------|
| `mean_embedding.py` | Compute mean embedding vector from saved embeddings |
| `mmd_table.py` | Compute and visualize MMD matrix between modalities |
| `energy_dist.py` | Compute energy distance between embedding distributions |
| `tsne.py` | Generate t-SNE visualizations of embeddings |
| `line_plot_performance_discrepancy.py` | Plot correlation vs late fusion performance |
| `line_plot_similarity_discrepancy.py` | Plot MMD/energy distance vs performance |

### Implementation Details (from paper)

- Action classes: 10 (shared subset between Sims4Action, Toyota Smarthome, ETRI)
- Evaluation metric: Mean per-class accuracy (balanced accuracy)
- Late fusion operates on class probability scores
- YOLO not included in MMD experiments (different embedding size from S3D)
- Energy distance requires the `dcor` library

---

## Data Preprocessing

### Generate Skeleton Modalities from AlphaPose

```bash
python utils/generate_skeletons_heatmaps.py \
    --root_dir /path/to/alphapose_results \
    --result_dir /path/to/output
```

Produces both heatmaps and limbs modalities.

### Generate Optical Flow

```bash
python utils/generate_optical_flow.py \
    --root_dir /path/to/rgb_videos \
    --result_dir /path/to/output \
    --n_workers 3
```

---

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | `sims_video`, `sims_video_multimodal`, `adl`, `YOLO_detections_only`, `sims_video_with_YOLO_detections` |
| `--model_vid` | `s3d`, `i3d`, `YOLO_mlp`, `s3d_yolo_fusion` |
| `--modality` | Single modality: `heatmaps`, `limbs`, `optical_flow`, `rgb` |
| `--modalities` | Multiple modalities for early fusion |
| `--n_channels` | Total input channels (sum of all modalities) |
| `--G_path` | Domain generator checkpoint for inference (MMGen) |
| `--split_policy` | `frac`, `cross-subject`, `cross-view-1`, `cross-view-2` |
| `--test_only` | Run evaluation only (no training) |
| `--pretrained_model` | Path to pretrained model checkpoint |

### ModSelect / YOLO Arguments

| Argument | Description |
|----------|-------------|
| `--detections_root` | Root folder containing YOLO detection CSVs |
| `--yolo_arch` | MLP architecture: `SimpleNet`, `BaseNet`, `TanyaNet`, `PyramidNet`, `LongNet`, `LastNet` |
| `--pretrained_YOLO_mlp` | Pretrained YOLO MLP checkpoint |
| `--pretrained_s3d_yolo_fusion` | Pretrained S3D+YOLO fusion model |
| `--fine_tune_yolo_mlp` | Fine-tune YOLO MLP submodel |

### L2A-OT Specific Arguments

| Argument | Description | Maps to |
|----------|-------------|---------|
| `--pretrained_model_C` | Frozen action classifier for classification loss | AC_f |
| `--pretrained_model_DGC` | Action classifier checkpoint | AC |
| `--pretrained_model_G` | Domain generator checkpoint | DG |
| `--num_iterations_D` | Domain classifier training iterations | DC |
| `--num_iterations_G` | Domain generator + action classifier training iterations | DG + AC |
| `--test_classifier_only` | Evaluation mode (no training) | — |

---

## Monitoring

```bash
tensorboard --logdir experiments/<exp-folder>/logs
# or for L2A-OT:
tensorboard --logdir L2A-OT/runs/<exp_tag>
```

---

## Project Structure

```
├── main.py                 # Entry point for action classifier (AC) training
├── lib/                    # Model architectures (S3D, I3D, YOLO MLP)
├── datasets/               # Data loaders
├── training/               # Training loops
├── testing/                # Evaluation scripts
├── utils/                  # Preprocessing and fusion utilities
│   ├── late_fusion_*.py    # Late fusion strategies (Borda, Sum, etc.)
│   ├── generate_*.py       # Data preprocessing scripts
│   └── modselect_analysis/ # ModSelect analysis scripts
│       ├── mean_embedding.py   # Mean embedding computation
│       ├── mmd_table.py        # MMD matrix computation
│       ├── energy_dist.py      # Energy distance computation
│       ├── tsne.py             # t-SNE visualization
│       └── line_plot_*.py      # Discrepancy vs performance plots
└── L2A-OT/                 # Domain generation training (MMGen)
    ├── main_SIMS_S3D.py    # Main training script (DC, DG, AC)
    ├── model.py            # Domain generator (DG) architecture (StarGAN-style)
    ├── resnet.py           # Domain classifier (DC) - ResNet18
    ├── lib/                # S3D backbone for AC and AC_f
    └── utils/              # Sinkhorn loss, augmentation
```

---

## References

- MMGen is based on [L2A-OT: Learning to Generate Novel Domains for Domain Generalization (ECCV 2020)](https://arxiv.org/abs/2007.03304)
- S3D architecture from [Rethinking Spatiotemporal Feature Learning (ECCV 2018)](https://arxiv.org/abs/1712.04851)
