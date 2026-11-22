# ST-JEMA: Spatio-Temporal Joint Embedding Masked Autoencoder

Self-supervised pre-training framework for dynamic brain network analysis using fMRI data.

ST-JEMA learns brain representations by masking and reconstructing both **node features** and **edge connectivity** across spatial and temporal dimensions, using a JEPA-style architecture with an EMA target encoder.

This repository also provides **ST-MAE** (Choi et al., *Temporal Graph Learning Workshop @ NeurIPS 2023*), our previous generative framework that reconstructs the input node embeddings and edges. It shares the same encoder and is selected with `ssl_type: stmae` (see `configs/pretrain_stmae.yaml`).

## Installation

```bash
pip install -r requirements.txt
```

For GAT encoder support, additionally install `torch_scatter`:
```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
```

## Data Format

Place data under a root directory (e.g., `./data/`) with the following structure:

```
data/
└── {dataset}/
    ├── {dataset}_roi-{roi}.pkl    # Timeseries dict: {subject_id: ndarray[T, N]}
    └── {dataset}.csv              # Behavioral data with 'subject_id' column + target columns
```

- **Timeseries pickle**: Dictionary mapping subject IDs (str) to NumPy arrays of shape `[T, N]`, where `T` is the number of timepoints and `N` is the number of ROIs (e.g., 400 for Schaefer atlas).
- **Behavioral CSV**: Must contain a `subject_id` column and one or more target columns (e.g., `sex`, `interview_age`, `DX`).

ROI time series are obtained by averaging the preprocessed BOLD signal within each ROI of the atlas (Schaefer atlas with 400 ROIs in the paper).

## Quick Start

### 1. Pre-training

```bash
python pretrain.py \
    --config configs/pretrain_stjema.yaml \
    --data_dir ./data \
    --output_dir ./output/pretrain \
    --fold 0 \
    --gpu 0
```

To pre-train ST-MAE instead, use `--config configs/pretrain_stmae.yaml`; fine-tuning is identical for both methods.

Pre-training saves the last-epoch model (`model.pth`) and the model with the lowest validation SSL loss (`best_model.pth`). Fine-tuning loads `best_model.pth` by default (`use_bestval_model: true`).

### 2. Fine-tuning (Classification)

```bash
python finetune.py \
    --config configs/finetune_classification.yaml \
    --data_dir ./data \
    --output_dir ./output/finetune_sex \
    --pretrain_dir ./output/pretrain \
    --fold 0 \
    --gpu 0
```

### 3. Fine-tuning (Regression)

```bash
python finetune.py \
    --config configs/finetune_regression.yaml \
    --data_dir ./data \
    --output_dir ./output/finetune_age \
    --pretrain_dir ./output/pretrain \
    --fold 0 \
    --gpu 0
```

### 4. 5-fold Cross-validation and Linear Probing

Results in the paper are averaged over 5 folds. Run fine-tuning for each fold; all folds load the pre-trained model of fold 0 (`fix_pretrain_k: 0`):

```bash
for k in 0 1 2 3 4; do
    python finetune.py \
        --config configs/finetune_classification.yaml \
        --data_dir ./data \
        --output_dir ./output/finetune_sex \
        --pretrain_dir ./output/pretrain \
        --fold $k \
        --gpu 0
done
```

For linear probing, add `--freeze_encoder` to keep the pre-trained encoder fixed.

## Project Structure

```
ST-JEMA/
├── configs/
│   ├── pretrain_stjema.yaml           # Pre-training config (ST-JEMA)
│   ├── pretrain_stmae.yaml            # Pre-training config (ST-MAE)
│   ├── finetune_classification.yaml   # Classification fine-tuning config
│   └── finetune_regression.yaml       # Regression fine-tuning config
│
├── models/
│   ├── backbone.py      # STAGIN backbone (GIN/GAT, readout, temporal modules)
│   ├── stjema.py        # ST-JEMA model (masking, EMA target, reconstruction)
│   ├── stmae.py         # ST-MAE model (random masking, input reconstruction)
│   ├── losses.py        # SSL loss functions (SCE, cosine, MSE, BCE)
│   └── components.py    # Shared components (EdgeConverter, MLP-Mixer, Percentile)
│
├── data/
│   ├── dataset.py       # Dataset loading with k-fold cross-validation
│   └── bold.py          # Dynamic functional connectivity computation
│
├── utils/
│   ├── config.py        # YAML config loader
│   ├── seed.py          # Random seed utilities
│   └── logger.py        # Metrics logging (classification + regression)
│
├── pretrain.py          # Pre-training entry point
├── finetune.py          # Fine-tuning entry point
├── requirements.txt
└── README.md
```

## Key Configuration Parameters

### Pre-training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ema_decay` / `ema_decay_final` | EMA momentum range (linear schedule) | 0.996 / 1.0 |
| `num_pred_masks` | Number of prediction mask blocks | 4 |
| `pred_mask_scale` | Block size range (fraction of nodes) | [0.10, 0.15] |
| `spatial_rate` | Balance between spatial and temporal loss | 0.5 |
| `ssl_lambda_node` / `ssl_lambda_edge` | Weights of the node and edge losses | 1.0 / 1e-4 |
| `node_decoder_type` | Decoder architecture | `mlp-mixer` |

### Fine-tuning

| Parameter | Description | Default |
|-----------|-------------|---------|
| `time_aggregation` | Temporal aggregation of window-level representations | `mean` |
| `freeze_pretrain` | Freeze pretrained encoder weights | `false` |
| `num_finetune_subsample` | Number of labeled samples (null = all) | `null` |
| `use_bestval_model` | Load the pre-trained model with the lowest validation SSL loss | `true` |
| `eval_full_length` | Evaluate on the full-length time series (training uses random crops of `dynamic_length`) | `true` |

## Architecture Overview

```
                        Pre-training (SSL)
                        ==================
BOLD Signal → Sliding Window FC → Dynamic Adjacency [B, T, N, N]
                                         ↓
                              ┌─── Block Masking ───┐
                              ↓                     ↓
                       Online Encoder          EMA Target Encoder
                        (GNN × L)               (GNN × L, no grad)
                              ↓                     ↓
                        Masked Features        Target Features
                              ↓                     ↓
                    ┌─────────┴─────────┐           │
                    ↓                   ↓           │
              Node Decoder        Edge Decoder      │
              (MLP-Mixer)         (MLP-Mixer)       │
                    ↓                   ↓           │
              Node Recon Loss    Edge Recon Loss ←──┘
              (Spatial+Temporal) (Spatial+Temporal)

                        Fine-tuning
                        ===========
                 Dynamic Adjacency [B, T, N, N]
                              ↓
                    Pretrained Encoder (GNN × L)
                              ↓
                      Readout (SERO/GARO)
                              ↓
                  Mean Pooling (Temporal)
                              ↓
                    Classification / Regression Head
```
