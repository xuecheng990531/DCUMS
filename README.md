# DCUMS — Difficulty-Controlled Uncertainty Modulation for Segmentation

Active learning framework for medical image segmentation (polyp detection) that modulates training loss and sampling strategy by per-class boundary quality. Supports iterative active learning loops with multiple uncertainty sampling strategies and ablation experiments.

## Overview

The core idea of DCUMS is to **weight the loss spatially** during training using a difficulty map derived from class-wise boundary quality proxies (BQP). Classes with lower boundary quality (harder to segment) receive higher weights, forcing the model to focus on difficult regions near object boundaries.

### Key formula

```
class_weights = 1 + α · log(b · (1 − quality) + 1) · ub
where b = e^(1/α) − 1
```

- `α` (alpha): controls the shape of difficulty modulation
- `ub`: upper bound on the difficulty coefficient
- `quality`: EMA-smoothed boundary IoU per class

## Project Structure

```
DCUMS/
├── models/                  # Network architectures
│   ├── trans_new.py         # TransUNet with BQP module
│   ├── transreunet_class.py # TransResUNet (AttentionUNet variant)
│   ├── unet.py              # Standard UNet2D
│   ├── VIT.py               # Vision Transformer backbone
│   └── decoder.py           # Decoder blocks
├── utils/                   # Shared utilities
│   ├── dataset.py           # CustomDataset with albumentations
│   ├── loss.py              # Combined CE + Dice loss
│   ├── metrics.py           # IoU, Dice, Precision, Recall, Hausdorff
│   └── utils.py             # Helper functions
├── experiment_logs/         # Training logs per experiment
├── ablations/               # Saved model checkpoints per ablation
├── swanlog/                 # SwanLab experiment tracking
├── data/                    # Dataset paths (train/val/unlabelled/test)
├── pth/                     # Pretrained model weights
├── inference_results/       # Inference outputs
├── compare_methods/         # Comparison baselines (BALD, CoreSet, CSAL, PAAL, etc.)
│
├── loop_class.py            # Main active learning loop (DCUMS + Entropy)
├── al_unet_entropy_boundary.py  # Active learning with UNet + BQP entropy
├── model_train_val_class.py # Train/validate with DCUMS difficulty modulation
├── momentum_ablation.py     # EMA momentum (m) ablation study
├── alpha_ablation_noal.py   # α parameter ablation (no active learning)
├── ub_ablation.py           # ub parameter ablation
│
├── entropy_sampling.py      # Entropy-based sampling
├── mc_dropout_sampling.py   # MC Dropout sampling
├── random_sampling.py       # Random sampling baseline
├── compute_confidence_interval.py  # Statistical analysis
├── coverage_visualization.py       # Coverage plots
├── generate_entropy_maps.py        # Entropy heatmap generation
├── inference_script.py      # Batch inference
├── delete_low_iou.py        # Data cleaning utility
├── split_image.py           # Sliding window tile splitting
├── stitch_tiles.py          # Recombine tiled predictions
│
├── plot_alpha_ablation.py   # α ablation plotting
├── plot_alpha_smoothed_nature.py   # Smoothed loss curves (Nature style)
├── plot_alpha_metrics_nature.py    # Validation metrics bar chart
├── export_alpha_smoothed.py        # Export smoothed losses to Excel
└── export_alpha_val_metrics.py     # Export val metrics to Excel
```

## Supported Sampling Strategies

| Strategy | Description |
|----------|-------------|
| **DCUMS** | Difficulty-Controlled Uncertainty Modulation (entropy weighted by BQP) |
| **Entropy** | Predictive entropy sampling |
| **MC Dropout** | Monte Carlo dropout uncertainty |
| **Random** | Uniform random sampling (baseline) |
| **BALD** | Bayesian Active Learning by Disagreement |
| **CoreSet** | Core-set selection |
| **CSAL** / **PAAL** / **VDIsNet** | Comparison baselines |
| **Weak Annotation** | Weakly-supervised baseline |

## Datasets

- **Kvasir-SEG** — gastrointestinal polyp segmentation (located at `/icislab/volume1/lxc/polyp_data/kvasir/`)
- **FIVES** — fundus image vessel segmentation (`fives_data/`)
- **CVC-ClinicDB** — colonoscopy polyp segmentation (`data/cvc/`)

## Active Learning Loop

Each active learning iteration follows this pipeline:

1. **Uncertainty scoring** — compute per-sample uncertainty on the unlabeled pool via entropy + BQP weighting
2. **Sampling** — draw `K` samples using the uniform-mix strategy with `τ`-smoothing
3. **Annotation simulation** — move sampled indices from unlabeled to labeled pool
4. **Training** — train with DCUMS loss modulation using warmup rounds
5. **Evaluation** — validate on held-out set and test on the full test set

## Usage

### Run the main active learning loop

```bash
python loop_class.py
```

### Run α (alpha) ablation (no active learning)

```bash
python alpha_ablation_noal.py
```

### Run ub ablation

```bash
python ub_ablation.py
```

### Run momentum ablation

```bash
python momentum_ablation.py
```

### Run standard training with DCUMS

```bash
python model_train_val_class.py
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `α` (alpha) | 0.3 / 0.5 | Difficulty modulation sharpness |
| `ub` | 0.2 | Upper bound on class weight |
| `base_momentum` | 0.999 | EMA momentum for BQP |
| `warmup_rounds` | 2 | Rounds before DCUMS activates |
| `lr` | 1e-4 | Learning rate (AdamW) |
| `batch_size` | 12 | Batch size |
| `max_iters` | 2000 | Training iterations per AL round |
| `val_interval` | 40 | Validation frequency (iterations) |
| `AL_BUDGET_RATIO` | 0.1–0.15 | Fraction of unlabeled pool to sample |

## Model Architectures

- **TransUNet** — Transformer + CNN U-Net hybrid with boundary quality module
- **AttentionUNet** — U-Net with attention gates and class quality proxy
- **UNet2D** — Standard 2D U-Net

All models maintain a `class_quality` buffer (EMA-updated per-class boundary IoU) used for difficulty modulation.

## Experiment Tracking

All runs are tracked via [SwanLab](https://swanlab.ai/) (workspace: `xuecheng`). Projects:
- `alpha_ablation_noal` — α parameter grid search
- `momentum_ablation` — EMA momentum tuning
- `ub_ablation` — ub parameter sweep
- `active_loop` — active learning experiments

## Dependencies

- Python 3.8+
- PyTorch ≥ 1.10
- CUDA (recommended, AMP supported)
- einops, albumentations, scikit-learn, scipy
- tqdm, numpy, matplotlib, pandas
- swanlab (experiment tracking)
- openpyxl (Excel export)

## License

Research code — for academic use.
