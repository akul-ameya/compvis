# Adaptive Synthetic Budget Allocation for Low-Data Image Classification

In low-data image classification, diffusion-generated synthetic data has **class-dependent utility**. This project predicts that utility from class-level properties and allocates synthetic budget more effectively than uniform augmentation under a fixed generation budget.

**Primary dataset:** Tiny ImageNet (200 classes, 64×64, trained at 224×224)  
**Generalisation dataset:** CIFAR-100 (reduced experimental track, ResNet-18 only)  
**Backbones:** ResNet-18 and **MobileNetV3-Small** (ImageNet pretrained)  
**GPU:** NVIDIA RTX 4060 class hardware (local runs)

## Research Questions

1. Does synthetic augmentation recover low-data performance?
2. Is the benefit uniform across classes?
3. Can we predict which classes will benefit or be harmed?
4. Can we allocate synthetic budget better than a uniform policy?

## Experimental Design

### Pipelines

| Pipeline | Training Data | Purpose |
|----------|--------------|---------|
| **Baseline** | 5% real (25 images/class) | Low-data floor |
| **Uniform DiffusionBoost** | 5% real + uniform synthetic | Standard augmentation |
| **Adaptive DiffusionBoost** | 5% real + policy-allocated synthetic (same total budget) | Targeted augmentation |
| **Ceiling** | 100% real | Full-data upper bound |

### Synthetic Budget Scaling (Tiny ImageNet)

Ratios **5×, 10×, 15×** per class (125 / 250 / 375 synthetic images per class). Cached under `data/synthetic/tiny_imagenet/` (or legacy `data/synthetic_sd/` auto-linked).

### Allocation Policies

| Policy | Strategy |
|--------|----------|
| **Uniform** | Equal synthetic count per class (reference total budget) |
| **Hard-class** | Budget ∝ (1 − baseline accuracy) |
| **Uncertainty-based** | Budget ∝ mean prediction entropy |
| **Predicted-utility** | Ridge regression on class diagnostics; budget ∝ max(0, predicted gain) |

Predicted-utility targets are **per-class accuracy under Uniform 15× minus Baseline**, fitted after Uniform 15× completes.

### Evaluation

`evaluate_stage2` (per run): Top-1 / macro / worst-k, ECE, **temperature scaling** (optimal \(T\), ECE after scaling), corruption suite, per-class accuracy, **linear probe** on frozen features (sklearn), **feature-covariance eigen-spectrum** + effective rank (and `eval_eigenvalues.png`), **linear CKA vs same-arch baseline** (`eval_cka_heatmap.png`). See `src/evaluation/stage2_eval.py` and `src/evaluation/eval_extras.py`.

### Synthetic-aware loss & FID

- **Weighted CE:** `training.synthetic_aware_loss` in YAML; distance-to-centroid weights using a **frozen baseline** of the same architecture (`src/training/synthetic_loss.py`). The notebook passes `baseline_ckpt_same_arch` for uniform / adaptive / ceiling.
- **Global FID:** `src/evaluation/fid_stage2.py` + **clean-fid** (`results/{dataset}/fid_cache/`). Notebook flag `RUN_FID`.

## Repository Layout

```
Group/
├── configs/
│   ├── tiny_imagenet.yaml       # Stage 2 Tiny ImageNet experiment config
│   ├── cifar100.yaml            # CIFAR-100 reduced-track config
│   └── stage1_config.yaml
├── notebooks/
│   ├── stage1_tinyimagenet_poc.ipynb
│   └── stage2_experiments.ipynb # Single Stage 2 entry: Run All
├── src/
│   ├── config.py                # YAML → dataclass loader
│   ├── migration/               # Legacy synthetic + checkpoint links
│   ├── synthesis/generate.py    # SD v1.5 generation (Tiny + CIFAR-100)
│   ├── data/
│   │   ├── tiny_imagenet.py, cifar100.py, registry.py
│   │   ├── synthetic_dataset.py, transforms.py, generate_synthetic.py
│   ├── models/backbone.py, resnet_baseline.py
│   ├── allocation/policies.py
│   ├── training/train_eval.py, stage2_train.py
│   ├── evaluation/stage2_eval.py, eval_extras.py, fid_stage2.py
│   ├── metrics/, stage2/orchestrator.py
├── results/                     # Created at run time: dataset/pipeline/arch/timestamp/
├── figures/stage2/
├── stage1_report_draft.md, stage2_report_draft.md
├── requirements.txt, notes.txt
└── README.md
```

## Quick Start

```bash
python -m venv venv && venv\Scripts\activate
# PyTorch first (use your CUDA build from pytorch.org; avoids overwriting with default PyPI torch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-others.txt

# Stage 1
jupyter notebook notebooks/stage1_tinyimagenet_poc.ipynb

# Synthetic pool (Tiny ImageNet) — resumable
python -m src.data.generate_synthetic --images-per-class 375

# Stage 2 — one notebook, Run All (see flags inside)
jupyter notebook notebooks/stage2_experiments.ipynb
```

`requirements.txt` still pins torch/torchvision/torchaudio for reproducibility; installing it wholesale can replace a CUDA build from PyTorch’s index—prefer **`requirements-others.txt`** after installing PyTorch as above.

Stage 2 auto-detects `data/synthetic_sd/` and links it to `data/synthetic/tiny_imagenet/` when needed. CIFAR-100 synthetic images are generated into `data/synthetic/cifar100/` when you enable that step in the notebook.

## References

- Ho et al. (2020); Dhariwal & Nichol (2021); Rombach et al. (2022) — diffusion  
- Shorten & Khoshgoftaar (2019) — augmentation survey  
- Guo et al. (2017) — calibration  
- Kornblith et al. (2019) — CKA  
- He et al. (2016); Howard et al. (2019) — ResNet / MobileNetV3  
