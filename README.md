# Adaptive Synthetic Budget Allocation for Low-Data Image Classification Under Compute Constraints

In low-data image classification, diffusion-generated synthetic data has **class-dependent utility**. This project predicts that utility from class-level properties and exploits it to allocate synthetic budget more effectively than uniform augmentation under fixed compute.

**Dataset:** Tiny ImageNet (200 classes, 64x64)
**Backbones:** ResNet-18, MobileNetV3 (both ImageNet pretrained)
**GPU:** NVIDIA RTX 4060 (8 GB VRAM), 64 GB system RAM

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
| **Adaptive DiffusionBoost** | 5% real + class-aware allocation (same total budget) | Targeted augmentation |
| **Ceiling** | 100% real (500 images/class) | Full-data upper bound |

### Synthetic Budget Scaling

Synthetic budgets tested: **0x, 5x, 10x, 15x** the real data per class (0, 125, 250, 375 synthetic images/class). The scaling study characterises the dose-response relationship between synthetic data quantity and downstream metrics.

### Allocation Policies

Under a fixed total synthetic budget, we compare:

| Policy | Strategy |
|--------|----------|
| **Uniform** | Equal budget to every class |
| **Hard-class** | More budget to classes with low baseline accuracy |
| **Uncertainty-based** | More budget to classes with high prediction entropy |
| **Fidelity-aware** | More budget to classes with high synthetic quality |
| **Predicted-utility** | Budget proportional to predicted per-class gain |

The predicted-utility policy is trained on class-level features (baseline accuracy, entropy, synthetic fidelity, feature compactness) to predict per-class gain from augmentation.

### Evaluation Axes

| Axis | Metrics |
|------|---------|
| **Accuracy** | Top-1, macro, worst-20-class |
| **Calibration** | ECE, reliability diagrams, temperature-scaled ECE |
| **Robustness** | Gaussian noise, blur, brightness shift |
| **Per-class** | Per-class gain/loss, harmful class detection |
| **Synthetic quality** | FID, per-class fidelity proxy |
| **Feature coverage** | Compactness, separation, centroid margin |
| **Representation** | Eigenvalue spectrum, linear probe, CKA |
| **Efficiency** | Gain per 1k generated images, gain per GPU-minute |

## Repository Layout

```
Group/
├── notebooks/
│   ├── stage1_tinyimagenet_poc.ipynb   # Stage 1: EDA + baseline PoC
│   └── stage2_full_experiments.ipynb   # Stage 2: full experiments
├── src/
│   ├── data/
│   │   ├── tiny_imagenet.py            # Download, subset, full-train, val datasets
│   │   ├── transforms.py              # Train/val/corruption transforms
│   │   ├── synthetic.py               # Synthetic + combined dataset classes
│   │   └── generate_synthetic.py      # Stable Diffusion generation (CLI + importable)
│   ├── models/
│   │   └── resnet_baseline.py         # ResNet-18 + MobileNetV3 with feature extraction
│   ├── training/
│   │   └── train_eval.py              # Training loop, early stopping, cosine LR
│   └── metrics/
│       ├── metrics.py                 # Accuracy, ECE, robustness, eigenvalue analysis
│       └── cka.py                     # Linear/RBF CKA
├── configs/
│   ├── stage1_config.yaml
│   └── stage2_config.yaml
├── stage1_report_draft.md
├── stage2_report_draft.md
├── requirements.txt
├── notes.txt
└── README.md
```

## Quick Start

```bash
# 1. Environment
python -m venv venv && venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn scikit-learn tqdm pyyaml scipy

# 2. Stage 1
jupyter notebook notebooks/stage1_tinyimagenet_poc.ipynb

# 3. Generate synthetic data (overnight)
pip install diffusers transformers accelerate
python -m src.data.generate_synthetic --images-per-class 375

# 4. Stage 2
jupyter notebook notebooks/stage2_full_experiments.ipynb
```

See `notes.txt` for detailed step-by-step instructions.

## Contributions

1. A controlled Tiny ImageNet benchmark for synthetic augmentation under fixed compute
2. A class-level utility framework identifying beneficial, neutral, and harmful augmentation regimes
3. Adaptive synthetic budget allocation policies that outperform uniform allocation at equal total budget
4. Multi-axis evaluation across accuracy, calibration, robustness, and representation geometry on two architectures

## References

- Ho et al. (2020) — Denoising Diffusion Probabilistic Models
- Dhariwal & Nichol (2021) — Diffusion Models Beat GANs on Image Synthesis
- Rombach et al. (2022) — High-Resolution Image Synthesis with Latent Diffusion Models
- Shorten & Khoshgoftaar (2019) — A Survey on Image Data Augmentation
- Guo et al. (2017) — On Calibration of Modern Neural Networks
- Kornblith et al. (2019) — Similarity of Neural Network Representations Revisited
- Heusel et al. (2017) — GANs Trained by a Two Time-Scale Update Rule Converge (FID)
- He et al. (2016) — Deep Residual Learning for Image Recognition
- Howard et al. (2019) — Searching for MobileNetV3
