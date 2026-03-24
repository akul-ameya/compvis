# Stage 2 Report Draft — Adaptive Synthetic Budget Allocation for Low-Data Image Classification

> IEEE Conference Template. Replace all **X/Y/Z** placeholders with actual results. Figures saved to `figures/stage2/`. Metrics and checkpoints are logged under `results/{dataset}/{pipeline}/{architecture}/{timestamp}/`.

## I. Introduction

Standard synthetic data augmentation treats all classes equally, allocating the same number of generated images regardless of class difficulty, synthetic fidelity, or downstream utility. We argue this uniform policy is suboptimal: under a fixed synthetic budget, synthetic data has class-dependent value, and that value is predictable from measurable class properties.

We study **Tiny ImageNet** (200 classes, 64×64 native resolution, 224×224 training) as the primary benchmark. Our four pipelines are: (1) **Baseline** — ResNet-18 or **MobileNetV3-Small** trained on 5% real data (25 images/class); (2) **Uniform DiffusionBoost** — the same backbone trained on 5% real data plus uniformly allocated synthetic images from Stable Diffusion v1.5; (3) **Adaptive DiffusionBoost** — the same total synthetic budget allocated per class according to an explicit policy (hard-class, uncertainty-based, or predicted-utility); and (4) **Ceiling** — the same backbone trained on 100% real data. Experiments are replicated on both backbones on Tiny ImageNet to test cross-architecture behaviour.

We additionally report a **CIFAR-100 generalisation track** (100 classes, 32×32 native, 224×224 training): ResNet-18 only — Baseline, Uniform 15×, per-class **utility** targets (Uniform − Baseline), allocations (**hard_class** + **predicted_utility**), **Adaptive 15×** using `scope.cifar_adaptive_policy` in `configs/cifar100.yaml` (default **predicted_utility**), and Ceiling — without the full scaling grid or second backbone.

Our contributions are: (a) a synthetic scaling study on Tiny ImageNet (5×, 10×, 15×) characterising the dose–response relationship; (b) class-level diagnostics and a ridge-based **predicted-utility** allocator whose targets are observed gains under Uniform 15×; (c) comparison of adaptive policies against uniform allocation at equal total budget; and (d) multi-axis evaluation (accuracy, calibration, corruption robustness, per-class analysis; extended geometry analyses as in Section II).

## II. Methodology

### Synthetic Data Generation and Quality Assessment

We use Stable Diffusion v1.5 with DPM-Solver++ (25 steps, guidance scale 7.5), mixed precision, and 512×512 generation followed by downscaling to the dataset’s native resolution (64×64 for Tiny ImageNet, 32×32 for CIFAR-100). Up to **375** images per class are cached once under `data/synthetic/{dataset}/` and reused for all experiments. Prompts use four fixed templates with human-readable class names (WordNet synonyms for Tiny ImageNet; CIFAR-100 class names for the second dataset).

**Global FID:** Fréchet Inception Distance between the **entire 5% real subset** (exported to PNG cache) and a **uniformly subsampled synthetic pool** at each budget (5× / 10× / 15× Tiny; 15× CIFAR), **one FID per ratio**. Implemented in `src/evaluation/fid_stage2.py` via **clean-fid** (`pip install clean-fid`); summaries under `results/{dataset}/fid_cache/fid_summary.json`. Not per-class FID (too few real images per class).

### Baseline and Class-Level Diagnostics

We first train each backbone (ResNet-18 and MobileNetV3) on 5% real data only. From this baseline we extract per-class properties: accuracy, mean confidence, prediction entropy, feature compactness (mean distance to class centroid), feature separation (distance to nearest other centroid), and nearest-centroid margin. These measurements inform all allocation policies.

### Allocation Policies

Given a fixed total synthetic budget B (equal to uniform 15× on Tiny ImageNet: 200 × 375, and analogously 100 × 375 on CIFAR-100 when applicable), each policy distributes integer counts per class with **normalisation and per-class clamping** to `[min_floor, max_cap]` (defaults 50 and 375).

- **Uniform.** Each class receives B / K images (K = number of classes), with remainder spread across classes.
- **Hard-class.** Budget proportional to (1 − baseline per-class accuracy).
- **Uncertainty-based.** Budget proportional to mean prediction entropy on the validation set.
- **Predicted-utility.** Ridge regression predicts utility(c) = accuracy_c(Uniform 15×) − accuracy_c(Baseline); allocation is proportional to max(0, predicted utility). **Uniform 15× must be trained before** fitting this model. Hard-class and uncertainty policies depend only on Baseline diagnostics and may be prepared in parallel with Uniform 15× training.

Cross-architecture check: allocation CSVs derived from ResNet-18 diagnostics are also used to build training sets for MobileNetV3-Small where noted.

### Utility Prediction Model

We define utility(c) = accuracy_c(Uniform 15×) − accuracy_c(Baseline) on the validation set. A **ridge** model with cross-validated α predicts utility from class-level features (baseline accuracy, mean confidence, entropy, feature compactness, separation, nearest-centroid margin, and optional synthetic-quality proxy). We report training R² and cross-validated R² where sample size permits.

### Training Protocol

Each pipeline trains a fresh ImageNet-pretrained backbone with a dataset-specific head (200-way Tiny ImageNet, 100-way CIFAR-100) using **AdamW** (lr = 3×10⁻⁴, weight decay = 0.01), **cosine annealing** (T_max = max epochs), **mixed precision**, and **early stopping** on validation Top-1 (patience 7, max 30 epochs, batch size 64). Uniform DiffusionBoost on Tiny ImageNet uses synthetic budgets **5×, 10×, and 15×** per class; adaptive policies are trained at the **15×** total budget unless stated otherwise.

**Synthetic-aware loss (ablation):** sample weights downweight synthetic images whose frozen-backbone features lie far from the real class centroid and upweight close ones; real samples keep unit weight. Full ablation tables compare Adaptive runs with standard cross-entropy versus this reweighting.

### Evaluation Framework

**Accuracy.** Top-1, macro (mean per-class), and worst-20-class accuracy on the 10,000-image validation set.

**Calibration.** ECE (15 bins), reliability diagrams, and post-hoc **temperature scaling**: grid search for scalar \(T\) minimising NLL on validation logits; report `ece` and `ece_after_scaling` in `metrics.json` (`temperature_scaling` block).

**Robustness.** Accuracy under Gaussian noise (σ=0.1), Gaussian blur (k=5), and brightness shift (+0.2).

**Per-class analysis.** Per-class accuracy delta, classification into beneficial/neutral/harmful categories, and harm detection analysis.

**Feature coverage.** Within-class compactness, between-class separation, and nearest-centroid margin in penultimate feature space.

**Representation geometry.** **Eigenvalue spectrum** of validation feature covariance (top eigenvalues + effective rank in `metrics.json`; `eval_eigenvalues.png` per run). **Linear probe:** sklearn multinomial logistic regression on frozen penultimate features (5% real train → val). **Linear CKA** vs the **baseline** model on paired validation batches (`linear_cka_vs_ref`, `eval_cka_heatmap.png`). Implemented in `src/evaluation/eval_extras.py` and `evaluate_stage2`.

**Compute-normalised benefit.** Gain per 1,000 generated images and gain per GPU-minute across policies.

## III. Results

### Accuracy and Architecture Consistency

| Pipeline | R18 Top-1 | R18 Macro | R18 Worst-20 | MV3-Small Top-1 | MV3-Small Macro | MV3-Small Worst-20 |
|----------|----------|----------|-------------|----------|----------|-------------|
| Baseline | **X1**% | **M1**% | **W1**% | **X4**% | **M4**% | **W4**% |
| Uniform 15x | **X2**% | **M2**% | **W2**% | **X5**% | **M5**% | **W5**% |
| Adaptive 15x | **X2a**% | **M2a**% | **W2a**% | **X5a**% | **M5a**% | **W5a**% |
| Ceiling | **X3**% | **M3**% | **W3**% | **X6**% | **M6**% | **W6**% |

Uniform DiffusionBoost recovers **R_unif**% of the Ceiling gap. Adaptive allocation recovers **R_adap**%, an improvement of **D**pp over uniform at the same total budget. The gain is concentrated in worst-class metrics: worst-20 accuracy improves by **Dw**pp under adaptive vs. uniform allocation.

*See Figure 1 — Learning curves for all pipelines, both architectures.*

### Synthetic Data Scaling

| Synth ratio | Synth/class | R18 Top-1 | R18 Macro | R18 ECE |
|------------|-----------|----------|----------|---------|
| 0x (Baseline) | 0 | **X1**% | **M1**% | **E1** |
| 5x | 125 | **A5**% | **Am5**% | **E5** |
| 10x | 250 | **A10**% | **Am10**% | **E10** |
| 15x | 375 | **X2**% | **M2**% | **E2** |

*See Figure 2 — Scaling curves: accuracy, macro accuracy, and ECE vs. synthetic budget.*

### Allocation Policy Comparison (at 15x total budget)

| Policy | R18 Top-1 | R18 Macro | R18 Worst-20 | Helped/Harmed |
|--------|----------|----------|-------------|--------------|
| Uniform | **X2**% | **M2**% | **W2**% | **Nh**/**Nd** |
| Hard-class | **Xh**% | **Mh**% | **Wh**% | **Nhh**/**Ndh** |
| Uncertainty | **Xu**% | **Mu**% | **Wu**% | **Nhu**/**Ndu** |
| Predicted-utility | **Xp**% | **Mp**% | **Wp**% | **Nhp**/**Ndp** |

*See Figure 3 — Policy comparison bar chart.*

The predicted-utility policy achieves the highest macro and worst-class accuracy while reducing the number of harmed classes from **Nd** to **Ndp**. This confirms that class-aware allocation outperforms uniform augmentation under the same total synthetic budget.

### Per-Class Utility and Harm Detection

Per-class accuracy delta (Uniform DiffBoost - Baseline): **N_pos** classes improve, **N_neg** degrade, **N_zero** unchanged. {Describe patterns: e.g., concrete objects improve, fine-grained categories degrade.}

Under adaptive allocation, the number of harmed classes drops from **N_neg** to **N_neg_adap**. The most harmful category {description} is successfully identified and deprioritised.

*See Figure 4 — Per-class delta sorted, uniform vs. adaptive overlay.*

### Utility Prediction Model

A ridge regression model predicting per-class utility from baseline accuracy, entropy, fidelity, compactness, and separation achieves R² = **R2** (10-fold CV). Feature importances: {list top predictors}. The model transfers across backbones: policies designed on ResNet-18 class properties improve MobileNetV3 performance by **Dt**pp over uniform.

### Synthetic Quality (FID)

Global FID (5% real subset vs synthetic subsampled at each budget): **F_5x**, **F_10x**, **F_15x**. Optional correlation between per-class fidelity proxy and per-class utility: **r_fid**.

### Calibration and Temperature Scaling

| Pipeline | Raw ECE | Temp T | Cal ECE |
|----------|---------|--------|---------|
| Baseline | **E1** | **T1** | **E1c** |
| Uniform 15x | **E2** | **T2** | **E2c** |
| Adaptive 15x | **E2a** | **T2a** | **E2ac** |
| Ceiling | **E3** | **T3** | **E3c** |

*See Figure 5 — Reliability diagrams: raw and temperature-scaled.*

### Corruption Robustness

| Pipeline | Clean | Noise | Blur | Brightness |
|----------|-------|-------|------|------------|
| Baseline | X1% | N1% | B1% | Br1% |
| Uniform 15x | X2% | N2% | B2% | Br2% |
| Adaptive 15x | X2a% | N2a% | B2a% | Br2a% |
| Ceiling | X3% | N3% | B3% | Br3% |

*See Figure 6 — Grouped bar chart.*

{Discuss clean-robustness tradeoff: does adaptive allocation give a better tradeoff than uniform?}

### Feature Coverage

| Pipeline | Compactness | Separation | Centroid Margin |
|----------|------------|-----------|----------------|
| Baseline | **Cp1** | **Sp1** | **Mg1** |
| Uniform 15x | **Cp2** | **Sp2** | **Mg2** |
| Adaptive 15x | **Cp2a** | **Sp2a** | **Mg2a** |
| Ceiling | **Cp3** | **Sp3** | **Mg3** |

Adaptive allocation {improves/maintains} between-class separation while maintaining within-class compactness, whereas uniform allocation {collapses some classes / causes overlap}.

### Representation Geometry

**Eigenvalue spectrum.** Effective rank: Baseline **R1**, Uniform **R2**, Adaptive **R2a**, Ceiling **R3**.

**Linear probe.** Baseline **P1**%, Uniform **P2**%, Adaptive **P2a**%, Ceiling **P3**%.

**CKA.** Uniform-Ceiling = **C_UC**, Adaptive-Ceiling = **C_AC**. If C_AC > C_UC, adaptive allocation produces representations structurally closer to the full-data model.

*See Figures 7-8 — Eigenvalue overlay, CKA heatmap.*

### Compute-Normalised Benefit

| Policy | Gain/1k images | Gain/GPU-min |
|--------|---------------|-------------|
| Uniform | **Gu** pp | **Gum** pp |
| Hard-class | **Gh** pp | **Ghm** pp |
| Predicted-utility | **Gp** pp | **Gpm** pp |

Adaptive allocation achieves **X**x better data efficiency than uniform augmentation (gain per generated image).

## IV. Discussion

**Class-dependent utility.** Synthetic augmentation is not uniformly beneficial. {N_neg} of 200 classes are harmed by uniform augmentation. These classes share properties: {low synthetic fidelity / fine-grained ambiguity / class overlap}. The utility prediction model captures this with R² = **R2**, enabling principled budget allocation.

**Adaptive vs. uniform.** At equal total budget, adaptive allocation improves macro accuracy by **Dm**pp and worst-class accuracy by **Dw**pp over uniform allocation. The benefit is concentrated in the tail of the performance distribution — the classes that uniform augmentation neglects or harms.

**Cross-architecture transfer.** Allocation policies designed on ResNet-18 transfer to MobileNetV3, confirming that class-level utility is not model-specific noise but reflects underlying dataset and generation properties.

**Clean-robustness tradeoff.** {Discuss whether adaptive allocation gives a better Pareto frontier than uniform.}

**Limitations.** Single generation model (SD v1.5); CIFAR-100 track is intentionally reduced (ResNet-18, four runs); utility prediction uses ridge regression; synthetic-aware loss ablations add training cost. Future work could add prompt tuning, alternative backbones, or larger-scale datasets.

## V. Conclusion

We demonstrate that uniform synthetic augmentation is suboptimal under a fixed synthetic budget. By measuring class-level baseline properties and synthetic fidelity, we predict which classes will benefit from augmentation and allocate synthetic budget accordingly. Adaptive allocation improves macro and worst-class accuracy over uniform augmentation at equal total cost, reduces the number of harmed classes, and produces representations closer to the full-data model. These findings reframe synthetic augmentation from a uniform data-scaling technique to a class-aware resource allocation problem.

## VI. References

- J. Ho et al., "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020.
- P. Dhariwal and A. Nichol, "Diffusion Models Beat GANs on Image Synthesis," *NeurIPS*, 2021.
- R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR*, 2022.
- T. Shorten and T. M. Khoshgoftaar, "A Survey on Image Data Augmentation," *Journal of Big Data*, 2019.
- C. Guo et al., "On Calibration of Modern Neural Networks," *ICML*, 2017.
- S. Kornblith et al., "Similarity of Neural Network Representations Revisited," *ICML*, 2019.
- M. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge," *NeurIPS*, 2017.
- K. He et al., "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
- A. Howard et al., "Searching for MobileNetV3," *ICCV*, 2019.

## Figures

- **Figure 1** — Learning curves (loss, accuracy, LR) for all pipelines, both architectures.
- **Figure 2** — Synthetic data scaling curves: accuracy, macro, ECE vs. budget.
- **Figure 3** — Allocation policy comparison at 15x budget.
- **Figure 4** — Per-class accuracy delta: uniform vs. adaptive, sorted.
- **Figure 5** — Reliability diagrams: raw and temperature-scaled.
- **Figure 6** — Corruption robustness grouped bar chart.
- **Figure 7** — Covariance eigenvalue spectrum overlay.
- **Figure 8** — Linear CKA similarity heatmap.
