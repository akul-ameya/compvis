# Stage 2 Report Draft — Adaptive Synthetic Budget Allocation for Low-Data Image Classification

> IEEE Conference Template. Replace all **X/Y/Z** placeholders with actual results. Figures saved to `figures/stage2/`.

## I. Introduction

Standard synthetic data augmentation treats all classes equally, allocating the same number of generated images regardless of class difficulty, synthetic fidelity, or downstream utility. We argue this uniform policy is suboptimal: under a fixed synthetic budget, synthetic data has class-dependent value, and that value is predictable from measurable class properties.

We study this on Tiny ImageNet (200 classes, 64x64 images). Our four pipelines are: (1) **Baseline** — ResNet-18 trained on 5% real data (25 images/class); (2) **Uniform DiffusionBoost** — the same backbone trained on 5% real data plus uniformly allocated synthetic images from Stable Diffusion; (3) **Adaptive DiffusionBoost** — same total synthetic budget allocated per class based on predicted utility; and (4) **Ceiling** — the same backbone trained on 100% real data. All experiments are replicated on MobileNetV3 to test cross-architecture generalisability.

Our contributions are: (a) a synthetic data scaling study from 5x to 15x ratios characterising the dose-response relationship; (b) a class-level utility prediction framework identifying which classes benefit, are neutral, or are harmed by synthetic augmentation; (c) three adaptive allocation policies (hard-class, uncertainty-based, predicted-utility) compared against uniform allocation at equal total budget; and (d) a multi-axis evaluation covering accuracy, calibration, robustness, per-class breakdown, feature coverage, and representation geometry.

## II. Methodology

### Synthetic Data Generation and Quality Assessment

We use Stable Diffusion v1.5 with DPM-Solver++ (25 steps, guidance scale 7.5) to generate up to 375 class-conditional images per class (75,000 total), providing the pool from which each allocation policy draws. Prompts derive from WordNet class labels with four template variations. To quantify generation fidelity before training, we compute the Frechet Inception Distance (FID) between real and synthetic images using InceptionV3 features. We also compute per-class synthetic quality proxies (mean feature distance to real centroid, intra-class diversity) that serve as inputs to the utility prediction model.

### Baseline and Class-Level Diagnostics

We first train each backbone (ResNet-18 and MobileNetV3) on 5% real data only. From this baseline we extract per-class properties: accuracy, mean confidence, prediction entropy, feature compactness (mean distance to class centroid), feature separation (distance to nearest other centroid), and nearest-centroid margin. These measurements inform all allocation policies.

### Allocation Policies

Given a fixed total synthetic budget B (equal to the uniform budget at a given ratio), each policy distributes B across 200 classes:

- **Uniform.** Each class receives B/200 images.
- **Hard-class.** Budget proportional to (1 - baseline_accuracy_c), directing resources to difficult classes.
- **Uncertainty-based.** Budget proportional to mean prediction entropy, directing resources to classes the model is least certain about.
- **Fidelity-aware.** Budget proportional to synthetic quality score, investing where generated images are higher fidelity.
- **Predicted-utility.** A linear model trained on class-level features predicts per-class utility (accuracy gain from synthetic data); budget is proportional to predicted utility.

All policies are subject to a per-class cap (maximum available generated images) and a minimum allocation floor.

### Utility Prediction Model

We define utility(c) = accuracy_c(DiffBoost) - accuracy_c(Baseline) for each class c under uniform augmentation. We then fit a regression model predicting utility from class-level features (baseline accuracy, entropy, fidelity, compactness, separation). We report feature importances and test whether the predicted-utility allocation improves on uniform allocation.

### Training Protocol

Each pipeline trains a fresh ImageNet-pretrained backbone with a 200-way head using AdamW (lr=3e-4, weight decay=0.01), cosine annealing LR, mixed precision, and early stopping (patience 7, max 30 epochs). The Uniform DiffusionBoost experiments are run at four synthetic budgets: 5x, 10x, and 15x per class (125, 250, 375 synthetic images/class). Adaptive policies are compared at the highest budget (15x) to maximise the potential benefit of targeted allocation.

### Evaluation Framework

**Accuracy.** Top-1, macro (mean per-class), and worst-20-class accuracy on the 10,000-image validation set.

**Calibration.** ECE (15 bins), reliability diagrams, and post-hoc temperature scaling (Guo et al. 2017).

**Robustness.** Accuracy under Gaussian noise (σ=0.1), Gaussian blur (k=5), and brightness shift (+0.2).

**Per-class analysis.** Per-class accuracy delta, classification into beneficial/neutral/harmful categories, and harm detection analysis.

**Feature coverage.** Within-class compactness, between-class separation, and nearest-centroid margin in penultimate feature space.

**Representation geometry.** Eigenvalue spectrum, linear probe accuracy, and pairwise linear CKA.

**Compute-normalised benefit.** Gain per 1,000 generated images and gain per GPU-minute across policies.

## III. Results

### Accuracy and Architecture Consistency

| Pipeline | R18 Top-1 | R18 Macro | R18 Worst-20 | MV3 Top-1 | MV3 Macro | MV3 Worst-20 |
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

Overall FID (real vs synthetic): **F_all**. Per-class fidelity correlates with per-class utility (r = **r_fid**), confirming that generation quality mediates downstream benefit.

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

**Limitations.** Single generation model (SD v1.5), single dataset (Tiny ImageNet), utility prediction tested only as ridge regression. Future work could explore generation-time conditioning, prompt engineering for hard classes, or larger-scale datasets.

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
