# Discovery of Hidden Symmetries and Conservation Laws
### End-to-End ML Pipeline on Rotated MNIST — Research Report


## Problem Statement

Recent advances in symmetry-aware machine learning have demonstrated that models which explicitly encode physical symmetries are more robust, data-efficient, and interpretable. While symmetries of the Standard Model (e.g. SO(2), SO(3), Lorentz group) are well-understood in 4-momentum space, they become **elusive in abstract representation spaces** — the latent spaces of modern neural encoders.

This project addresses the following research question:

> *Given only an abstract representation (a VAE latent code), can a neural network discover that SO(2) rotation is a symmetry of the data, without being told the group structure in advance?*

The pipeline proceeds in four stages: dataset construction → latent space learning → supervised symmetry mapping → unsupervised symmetry discovery, capped by a bonus rotation-invariant classifier.

**Reference papers:**
- Forestano et al. (2023) — *Deep Learning Symmetries* ([arXiv:2109.09721](https://arxiv.org/abs/2109.09721))
- Moskalev et al. (2023) — *LieGG* ([arXiv:2301.05638](https://arxiv.org/pdf/2301.05638v1))
- Keurti et al. (2023) — *Homomorphism Autoencoder* ([arXiv:2302.00236](https://arxiv.org/abs/2302.00236))

---

## Task 1 — Dataset Preparation

### Protocol

- **Source**: Vanilla MNIST, filtered to **digits {1, 2}** only (computational budget constraint)
- **Rotation Augmentation**: Each sample rotated in **steps of 30°** covering the full SO(2) orbit: {0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330°}
- **Storage**: Saved as HDF5 with three arrays: `images`, `labels`, `angles`

### Dataset Statistics

| Split | Samples | Orbit Factor | Shape |
|-------|---------|-------------|-------|
| Train | 12,700 base → **152,400** | ×12 | (152400, 28, 28) |
| Test  | 2,167 base → **26,004**   | ×12 | (26004, 28, 28)  |

| Digit | Train count | Test count |
|-------|------------|-----------|
| 1     | 6,742       | 1,135      |
| 2     | 5,958       | 1,032      |

> Each training sample appears 12 times, once at each rotation angle, forming a **complete SO(2) orbit** in the discrete Z₁₂ subgroup.

### Dataset Preview

![Dataset Preview — one sample per (digit, angle)](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/Task1_Data_prep/orbit_visualization.png)

*One sample per (digit × angle) pair. Top row: digit 1, bottom row: digit 2. Columns span 0° → 330° in 30° steps.*

---

## Task 1 — Latent Space Creation: VAE Ablation

Four VAE variants were trained and evaluated. All share the same convolutional encoder/decoder backbone (~3.1M parameters, ResBlock + GroupNorm + SiLU) with a 16-dimensional latent space, trained for 50 epochs on a Tesla T4 GPU.

### Architecture

```
Encoder: Input(1×28×28)
  → Conv(1→32, k=3, s=2)  → ResBlock(32) → ResBlock(32)
  → Conv(32→64, k=3, s=2) → ResBlock(64) → ResBlock(64)
  → Flatten → Linear(→μ₁₆, →logσ₁₆)     [1,560,928 params]

Decoder: z₁₆
  → Linear → Reshape(64×7×7)
  → TransposedConv(64→32) → ResBlock(32)
  → TransposedConv(32→1)  → Sigmoid        [1,554,593 params]
```

### VAE Variants

| Exp | Name | Loss Function | Key Hyperparameters |
|-----|------|---------------|---------------------|
| 1 | **Baseline VAE** | `BCE + KL` | β=1 |
| 2 | **β-VAE** | `BCE + β·KL` | β=4.0 |
| 3 | **Capacity VAE** (β-VAE-H) | `BCE + γ·\|KL − C(t)\|` | γ=100, C_max=25 |
| 4 | **SO(2)-Invariant VAE** | `BCE + γ·\|KL−C\| + λ·‖z(x)−z(R_θx)‖²` | λ_inv=10.0 |

### Evaluation Metrics

- **RIE** (Rotation Invariance Error): Mean L2 distance between `z(x)` and `z(R_θ x)` — *lower is better*
- **LP-AUC** (Linear Probe AUC): ROC-AUC of a logistic regression classifier on latent codes — *higher is better*
- **OCS** (Orbit Circularity Score): How closely rotation orbits trace circles in PCA-projected latent space

---

### Exp 1 — Baseline VAE

#### Training Curves
![Baseline VAE Training Curves](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/baseline_vae/A_training_curves.png)

*Left: Total loss and reconstruction (BCE). Centre: KL divergence. Right: Val ELBO. Converges to val ELBO = 81.65.*

#### Evaluation Panel
![Baseline VAE Eval Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/baseline_vae/E_eval_metrics.png)

*Clockwise from top-left: t-SNE coloured by digit / by angle; RIE per rotation angle; orbit traces in PCA space; OCS histogram. Orbits show no circular structure — the encoder treats each rotated sample independently.*

---

### Exp 2 — β-VAE (β=4)

#### Evaluation Panel
![Beta-VAE Eval Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/beta_vae/E_eval_metrics.png)

*β=4 forces stronger posterior regularisation. RIE drops 4.09 → 2.96. t-SNE clusters tighten. LP-AUC = 0.9984, the best across all VAE variants.*

---

### Exp 3 — Capacity VAE  (Selected Backbone)

#### Training Curves
![Capacity VAE Training Curves](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/capacity_vae/A_training_curves.png)

*Four-panel log: Reconstruction, KL, capacity target C(t) (annealed 0→25), and Val ELBO. C stabilises at 25 around epoch 42, forcing exactly 3 highly active latent dimensions.*

#### Evaluation Panel
![Capacity VAE Eval Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/capacity_vae/E_eval_metrics.png)

*With capacity annealing, t-SNE separates digits cleanly and angle-coloured clusters form visible arcs. RIE = 1.73, OCS = 0.832. Orbit traces approach — but don't perfectly form — circles in PCA space.*

#### Per-Angle Reconstructions
![Capacity VAE Per-Angle Reconstructions](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/capacity_vae/C_reconstructions.png)

*Top: originals at 0°–150°. Middle: reconstructions. Bottom: absolute residual. Reconstruction quality is stable across all angles with no systematic degradation.*


#### Generated Samples & Latent Interpolation
![Capacity VAE Generated Samples](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/capacity_vae/D_latent_traversal.png)

*Top: random samples from N(0,I). Bottom: latent interpolations between digits 1 and 2. Sharp, diverse samples and smooth interpolations confirm a well-structured latent manifold.*

---

### Exp 4 — SO(2)-Invariant VAE

#### Evaluation Panel
![SO(2)-Invariant VAE Eval Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/equivariant_vae/E_eval_metrics.png)

*Explicit equivariance penalty achieves best RIE = 0.86 (5× over baseline). t-SNE shows angle clusters fully collapsed — the encoder has learned rotation invariance. Trade-off: OCS = 0.825.*

#### Per-Angle Reconstructions
![SO(2)-Invariant VAE Per-Angle Reconstructions](ihttps://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/equivariant_vae/C_reconstructions.png)

*Reconstructions remain sharp across all angles. The decoder compensates for the collapsed angle information.*


#### Generated Samples & Latent Interpolation
![SO(2)-Invariant VAE Generated Samples](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task1_latent_representation_learning/equivariant_vae/D_latent_traversal.png)

*Samples show less angular diversity than Capacity VAE — expected, since the model discards orientation information.*

---

### VAE Results Summary

| Experiment | Val ELBO (β=1) | RIE ↓ | LP-AUC ↑ | OCS | Top Active Dims |
|------------|----------------|--------|----------|-----|-----------------|
| Exp1 — Baseline VAE | **81.65** | 4.0885 | 0.9856 | 0.8384 | [12,4,14,6] |
| Exp2 — β-VAE (β=4) | 85.56 | 2.9607 | **0.9984** | **0.8398** | [6,2,13,3] |
| Exp3 — **Capacity VAE**  | 98.21 | 1.7252 | 0.9982 | 0.8318 | [1,4,7,12] |
| Exp4 — SO(2)-Invariant VAE | 108.30 | **0.8597** | 0.9968 | 0.8250 | [3,0,13,15] |

---

## Task 2 — Supervised Symmetry Discovery

Using the frozen Capacity VAE encoder, three flow models learn the latent-space transformation for +30° rotation, evaluated on all 11 non-trivial angles via multi-step rollout.

### Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| M1 | Mean Latent MSE across all target angles | ↓ |
| M2 | Oracle drift after full 360° cycle | ↓ |
| M4 | Cycle closure `‖z₀ − g¹²(z₀)‖ / ‖z₀‖` | ↓ |

---

### Exp 1 — Heun-RK2 Flow

141K-param MLP integrated with second-order Heun (RK2):
```
z_{t+1} = z_t + 0.5·(k₁ + k₂),   k₁ = g(z_t),  k₂ = g(z_t + k₁)
```

#### Training Curves
![Heun-RK2 Training Curves](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task2_supervised_symmetry_discovery/heun_flow/results.png)

*Supervised, multi-step, cycle closure, and oracle-logit losses. Convergence is fast (~5 epochs) and stable.*

#### Orbit Visualisation
![Heun-RK2 Orbit Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task2_supervised_symmetry_discovery/heun_flow/orbit.png)

*Each row: base sample at 0°, then latent rollouts at 30°, 60°, …, 330° decoded to pixel space. The model accurately generates all 12 orientations from a single 30°-step generator.*

---

### Exp 3 — Bi-Directional Spectral MLP

Spectral normalisation (σ_max(W) ≤ 1, Lip = 1.0001) with bidirectional loss jointly training forward (+30°) and approximate inverse steps.

#### Training Curves
![Spectral Flow Training Curves](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task2_supervised_symmetry_discovery/bidir_spectral/results.png)

*Four-component loss: forward, backward, invertibility, closure. Spectral norm keeps training smooth — no spikes.*

#### Orbit Visualisation
![Spectral Flow Orbit Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task2_supervised_symmetry_discovery/bidir_spectral/orbit.png)

*Visual quality on par with Heun-RK2, consistent with matching M1 = 0.00093.*

---

### Exp 4 — Hybrid Lie Algebra + Nonlinear Residual

```
T(z, θ) = expm(θ·A)·z  +  MLP([z; sin(θ), cos(θ), sin(2θ), cos(2θ)])
```
Skew-symmetry enforced by construction (A = L − Lᵀ).

#### Training Curves
![Hybrid Lie Training Curves](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task2_supervised_symmetry_discovery/hybrid_lie_residual/results.png)

*Supervised, composition, inverse, orbit, and regularisation losses.*

#### Orbit Visualisation
![Hybrid Lie Orbit Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task2_supervised_symmetry_discovery/hybrid_lie_residual/orbit.png)

*Excellent quality at small angles (30°–180°) but visible degradation at large angles (270°–330°) — the 16×16 skew-symmetric matrix cannot fully capture latent manifold curvature at large displacements.*

---

### Task 2 Results

| Experiment | M1 MSE ↓ | M2 Drift ↓ | M4 Cycle ↓ | Lipschitz |
|------------|----------|------------|------------|-----------|
| **Heun-RK2** | 0.00094 | **0.00021** | 0.09354 | — |
| **Bi-Dir Spectral** | **0.00093** | 0.00159 | — | **1.0001** |
| Hybrid Lie+Residual | 0.03682 | 0.42335 | **0.00014** | — |

**Per-angle M1:**

| Angle | Heun-RK2 | Spectral | Lie+Residual |
|-------|----------|----------|--------------|
| 30°–180° | 0.00081–0.00138 | 0.00054–0.00166 | 0.00080–0.00135 |
| 210°–330° | 0.00073–0.00138 | 0.00105–0.00166 | 0.03180–0.14313 |

---

## Task 3 — Unsupervised Symmetry Discovery

Without angle labels, three methods discover symmetry generators from the sole constraint of oracle logit preservation: `ψ(z + ε·G(z)) ≈ ψ(z)`.

A binary oracle ψ (3-layer MLP, val_acc = 99.57%) is trained first. Then N_g generators are trained unsupervised.


### Exp 2 — Jacobian Null-Space Method

Generators constrained to `ker(J_ψ(z))`, enforcing exact first-order logit preservation.

#### Evaluation Panel
![Jacobian Eval Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task3_unsupervised_symmetry_discovery/equivariant_classifier/results.png)

*Training dynamics for G₁ and G₂; null-space alignment score; inter-generator orthogonality maintained throughout training.*

#### Generator Visualisation
![Jacobian Generator Visualisation](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task3_unsupervised_symmetry_discovery/equivariant_classifier/orbits.png)

*Latent orbits (top) and decoded sequences (bottom) for both generators. Both produce smooth transformations preserving digit identity.*

---

### Exp 3 — Contrastive Symmetry Discovery

Triplet-style loss pulling same-class orbit samples together, pushing cross-class apart, with antisymmetry `G(-G(z)) ≈ z`.

#### Evaluation Panel
![Contrastive Eval Panel](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task3_unsupervised_symmetry_discovery/spectral_symmetry/results.png)

*Pull, push, oracle, antisymmetry training losses; val triplet accuracy reaching 99.80%; intra/inter distance ratio.*

#### Generator Visualisation
![Contrastive Generator Visualisation](https://github.com/tanishmohokar/GSOC_2026_E2E_ML4SCI_Hidden_Symmetries/blob/main/task3_unsupervised_symmetry_discovery/spectral_symmetry/orbits.png)

*Orbits are less circular than Exp1/2 — the push loss distorts the flow field, visible as irregular arcs in PCA space.*

---

### Task 3 Results

| Method | Rot. Alignment ↑ | Oracle Drift ↓ | Null-Space Align | Gram Rank |
|--------|-----------------|----------------|-----------------|-----------|
| **Oracle Flow** | **0.2624** | — | — | **2** |
| **Jacobian Null-Space** | 0.2621 | 0.0105–0.0295 | **0.65 / 0.59** | 2 |
| Contrastive | 0.1942 | 0.75977 | — | — |

> All oracle-preserving methods independently recover **rank-2 symmetry**, consistent with Z₁₂ ⊂ SO(2).

---

## Bonus Task — Rotation-Invariant Network

Four approaches build a rotation-invariant classifier trained on 0°-only data, tested across all 12 angles. Invariance headroom to close: **0.1593 accuracy points**.

### Results

| Method | Mean Acc ↑ | Δ vs Baseline | Gap Closed |
|--------|-----------|--------------|------------|
| MLP-NoAug (baseline) | 0.8496 | — | 0% |
| SWOP K=6 | 0.8300 | −0.020 | — |
| SWOP K=12 | 0.8079 | −0.042 | −26% |
| ACEC-soft | 0.8246 | −0.031 | −19% |
| ACEC-hard | 0.7847 | −0.065 | −41% |
| **CLC** | **0.9812** | **+0.132** | **83%** |
| MLP-FullAug *(oracle)* | 0.9947 | +0.145 | 100% |

**CLC** (Continuous Latent Canonicalization) dominates — a learned canonicalization network maps each latent code to a canonical orientation before classification. **ACEC** (Algebra-Commutant constraint) hurts performance in both soft and hard modes, as the global linear commutant constraint `[W,A]=0` is too restrictive for the empirical decision boundary.

---

## Master Comparison Table

### Task 1 — VAE Latent Space

| Model | Val ELBO ↑ | RIE ↓ | LP-AUC ↑ | OCS |
|-------|-----------|--------|----------|-----|
| Baseline VAE | **81.65** | 4.0885 | 0.9856 | 0.8384 |
| β-VAE (β=4) | 85.56 | 2.9607 | **0.9984** | **0.8398** |
| Capacity VAE   | 98.21 | 1.7252 | 0.9982 | 0.8318 |
| SO(2)-Invariant VAE | 108.30 | **0.8597** | 0.9968 | 0.8250 |

### Task 2 — Supervised Flow

| Model | M1 (MSE) ↓ | M2 (Drift) ↓ | Properties |
|-------|------------|-------------|------------|
| **Heun-RK2** | 0.00094 | **0.00021** | Empirically optimal |
| **Bi-Dir Spectral** | **0.00093** | 0.00159 | Lipschitz = 1.0001 |
| Hybrid Lie+Residual | 0.03682 | 0.42335 | Exact group structure |

### Task 3 — Unsupervised Discovery

| Method | Rot. Alignment ↑ | Oracle Drift ↓ | Rank |
|--------|-----------------|----------------|------|
| **Oracle Flow** | **0.2624** | — | 2 |
| Jacobian Null-Space | 0.2621 | 0.010 | 2 |
| Contrastive | 0.1942 | 0.760 | — |

### Bonus — Rotation-Invariant Classifier

| Method | Mean Acc ↑ | Gap Closed |
|--------|-----------|------------|
| Baseline | 0.8496 | 0% |
| **CLC** | **0.9812** | **83%** |
| ACEC-soft | 0.8246 | −19% |
| Oracle | 0.9947 | 100% |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  TASK 1: Dataset                                                │
│  MNIST {1,2} → 12× rotation → HDF5 (152,400 train / 26,004 test)│
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  TASK 1: VAE (best: Capacity VAE, γ=100, C_max=25)              │
│  Conv-ResNet Encoder → μ,σ (ℝ¹⁶) → Reparameterize              │
│  → Conv-ResNet Decoder → Sigmoid(28×28)                         │
│  Metrics: RIE=1.73, AUC=0.998, OCS=0.832                       │
└────────────────────────┬────────────────────────────────────────┘
                         │  frozen encoder  z = μ(x)
          ┌──────────────┼──────────────┐
          │              │              │
┌─────────▼───────┐  ┌───▼────────┐  ┌─▼──────────────────────┐
│ TASK 2          │  │ TASK 3     │  │ BONUS TASK             │
│ Supervised Flow │  │ Unspv.     │  │ Invariant Classifiers  │
│ g: z → z+δ(30°) │  │ Oracle ψ   │  │ CLC: canon→classify    │
│ Best: Heun-RK2  │  │ + Gen. G   │  │                        │
│ M1 = 0.00094    │  │ Rank = 2   │  │ Best: CLC 98.12%       │
│                 │  │ Align=0.26 │  │ vs baseline 84.96%     │
└─────────────────┘  └────────────┘  └────────────────────────┘
```

---

## Key Findings

1. **Explicit symmetry penalties are most effective**: The SO(2)-Invariant VAE achieves 5× lower RIE than the baseline by directly penalising `‖z(x) − z(R_θx)‖²`.

2. **Simple MLPs represent group actions near-perfectly**: Heun-RK2 and Spectral flows (141K params) achieve latent MSE ≈ 0.00094 — confirming the Capacity VAE latent space admits a near-linear SO(2) representation.

3. **Unsupervised methods recover correct group rank**: All oracle-preserving methods independently discover **rank-2 generators**, matching Z₁₂ ⊂ SO(2). Cosine similarity ~0.26 with the true generator is strong for a fully unsupervised method with inherent basis ambiguity.

4. **Learned canonicalization dominates orbit pooling**: CLC (98.1%) closes 83% of the invariance gap; SWOP (80.8%) and ACEC degrade performance, showing that imperfect generator rollout introduces more noise than the invariance it provides.

5. **Mathematical rigour vs. empirical accuracy**: The Hybrid Lie model achieves perfect group properties (skew error = 0, composition error = 4×10⁻⁷) but underperforms at large angles, highlighting a fundamental tension in learned symmetry representations.

---

## References

```bibtex
@article{forestano2023deep,
  title={Deep Learning Symmetries and Their Lie Groups, Algebras, and Subalgebras},
  author={Forestano, Roy T. and others},
  journal={arXiv:2109.09721}, year={2023}
}

@article{moskalev2023liegg,
  title={LieGG: Studying Learned Lie Group Generators},
  author={Moskalev, Artem and others},
  journal={arXiv:2301.05638}, year={2023}
}

@article{keurti2023homomorphism,
  title={Homomorphism AutoEncoder — Learning Group Structured Representations},
  author={Keurti, Hamza and others},
  journal={arXiv:2302.00236}, year={2023}
}
```

---

