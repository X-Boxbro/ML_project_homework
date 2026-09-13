# Support Vector Machine Experiment (Lab 2)

**[English](README.md) | [简体中文](README.zh-CN.md)**

Nankai University Machine Learning Lab 2: Support Vector Machine (SVM). A hand-written SMO (Sequential Minimal Optimization) implementation solves the SVM dual optimization problem (linear kernel), accompanied by an Augmented Lagrangian Method (ALM) implementation for comparison. The lab covers linearly separable / inseparable cases, the effect of the penalty parameter C, soft vs. hard margin, and convergence analysis. A fixed random seed makes results fully reproducible.

## Table of Contents

- [1. Background](#1-background)
- [2. Dataset](#2-dataset)
- [3. Algorithm](#3-algorithm)
- [4. Experiments and Results](#4-experiments-and-results)
  - [4.1 Experiment 1: Linearly Separable SVM](#41-experiment-1-linearly-separable-svm)
  - [4.2 Experiment 2: Linearly Inseparable Data and Soft Margin](#42-experiment-2-linearly-inseparable-data-and-soft-margin)
  - [4.3 Experiment 3: SMO Convergence Analysis](#43-experiment-3-smo-convergence-analysis)
  - [4.4 Experiment 4: Effect of the Penalty Parameter C](#44-experiment-4-effect-of-the-penalty-parameter-c)
  - [4.5 Experiment 5: Soft Margin vs Hard Margin Comparison](#45-experiment-5-soft-margin-vs-hard-margin-comparison)
  - [4.6 Augmented Lagrangian Method Comparison](#46-augmented-lagrangian-method-comparison)
- [5. Reproduction Notes and Known Issues](#5-reproduction-notes-and-known-issues)
- [6. Repository Structure](#6-repository-structure)
- [7. Environment and Usage](#7-environment-and-usage)
- [8. References](#8-references)

---

## 1. Background

The Support Vector Machine (SVM) is a classic linear binary classifier whose core idea is to find a maximum-margin separating hyperplane. With the kernel trick and soft margin, SVM can handle nonlinear and linearly inseparable problems. This lab starts from the dual formulation and implements SVM solving via a hand-written SMO algorithm, then uses several comparative experiments to understand how key hyperparameters (C, kernel) and algorithmic variants (SMO vs. Augmented Lagrangian) affect model behavior.

## 2. Dataset

Two-class 2D Gaussian samples (default `seed=42`):

| Parameter | Class 1 (+1) | Class 2 (-1) |
|---|---|---|
| Center | (1, 1) | (6, 6) |
| Std per dimension σ | 1.0 | 1.0 |
| Samples per class n | 100 | 100 |

With the default centers, ‖c₂−c₁‖ ≈ 7.07 ≫ 2σ ≈ 2.83, so the default configuration is linearly separable. Reducing the distance between centers yields overlapping data for soft-margin experiments.

## 3. Algorithm

**Primal problem (soft margin)**:

```text
min_{w,b,ξ}  (1/2)||w||² + C · Σ ξᵢ
s.t.  yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0
```

**Dual problem**:

```text
min_α  (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ) − Σᵢ αᵢ
s.t.   Σᵢ αᵢyᵢ = 0,  0 ≤ αᵢ ≤ C
```

Decision function: `f(x) = Σᵢ αᵢyᵢ(xᵢᵀx) + b`, where `b = mean_{sv}( y_sv − Σᵢ αᵢyᵢ(xᵢᵀx_sv) )`.

- **SMO algorithm**: each step selects a pair of Lagrange multipliers (αᵢ, αⱼ) and solves them analytically while keeping all other multipliers fixed, thereby decomposing the large QP problem into a sequence of two-variable subproblems. This implementation includes heuristic pair selection, update, KKT checking, and support vector identification.
- **Augmented Lagrangian (AL) method**: the equality constraint `Σαᵢyᵢ = 0` is folded into the objective via a Lagrange multiplier λ and a penalty term ρ; at each outer iteration, with λ and ρ fixed, the bounded subproblem is solved with L-BFGS-B, then λ and ρ are updated until the constraint violation is sufficiently small.
- **Support vector identification (from KKT conditions)**:
  - `αᵢ = 0` → `yᵢ(wᵀxᵢ + b) ≥ 1` (non-support vector; correctly classified and outside the margin)
  - `0 < αᵢ < C` → `yᵢ(wᵀxᵢ + b) = 1` (support vector on the margin boundary)
  - `αᵢ = C` → `yᵢ(wᵀxᵢ + b) ≤ 1` (support vector inside the margin, possibly a misclassified point)

## 4. Experiments and Results

### 4.1 Experiment 1: Linearly Separable SVM

Train an SVM on the default separable data (C=100, approximating a hard margin) and plot the data points, decision boundary, margin boundaries, and support vectors.

- Output image: `svm_separable.png` (left: scatter plot; right: SVM result with support vectors marked).

### 4.2 Experiment 2: Linearly Inseparable Data and Soft Margin

Reduce the distance between the two class centers to create overlapping data, then train a soft-margin SVM under different C values and observe how the decision boundary, support vector distribution, and margin width change.

- Output image: `svm_nonseparable.png` (multiple subplots showing classification results for different C).

### 4.3 Experiment 3: SMO Convergence Analysis

Record the dual objective value across SMO iterations and plot the convergence curve, observing how the objective decreases and stabilizes with increasing iterations.

- Output image: `svm_convergence.png` (dual objective vs. iteration count).

### 4.4 Experiment 4: Effect of the Penalty Parameter C

On a fixed dataset, sweep C from small values (soft-margin tendency) to large values (hard-margin tendency), recording for each C the total number of support vectors, the number of boundary support vectors (0 < α < C), and the margin width `2/‖w‖`. Results are plotted and tabulated.

- Output image: `svm_c_effect.png` (three subplots: total SV count, boundary SV count, and margin width vs. C).
- Terminal table: per-C total SV count, boundary SV count, and margin width.

### 4.5 Experiment 5: Soft Margin vs Hard Margin Comparison

On the same separable data, train SVMs with C=0.1 (soft margin), C=1.0 (intermediate), and C=100 (near hard margin) and visually compare the decision boundaries and support vector distributions.

- Output image: `svm_soft_vs_hard.png` (three side-by-side subplots).

### 4.6 Augmented Lagrangian Method Comparison

Solve the SVM on the same data using the Augmented Lagrangian method, compare its convergence curve (objective / constraint violation vs. outer iteration) with SMO, and report both methods' support vector counts, boundary parameters, etc.

- Output images: `svm_al_convergence.png` (ALM convergence curve), `svm_smo_vs_al.png` (SMO vs. AL comparison).
- Terminal output: side-by-side comparison of w, b, support vector counts, etc.

## 5. Reproduction Notes and Known Issues

- The augmented Lagrangian inner subproblem uses `scipy.optimize.minimize` (L-BFGS-B), so in addition to numpy and matplotlib the experiment also requires scipy.
- All random data generation uses a fixed seed; repeated runs give identical results; images are saved at 150 dpi to the script's directory.
- Some experiments (e.g., C sweep, convergence curve) involve many iterations; the first run may take a noticeable amount of time.

## 6. Repository Structure

```text
svm/
├── README.md             # English documentation (rendered on the GitHub homepage)
├── README.zh-CN.md       # Chinese documentation (this document's counterpart)
├── SVM_example.m         # Course-provided MATLAB reference example
└── svm_manual.py         # Hand-written SVM: SMO + Augmented Lagrangian + all experiments
```

Images generated at runtime (`svm_*.png`) are not committed; see the root `.gitignore`.

## 7. Environment and Usage

```bash
pip install numpy matplotlib scipy
python svm_manual.py
```

In a headless environment (server / CI), run `export MPLBACKEND=Agg` first. The script runs all 5 main experiments plus the ALM comparison in one execution and prints per-stage results to the terminal.

## 8. References

- Cortes, C. & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Platt, J. C. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines. *Advances in Kernel Methods — SVM*, 185–208.
- Li Hang. *Statistical Learning Methods (2nd Edition)*, Chapter 7: Support Vector Machines.
- Course-provided MATLAB example `SVM_example.m`.
