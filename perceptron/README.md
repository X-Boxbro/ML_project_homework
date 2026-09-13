# Perceptron Experiment (Lab 1)

**[English](README.md) | [简体中文](README.zh-CN.md)**

Nankai University Machine Learning Lab 1: Perceptron binary classification. The course-provided MATLAB reference implementation (`perception_example.m`) is migrated to Python, comprising a main experiment and 5 bonus questions. A fixed random seed makes results fully reproducible.

## Table of Contents

- [1. Background](#1-background)
- [2. Dataset](#2-dataset)
- [3. Algorithm](#3-algorithm)
- [4. Experiments and Results](#4-experiments-and-results)
  - [4.1 Main Experiment](#41-main-experiment)
  - [4.2 Bonus 1: Iteration Process under Different Learning Rates](#42-bonus-1-iteration-process-under-different-learning-rates)
  - [4.3 Bonus 2: Removing the Misclassification Check](#43-bonus-2-removing-the-misclassification-check)
  - [4.4 Bonus 3: Linearly Inseparable Data](#44-bonus-3-linearly-inseparable-data)
  - [4.5 Bonus 4: Effect of Cluster Center Distance](#45-bonus-4-effect-of-cluster-center-distance)
  - [4.6 Bonus 5: High-Dimensional Extension (100D)](#46-bonus-5-high-dimensional-extension-100d)
- [5. Reproduction Notes and Known Issues](#5-reproduction-notes-and-known-issues)
- [6. Repository Structure](#6-repository-structure)
- [7. Environment and Usage](#7-environment-and-usage)
- [8. References](#8-references)

---

## 1. Background

The perceptron (Rosenblatt, 1958) is one of the earliest machine learning models and forms the foundation of modern neural networks and support vector machines. It is a linear binary classifier: `f(x) = sign(w·x + b)`, which searches for a separating hyperplane by updating parameters only on misclassified samples. This lab implements the perceptron primal form on a 2D synthetic dataset and systematically observes convergence behavior and decision boundary evolution.

## 2. Dataset

Two-class 2D Gaussian samples (`seed=42`, train and test sets are independently sampled):

| Parameter | Class 1 (+1) | Class 2 (-1) |
|---|---|---|
| Center | (1, 1) | (3, 4) |
| Std per dimension σ | 1.0 | 1.0 |
| Training samples n | 100 | 100 |
| Test samples m | 10 | 10 |

The distance between centers ‖c₂−c₁‖ ≈ 3.61 is slightly larger than 2σ ≈ 2.83, so the data are "mostly separable" but a few overlap samples exist (this directly affects whether the main experiment fully converges, see §4.1).

## 3. Algorithm

**Model**: `f(x) = sign(w·x + b)`

**Primal update rule** (applied when a sample is misclassified, i.e., `y_i(w·x_i + b) ≤ 0`):

```text
w ← w + lr · y_i · x_i
b ← b + lr · y_i
```

- **Convergence** (Novikoff's theorem): when data are linearly separable, the number of misclassification updates is bounded and the algorithm converges in a finite number of steps.
- **Pocket algorithm**: because this dataset contains overlapping samples and is not strictly linearly separable, after each full pass through the data we check the "cumulative correct count" and, if it exceeds the historical best, store the current `(w, b)` in the pocket. The pocket's best parameters are returned at the end, preventing the final-epoch parameters from degrading.
- Learning rate `lr = 0.1` (a smaller value suppresses oscillation in the overlap region); maximum 500 passes.

## 4. Experiments and Results

### 4.1 Main Experiment

Pipeline: generate data → plot training scatter → train → plot decision boundary → evaluate on test set → plot train+test+boundary → result analysis.

Example run (`seed=42`):

| Metric | Value |
|---|---|
| Final decision boundary (pocket best) | −0.2785·x1 − 1.2619·x2 + 3.6000 = 0 |
| Pocket best training accuracy | 194/200 (97.0%) |
| Training error rate | 3.00% (6/200) |
| Test error rate | 5.00% (1/20) |
| Distance from class centers to boundary | 1.59 / 1.77 (roughly balanced; boundary position reasonable) |

Because a few samples overlap, each pass still misclassifies about 4 samples and the algorithm does not fully converge within 500 passes; the pocket mechanism guarantees the returned parameters are the best seen throughout training.

Output images: `fig1_training_data.png` (training data), `fig2_classification.png` (decision boundary), `fig3_train_test.png` (train + test + boundary).

### 4.2 Bonus 1: Iteration Process under Different Learning Rates

Fix the data and train with `lr ∈ {0.1, 0.5, 1.0}`, plotting the decision boundary after the 1st, 5th, 10th, 20th, 50th, and 100th updates (3×6 subplot grid). Also report the misclassification count and final `w, b` after 100 passes for each learning rate.

- **Observation**: the larger the learning rate, the larger the single-step update, and the more the boundary oscillates in the overlap region; `lr=0.1` is the most stable.
- Output: `add_fig1_lr_iterations.png` + terminal summary table.

### 4.3 Bonus 2: Removing the Misclassification Check

After removing `if y_i(w·x_i+b) ≤ 0`, **every sample unconditionally executes the update**. Parameters then degenerate into a cumulative sum of training-set label information (`b` converges toward a constant, while `‖w‖` grows **linearly** with the number of updates), and the boundary direction is dominated by the overall sample distribution and drifts violently.

- Output: `add_fig2_no_check.png` ((a) ‖w‖ evolution with vs. without the check; (b) correct-version boundary; (c) incorrect-version boundary) and `add_fig2_no_check_detail.png` (incorrect-version boundary at the 1st/20th/50th/100th pass).

### 4.4 Bonus 3: Linearly Inseparable Data

Construct strongly overlapping data: centers (0, 0) and (1, 1), distance 1.41 < 2σ ≈ 2.83.

- **Conclusion**: the loop does **not** terminate — the number of misclassified samples **oscillates** within a fixed band (after 500 passes misclassifications remain), and `‖w‖` **grows without bound** across passes (updates keep occurring, ‖w‖ is unbounded). This validates that Novikoff's convergence theorem requires "linear separability" as a premise.
- Output: `add_fig3_inseparable.png` ((a) misclassification count vs. epoch; (b) ‖w‖ vs. epoch).

### 4.5 Bonus 4: Effect of Cluster Center Distance

Fix class 1 center at (1, 1) and move class 2 center from (3, 3) to (10, 10), covering 5 center distances `d ≈ 2.0 / 3.6 / 5.7 / 8.5 / 12.7`.

- **Conclusion**: the misclassification rate decreases monotonically as center distance increases — at d=2.0 the two classes heavily overlap with a misclassification rate around 10%; for d ≥ 8.5 the data are basically separable; at d=12.7 they are strictly separable (0 misclassifications).
- Output: `add_fig4_center_distance.png` (5 subplots) + terminal summary table (misclassification count and `w, b` per distance).

### 4.6 Bonus 5: High-Dimensional Extension (100D)

Extend data generation to 100 dimensions (class centers are the all-1 and all-3 vectors) and train a 100D perceptron, then compare with the 2D version:

- Because the center distance accumulates with dimension (‖c₂−c₁‖ = √(100·4) = 20, while per-dimension noise contributes √100·1 = 10), **the higher the dimension, the higher the signal-to-noise ratio**. The 100D training/test accuracy is close to 100%, significantly outperforming the 2D version.
- Numerical results are printed directly to the terminal.

## 5. Reproduction Notes and Known Issues

- **`perceptron.py`'s `history` list is never populated**: the function docstring says it should record "misclassification count per pass", but the loop lacks `history.append(...)`, so step [7] of the main program always prints "converged passes / total updates" as 0. This does not affect training, the pocket parameters, or any image output.
- **Bonus 1 subplot/title mismatch**: `epoch_wb[col]` takes indices 0–5 (i.e., the boundary after the 0th–5th updates), while subplot titles say "1st/5th/10th/20th/50th/100th iteration". For strict correspondence, indexing by `check_epochs` should be used instead.
- All experiments use a fixed random seed; repeated runs give identical results; images are saved at 150 dpi to the script's directory.

## 6. Repository Structure

```text
perceptron/
├── README.md             # English documentation (rendered on the GitHub homepage)
├── README.zh-CN.md       # Chinese documentation (this document's counterpart)
├── perception_example.m  # Course-provided MATLAB reference implementation
├── perceptron.py         # Main experiment: perceptron primal form + Pocket algorithm
└── perceptron_add.py     # Bonus questions 1–5
```

Images generated at runtime (`fig*.png`, `add_fig*.png`) are not committed; see the root `.gitignore`.

## 7. Environment and Usage

```bash
pip install numpy matplotlib
python perceptron.py        # Main experiment
python perceptron_add.py    # Bonus questions 1–5 (all executed in one run)
```

In a headless environment (server / CI), run `export MPLBACKEND=Agg` first.

## 8. References

- Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408.
- Li Hang. *Statistical Learning Methods (2nd Edition)*, Chapter 2: Perceptron.
- Course-provided MATLAB example `perception_example.m`.
