# Machine Learning Course Experiments (project_for_MLcourse)

**[English](README.md) | [简体中文](README.zh-CN.md)**

Nankai University Machine Learning course experiments. All experiments are migrated/extended from the official MATLAB examples to Python. Each subdirectory is an independent experiment with no cross-dependencies and can be run standalone. Bilingual (EN/中文) READMEs are provided in every subdirectory.

## Repository Layout

| Directory | Experiment | Description |
|---|---|---|
| [`perceptron/`](perceptron/) | Lab 1: Perceptron | Perceptron primal form (Pocket algorithm) training and decision-boundary visualization; bonuses: iteration process under different learning rates, effect of removing the misclassification check, linearly-inseparable behavior, effect of cluster-center distance, and 100-D high-dimensional extension |
| [`svm/`](svm/) | Lab 2: Support Vector Machine | Hand-written SMO solver for the SVM dual problem (linear kernel), with an Augmented Lagrangian implementation for comparison; experiments on separable / inseparable data, effect of C, soft vs. hard margin, and convergence analysis |
| [`Kmeans/`](Kmeans/) | Lab 3: K-means Clustering | Hand-written K-means (9-cluster dataset) with 10 random-initialization runs and SSE summary; bonus: 6-cluster dataset (Kmeans2) |
| [`superconductivity_regression/`](superconductivity_regression/) | Final Project: Superconductivity Critical Temperature Regression | UCI superconductivity regression; v10 five-model stacking ensemble (XGB + LGB + ET + Cubist + HGB) + XGBoost meta-learner HPO + isotonic calibration, Test RMSE 8.84 vs Baseline 10.22. See its [README](superconductivity_regression/README.md) for full details. |

## Environment

Labs 1-3 need only `numpy` and `matplotlib` (Lab 2 additionally needs `scipy`):

```bash
pip install numpy matplotlib scipy
```

Lab 4 dependencies: see [`superconductivity_regression/requirements.txt`](superconductivity_regression/requirements.txt).

## Quick Start

```bash
python perceptron/perceptron.py        # Lab 1: Perceptron
python perceptron/perceptron_add.py    # Lab 1: Bonus 1-5
python svm/svm_manual.py               # Lab 2: SVM (SMO + Augmented Lagrangian)
python Kmeans/kmeans.py                # Lab 3: K-means (incl. bonus)
```

Lab 4 training / inference: see [superconductivity_regression/README.md](superconductivity_regression/README.md#9-environment-and-usage).

## Notes

- Course slides (pptx), lab reports (docx / pdf) and similar document files are excluded from the repository via the root `.gitignore`
- `superconductivity_regression/` keeps its own `.gitignore` and `.gitattributes` (enforcing LF line endings)
