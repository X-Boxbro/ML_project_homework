# K-means Clustering Experiment (Lab 3)

**[English](README.md) | [简体中文](README.zh-CN.md)**

Nankai University Machine Learning Lab 3: K-means clustering. A hand-written K-means implementation (alternating assignment and update steps, convergence judged by centroid shift distance) is evaluated on a 9-cluster rectangular-distribution dataset over 10 independent random-initialization runs, with SSE summarized across runs. The bonus question repeats the same procedure on a 6-cluster dataset (matching MATLAB `Kmeans2_student.m`). A fixed sample-generation seed makes the dataset reproducible; results across initializations differ as expected.

## Table of Contents

- [1. Background](#1-background)
- [2. Dataset](#2-dataset)
- [3. Algorithm](#3-algorithm)
- [4. Experiments and Results](#4-experiments-and-results)
  - [4.1 Main Experiment: 9-Cluster Dataset, 10 Random Initializations](#41-main-experiment-9-cluster-dataset-10-random-initializations)
  - [4.2 Bonus: 6-Cluster Dataset (Kmeans2)](#42-bonus-6-cluster-dataset-kmeans2)
- [5. Reproduction Notes and Known Issues](#5-reproduction-notes-and-known-issues)
- [6. Repository Structure](#6-repository-structure)
- [7. Environment and Usage](#7-environment-and-usage)
- [8. References](#8-references)

---

## 1. Background

K-means is one of the most basic unsupervised clustering algorithms. Its goal is to partition the data into K clusters so that the sum of squared distances from each sample to its assigned cluster center (i.e., Inertia / SSE) is minimized. The algorithm alternates between an "assignment" step and an "update" step and typically converges quickly to a local optimum. This lab implements K-means from scratch on 2D regularly-distributed data and uses multiple random-initialization runs to illustrate how initialization affects final clustering quality.

## 2. Dataset

Two 2D synthetic datasets, both obtained by uniformly sampling points and retaining only those falling inside a set of rectangular regions (discarding points in the white gaps):

### Main experiment dataset (9 clusters)

2000 candidate points are uniformly generated in `[0,10] × [0,10]`; only points falling into the following 9 rectangles are retained:

| Region ID | x range | y range |
|---|---|---|
| 1 | (0, 3) | (0, 3) |
| 2 | (0, 3) | (3.5, 6.5) |
| 3 | (0, 3) | (7, 10) |
| 4 | (3.5, 6.5) | (0, 3) |
| 5 | (3.5, 6.5) | (3.5, 6.5) |
| 6 | (3.5, 6.5) | (7, 10) |
| 7 | (7, 10) | (0, 3) |
| 8 | (7, 10) | (3.5, 6.5) |
| 9 | (7, 10) | (7, 10) |

These regions correspond to the ideal 9 clusters (shown only as a visual reference; K-means itself does not use the ground-truth labels).

### Bonus dataset (6 clusters, Kmeans2)

Again 2000 candidate points are uniformly sampled in `[0,10] × [0,10]`; points inside the following 6 rectangles are retained:

| Region ID | x range | y range |
|---|---|---|
| 1 | (0, 6) | (0, 6) |
| 2 | (7, 10) | (0, 3) |
| 3 | (7, 10) | (3, 6) |
| 4 | (0, 3) | (7, 10) |
| 5 | (3, 6) | (7, 10) |
| 6 | (7, 10) | (7, 10) |

## 3. Algorithm

**Input**: data matrix X (n × d), number of clusters K, maximum iterations max_iters, convergence tolerance tol, random seed seed.

**Initialization**: randomly select K sample points (without replacement) as the initial centroids.

**Iteration (up to max_iters times)**:

1. **Assignment step**: for each sample, compute its Euclidean distance to all K centroids and assign it to the cluster of the nearest centroid.
2. **Update step**: for each cluster, compute the mean of all its members and use it as the new centroid. If a cluster becomes empty, pick a random sample as its centroid (to avoid a crash).
3. **Convergence check**: if the norm of the shift of all centroids is below tol, terminate early.

**Objective function (Inertia / SSE)**:

```text
SSE = Σ_{i=1}^{n} ‖x_i − μ_{c(i)}‖²
```

where μ_{c(i)} is the centroid of the cluster to which sample x_i is assigned.

**Limitations**:
- Results depend on initialization; different initializations may converge to different local optima (demonstrated in this lab via 10 repeated runs).
- Requires the user to specify K in advance.
- Sensitive to noise and outliers (the mean is easily pulled by extreme points).

## 4. Experiments and Results

### 4.1 Main Experiment: 9-Cluster Dataset, 10 Random Initializations

On the 9-cluster dataset with K=9, keep the sample-generation seed fixed at 42 (so every run uses the same set of points), and run K-means 10 times with initialization seeds 0 through 9. Record the iteration count and final SSE of each run, and produce:

- Ground-truth reference plot: `kmeans_ground_truth.png` (colored by region, for visual reference only; not used by the algorithm).
- 10-run summary plot: `kmeans_results_10_experiments.png` (2×5 subplot grid, each subplot showing one run's clustering result and centroids).
- Summary statistics table: `kmeans_summary_table.png` (experiment ID, iteration count, SSE for each run, plus the best/worst/median SSE summary).

Terminal output (identical across runs because the initialization seeds are fixed):

```
Experiment  1: iterations=  5, SSE=XXXX.XX
Experiment  2: iterations=  4, SSE=XXXX.XX
...
```

### 4.2 Bonus: 6-Cluster Dataset (Kmeans2)

Switch to the 6-cluster dataset with K=6 and repeat the same 10-random-initialization procedure (initialization seeds 0–9):

- Ground-truth reference plot: `kmeans2_ground_truth.png`.
- 10-run summary plot: `kmeans2_results_10_experiments.png` (2×5 subplot grid).
- Terminal printout of per-run iteration count and SSE.

## 5. Reproduction Notes and Known Issues

- **Sample generation uses a fixed seed=42**, so every run uses the same points; however, the initialization seed cycles through 0–9 across the 10 runs, so different initializations yield different clusterings and SSE values — this is the intended behavior, used to demonstrate initialization sensitivity.
- Images are saved at 150 dpi to the project root directory (`BASE_DIR` points to the repository root).
- K-means itself does not use the ground-truth labels; the "ground-truth reference plot" is for visual comparison only and does not enter the algorithm.

## 6. Repository Structure

```text
Kmeans/
├── README.md             # English documentation (rendered on the GitHub homepage)
├── README.zh-CN.md       # Chinese documentation (this document's counterpart)
├── Kmeans_example.m      # Course-provided MATLAB reference example (9 clusters)
├── Kmeans2_example.m     # Course-provided MATLAB reference example (6 clusters, bonus)
└── kmeans.py             # Hand-written K-means: main experiment + bonus
```

Images generated at runtime (`kmeans*.png`, `kmeans2*.png`) are not committed; see the root `.gitignore`.

## 7. Environment and Usage

```bash
pip install numpy matplotlib
python Kmeans/kmeans.py
```

In a headless environment (server / CI), run `export MPLBACKEND=Agg` first. The script runs both the main experiment and the bonus question in one execution and prints per-run iteration counts and SSE to the terminal.

## 8. References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
- Li Hang. *Statistical Learning Methods (2nd Edition)*, Chapter 9: Clustering.
- Course-provided MATLAB examples `Kmeans_example.m`, `Kmeans2_example.m`.
