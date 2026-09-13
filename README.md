# Superconductivity Critical Temperature Regression

**[English](README.md) | [简体中文](README.zh-CN.md)**

A regression task on the UCI Superconductivity dataset: predicting a material's critical temperature (`critical_temp`) from statistical features derived from its elemental composition.
This repository is the final project for a Machine Learning course. It contains a reproduction of the official baseline and the final submitted v10 five-model stacking ensemble.

## Table of Contents

- [1. Background](#1-background)
- [2. Dataset](#2-dataset)
- [3. Evaluation Metrics](#3-evaluation-metrics)
- [4. Baseline Performance](#4-baseline-performance)
- [5. Final Approach: v10 Five-Model Stacking Ensemble](#5-final-approach-v10-five-model-stacking-ensemble)
- [6. Results](#6-results)
- [7. Version History](#7-version-history)
- [8. Repository Structure](#8-repository-structure)
- [9. Environment and Usage](#9-environment-and-usage)
- [10. Reproduction Notes and Known Discrepancies](#10-reproduction-notes-and-known-discrepancies)
- [11. References](#11-references)

---

## 1. Background

Below a certain critical temperature (Tc), superconducting materials exhibit zero electrical resistance and perfect diamagnetism. The higher the critical temperature, the greater a material's potential for practical applications — which is why the search for high-Tc materials has long been a central problem in materials science.

Traditional experimental screening is expensive and slow. Being able to predict the critical temperature from a material's composition alone would dramatically accelerate the discovery of new materials. This project applies machine learning to regress `critical_temp` from statistical features of elemental composition.

## 2. Dataset

Source: UCI Machine Learning Repository — Superconductivity Data (Hamidieh, 2018).

Two complementary tabular files are used, both with 17,010 samples:

| File | Description |
|---|---|
| `data/ml_train.csv` | Statistical features derived from the properties of 86 elements (81 features + `critical_temp`) |
| `data/unique_m_train.csv` | Elemental mole fractions of each material (86-column sparse matrix) (86 columns + `critical_temp` + `material`) |

How the 81 statistical features are constructed: for each of the 8 elemental properties (atomic mass `atomic_mass`, first ionization energy `fie`, atomic radius `atomic_radius`, density `Density`, electron affinity `ElectronAffinity`, fusion heat `FusionHeat`, thermal conductivity `ThermalConductivity`, valence `Valence`), 10 statistics are computed (`mean` / `wtd_mean` / `gmean` / `wtd_gmean` / `entropy` / `wtd_entropy` / `range` / `wtd_range` / `std` / `wtd_std`), giving 8 × 10 = 80 features, plus `number_of_elements`, for a total of 81.

**Why the problem is hard:** material properties are related through complex nonlinear interactions, and the target is strongly long-tailed — `critical_temp` has mean 34.48 K, standard deviation 34.25 K and skewness 0.855, with far more low-temperature than high-temperature samples. Models tend to make larger errors in certain temperature ranges.

## 3. Evaluation Metrics

Three regression metrics are used, with RMSE as the primary metric:

| Metric | Description |
|---|---|
| RMSE (Root Mean Squared Error) | Primary metric. Sensitive to large errors; good for checking whether the model fails badly on high-critical-temperature samples |
| MAE (Mean Absolute Error) | A robust measure of average absolute error |
| R² (Coefficient of Determination) | The proportion of variance in the target explained by the model |

## 4. Baseline Performance

The course-provided baseline model (HistGradientBoostingRegressor):

```python
HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, l2_regularization=0.1,
    early_stopping=True, random_state=42,
)
```

Performance on the official split (17,010 train / 4,253 test):

| Metric | Baseline |
|---|---|
| RMSE | 10.2201 |
| MAE | 6.2628 |
| R² | 0.9111 |

Raw output is in `baseline_results.txt`.

## 5. Final Approach: v10 Five-Model Stacking Ensemble

`train.py` implements a two-stage stacking ensemble with post-hoc calibration:

```text
                     ┌─────────────────────────────────────┐
   209 features      │  Stage 1: 5 heterogeneous base      │
   (81 statistical   │  models                             │
    + 93 derived     │  ┌───────┐ ┌────────┐ ┌───────┐    │
    + 35 element-    │  │XGBoost│ │LightGBM│ │  ET   │    │
      level)         │  └───────┘ └────────┘ └───────┘    │
        │            │  ┌──────────────┐ ┌─────────────┐  │
        ▼            │  │ Cubist       │ │    HGB      │  │
   variance filter + │  │ (ET + Ridge) │ │             │  │
   correlation prune │  └──────────────┘ └─────────────┘  │
   209 → 166         └──────────────────┬──────────────────┘
        │                               │ Repeated 5×2-fold OOF
        ▼                               ▼
   + 15 element PCA          ┌────────────────────────┐
   166 → 181                 │ Stage 2: XGBoost       │
                             │ meta-learner (25-trial)│
                             └───────────┬────────────┘
                                         ▼
                             ┌────────────────────────┐
                             │ IsotonicRegression     │
                             │ calibration            │
                             └───────────┬────────────┘
                                         ▼
                              inverse transform (sqrt)
                              → average over 3 seeds
```

### 5.1 Feature Engineering

Two families of derived features are built on top of the 81 raw statistical features:

**Statistical derived features (`build_features`, 93 features)** — ratios, interactions, and physics proxies over the elemental properties:

- Coefficient of variation and ratios: `cv_{group}`, `range_over_mean_{group}`, `gm_over_am_{group}`, `norm_entropy_{group}`, `wtd_vs_mean_{group}`
- Thermal–electrical–mechanical cross terms: `fie_x_ea`, `tc_x_dens`, `fie_x_tc`, `ea_x_tc`, `fh_x_tc`, `ar_x_fh`, `val_tc_dens_3way`, etc.
- Non-monotonic transforms: `log_mass`, `log_tc`, `log_dens`, `sqrt_tc`, `sqrt_ar`, `mass_over_valence`
- Physics-inspired proxies:
  - `mc_llan_proxy` — simplified proxy for the McMillan formula
  - `val_electron_density`, `packing_proxy` — crystal-structure proxies
  - `debye_proxy = sqrt(FusionHeat / atomic_mass)` — Debye temperature proxy (a core BCS parameter)
  - `radius_ratio_max_min` — a variant of the Goldschmidt tolerance factor
  - `ion_polar_proxy` — ionic polarizability proxy
  - `el_phonon_coupling` — electron–phonon coupling strength proxy (λ in BCS theory)
  - `band_filling` — band filling proxy
- Elemental heterogeneity: `fie_val_heterogeneity`, `fie_wtd_minus_mean`, `entropy_tc_product`

**Element-level features (`build_element_features`, 35 features)** — extracted from the 86 elemental mole-fraction columns:

- Composition statistics: `n_element_types`, `max_element_frac`, `element_concentration`, `element_entropy`, `element_entropy_norm`
- Chemical family indicators: `is_cuprate`, `is_iron_based`, `is_bcuprates`, `is_rare_earth_based`, `transition_metal_frac`, `rare_earth_frac`
- Electronegativity statistics: `max/min_electroneg_in_material`, `electronegativity_spread`
- Periodic table position: `mendeleev_mean`, `mendeleev_var`, `dom_group`, `dom_period`, `group_x_period`
- Valence electron concentration (VEC, Matthias' rule): `vec_mean`, `vec_var`

Feature selection pipeline (every fitting step is performed on the training split only, preventing data leakage):

```text
209 raw features
   │  VarianceThreshold(0.001) + correlation pruning (threshold 0.95)
   ▼
166 features
   │  concatenate 15 element-PCA components (77/86 sparse columns, 89.3% explained variance)
   ▼
181 final features
```

### 5.2 Training Strategy

| Strategy | Configuration |
|---|---|
| Target transform | Selected adaptively among `log1p` / `sqrt` / `power` (fitted on train only); `sqrt` chosen for this run |
| Data split | 90 / 10 train-test (`random_state=42`); 15,309 train / 1,701 test |
| Base models | XGBoost, LightGBM, ExtraTrees, Cubist (ExtraTrees + Ridge, two-stage), HistGradientBoosting |
| Iteration caps | XGB/LGB = 1500, HGB = 1000, ET = 800, Cubist = 600; XGB/LGB use `early_stopping_rounds=50` |
| Base-model HPO | Optuna TPE; trials XGB=25 / LGB=25 / ET=20 / Cubist=10 / HGB=10; uses StratifiedKFold over 5 Tc bins `[10, 30, 50, 77]` so every fold has a consistent temperature distribution |
| Stacking CV | RepeatedKFold(n_splits=5, n_repeats=2) — 10 folds in total, generating OOF predictions |
| Meta-learner | XGBoost, tuned by a 25-trial Optuna search on the OOF prediction matrix (`depth∈[2,4]`, `n_est∈[50,300]`, `lr∈[0.01,0.3]`, `alpha/lambda∈[0.01,50]`, `subsample/colsample∈[0.5,1.0]`), scored by an inner 5-fold CV |
| SMOGN augmentation | Resample `Tc > 50 K` samples with replacement to 2.0×, adding Gaussian noise with std = 0.5%; applied to each fold's training set only |
| Post-hoc calibration | IsotonicRegression (`out_of_bounds="clip"`), fitted on the ensemble's train predictions and applied to both train and test |
| Multi-seed | 3 seeds `[42, 123, 2026]`; final prediction is the average |
| GPU | Auto-detected via `nvidia-smi`: XGBoost uses `cuda`, LightGBM attempts `gpu` with automatic CPU fallback |

## 6. Results

### 6.1 Final Model v10 vs. Baseline

| Metric | Baseline | v10 Ensemble |
|---|---|---|
| Test RMSE | 10.2201 | 8.8406 |
| Test MAE | 6.2628 | 4.7668 |
| Test R² | 0.9111 | 0.9323 |
| Train RMSE | — | 4.6658 |
| Train MAE | — | 2.2681 |
| Train R² | — | 0.9815 |
| Test-Train gap | — | 4.1749 |
| Training time | seconds | 2,444 s (~41 min) |

Against the baseline, Test RMSE drops by roughly 1.38 (13.5%) and R² improves by roughly 2.1 percentage points.

### 6.2 Key Intermediate Metrics

Per-seed Stack OOF (out-of-fold) RMSE:

| Seed | Stack OOF RMSE | MAE | R² |
|---|---|---|---|
| 42 | 8.8761 | 5.0006 | 0.9329 |
| 123 | 8.8923 | 4.9942 | 0.9327 |
| 2026 | 9.0987 | 5.0653 | 0.9295 |

Effect of isotonic calibration on the final ensemble:

| | Train MAE | Test RMSE |
|---|---|---|
| Before calibration | 4.7841 | 8.8268 |
| After calibration | 4.6658 | 8.8406 |

Calibration clearly improves the training fit (−0.12) but slightly worsens the test set (+0.014), indicating that in this configuration isotonic regression mainly corrects a training-side systematic bias and offers limited generalization gain.

The complete per-fold, per-seed training log is written to `training_log.txt` when the script runs.

## 7. Version History

This project went through 13 iterations (`train_v1` → `train_v13`), progressively introducing elemental composition features, Mendeleev number / VEC, periodic table position, Debye temperature proxy and BCS-theory features, and comparing meta-learner designs including SLSQP constrained blending, an XGBoost nonlinear meta-learner, an ET residual corrector, and sample weighting.

The complete record of every version — design rationale, parameter changes and performance comparisons — is documented in [`docs/版本更新优化说明.md`](docs/版本更新优化说明.md) (in Chinese).

The v10 submitted here is the best-performing version of the entire series (pre-calibration Test RMSE 8.8239, the best record across all 13 versions). The deep-HPO, 5-seed and sample-weighting experiments of v11–v13 all failed to break this record; in particular, v13's sample weighting clearly regressed (Test RMSE 8.9515), confirming that the optimization objective must align with the evaluation metric (RMSE).

## 8. Repository Structure

```text
.
├── README.md                     # English documentation (this document; GitHub renders this on the homepage)
├── README.zh-CN.md               # Chinese documentation
├── docs/
│   └── 版本更新优化说明.md       # Detailed experiment log of versions v1–v13 (Chinese)
├── train.py                      # Final model: 5-model stacking + meta-learner HPO + isotonic calibration
├── test.py                       # Inference script (pairs with the model.pkl saved by train.py)
├── requirements.txt              # Dependencies
├── baseline_results.txt          # Raw baseline output
└── .gitignore
```

Not included in the repository (see `.gitignore`):

- `data/` — Dataset; download from UCI
- `hpo_cache_v10/` — Hyperparameter search cache, generated automatically when `train.py` runs
- `model.pkl` — Trained model, ~850 MB, exceeding GitHub's 100 MB per-file limit
- `training_log.txt` — Training log, generated automatically after a run

## 9. Environment and Usage

### 9.1 Dependencies

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
xgboost>=1.6
lightgbm>=3.3
optuna>=3.0
scipy>=1.7
joblib>=1.1
```

### 9.2 Data Preparation

Download from UCI Superconductivity Data and arrange as:

```text
data/
├── ml_train.csv
└── unique_m_train.csv
```

### 9.3 Training

```bash
python train.py --train_data data/ml_train.csv --elem_data data/unique_m_train.csv
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `model_v10.pkl` | Output path for the trained model |
| `--n_trials` | 35 | Maximum HPO trials (per-model overrides apply) |
| `--no_cache` | off | Skip the HPO cache and re-run all hyperparameter searches |

The first run performs the full HPO search; subsequent runs load from `hpo_cache_v10/`, cutting training time substantially.

### 9.4 Inference

```bash
python test.py --test_data data/ml_test.csv \
               --elem_data data/unique_m_test.csv \
               --model_path model.pkl \
               --pred_output predictions.csv
```

Both `--test_data` and `--elem_data` are required (the model was trained on statistical and elemental-composition features jointly). If the test data contains a `critical_temp` column, the script additionally reports RMSE / MAE / R² plus a per-temperature-bin breakdown (<10K / 10–30K / 30–77K / >77K).

## 10. Reproduction Notes and Known Discrepancies

**`test.py` and `train.py` average over seeds in different spaces.**
`train.py` inverse-transforms each seed first and then averages in the original space; `test.py` averages in the transformed space and inverse-transforms once at the end. The difference is exactly the cross-seed variance in sqrt space: `mean(p) − (mean√p)² = Var(√p) ≥ 0`. Consequently `test.py`'s output is systematically slightly lower and will not reproduce `train.py`'s reported Test RMSE digit-for-digit. Since the per-seed predictions are close to one another, the magnitude is small.

**`inverse_transform` behaves differently on the `power` branch.**
`train.py` raises `ValueError` when no `PowerTransformer` is supplied, whereas `test.py` falls back to plain clipping. This run uses the `sqrt` transform, so that branch never executes and the difference does not affect results.

**Runtime and randomness.** Optuna search and early stopping are stochastic, so repeated runs yield slightly different results. The v10 result recorded in `docs/版本更新优化说明.md` is Test RMSE 8.8345 (pre-calibration 8.8239); the figures used in this document (8.8406 / pre-calibration 8.8268) come from a different run of the same configuration.

**One piece of dead code in `train.py`.** The `n_total` variable is assigned but never used, and can be safely removed.

**A sizable generalization gap.** The Test-Train gap is 4.17, so the model still overfits noticeably. This motivated the deep-HPO, residual-corrector and sample-weighting experiments in v11, v12 and v13, none of which achieved a substantive breakthrough in Test RMSE — evidence that this framework has reached the information ceiling of the dataset.

## 11. References

- Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. *Computational Materials Science*, 154, 346–354.
- UCI Machine Learning Repository — Superconductivity Data
- McMillan, W. L. (1968). Transition temperature of strong-coupled superconductors. *Physical Review*, 167(2), 331.
