超导临界温度预测（Superconductivity Critical Temperature Regression）

基于 UCI 超导材料数据集的回归任务：由材料的元素组成统计特征预测其临界温度 `critical_temp`。
本仓库为机器学习课程期末大作业，包含官方 Baseline 的复现，以及最终提交的 v10 五模型堆叠（Stacking）集成模型。
English Version
目录
	1. 任务背景
	2. 数据集
	3. 评估指标
	4. Baseline 性能
	5. 最终方案：v10 五模型堆叠集成
	6. 实验结果
	7. 版本迭代记录
	8. 仓库结构
	9. 环境与运行
	10. 复现说明与已知差异
	11. 参考
1. 任务背景
	超导材料在低于某一临界温度（Critical Temperature, Tc）时会表现出零电阻和完全抗磁性等特殊性质。临界温度越高，材料在实际应用中的潜力通常越大，因此寻找高临界温度材料一直是材料科学中的核心问题。
	传统实验筛选成本高、周期长。若能利用已有材料的组成信息预测临界温度，就可以大幅加速新材料的发现过程。本任务即在此背景下，用机器学习方法从材料的元素组成统计特征回归预测 `critical_temp`。
2. 数据集
	来源：UCI Machine Learning Repository — Superconductivity Data（Hamidieh, 2018）。
	本实验使用两个互补的表格文件，均为 17010 条样本：
	
		`data/ml_train.csv`
			由 86 种元素性质派生出的统计特征
			81 维特征 + `critical_temp`
		`data/unique_m_train.csv`
			材料的元素组成摩尔分数（86 列稀疏矩阵）
			86 列 + `critical_temp` + `material`
	81 维统计特征的构造方式：对 8 类元素属性
（原子质量 `atomic_mass`、第一电离能 `fie`、原子半径 `atomic_radius`、密度 `Density`、电子亲和能 `ElectronAffinity`、熔化热 `FusionHeat`、热导率 `ThermalConductivity`、价电子数 `Valence`）
各计算 10 个统计量
（`mean` / `wtd_mean` / `gmean` / `wtd_gmean` / `entropy` / `wtd_entropy` / `range` / `wtd_range` / `std` / `wtd_std`），
共 8 × 10 = 80 维，再加 1 维 `number_of_elements`，合计 81 维。
	问题难点：材料性质之间存在复杂的非线性关系；且目标分布长尾显著——`critical_temp` 均值 34.48 K、标准差 34.25 K，偏度 0.855，低温样本远多于高温样本，模型在某些温度区间容易出现较大预测误差。
3. 评估指标
	本实验采用三种回归评估指标，主指标为 RMSE：
	
		RMSE（均方根误差）
			主指标。对大误差敏感，适合检查模型在高临界温度样本上是否严重失误
		MAE（平均绝对误差）
			稳健地反映平均绝对误差大小
		R²（决定系数）
			衡量模型对目标变量方差的解释比例
4. Baseline 性能
	课程提供的 Baseline 模型（HistGradientBoostingRegressor）：
	```python
HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, l2_regularization=0.1,
    early_stopping=True, random_state=42,
)
```
	在官方划分（Train 17010 条 / Test 4253 条）上的性能：
	
		RMSE
			10.2201
		MAE
			6.2628
		R²
			0.9111
	原始输出见 `baseline_results.txt`。
5. 最终方案：v10 五模型堆叠集成
	`train.py` 实现的是一个两阶段堆叠集成 + 后校准的架构：
	```
                     ┌─────────────────────────────────────┐
   209 维特征         │  Stage 1：5 个异构基模型            │
   （81 统计          │  ┌───────┐ ┌────────┐ ┌───────┐    │
     + 93 派生        │  │XGBoost│ │LightGBM│ │  ET   │    │
     + 35 元素级）    │  └───────┘ └────────┘ └───────┘    │
        │             │  ┌──────────────┐ ┌─────────────┐  │
        ▼             │  │ Cubist       │ │    HGB      │  │
   方差过滤 + 相关性   │  │ (ET + Ridge) │ │             │  │
   209 → 166          │  └──────────────┘ └─────────────┘  │
        │             └──────────────────┬──────────────────┘
        ▼                                │ Repeated 5×2-fold OOF
   + 15 维元素 PCA                        ▼
   166 → 181                   ┌────────────────────────┐
                               │ Stage 2：XGBoost       │
                               │ 元学习器（25-trial）   │
                               └───────────┬────────────┘
                                           ▼
                               ┌────────────────────────┐
                               │ IsotonicRegression 校准 │
                               └───────────┬────────────┘
                                           ▼
                                逆变换（sqrt）→ 3 种子平均
```
	5.1 特征工程
		在 81 维原始统计特征基础上，构建了两大类派生特征：
		统计派生特征（`build_features`，93 个）——围绕元素属性的比率、交互与物理代理量：
		变异系数与比率：`cv_{group}`、`range_over_mean_{group}`、`gm_over_am_{group}`、`norm_entropy_{group}`、`wtd_vs_mean_{group}`
		热-电-机械交叉项：`fie_x_ea`、`tc_x_dens`、`fie_x_tc`、`ea_x_tc`、`fh_x_tc`、`ar_x_fh`、`val_tc_dens_3way` 等
		非单调变换：`log_mass`、`log_tc`、`log_dens`、`sqrt_tc`、`sqrt_ar`、`mass_over_valence`
		物理启发式代理量：
			`mc_llan_proxy` —— McMillan 公式的简化代理
			`val_electron_density`、`packing_proxy` —— 晶体结构代理量
			`debye_proxy = sqrt(FusionHeat / atomic_mass)` —— 德拜温度代理（BCS 理论核心参数）
			`radius_ratio_max_min` —— Goldschmidt 容差因子变体
			`ion_polar_proxy` —— 离子极化率代理
			`el_phonon_coupling` —— 电子-声子耦合强度代理（BCS 理论中的 λ）
			`band_filling` —— 能带填充代理
		元素异质性：`fie_val_heterogeneity`、`fie_wtd_minus_mean`、`entropy_tc_product`
		元素级特征（`build_element_features`，35 个）——从 86 列元素摩尔分数中提取：
		组成统计：`n_element_types`、`max_element_frac`、`element_concentration`、`element_entropy`、`element_entropy_norm`
		化学家族标识：`is_cuprate`、`is_iron_based`、`is_bcuprates`、`is_rare_earth_based`、`transition_metal_frac`、`rare_earth_frac`
		电负性统计：`max/min_electroneg_in_material`、`electronegativity_spread`
		周期表位置：`mendeleev_mean`、`mendeleev_var`、`dom_group`、`dom_period`、`group_x_period`
		价电子浓度（VEC，Matthias 定则）：`vec_mean`、`vec_var`
		特征选择流程（所有拟合步骤仅在 Train 上进行，避免信息泄漏）：
		```
209 维原始特征
   │  VarianceThreshold(0.001) + 相关性剔除（阈值 0.95）
   ▼
166 维
   │  拼接 15 维元素 PCA 分量（77/86 稀疏列，解释方差 89.3%）
   ▼
181 维最终特征
```
	5.2 训练策略
		目标变换
			在 `log1p` / `sqrt` / `power` 中自适应选择（仅用 Train 拟合），本次选中 `sqrt`
		数据划分
			90 / 10 Train-Test（`random_state=42`），Train 15309 条 / Test 1701 条
		基模型
			XGBoost、LightGBM、ExtraTrees、Cubist（ExtraTrees + Ridge 两阶段）、HistGradientBoosting
		迭代上限
			XGB/LGB = 1500，HGB = 1000，ET = 800，Cubist = 600；XGB/LGB 配 `early_stopping_rounds=50`
		基模型 HPO
			Optuna TPE，trial 数 XGB=25 / LGB=25 / ET=20 / Cubist=10 / HGB=10；使用 StratifiedKFold（按 Tc 分 5 箱 `[10, 30, 50, 77]`）保证每折温度分布一致
		堆叠 CV
			RepeatedKFold(n_splits=5, n_repeats=2)，共 10 折生成 OOF 预测
		元学习器
			XGBoost，在 OOF 预测矩阵上做 25-trial Optuna 搜索（`depth∈[2,4]`、`n_est∈[50,300]`、`lr∈[0.01,0.3]`、`alpha/lambda∈[0.01,50]`、`subsample/colsample∈[0.5,1.0]`），内部 5-fold CV 评估
		SMOGN 增广
			对 `Tc > 50 K` 样本有放回重采样至 2.0 倍，加入 std = 0.5% 的高斯噪声；仅在每折训练集上应用
		后校准
			IsotonicRegression（`out_of_bounds="clip"`），在集成的 Train 预测上拟合，同时作用于 Train 与 Test
		多种子
			3 个种子 `[42, 123, 2026]`，最终预测取平均
		GPU
			自动检测（`nvidia-smi`）：XGB 用 `cuda`，LightGBM 尝试 `gpu`，失败自动回退 CPU
6. 实验结果
	6.1 最终模型 v10 vs. Baseline
		
			Test RMSE
				10.2201
				8.8406
			Test MAE
				6.2628
				4.7668
			Test R²
				0.9111
				0.9323
			Train RMSE
				—
				4.6658
			Train MAE
				—
				2.2681
			Train R²
				—
				0.9815
			Test-Train Gap
				—
				4.1749
			训练耗时
				秒级
				2444 s（约 41 min）
		相对 Baseline，Test RMSE 下降约 1.38（13.5%），R² 提升约 2.1 个百分点。
	6.2 关键过程指标
		各单种子的 Stack OOF（折外）RMSE：
		
			42
				8.8761
				5.0006
				0.9329
			123
				8.8923
				4.9942
				0.9327
			2026
				9.0987
				5.0653
				0.9295
		等渗校准对最终集成的影响：
		
			校准前
				4.7841
				8.8268
			校准后
				4.6658
				8.8406
		校准显著改善了训练集拟合（−0.12），但测试集略微变差（+0.014），说明在该配置下等渗回归主要修正了训练侧的系统性偏差，对泛化的增益有限。
		完整逐折、逐种子的训练日志见运行后生成的 `training_log.txt`。
7. 版本迭代记录
	本项目从 `train_v1` 到 `train_v12` 共迭代了 12 个版本，逐步引入元素组成特征、Mendeleev 数 / VEC、周期表位置、德拜温度代理、BCS 理论特征等，并比较了 SLSQP 约束混合、XGBoost 非线性元学习器、ET 残差修正等多种元学习器方案。
	每个版本的完整实验记录、设计动机、参数变更与性能对比见 `版本更新优化说明.txt`。
	本仓库提交的 v10 为全系列性能最优版本（校准前 Test RMSE 8.8239，为 12 个版本中的最佳记录）。
8. 仓库结构
	```
.
├── train.py                       # 最终模型：5 模型堆叠集成 + 元学习器 HPO + 等渗校准
├── test.py                        # 推理脚本（与 train.py 保存的 model.pkl 配套）
├── requirements.txt               # 依赖清单
├── baseline_results.txt           # Baseline 运行的原始输出
├── 版本更新优化说明.txt            # 12 个版本的详细实验记录与分析
├── README.md                      # 本文档
├── README_EN.md                   # 英文版
└── .gitignore
```
	未包含在仓库中的内容（见 `.gitignore`）：
	
		`data/`
			数据集，需从 UCI 自行下载
		`hpo_cache_v10/`
			超参数搜索缓存，运行 `train.py` 时自动生成
		`model.pkl`
			训练好的模型，约 850 MB，超出 GitHub 单文件 100 MB 限制
		`training_log.txt`
			训练日志，运行后自动生成
9. 环境与运行
	9.1 依赖
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
	9.2 数据准备
		从 UCI Superconductivity Data 下载后，放置为：
		```
data/
├── ml_train.csv
└── unique_m_train.csv
```
	9.3 训练
		```bash
python train.py --train_data data/ml_train.csv --elem_data data/unique_m_train.csv
```
		可选参数：
		
			`--model_path`
				`model_v10.pkl`
				模型输出路径
			`--n_trials`
				35
				HPO 最大 trial 数（各模型有独立覆盖值）
			`--no_cache`
				关闭
				跳过 HPO 缓存，重新执行全部超参数搜索
		首次运行会执行完整的 HPO 搜索；之后从 `hpo_cache_v10/` 加载缓存，训练时间大幅缩短。
	9.4 推理
		```bash
python test.py --test_data data/ml_test.csv \
               --elem_data data/unique_m_test.csv \
               --model_path model.pkl \
               --pred_output predictions.csv
```
		`--test_data` 与 `--elem_data` 均为必填（模型在训练时同时使用了统计特征与元素组成特征）。若测试数据包含 `critical_temp` 列，脚本会额外输出 RMSE / MAE / R² 及分温度区间（<10K / 10–30K / 30–77K / >77K）的误差明细。
10. 复现说明与已知差异
	`test.py` 与 `train.py` 的多种子平均发生在不同空间。
`train.py` 对每个种子先逆变换、再在原始空间取平均；`test.py` 先在变换空间取平均、再统一逆变换。两者之差恰好等于各种子预测在 sqrt 空间上的方差 `mean(p) − (mean√p)² = Var(√p) ≥ 0`，因此 `test.py` 的输出会系统性地略低，不会与 `train.py` 报告的 Test RMSE 逐位相同。由于各种子预测彼此接近，该差异量级较小。
	`inverse_transform` 在 `power` 变换分支上行为不同。
`train.py` 在缺少 `PowerTransformer` 时抛 `ValueError`，`test.py` 则退化为直接裁剪。本次训练实际使用 `sqrt` 变换，该分支不会执行，故不影响结果。
	训练耗时与随机性。 Optuna 搜索与早停带有随机性，重复运行会得到略有差异的结果。`版本更新优化说明.txt` 中记录的 v10 结果为 Test RMSE 8.8345（校准前 8.8239），与本文档采用的本次运行结果 8.8406（校准前 8.8268）来自同一配置的不同次运行。
	`train.py` 中存在一处死代码。 `n_total` 变量赋值后未被使用，可安全删除。
	泛化间隙较大。 Test-Train gap 为 4.17，模型仍存在明显过拟合，这也是后续版本（v11、v12）尝试深度 HPO 与残差修正的动机；但两者均未能在 Test RMSE 上取得实质突破，表明该框架已接近此数据集的信息上限。
11. 参考
	Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. Computational Materials Science, 154, 346–354.
	UCI Machine Learning Repository — Superconductivity Data
	McMillan, W. L. (1968). Transition temperature of strong-coupled superconductors. Physical Review, 167(2), 331.

English Version
A regression task on the UCI Superconductivity dataset: predicting a material's critical temperature (`critical_temp`) from statistical features derived from its elemental composition.
This repository is the final project for a Machine Learning course. It contains a reproduction of the official baseline and the final submitted v10 five-model stacking ensemble.

Table of Contents
	1. Background
	2. Dataset
	3. Evaluation Metrics
	4. Baseline Performance
	5. Final Approach: v10 Five-Model Stacking Ensemble
	6. Results
	7. Version History
	8. Repository Structure
	9. Environment and Usage
	10. Reproduction Notes and Known Discrepancies
	11. References
1. Background
	Below a certain critical temperature (Tc), superconducting materials exhibit zero electrical resistance and perfect diamagnetism. The higher the critical temperature, the greater a material's potential for practical applications — which is why the search for high-Tc materials has long been a central problem in materials science.
	Traditional experimental screening is expensive and slow. Being able to predict the critical temperature from a material's composition alone would dramatically accelerate the discovery of new materials. This project applies machine learning to regress `critical_temp` from statistical features of elemental composition.
2. Dataset
	Source: UCI Machine Learning Repository — Superconductivity Data (Hamidieh, 2018).
	Two complementary tabular files are used, both with 17,010 samples:
	
		`data/ml_train.csv`
			Statistical features derived from the properties of 86 elements
			81 features + `critical_temp`
		`data/unique_m_train.csv`
			Elemental mole fractions of each material (86-column sparse matrix)
			86 columns + `critical_temp` + `material`
	How the 81 statistical features are constructed: for each of 8 elemental properties
(atomic mass `atomic_mass`, first ionization energy `fie`, atomic radius `atomic_radius`, density `Density`, electron affinity `ElectronAffinity`, fusion heat `FusionHeat`, thermal conductivity `ThermalConductivity`, valence `Valence`),
10 statistics are computed
(`mean` / `wtd_mean` / `gmean` / `wtd_gmean` / `entropy` / `wtd_entropy` / `range` / `wtd_range` / `std` / `wtd_std`),
giving 8 × 10 = 80 features, plus `number_of_elements`, for a total of 81.
	Why the problem is hard: material properties are related through complex nonlinear interactions, and the target is strongly long-tailed — `critical_temp` has mean 34.48 K, standard deviation 34.25 K and skewness 0.855, with far more low-temperature than high-temperature samples. Models tend to make larger errors in certain temperature ranges.
3. Evaluation Metrics
	Three regression metrics are used, with RMSE as the primary metric:
	
		RMSE (Root Mean Squared Error)
			Primary metric. Sensitive to large errors; good for checking whether the model fails badly on high-critical-temperature samples
		MAE (Mean Absolute Error)
			A robust measure of average absolute error
		R² (Coefficient of Determination)
			The proportion of variance in the target explained by the model
4. Baseline Performance
	The course-provided baseline model (HistGradientBoostingRegressor):
	```python
HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, l2_regularization=0.1,
    early_stopping=True, random_state=42,
)
```
	Performance on the official split (17,010 train / 4,253 test):
	
		RMSE
			10.2201
		MAE
			6.2628
		R²
			0.9111
	Raw output is in `baseline_results.txt`.
5. Final Approach: v10 Five-Model Stacking Ensemble
	`train.py` implements a two-stage stacking ensemble with post-hoc calibration:
	```
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
	5.1 Feature Engineering
		Two families of derived features are built on top of the 81 raw statistical features:
		Statistical derived features (`build_features`, 93 features) — ratios, interactions, and physics proxies over the elemental properties:
		Coefficient of variation and ratios: `cv_{group}`, `range_over_mean_{group}`, `gm_over_am_{group}`, `norm_entropy_{group}`, `wtd_vs_mean_{group}`
		Thermal–electrical–mechanical cross terms: `fie_x_ea`, `tc_x_dens`, `fie_x_tc`, `ea_x_tc`, `fh_x_tc`, `ar_x_fh`, `val_tc_dens_3way`, etc.
		Non-monotonic transforms: `log_mass`, `log_tc`, `log_dens`, `sqrt_tc`, `sqrt_ar`, `mass_over_valence`
		Physics-inspired proxies:
			`mc_llan_proxy` — simplified proxy for the McMillan formula
			`val_electron_density`, `packing_proxy` — crystal-structure proxies
			`debye_proxy = sqrt(FusionHeat / atomic_mass)` — Debye temperature proxy (a core BCS parameter)
			`radius_ratio_max_min` — a variant of the Goldschmidt tolerance factor
			`ion_polar_proxy` — ionic polarizability proxy
			`el_phonon_coupling` — electron–phonon coupling strength proxy (λ in BCS theory)
			`band_filling` — band filling proxy
		Elemental heterogeneity: `fie_val_heterogeneity`, `fie_wtd_minus_mean`, `entropy_tc_product`
		Element-level features (`build_element_features`, 35 features) — extracted from the 86 elemental mole-fraction columns:
		Composition statistics: `n_element_types`, `max_element_frac`, `element_concentration`, `element_entropy`, `element_entropy_norm`
		Chemical family indicators: `is_cuprate`, `is_iron_based`, `is_bcuprates`, `is_rare_earth_based`, `transition_metal_frac`, `rare_earth_frac`
		Electronegativity statistics: `max/min_electroneg_in_material`, `electronegativity_spread`
		Periodic table position: `mendeleev_mean`, `mendeleev_var`, `dom_group`, `dom_period`, `group_x_period`
		Valence electron concentration (VEC, Matthias' rule): `vec_mean`, `vec_var`
		Feature selection pipeline (every fitting step is performed on the training split only, preventing data leakage):
		```
209 raw features
   │  VarianceThreshold(0.001) + correlation pruning (threshold 0.95)
   ▼
166 features
   │  concatenate 15 element-PCA components (77/86 sparse columns, 89.3% explained variance)
   ▼
181 final features
```
	5.2 Training Strategy
		Target transform
			Selected adaptively among `log1p` / `sqrt` / `power` (fitted on train only); `sqrt` chosen for this run
		Data split
			90 / 10 train-test (`random_state=42`); 15,309 train / 1,701 test
		Base models
			XGBoost, LightGBM, ExtraTrees, Cubist (ExtraTrees + Ridge, two-stage), HistGradientBoosting
		Iteration caps
			XGB/LGB = 1500, HGB = 1000, ET = 800, Cubist = 600; XGB/LGB use `early_stopping_rounds=50`
		Base-model HPO
			Optuna TPE; trials XGB=25 / LGB=25 / ET=20 / Cubist=10 / HGB=10; uses StratifiedKFold over 5 Tc bins `[10, 30, 50, 77]` so every fold has a consistent temperature distribution
		Stacking CV
			RepeatedKFold(n_splits=5, n_repeats=2) — 10 folds in total, generating OOF predictions
		Meta-learner
			XGBoost, tuned by a 25-trial Optuna search on the OOF prediction matrix (`depth∈[2,4]`, `n_est∈[50,300]`, `lr∈[0.01,0.3]`, `alpha/lambda∈[0.01,50]`, `subsample/colsample∈[0.5,1.0]`), scored by an inner 5-fold CV
		SMOGN augmentation
			Resample `Tc > 50 K` samples with replacement to 2.0×, adding Gaussian noise with std = 0.5%; applied to each fold's training set only
		Post-hoc calibration
			IsotonicRegression (`out_of_bounds="clip"`), fitted on the ensemble's train predictions and applied to both train and test
		Multi-seed
			3 seeds `[42, 123, 2026]`; final prediction is the average
		GPU
			Auto-detected via `nvidia-smi`: XGBoost uses `cuda`, LightGBM attempts `gpu` with automatic CPU fallback
6. Results
	6.1 Final Model v10 vs. Baseline
		
			Test RMSE
				10.2201
				8.8406
			Test MAE
				6.2628
				4.7668
			Test R²
				0.9111
				0.9323
			Train RMSE
				—
				4.6658
			Train MAE
				—
				2.2681
			Train R²
				—
				0.9815
			Test-Train gap
				—
				4.1749
			Training time
				seconds
				2,444 s (~41 min)
		Against the baseline, Test RMSE drops by roughly 1.38 (13.5%) and R² improves by roughly 2.1 percentage points.
	6.2 Key Intermediate Metrics
		Per-seed Stack OOF (out-of-fold) RMSE:
		
			42
				8.8761
				5.0006
				0.9329
			123
				8.8923
				4.9942
				0.9327
			2026
				9.0987
				5.0653
				0.9295
		Effect of isotonic calibration on the final ensemble:
		
			Before calibration
				4.7841
				8.8268
			After calibration
				4.6658
				8.8406
		Calibration clearly improves the training fit (−0.12) but slightly worsens the test set (+0.014), indicating that in this configuration isotonic regression mainly corrects a training-side systematic bias and offers limited generalization gain.
		The complete per-fold, per-seed training log is written to `training_log.txt` when the script runs.
7. Version History
	This project went through 12 iterations (`train_v1` → `train_v12`), progressively introducing elemental composition features, Mendeleev number / VEC, periodic table position, Debye temperature proxy and BCS-theory features, and comparing meta-learner designs including SLSQP constrained blending, an XGBoost nonlinear meta-learner, and an ET residual corrector.
	The complete record of every version — design rationale, parameter changes and performance comparisons — is documented in `版本更新优化说明.txt` (in Chinese).
	The v10 submitted here is the best-performing version of the entire series (pre-calibration Test RMSE 8.8239, the best record across all 12 versions).
8. Repository Structure
	```
.
├── train.py                       # Final model: 5-model stacking + meta-learner HPO + isotonic calibration
├── test.py                        # Inference script (pairs with the model.pkl saved by train.py)
├── requirements.txt               # Dependencies
├── baseline_results.txt           # Raw baseline output
├── 版本更新优化说明.txt            # Detailed record and analysis of all 12 versions
├── README.md                      # Chinese documentation
├── README_EN.md                   # This document
└── .gitignore
```
	Not included in the repository (see `.gitignore`):
	
		`data/`
			Dataset; download from UCI
		`hpo_cache_v10/`
			Hyperparameter search cache, generated automatically when `train.py` runs
		`model.pkl`
			Trained model, ~850 MB, exceeding GitHub's 100 MB per-file limit
		`training_log.txt`
			Training log, generated automatically after a run
9. Environment and Usage
	9.1 Dependencies
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
	9.2 Data Preparation
		Download from UCI Superconductivity Data and arrange as:
		```
data/
├── ml_train.csv
└── unique_m_train.csv
```
	9.3 Training
		```bash
python train.py --train_data data/ml_train.csv --elem_data data/unique_m_train.csv
```
		Optional arguments:
		
			`--model_path`
				`model_v10.pkl`
				Output path for the trained model
			`--n_trials`
				35
				Maximum HPO trials (per-model overrides apply)
			`--no_cache`
				off
				Skip the HPO cache and re-run all hyperparameter searches
		The first run performs the full HPO search; subsequent runs load from `hpo_cache_v10/`, cutting training time substantially.
	9.4 Inference
		```bash
python test.py --test_data data/ml_test.csv \
               --elem_data data/unique_m_test.csv \
               --model_path model.pkl \
               --pred_output predictions.csv
```
		Both `--test_data` and `--elem_data` are required (the model was trained on statistical and elemental-composition features jointly). If the test data contains a `critical_temp` column, the script additionally reports RMSE / MAE / R² plus a per-temperature-bin breakdown (<10K / 10–30K / 30–77K / >77K).
10. Reproduction Notes and Known Discrepancies
	`test.py` and `train.py` average over seeds in different spaces.
`train.py` inverse-transforms each seed first and then averages in the original space; `test.py` averages in the transformed space and inverse-transforms once at the end. The difference is exactly the cross-seed variance in sqrt space: `mean(p) − (mean√p)² = Var(√p) ≥ 0`. Consequently `test.py`'s output is systematically slightly lower and will not reproduce `train.py`'s reported Test RMSE digit-for-digit. Since the per-seed predictions are close to one another, the magnitude is small.
	`inverse_transform` behaves differently on the `power` branch.
`train.py` raises `ValueError` when no `PowerTransformer` is supplied, whereas `test.py` falls back to plain clipping. This run uses the `sqrt` transform, so that branch never executes and the difference does not affect results.
	Runtime and randomness. Optuna search and early stopping are stochastic, so repeated runs yield slightly different results. The v10 result recorded in `版本更新优化说明.txt` is Test RMSE 8.8345 (pre-calibration 8.8239); the figures used in this document (8.8406 / pre-calibration 8.8268) come from a different run of the same configuration.
	One piece of dead code in `train.py`. The `n_total` variable is assigned but never used, and can be safely removed.
	A sizable generalization gap. The Test-Train gap is 4.17, so the model still overfits noticeably. This motivated the deep-HPO and residual-corrector experiments in v11 and v12, neither of which achieved a substantive breakthrough in Test RMSE — evidence that this framework has reached the information ceiling of the dataset.
11. References
	Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. Computational Materials Science, 154, 346–354.
	UCI Machine Learning Repository — Superconductivity Data
	McMillan, W. L. (1968). Transition temperature of strong-coupled superconductors. Physical Review, 167(2), 331.
