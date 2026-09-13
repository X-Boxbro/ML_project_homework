# 机器学习课程实验仓库 (project_for_MLcourse)

南开大学《机器学习》课程实验集合。所有实验均从 MATLAB 官方示例迁移/扩展为 Python 实现，每个子目录为一个独立实验，互不依赖，可单独运行。

## 目录结构

| 目录 | 实验 | 主要内容 |
|---|---|---|
| [`perceptron/`](perceptron/) | 实验一：感知机 | 感知机原始形式（Pocket 算法）训练与分类面可视化；附加题：不同学习率迭代过程、去掉错分判断的影响、线性不可分行为、数据中心距离与高维（100 维）扩展 |
| [`svm/`](svm/) | 实验二：支持向量机 | 手写 SMO 算法求解 SVM 对偶问题（线性核），含增广拉格朗日法对比实现；线性可分/不可分实验、C 值影响、软/硬间隔、收敛性分析 |
| [`Kmeans/`](Kmeans/) | 实验三：K-means 聚类 | 手写 K-means（9 簇数据集），10 次随机初始化实验与 SSE 汇总；附加题：6 簇数据集（Kmeans2） |
| [`superconductivity_regression/`](superconductivity_regression/) | 期末大作业：超导临界温度回归 | UCI 超导数据集回归任务；v10 五模型 Stacking 集成（XGB + LGB + ET + Cubist + HGB）+ XGBoost 元学习器 HPO + 等渗校准，Test RMSE 8.84 vs Baseline 10.22。详见其 [README](superconductivity_regression/README.zh-CN.md) |

## 运行环境

实验一~三仅需 `numpy` 与 `matplotlib`：

```bash
pip install numpy matplotlib
```

实验四依赖见 [`superconductivity_regression/requirements.txt`](superconductivity_regression/requirements.txt)。

## 快速运行

```bash
python perceptron/perceptron.py        # 实验一：感知机
python perceptron/perceptron_add.py    # 实验一：附加题 1-5
python svm/svm_manual.py               # 实验二：SVM (SMO)
python Kmeans/kmeans.py                # 实验三：K-means
```

实验四训练/推理命令见 [superconductivity_regression/README.zh-CN.md](superconductivity_regression/README.zh-CN.md#9-环境与运行)。

## 说明

- 课件（pptx）、实验报告（docx/pdf）等文档类文件不上传仓库，见根目录 `.gitignore`
- `superconductivity_regression/` 中保留了项目专属的 `.gitignore` 与 `.gitattributes`（统一 LF 换行符）
