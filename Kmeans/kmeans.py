# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

# 配置中文字体（Windows 系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def generate_data(seed=None):
    """
    生成数据，匹配 MATLAB Kmeans_student.m 的数据生成方式。
    在 [0,10] x [0,10] 区域内均匀生成 2000 个点，
    只保留落在 9 个矩形区域内的点，去掉白色间隔中的点。
    """
    if seed is not None:
        np.random.seed(seed)

    # 初始生成 2000 个均匀分布的点
    n_init = 2000
    X_raw = np.random.rand(n_init, 2) * 10
    Y = np.zeros(n_init, dtype=int)

    # 9 个有效矩形区域的边界 (x_min, x_max, y_min, y_max)
    regions = [
        (0, 3, 0, 3),
        (0, 3, 3.5, 6.5),
        (0, 3, 7, 10),
        (3.5, 6.5, 0, 3),
        (3.5, 6.5, 3.5, 6.5),
        (3.5, 6.5, 7, 10),
        (7, 10, 0, 3),
        (7, 10, 3.5, 6.5),
        (7, 10, 7, 10),
    ]

    for i in range(n_init):
        for label_idx, (x_min, x_max, y_min, y_max) in enumerate(regions, start=1):
            if (x_min < X_raw[i, 0] < x_max and y_min < X_raw[i, 1] < y_max):
                Y[i] = label_idx
                break

    # 只保留落入 9 个区域的点，去掉白色间隔中的点
    mask = Y > 0
    X = X_raw[mask]
    return X


def initialize_centroids(X, k, seed=None):
    """
    随机初始化：随机选择 k 个样本点作为初始类中心。
    """
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices].copy()


def assign_clusters(X, centroids):
    """
    指派步骤：计算每个样本到所有类中心的欧氏距离，
    将其指派到距离最近的类中心所属的类。
    """
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    """
    更新类中心步骤：计算每个类中所有样本的均值，
    将该均值作为该类新的类中心。
    如果某个类为空，则随机选取一个样本作为其类中心。
    """
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        members = X[labels == i]
        if len(members) > 0:
            new_centroids[i] = members.mean(axis=0)
        else:
            new_centroids[i] = X[np.random.randint(len(X))]
    return new_centroids


def compute_inertia(X, labels, centroids):
    """
    计算目标函数值（Inertia）：所有样本到其所属类中心的
    平方欧氏距离之和（SSE）。
    """
    total = 0.0
    for i in range(len(centroids)):
        members = X[labels == i]
        if len(members) > 0:
            total += np.sum((members - centroids[i]) ** 2)
    return total


def kmeans(X, k, max_iters=300, tol=1e-6, seed=None):
    """
    K-means 聚类算法。

    参数
    ----------
    X : ndarray，形状为 (n_samples, n_features)
        输入数据。
    k : int
        聚类数目。
    max_iters : int
        最大迭代次数。
    tol : float
        基于类中心移动距离的收敛阈值。
    seed : int, optional
        随机种子，用于结果复现。

    返回值
    -------
    labels : ndarray
        每个样本的预测类别标签。
    centroids : ndarray
        最终的类中心坐标。
    n_iter : int
        实际迭代次数。
    inertia : float
        目标函数值（平方距离和，SSE）。
    """
    centroids = initialize_centroids(X, k, seed)
    labels = np.zeros(len(X), dtype=int)

    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    inertia = compute_inertia(X, labels, centroids)
    return labels, centroids, iteration + 1, inertia


def plot_result(X, labels, centroids, k, ax, title):
    """
    在指定的 axes 上绘制聚类结果。
    不同类别使用不同的颜色和形状区分，
    类中心用黑色 X 标记。
    """
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#ffff33', '#a65628', '#f781bf', '#999999']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

    for i in range(k):
        mask = labels == i
        if np.sum(mask) > 0:
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=colors[i], marker=markers[i],
                       s=30, alpha=0.6, linewidths=0.5,
                       label=f'簇 {i + 1}')

    # 用黑色 X 标记聚类中心
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='X', s=200,
               edgecolors='white', linewidths=1.5,
               zorder=5, label='类中心')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('x 轴')
    ax.set_ylabel('y 轴')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)


def plot_ground_truth(X, k, ax):
    """
    绘制原始 9 区域真实标签（仅作参考，K-means 本身不使用真实标签）。
    """
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#ffff33', '#a65628', '#f781bf', '#999999']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

    regions = [
        (0, 3, 0, 3), (0, 3, 3.5, 6.5), (0, 3, 7, 10),
        (3.5, 6.5, 0, 3), (3.5, 6.5, 3.5, 6.5), (3.5, 6.5, 7, 10),
        (7, 10, 0, 3), (7, 10, 3.5, 6.5), (7, 10, 7, 10),
    ]

    for i, (x_min, x_max, y_min, y_max) in enumerate(regions):
        mask = ((X[:, 0] > x_min) & (X[:, 0] < x_max) &
                (X[:, 1] > y_min) & (X[:, 1] < y_max))
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[i], marker=markers[i],
                   s=30, alpha=0.6, linewidths=0.5,
                   label=f'区域 {i + 1}')

    ax.set_title('真实标签（参考）', fontsize=13)
    ax.set_xlabel('x 轴')
    ax.set_ylabel('y 轴')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)


def generate_data_bonus(seed=None):
    """
    生成附加题数据，匹配 MATLAB Kmeans2_student.m 的数据生成方式。
    在 [0,10] x [0,10] 区域内均匀生成 2000 个点，
    只保留落在 6 个矩形区域内的点，去掉白色间隔中的点。
    """
    if seed is not None:
        np.random.seed(seed)

    n_init = 2000
    X_raw = np.random.rand(n_init, 2) * 10
    Y = np.zeros(n_init, dtype=int)

    # 6 个有效矩形区域的边界 (x_min, x_max, y_min, y_max)
    regions = [
        (0, 6, 0, 6),        # 区域 1：左下大区域
        (7, 10, 0, 3),        # 区域 2：右下上
        (7, 10, 3, 6),        # 区域 3：右下中
        (0, 3, 7, 10),        # 区域 4：左上
        (3, 6, 7, 10),        # 区域 5：中上
        (7, 10, 7, 10),       # 区域 6：右边上
    ]

    for i in range(n_init):
        for label_idx, (x_min, x_max, y_min, y_max) in enumerate(regions, start=1):
            if (x_min < X_raw[i, 0] < x_max and y_min < X_raw[i, 1] < y_max):
                Y[i] = label_idx
                break

    mask = Y > 0
    X = X_raw[mask]
    return X


def plot_ground_truth_bonus(X, k, ax):
    """
    绘制附加题原始 6 区域真实标签（仅作参考）。
    """
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    markers = ['o', 's', '^', 'D', 'v', '<']

    regions = [
        (0, 6, 0, 6),
        (7, 10, 0, 3),
        (7, 10, 3, 6),
        (0, 3, 7, 10),
        (3, 6, 7, 10),
        (7, 10, 7, 10),
    ]

    for i, (x_min, x_max, y_min, y_max) in enumerate(regions):
        mask = ((X[:, 0] > x_min) & (X[:, 0] < x_max) &
                (X[:, 1] > y_min) & (X[:, 1] < y_max))
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[i], marker=markers[i],
                   s=30, alpha=0.6, linewidths=0.5,
                   label=f'区域 {i + 1}')

    ax.set_title('附加题真实标签（参考）', fontsize=13)
    ax.set_xlabel('x 轴')
    ax.set_ylabel('y 轴')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)


def plot_result_bonus(X, labels, centroids, k, ax, title):
    """
    在指定的 axes 上绘制附加题聚类结果（6 个簇）。
    """
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    markers = ['o', 's', '^', 'D', 'v', '<']

    for i in range(k):
        mask = labels == i
        if np.sum(mask) > 0:
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=colors[i], marker=markers[i],
                       s=30, alpha=0.6, linewidths=0.5,
                       label=f'簇 {i + 1}')

    ax.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='X', s=200,
               edgecolors='white', linewidths=1.5,
               zorder=5, label='类中心')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('x 轴')
    ax.set_ylabel('y 轴')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)


def run_kmeans_and_plot_bonus():
    """
    附加题：使用 Kmeans2_student.m 的数据生成方式，
    运行 10 次 K-means 实验（K=6），绘制结果。
    """
    print("\n" + "=" * 60)
    print("附加题：Kmeans2 数据集（6 个簇，10 次实验）")
    print("=" * 60)

    K = 6
    N_EXPERIMENTS = 10
    X = generate_data_bonus(seed=42)
    print(f"生成数据：{len(X)} 个点，来自 6 个矩形区域（seed=42）")

    # 真实标签参考图
    fig_gt, ax_gt = plt.subplots(figsize=(7, 6))
    fig_gt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    plot_ground_truth_bonus(X, K, ax_gt)
    fig_gt.savefig(f'{BASE_DIR}/kmeans2_ground_truth.png', dpi=150, bbox_inches='tight')
    plt.close(fig_gt)
    print("已保存附加题真实标签参考图。")

    results = []

    for exp_idx in range(N_EXPERIMENTS):
        labels, centroids, n_iter, inertia = kmeans(
            X, K, max_iters=300, tol=1e-6, seed=exp_idx
        )
        results.append({
            'experiment': exp_idx + 1,
            'labels': labels,
            'centroids': centroids,
            'n_iter': n_iter,
            'inertia': inertia,
        })
        print(f"实验 {exp_idx + 1:2d}：迭代次数={n_iter:3d}，SSE={inertia:.2f}")

    # 10 次实验结果汇总图
    fig, axes = plt.subplots(2, 5, figsize=(25, 11))
    fig.suptitle(f'附加题 K-means 聚类结果（K={K}），共 10 次实验', fontsize=16, fontweight='bold')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, wspace=0.25, hspace=0.30)

    for idx, res in enumerate(results):
        ax = axes.flat[idx]
        plot_result_bonus(X, res['labels'], res['centroids'], K, ax,
                         f"实验 {idx + 1}  (迭代={res['n_iter']}, SSE={res['inertia']:.1f})")

    fig.savefig(f'{BASE_DIR}/kmeans2_results_10_experiments.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n已保存附加题 10 次实验结果图。")

    # 汇总统计表
    summary_fig, summary_ax = plt.subplots(figsize=(10, 8))
    summary_ax.set_title('附加题 K-means 汇总：10 次实验结果', fontsize=14, fontweight='bold')
    summary_ax.axis('off')

    cell_text = []
    for res in results:
        cell_text.append([
            f"实验 {res['experiment']}",
            str(res['n_iter']),
            f"{res['inertia']:.2f}",
        ])

    table = summary_ax.table(
        cellText=cell_text,
        colLabels=['实验编号', '迭代次数', 'SSE（误差平方和）'],
        cellLoc='center',
        loc='center',
        bbox=[0.15, 0.3, 0.7, 0.55]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D9E2F3')

    best = min(results, key=lambda r: r['inertia'])
    worst = max(results, key=lambda r: r['inertia'])
    median_res = sorted(results, key=lambda r: r['inertia'])[N_EXPERIMENTS // 2]
    best_idx = results.index(best)
    worst_idx = results.index(worst)

    summary_fig.text(0.5, 0.17,
                     f"最优：实验 {best_idx + 1}  (SSE={best['inertia']:.2f})   |   "
                     f"最差：实验 {worst_idx + 1}  (SSE={worst['inertia']:.2f})   |   "
                     f"中位：实验 {results.index(median_res) + 1}  (SSE={median_res['inertia']:.2f})",
                     ha='center', fontsize=11, style='italic',
                     bbox=dict(boxstyle='round', facecolor='#F2F2F2', alpha=0.8))

    summary_fig.savefig(f'{BASE_DIR}/kmeans2_summary_table.png',
                        dpi=150, bbox_inches='tight')
    plt.close(summary_fig)
    print("已保存附加题汇总统计表。")

    print("\n" + "-" * 60)
    print("所有附加题实验结果：")
    print(f"{'实验':>4}  {'迭代次数':>10}  {'SSE（误差平方和）':>18}")
    print("-" * 40)
    for res in results:
        marker = "  <-- 最优" if res['experiment'] == best['experiment'] else ""
        print(f"{res['experiment']:>4}  {res['n_iter']:>10}  {res['inertia']:>18.2f}{marker}")
    print("-" * 60)
    print("\n附加题输出文件：")
    print("  1. kmeans2_ground_truth.png              - 附加题真实标签参考图")
    print("  2. kmeans2_results_10_experiments.png     - 附加题 10 次实验结果汇总图")
    print("  3. kmeans2_summary_table.png              - 附加题汇总统计表")


def main():
    """主函数：生成数据，运行 10 次 K-means 实验，并绘制结果。"""
    K = 9                                    # 聚类数目
    N_EXPERIMENTS = 10                       # 实验次数

    # 生成数据（seed=42 保证数据可复现）
    X = generate_data(seed=42)
    print(f"生成数据：{len(X)} 个点，来自 9 个矩形区域（seed=42）")

    # 绘制真实标签参考图
    fig_gt, ax_gt = plt.subplots(figsize=(7, 6))
    fig_gt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    plot_ground_truth(X, K, ax_gt)
    fig_gt.savefig(f'{BASE_DIR}/kmeans_ground_truth.png',
                   dpi=150, bbox_inches='tight')
    plt.close(fig_gt)
    print("已保存真实标签参考图。")

    results = []

    # 运行 10 次 K-means 实验，每次使用不同随机种子
    for exp_idx in range(N_EXPERIMENTS):
        labels, centroids, n_iter, inertia = kmeans(
            X, K,
            max_iters=300,
            tol=1e-6,
            seed=exp_idx
        )
        results.append({
            'experiment': exp_idx + 1,
            'labels': labels,
            'centroids': centroids,
            'n_iter': n_iter,
            'inertia': inertia,
        })
        print(f"实验 {exp_idx + 1:2d}：迭代次数={n_iter:3d}，SSE={inertia:.2f}")

    # 将 10 次实验的结果绘制在同一张图中
    fig, axes = plt.subplots(2, 5, figsize=(25, 11))
    fig.suptitle(f'K-means 聚类结果（K={K}），共 10 次实验', fontsize=16, fontweight='bold')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, wspace=0.25, hspace=0.30)

    for idx, res in enumerate(results):
        ax = axes.flat[idx]
        plot_result(X, res['labels'], res['centroids'], K, ax,
                    f"实验 {idx + 1}  (迭代={res['n_iter']}, SSE={res['inertia']:.1f})")

    fig.savefig(f'{BASE_DIR}/kmeans_results_10_experiments.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n已保存 10 次实验结果图。")

    # 绘制汇总统计表
    summary_fig, summary_ax = plt.subplots(figsize=(10, 8))
    summary_ax.set_title('K-means 汇总：10 次实验结果', fontsize=14, fontweight='bold')
    summary_ax.axis('off')

    cell_text = []
    for res in results:
        cell_text.append([
            f"实验 {res['experiment']}",
            str(res['n_iter']),
            f"{res['inertia']:.2f}",
        ])

    table = summary_ax.table(
        cellText=cell_text,
        colLabels=['实验编号', '迭代次数', 'SSE（误差平方和）'],
        cellLoc='center',
        loc='center',
        bbox=[0.15, 0.3, 0.7, 0.55]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D9E2F3')

    best = min(results, key=lambda r: r['inertia'])
    worst = max(results, key=lambda r: r['inertia'])
    median_res = sorted(results, key=lambda r: r['inertia'])[N_EXPERIMENTS // 2]

    best_idx = results.index(best)
    worst_idx = results.index(worst)

    summary_fig.text(0.5, 0.17,
                     f"最优：实验 {best_idx + 1}  (SSE={best['inertia']:.2f})   |   "
                     f"最差：实验 {worst_idx + 1}  (SSE={worst['inertia']:.2f})   |   "
                     f"中位：实验 {results.index(median_res) + 1}  (SSE={median_res['inertia']:.2f})",
                     ha='center', fontsize=11, style='italic',
                     bbox=dict(boxstyle='round', facecolor='#F2F2F2', alpha=0.8))

    summary_fig.savefig(f'{BASE_DIR}/kmeans_summary_table.png',
                        dpi=150, bbox_inches='tight')
    plt.close(summary_fig)
    print("已保存汇总统计表。")

    print("\n" + "=" * 60)
    print("所有实验结果：")
    print(f"{'实验':>4}  {'迭代次数':>10}  {'SSE（误差平方和）':>18}")
    print("-" * 40)
    for res in results:
        marker = "  <-- 最优" if res['experiment'] == best['experiment'] else ""
        print(f"{res['experiment']:>4}  {res['n_iter']:>10}  {res['inertia']:>18.2f}{marker}")
    print("=" * 60)
    print("\n输出文件：")
    print("  1. kmeans_ground_truth.png              - 真实标签参考图")
    print("  2. kmeans_results_10_experiments.png    - 10 次实验结果汇总图")
    print("  3. kmeans_summary_table.png             - 汇总统计表")

    # 执行附加题
    run_kmeans_and_plot_bonus()


if __name__ == '__main__':
    main()
