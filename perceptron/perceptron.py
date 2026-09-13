"""
感知机实验 — 南开大学机器学习实验
将 MATLAB 实现迁移为 Python 版本
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


# ==================== 数据生成 ====================
def generate_data(n=100, m=10):
    """生成训练和测试数据"""
    center1 = np.array([1.0, 1.0])
    center2 = np.array([3.0, 4.0])

    # 训练数据
    X_train = np.vstack([
        center1 + np.random.randn(n, 2),
        center2 + np.random.randn(n, 2)
    ])
    y_train = np.hstack([np.ones(n), -np.ones(n)])

    # 测试数据
    X_test = np.vstack([
        center1 + np.random.randn(m, 2),
        center2 + np.random.randn(m, 2)
    ])
    y_test = np.hstack([np.ones(m), -np.ones(m)])

    return X_train, y_train, X_test, y_test, center1, center2


def plot_data(X_train, y_train, center1, center2, title="训练数据", filename="data.png"):
    """绘制训练数据散点图"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('white')

    mask1 = y_train == 1
    mask2 = y_train == -1

    ax.scatter(X_train[mask1, 0], X_train[mask1, 1],
               c='red', marker='o', s=80, label='class 1', zorder=3)
    ax.scatter(X_train[mask2, 0], X_train[mask2, 1],
               c='blue', marker='*', s=120, label='class 2', zorder=3)

    ax.set_xlabel('x axis', fontsize=18)
    ax.set_ylabel('y axis', fontsize=18)
    ax.legend(fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  保存图像: {filename}")
    plt.show()
    plt.close()


# ==================== 感知机模型 ====================
def perceptron_train(X, y, max_iter=500, lr=0.1):
    """
    感知机训练算法（原始形式，顺序遍历 + Pocket 口袋记录最优解）
    模型: f(x) = sign(w · x + b)

    Pocket 算法在每轮遍历后记录当前最优参数。
    较小的学习率 lr=0.1 可有效抑制参数振荡。

    参数:
        X: (N, 2) 数据矩阵
        y: (N,)  标签 (+1 或 -1)
        max_iter: 最大遍历轮数
        lr: 学习率

    返回:
        w_best: (2,) 最优权重向量（口袋中记录）
        b_best: float 最优偏置
        history: 每轮错分样本数记录
    """
    N = X.shape[0]
    w = np.zeros(2)
    b = 0.0

    w_best = np.zeros(2)
    b_best = 0.0
    best_correct = 0

    history = []

    for epoch in range(max_iter):
        error_count = 0
        for i in range(N):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
                error_count += 1

        scores = y * (np.dot(X, w) + b)
        correct = np.sum(scores > 0)

        if correct > best_correct:
            w_best = w.copy()
            b_best = b
            best_correct = correct

        if epoch == 0 or (epoch + 1) % 50 == 0 or error_count == 0:
            print(f"  第 {epoch + 1:3d} 轮: w=[{w[0]:8.4f}, {w[1]:8.4f}], b={b:7.4f}, "
                  f"错分数={error_count}, 累计正确={correct}/{N}")

        if error_count == 0:
            print(f"  收敛！共 {epoch + 1} 轮")
            break

    # 打印最终口袋结果
    final_scores = y * (np.dot(X, w_best) + b_best)
    final_correct = np.sum(final_scores > 0)
    print(f"  最终口袋参数 (最优): w=[{w_best[0]:8.4f}, {w_best[1]:8.4f}], b={b_best:.4f}, "
          f"正确率={final_correct}/{N} ({final_correct/N:.1%})")

    return w_best, b_best, history


def plot_classification_surface(X_train, y_train, w, b, title="分类结果", filename="classification.png"):
    """绘制分类面 x*w + b = 0"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('white')

    mask1 = y_train == 1
    mask2 = y_train == -1

    ax.scatter(X_train[mask1, 0], X_train[mask1, 1],
               c='red', marker='o', s=80, label='class 1', zorder=3)
    ax.scatter(X_train[mask2, 0], X_train[mask2, 1],
               c='blue', marker='*', s=120, label='class 2', zorder=3)

    # 绘制分类面
    x1 = np.linspace(-2, 7, 500)
    if np.abs(w[1]) > 1e-8:
        y1 = (-b - w[0] * x1) / w[1]
        ax.plot(x1, y1, 'k-', linewidth=1.5, label='classification surface')
    elif np.abs(w[0]) > 1e-8:
        x2 = -b / w[0]
        ax.axvline(x=x2, color='k', linewidth=1.5, label='classification surface')

    ax.set_xlabel('x axis', fontsize=18)
    ax.set_ylabel('y axis', fontsize=18)
    ax.legend(fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(-2, 7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  保存图像: {filename}")
    plt.show()
    plt.close()


def perceptron_predict(X, w, b):
    """使用感知机模型预测"""
    return np.sign(np.dot(X, w) + b)


def evaluate(X, y, w, b):
    """计算错误率"""
    preds = perceptron_predict(X, w, b)
    # 避免 sign(0) = 0 的问题
    preds[preds == 0] = 1
    errors = np.sum(preds != y)
    error_rate = errors / len(y)
    return error_rate, errors


def plot_train_test(X_train, y_train, X_test, y_test, w, b, filename="test.png"):
    """绘制训练集、测试集和分类面"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('white')

    # 训练集
    mask1_tr = y_train == 1
    mask2_tr = y_train == -1
    ax.scatter(X_train[mask1_tr, 0], X_train[mask1_tr, 1],
                c='red', marker='o', s=80, label='class 1: train', zorder=3)
    ax.scatter(X_train[mask2_tr, 0], X_train[mask2_tr, 1],
               c='blue', marker='*', s=120, label='class 2: train', zorder=3)

    # 测试集
    mask1_te = y_test == 1
    mask2_te = y_test == -1
    ax.scatter(X_test[mask1_te, 0], X_test[mask1_te, 1],
               c='green', marker='o', s=80, label='class 1: test', zorder=3)
    ax.scatter(X_test[mask2_te, 0], X_test[mask2_te, 1],
               c='green', marker='*', s=120, label='class 2: test', zorder=3)

    # 分类面
    x1 = np.linspace(-2, 7, 500)
    if np.abs(w[1]) > 1e-8:
        y1 = (-b - w[0] * x1) / w[1]
        ax.plot(x1, y1, 'k-', linewidth=1.5, label='classification surface')
    elif np.abs(w[0]) > 1e-8:
        x2 = -b / w[0]
        ax.axvline(x=x2, color='k', linewidth=1.5, label='classification surface')

    ax.set_xlabel('x axis', fontsize=18)
    ax.set_ylabel('y axis', fontsize=18)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title('训练集/测试集 + 分类面', fontsize=18)
    ax.set_xlim(-2, 7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  保存图像: {filename}")
    plt.show()
    plt.close()


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("感知机实验 — 南开大学机器学习")
    print("=" * 60)

    # 参数（与 MATLAB 一致）
    n = 100   # 每类训练样本数
    m = 10    # 每类测试样本数

    # 1. 数据生成
    print("\n[1] 数据生成")
    print(f"  每类训练样本数 n = {n}")
    print(f"  每类测试样本数 m = {m}")
    X_train, y_train, X_test, y_test, center1, center2 = generate_data(n, m)
    print(f"  类别1中心: {center1}，类别2中心: {center2}")
    print(f"  训练集: {X_train.shape[0]} 样本 (类别1: {n}，类别2: {n})")
    print(f"  测试集: {X_test.shape[0]} 样本 (类别1: {m}，类别2: {m})")

    # 2. 可视化训练数据
    print("\n[2] 可视化训练数据")
    plot_data(X_train, y_train, center1, center2,
              title="Training Data", filename="fig1_training_data.png")

    # 3. 训练感知机
    print("\n[3] 训练感知机模型")
    w, b, history = perceptron_train(X_train, y_train)
    print(f"  最终参数: w = {w}, b = {b:.4f}")
    print(f"  分类面方程: {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")

    # 4. 可视化分类结果
    print("\n[4] 可视化分类结果")
    plot_classification_surface(X_train, y_train, w, b,
                                title="Perceptron Classification",
                                filename="fig2_classification.png")

    # 5. 测试集评估
    print("\n[5] 测试集评估")
    train_error, train_errors = evaluate(X_train, y_train, w, b)
    test_error, test_errors = evaluate(X_test, y_test, w, b)
    print(f"  训练集错误数: {train_errors}/{len(y_train)}，错误率: {train_error:.2%}")
    print(f"  测试集错误数: {test_errors}/{len(y_test)}，错误率: {test_error:.2%}")

    # 6. 绘制训练集+测试集+分类面
    print("\n[6] 可视化训练集/测试集 + 分类面")
    plot_train_test(X_train, y_train, X_test, y_test, w, b,
                    filename="fig3_train_test.png")

    # 7. 实验分析
    print("\n[7] 实验分析")
    total_updates = sum(history)
    print(f"  感知机收敛所需轮数: {len(history)} 轮")
    print(f"  总参数更新次数: {total_updates}")
    print(f"  分类面法向量 w = {w.round(4)}")
    print(f"  分类面偏置 b = {b:.4f}")
    print(f"  分类面方程: {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")
    # 计算两类数据中心到分类面的距离（验证分类面是否合理）
    d1 = np.abs(np.dot(w, center1) + b) / np.linalg.norm(w)
    d2 = np.abs(np.dot(w, center2) + b) / np.linalg.norm(w)
    print(f"  类别1中心到分类面距离: {d1:.4f}")
    print(f"  类别2中心到分类面距离: {d2:.4f}")

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
