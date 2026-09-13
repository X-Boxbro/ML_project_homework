"""
感知机附加题 — 南开大学机器学习实验
附加题 1-5：严格按题目描述实现
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


# =============================================================================
# 数据生成（与 perceptron.py 完全一致）
# =============================================================================
def generate_data(n=100, m=10):
    """生成训练和测试数据，与 perceptron.py 完全一致"""
    center1 = np.array([1.0, 1.0])
    center2 = np.array([3.0, 4.0])

    X_train = np.vstack([
        center1 + np.random.randn(n, 2),
        center2 + np.random.randn(n, 2)
    ])
    y_train = np.hstack([np.ones(n), -np.ones(n)])

    X_test = np.vstack([
        center1 + np.random.randn(m, 2),
        center2 + np.random.randn(m, 2)
    ])
    y_test = np.hstack([np.ones(m), -np.ones(m)])

    return X_train, y_train, X_test, y_test, center1, center2


def plot_boundary(X_train, y_train, w, b, title="", filename=None):
    """绘制散点图 + 分类面"""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('white')

    m1 = y_train == 1
    m2 = y_train == -1
    ax.scatter(X_train[m1, 0], X_train[m1, 1], c='red', marker='o',
               s=60, label='class 1', zorder=3, alpha=0.8)
    ax.scatter(X_train[m2, 0], X_train[m2, 1], c='blue', marker='*',
               s=100, label='class 2', zorder=3, alpha=0.8)

    x1 = np.linspace(-2, 7, 500)
    if np.abs(w[1]) > 1e-8:
        y1 = (-b - w[0] * x1) / w[1]
        ax.plot(x1, y1, 'k-', linewidth=2, label='$wx+b=0$', zorder=4)
    elif np.abs(w[0]) > 1e-8:
        x2 = -b / w[0]
        ax.axvline(x=x2, color='k', linewidth=2, label='$wx+b=0$', zorder=4)

    ax.set_xlim(-2, 7)
    ax.set_xlabel('x1', fontsize=13)
    ax.set_ylabel('x2', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  保存: {filename}")
    plt.show()
    plt.close()


# =============================================================================
# 附加题 1：不同学习率 + 画出第10/50/100次迭代的分类面
# =============================================================================
def experiment_lr_iteration():
    print("\n" + "=" * 58)
    print("附加题 1：不同学习率下的迭代过程观察")
    print("=" * 58)

    X_train, y_train, X_test, y_test, center1, center2 = generate_data()
    N = X_train.shape[0]
    lrs = [0.1, 0.5, 1.0]
    check_epochs = [1, 5, 10, 20, 50, 100]

    fig, axes = plt.subplots(len(lrs), len(check_epochs), figsize=(20, 12))
    fig.suptitle('附加题1: 不同学习率下，第1/5/10/20/50/100次迭代的分类面', fontsize=16, y=1.01)

    for row, lr in enumerate(lrs):
        w = np.zeros(2)
        b = 0.0
        epoch_wb = [(0, w.copy(), b)]  # (epoch, w, b)

        for epoch in range(1, 201):
            for i in range(N):
                if y_train[i] * (np.dot(w, X_train[i]) + b) <= 0:
                    w = w + lr * y_train[i] * X_train[i]
                    b = b + lr * y_train[i]
            epoch_wb.append((epoch, w.copy(), b))
            if epoch >= max(check_epochs):
                break

        # 统计信息
        last_ec = 0
        for i in range(N):
            if y_train[i] * (np.dot(w, X_train[i]) + b) <= 0:
                last_ec += 1

        for col, ep in enumerate(check_epochs):
            ax = axes[row, col]
            ax.set_facecolor('#f9f9f9')
            _, wi, bi = epoch_wb[col]

            m1 = y_train == 1
            m2 = y_train == -1
            ax.scatter(X_train[m1, 0], X_train[m1, 1], c='red', marker='o',
                       s=30, alpha=0.7, zorder=3)
            ax.scatter(X_train[m2, 0], X_train[m2, 1], c='blue', marker='*',
                       s=50, alpha=0.7, zorder=3)

            x1 = np.linspace(-2, 7, 500)
            if np.abs(wi[1]) > 1e-8:
                y1 = (-bi - wi[0] * x1) / wi[1]
                ax.plot(x1, y1, 'k-', linewidth=1.8, zorder=4)
            elif np.abs(wi[0]) > 1e-8:
                ax.axvline(x=-bi / wi[0], color='k', linewidth=1.8, zorder=4)

            ax.set_xlim(-2, 7)
            ax.set_ylim(-2, 8)
            ax.set_xlabel('x1', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'lr={lr}\nx2', fontsize=10)
            if row == 0:
                ax.set_title(f'第 {ep} 次迭代', fontsize=11)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('add_fig1_lr_iterations.png', dpi=150, bbox_inches='tight')
    print("  保存: add_fig1_lr_iterations.png")
    plt.show()
    plt.close()

    # 打印每种学习率下的收敛信息
    print(f"\n  {'学习率':<10} {'100轮后错分数':<15} {'最终w':<25} {'最终b':<10}")
    print("  " + "-" * 60)
    for lr in lrs:
        w, b = np.zeros(2), 0.0
        for epoch in range(100):
            for i in range(N):
                if y_train[i] * (np.dot(w, X_train[i]) + b) <= 0:
                    w = w + lr * y_train[i] * X_train[i]
                    b = b + lr * y_train[i]
        ec = sum(1 for i in range(N) if y_train[i] * (np.dot(w, X_train[i]) + b) <= 0)
        print(f"  {lr:<10.1f} {ec:<15} [{w[0]:8.4f}, {w[1]:8.4f}]    {b:<10.4f}")


# =============================================================================
# 附加题 2：去掉 if yi(wiT xi + b) <= 0 判断，观察结果
# =============================================================================
def experiment_no_misclass_check():
    print("\n" + "=" * 58)
    print("附加题 2：去掉 if yi(wT x + b) <= 0 条件的结果")
    print("=" * 58)

    X_train, y_train, X_test, y_test, center1, center2 = generate_data()
    N = X_train.shape[0]
    lr = 0.1

    # --- 正确版本（有判断） ---
    w1, b1 = np.zeros(2), 0.0
    for epoch in range(100):
        for i in range(N):
            if y_train[i] * (np.dot(w1, X_train[i]) + b1) <= 0:
                w1 = w1 + lr * y_train[i] * X_train[i]
                b1 = b1 + lr * y_train[i]

    ec1 = sum(1 for i in range(N) if y_train[i] * (np.dot(w1, X_train[i]) + b1) <= 0)
    print(f"  正确版本（有判断）100轮后错分数: {ec1}/{N}")
    print(f"  w = [{w1[0]:.4f}, {w1[1]:.4f}], b = {b1:.4f}")

    # --- 错误版本（无判断，每个样本都更新） ---
    w2, b2 = np.zeros(2), 0.0
    w2_hist = [w2.copy()]
    b2_hist = [b2]
    for epoch in range(100):
        for i in range(N):
            # 去掉判断：每个样本都会执行更新
            w2 = w2 + lr * y_train[i] * X_train[i]
            b2 = b2 + lr * y_train[i]
        w2_hist.append(w2.copy())
        b2_hist.append(b2)

    ec2 = sum(1 for i in range(N) if y_train[i] * (np.dot(w2, X_train[i]) + b2) <= 0)
    print(f"\n  错误版本（无判断）100轮后错分数: {ec2}/{N}")
    print(f"  w = [{w2[0]:.4f}, {w2[1]:.4f}], b = {b2:.4f}")
    print(f"  （对比：w模长从0增长到{np.linalg.norm(w2):.2f}，参数严重偏移）")

    # --- 图2a：错误版本的 w 模长变化（与正确版本对比） ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('附加题2: 去掉判断条件后的影响', fontsize=15, y=1.01)

    # 左：两种版本的 w 模长变化
    w1_h = [np.zeros(2)]
    b1_h = [0.0]
    for epoch in range(100):
        for i in range(N):
            if y_train[i] * (np.dot(w1_h[-1], X_train[i]) + b1_h[-1]) <= 0:
                new_w = w1_h[-1] + lr * y_train[i] * X_train[i]
                new_b = b1_h[-1] + lr * y_train[i]
            else:
                new_w, new_b = w1_h[-1], b1_h[-1]
            w1_h.append(new_w)
            b1_h.append(new_b)
    w1_norms = [np.linalg.norm(v) for v in w1_h]
    w2_norms = [np.linalg.norm(v) for v in w2_hist]

    axes[0].plot(w1_norms, 'b-', linewidth=2, label='正确版本（有判断）')
    axes[0].plot(w2_norms, 'r--', linewidth=2, label='错误版本（无判断）')
    axes[0].set_xlabel('参数更新次数', fontsize=12)
    axes[0].set_ylabel('||w||', fontsize=12)
    axes[0].set_title('(a) 权重模长变化', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#f9f9f9')

    # 中：正确版本分类面
    axes[1].set_facecolor('#f9f9f9')
    m1 = y_train == 1
    m2 = y_train == -1
    axes[1].scatter(X_train[m1, 0], X_train[m1, 1], c='red', marker='o', s=40, alpha=0.7)
    axes[1].scatter(X_train[m2, 0], X_train[m2, 1], c='blue', marker='*', s=60, alpha=0.7)
    x1 = np.linspace(-2, 7, 500)
    if np.abs(w1[1]) > 1e-8:
        y1 = (-b1 - w1[0] * x1) / w1[1]
        axes[1].plot(x1, y1, 'k-', linewidth=2)
    axes[1].set_xlim(-2, 7)
    axes[1].set_xlabel('x1', fontsize=12)
    axes[1].set_ylabel('x2', fontsize=12)
    axes[1].set_title(f'(b) 正确版本\n错分数={ec1}/{N}', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # 右：错误版本分类面
    axes[2].set_facecolor('#f9f9f9')
    axes[2].scatter(X_train[m1, 0], X_train[m1, 1], c='red', marker='o', s=40, alpha=0.7)
    axes[2].scatter(X_train[m2, 0], X_train[m2, 1], c='blue', marker='*', s=60, alpha=0.7)
    if np.abs(w2[1]) > 1e-8:
        y2 = (-b2 - w2[0] * x1) / w2[1]
        axes[2].plot(x1, y2, 'k-', linewidth=2)
    axes[2].set_xlim(-2, 7)
    axes[2].set_xlabel('x1', fontsize=12)
    axes[2].set_ylabel('x2', fontsize=12)
    axes[2].set_title(f'(c) 错误版本（无判断）\n错分数={ec2}/{N}', fontsize=13)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('add_fig2_no_check.png', dpi=150, bbox_inches='tight')
    print("  保存: add_fig2_no_check.png")
    plt.show()
    plt.close()

    # --- 图2b：错误版本中分类面剧烈变化的详情（第1/20/50/100轮） ---
    w_check, b_check = [], []
    w_tmp3, b_tmp3 = np.zeros(2), 0.0
    for epoch in range(100):
        for i in range(N):
            w_tmp3 = w_tmp3 + lr * y_train[i] * X_train[i]
            b_tmp3 = b_tmp3 + lr * y_train[i]
        w_check.append(w_tmp3.copy())
        b_check.append(b_tmp3)

    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5))
    fig2.suptitle('附加题2: 错误版本中分类面的剧烈变化（第1/20/50/100轮）', fontsize=14, y=1.01)
    checkpoints = [0, 19, 49, 99]
    for ci, ck in enumerate(checkpoints):
        ax = axes2[ci]
        ax.set_facecolor('#f9f9f9')
        wi, bi = w_check[ck], b_check[ck]
        ax.scatter(X_train[m1, 0], X_train[m1, 1], c='red', marker='o', s=30, alpha=0.7)
        ax.scatter(X_train[m2, 0], X_train[m2, 1], c='blue', marker='*', s=50, alpha=0.7)
        x_range = np.linspace(-20, 20, 500)
        if np.abs(wi[1]) > 1e-8:
            y_range = (-bi - wi[0] * x_range) / wi[1]
            ax.plot(x_range, y_range, 'k-', linewidth=2)
        elif np.abs(wi[0]) > 1e-8:
            ax.axvline(x=-bi / wi[0], color='k', linewidth=2)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x1', fontsize=10)
        ax.set_ylabel('x2', fontsize=10)
        ax.set_title(f'第 {ck+1} 轮\n||w||={np.linalg.norm(wi):.1f}', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig('add_fig2_no_check_detail.png', dpi=150, bbox_inches='tight')
    print("  保存: add_fig2_no_check_detail.png")
    plt.show()
    plt.close()


# =============================================================================
# 附加题 3：线性不可分时循环能否终止？为什么？
# =============================================================================
def experiment_inseparable():
    print("\n" + "=" * 58)
    print("附加题 3：线性不可分时循环能否终止？")
    print("=" * 58)

    # 构造线性不可分数据：两类中心极近，大量重叠
    center1 = np.array([0.0, 0.0])
    center2 = np.array([1.0, 1.0])
    np.random.seed(42)
    X = np.vstack([
        center1 + np.random.randn(50, 2),
        center2 + np.random.randn(50, 2)
    ])
    y = np.hstack([np.ones(50), -np.ones(50)])
    N = 100

    w, b = np.zeros(2), 0.0
    err_hist = []
    max_show = 500

    print(f"\n  数据配置：center1={center1}, center2={center2}")
    print(f"  两类中心距离={np.linalg.norm(center1-center2):.2f}（< 2*sigma≈2.83，高度重叠）")
    print(f"  {'迭代轮数':<10} {'本轮错分数':<12} {'||w||':<10}")
    print("  " + "-" * 35)

    for epoch in range(max_show):
        ec = 0
        for i in range(N):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w = w + 0.1 * y[i] * X[i]
                b = b + 0.1 * y[i]
                ec += 1
        err_hist.append(ec)
        if epoch < 10 or epoch % 50 == 0:
            print(f"  {epoch+1:<10} {ec:<12} {np.linalg.norm(w):<10.4f}")
        if ec == 0:
            print(f"\n  收敛！共 {epoch+1} 轮")
            break

    if err_hist[-1] != 0:
        print(f"\n  未收敛！达到最大 {max_show} 轮后仍有 {err_hist[-1]} 个错分样本")
        print(f"  错分样本数一直在 {min(err_hist)} ~ {max(err_hist)} 之间振荡")

    # 图3a：错分样本数变化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('附加题3: 线性不可分数据的感知机训练', fontsize=15, y=1.01)

    axes[0].plot(range(1, len(err_hist)+1), err_hist, 'b-', linewidth=1.8)
    axes[0].set_xlabel('Epoch', fontsize=13)
    axes[0].set_ylabel('Misclassified Samples', fontsize=13)
    axes[0].set_title(f'(a) 错分样本数随Epoch变化\n(最高={max(err_hist)}, 最低={min(err_hist)}, 最终={err_hist[-1]})', fontsize=12)
    axes[0].axhline(y=min(err_hist), color='g', linestyle='--', alpha=0.6, label=f'最低={min(err_hist)}')
    axes[0].axhline(y=max(err_hist), color='r', linestyle='--', alpha=0.6, label=f'最高={max(err_hist)}')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#f9f9f9')

    # 图3b：w 模长变化
    w_tmp, b_tmp = np.zeros(2), 0.0
    w_norms = []
    for epoch in range(len(err_hist)):
        ec = 0
        for i in range(N):
            if y[i] * (np.dot(w_tmp, X[i]) + b_tmp) <= 0:
                w_tmp = w_tmp + 0.1 * y[i] * X[i]
                b_tmp = b_tmp + 0.1 * y[i]
                ec += 1
        w_norms.append(np.linalg.norm(w_tmp))
    axes[1].plot(range(1, len(err_hist)+1), w_norms, 'g-', linewidth=1.8)
    axes[1].set_xlabel('Epoch', fontsize=13)
    axes[1].set_ylabel('||w||', fontsize=13)
    axes[1].set_title('(b) 权重模长随Epoch变化\n（线性不可分时||w||持续增长）', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#f9f9f9')

    fig.tight_layout()
    fig.savefig('add_fig3_inseparable.png', dpi=150, bbox_inches='tight')
    print("\n  保存: add_fig3_inseparable.png")
    plt.show()
    plt.close()


# =============================================================================
# 附加题 4：改变数据中心，观察分类效果
# =============================================================================
def experiment_center_distance():
    print("\n" + "=" * 58)
    print("附加题 4：改变数据中心距离，观察分类效果")
    print("=" * 58)

    np.random.seed(42)
    base_c1 = np.array([1.0, 1.0])

    # 不同距离的配置
    configs = [
        {"label": "d=2.0", "c2": np.array([3.0, 3.0]),        "note": "高度重叠"},
        {"label": "d=3.6", "c2": np.array([3.0, 4.0]),        "note": "适中重叠（原始）"},
        {"label": "d=5.7", "c2": np.array([5.0, 5.0]),        "note": "轻度重叠"},
        {"label": "d=8.5", "c2": np.array([7.0, 7.0]),        "note": "基本可分"},
        {"label": "d=12.7","c2": np.array([10.0, 10.0]),      "note": "严格可分"},
    ]

    results = []
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('附加题4: 不同数据中心距离下的分类效果', fontsize=16, y=1.05)

    for idx, cfg in enumerate(configs):
        c1 = base_c1
        c2 = cfg["c2"]

        # 生成数据
        X = np.vstack([c1 + np.random.randn(100, 2), c2 + np.random.randn(100, 2)])
        y = np.hstack([np.ones(100), -np.ones(100)])
        dist = np.linalg.norm(c1 - c2)

        # 训练
        w, b = np.zeros(2), 0.0
        for epoch in range(500):
            for i in range(200):
                if y[i] * (np.dot(w, X[i]) + b) <= 0:
                    w = w + 0.1 * y[i] * X[i]
                    b = b + 0.1 * y[i]

        ec = sum(1 for i in range(200) if y[i] * (np.dot(w, X[i]) + b) <= 0)
        results.append({
            "label": cfg["label"],
            "dist": dist,
            "note": cfg["note"],
            "err": ec,
            "err_rate": ec / 200,
            "w": w,
            "b": b
        })

        print(f"  {cfg['label']:<8} 距离={dist:5.2f}  错分数={ec:3d}/200 ({ec/200:.1%})  "
              f"w=[{w[0]:7.4f},{w[1]:7.4f}]  b={b:7.4f}  [{cfg['note']}]")

        # 绘图
        ax = axes[idx]
        ax.set_facecolor('#f9f9f9')
        m1 = y == 1
        m2 = y == -1
        ax.scatter(X[m1, 0], X[m1, 1], c='red', marker='o', s=25, alpha=0.7)
        ax.scatter(X[m2, 0], X[m2, 1], c='blue', marker='*', s=50, alpha=0.7)

        x1 = np.linspace(-3, 14, 500)
        if np.abs(w[1]) > 1e-8:
            y1 = (-b - w[0] * x1) / w[1]
            ax.plot(x1, y1, 'k-', linewidth=2)
        elif np.abs(w[0]) > 1e-8:
            ax.axvline(x=-b / w[0], color='k', linewidth=2)

        ax.set_xlim(-3, 14)
        ax.set_xlabel('x1', fontsize=10)
        ax.set_ylabel('x2', fontsize=10)
        ax.set_title(f"{cfg['label']}\n错分={ec}/200 ({ec/200:.0%})\n{cfg['note']}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('add_fig4_center_distance.png', dpi=150, bbox_inches='tight')
    print("  保存: add_fig4_center_distance.png")
    plt.show()
    plt.close()


# =============================================================================
# 附加题 5：高维扩展（100维），训练感知机并报告准确率
# =============================================================================
def experiment_high_dimension():
    print("\n" + "=" * 58)
    print("附加题 5：高维扩展（100维）感知机")
    print("=" * 58)

    np.random.seed(42)
    dim = 100

    # 训练数据
    center1 = np.ones(dim) * 1.0
    center2 = np.ones(dim) * 3.0
    X_train = np.vstack([
        center1 + np.random.randn(100, dim),
        center2 + np.random.randn(100, dim)
    ])
    y_train = np.hstack([np.ones(100), -np.ones(100)])

    # 测试数据
    np.random.seed(99)
    X_test = np.vstack([
        center1 + np.random.randn(20, dim),
        center2 + np.random.randn(20, dim)
    ])
    y_test = np.hstack([np.ones(20), -np.ones(20)])

    print(f"  数据维度: {dim}")
    print(f"  训练样本: {X_train.shape[0]} (每类100个)")
    print(f"  测试样本: {X_test.shape[0]} (每类20个)")

    # 训练
    w, b = np.zeros(dim), 0.0
    print(f"\n  {'迭代':<8} {'本轮错分数':<15} {'累计正确数':<15} {'||w||':<12}")
    print("  " + "-" * 50)
    for epoch in range(500):
        ec = 0
        for i in range(200):
            if y_train[i] * (np.dot(w, X_train[i]) + b) <= 0:
                w = w + 0.1 * y_train[i] * X_train[i]
                b = b + 0.1 * y_train[i]
                ec += 1
        correct = sum(1 for i in range(200) if y_train[i] * (np.dot(w, X_train[i]) + b) > 0)
        if epoch < 10 or epoch % 50 == 0 or ec == 0:
            print(f"  {epoch+1:<8} {ec:<15} {correct:<15} {np.linalg.norm(w):<12.4f}")
        if ec == 0:
            print(f"\n  收敛！共 {epoch+1} 轮")
            break

    # 评估
    train_pred = np.sign(np.dot(X_train, w) + b)
    train_pred[train_pred == 0] = 1
    train_acc = np.mean(train_pred == y_train) * 100

    test_pred = np.sign(np.dot(X_test, w) + b)
    test_pred[test_pred == 0] = 1
    test_acc = np.mean(test_pred == y_test) * 100

    print(f"\n  训练集准确率: {train_acc:.2f}%  ({int(train_acc*2)}/200 正确)")
    print(f"  测试集准确率: {test_acc:.2f}%  ({int(test_acc*0.4)}/40 正确)")
    print(f"  最终 ||w|| = {np.linalg.norm(w):.4f}")

    # 与二维对比
    print(f"\n  --- 与二维对比 ---")
    np.random.seed(42)
    c1_2d = np.array([1.0, 1.0])
    c2_2d = np.array([3.0, 4.0])
    X2_train = np.vstack([c1_2d + np.random.randn(100, 2), c2_2d + np.random.randn(100, 2)])
    y2_train = np.hstack([np.ones(100), -np.ones(100)])
    w2, b2 = np.zeros(2), 0.0
    for epoch in range(500):
        for i in range(200):
            if y2_train[i] * (np.dot(w2, X2_train[i]) + b2) <= 0:
                w2 = w2 + 0.1 * y2_train[i] * X2_train[i]
                b2 = b2 + 0.1 * y2_train[i]
    p2 = np.sign(np.dot(X2_train, w2) + b2)
    p2[p2 == 0] = 1
    print(f"  2D 训练准确率: {np.mean(p2==y2_train)*100:.2f}%")
    print(f"  100D 训练准确率: {train_acc:.2f}%")


# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 60)
    print("感知机附加题 — 南开大学机器学习实验")
    print("=" * 60)

    experiment_lr_iteration()
    experiment_no_misclass_check()
    experiment_inseparable()
    experiment_center_distance()
    experiment_high_dimension()

    print("\n" + "=" * 60)
    print("所有附加题完成！")
    print("  add_fig1_lr_iterations.png    — 附加题1：不同学习率各迭代轮次分类面")
    print("  add_fig2_no_check.png          — 附加题2：去掉判断条件的影响")
    print("  add_fig2_no_check_detail.png  — 附加题2：分类面剧烈变化详情")
    print("  add_fig3_inseparable.png      — 附加题3：线性不可分数据训练")
    print("  add_fig4_center_distance.png   — 附加题4：不同数据中心距离分类效果")
    print("  (附加题5数值结果直接打印在终端)")
    print("=" * 60)


if __name__ == "__main__":
    main()
