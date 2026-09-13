"""
手动实现SVM（支持向量机）
核心：SMO算法（Sequential Minimal Optimization）求解对偶问题
同时包含增广拉格朗日法实现作为对比
支持线性可分和线性不可分两种情况
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# 设置中文字体
# ============================================================
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：数据生成
# ============================================================

def generate_data(n=100, center1=(1, 1), center2=(6, 6), seed=42):
    """
    生成二分类数据
    n: 每类样本数
    center1: 第一类中心
    center2: 第二类中心
    seed: 随机种子（保证可复现）
    """
    np.random.seed(seed)
    # 第一类数据
    X1 = np.random.randn(n, 2) + np.array(center1)
    y1 = np.ones(n)
    # 第二类数据
    X2 = np.random.randn(n, 2) + np.array(center2)
    y2 = -np.ones(n)
    # 合并
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    return X, y


def plot_data(X, y, title="训练数据", ax=None):
    """绘制数据分布图"""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'go', linewidth=1, markersize=8, label='class 1')
    ax.plot(X[y == -1, 0], X[y == -1, 1], 'b*', linewidth=1, markersize=8, label='class 2')
    ax.set_xlabel('x axis', fontsize=14)
    ax.set_ylabel('y axis', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    return ax


# ============================================================
# 第二部分：SVM 核心算法
# ============================================================

class SVM:
    """
    手动实现SVM

    方法一：SMO算法（Sequential Minimal Optimization）—— 默认方法
    方法二：增广拉格朗日法（Augmented Lagrangian Method）—— 用于对比分析

    SVM对偶问题：
        min_α  (1/2) α^T Q α - e^T α
        s.t.   y^T α = 0
               0 ≤ α_i ≤ C

    其中 Q_{ij} = y_i y_j K(x_i, x_j)，K为核函数（线性核: K(x_i, x_j) = x_i^T x_j）

    由对偶问题的KKT条件可推导支持向量的判定：
      - α_i = 0    →  y_i (w^T x_i + b) ≥ 1    （非支持向量，正确分类且在间隔外）
      - 0 < α_i < C →  y_i (w^T x_i + b) = 1    （边界上的支持向量）
      - α_i = C    →  y_i (w^T x_i + b) ≤ 1    （间隔内的支持向量，含误分类点）
    """

    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        """
        C: 惩罚参数（C越大，对误分类容忍度越低）
           C → ∞ 时等价于硬间隔SVM
        kernel: 核函数类型 ('linear' 或 'rbf')
        gamma: RBF核参数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vec_idx = None
        self.history = []
        self.X_train = None
        self.y_train = None

    def _kernel_matrix(self, X1, X2=None):
        """计算核矩阵 K(x_i, x_j)"""
        if X2 is None:
            X2 = X1
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            n1, n2 = X1.shape[0], X2.shape[0]
            K = np.zeros((n1, n2))
            for i in range(n1):
                diff = X1[i] - X2
                K[i, :] = np.exp(-self.gamma * np.sum(diff ** 2, axis=1))
            return K
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _decision_f(self, i, alpha, y, K):
        """计算决策函数值 f(x_i) = sum_j alpha_j y_j K(x_j, x_i) + b"""
        return (alpha * y) @ K[:, i] + self.b

    def _take_step(self, i1, i2, alpha, y, K, errors, eps):
        """
        SMO核心步骤：优化一对 α_i1 和 α_i2
        参考 Platt (1998) SMO 论文
        """
        if i1 == i2:
            return 0

        alpha1_old = alpha[i1]
        alpha2_old = alpha[i2]
        y1 = y[i1]
        y2 = y[i2]

        # 计算误差
        E1 = self._decision_f(i1, alpha, y, K) - y1
        E2 = self._decision_f(i2, alpha, y, K) - y2

        s = y1 * y2

        # 计算 L 和 H 边界
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha2_old + alpha1_old - self.C)
            H = min(self.C, alpha2_old + alpha1_old)

        if L >= H:
            return 0

        # 计算二阶导数
        k11 = K[i1, i1]
        k12 = K[i1, i2]
        k22 = K[i2, i2]
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alpha2_old + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # eta <= 0，数值问题，在边界上评估目标函数
            # 计算 L 和 H 处的目标函数值
            def obj_at(a2_val):
                a1 = alpha1_old + s * (alpha2_old - a2_val)
                f1 = y1 * (E1 + self.b) - alpha1_old * k11 - s * alpha2_old * k12
                f2 = y2 * (E2 + self.b) - s * alpha1_old * k12 - alpha2_old * k22
                Lobj = a1 * f1 + a2_val * f2 + 0.5 * a1 * a1 * k11 + \
                       0.5 * a2_val * a2_val * k22 + s * a1 * a2_val * k12
                return Lobj

            # 在 L, H 处评估
            a1_L = alpha1_old + s * (alpha2_old - L)
            Lobj = a1_L * (y1 * (E1 + self.b) - alpha1_old * k11 - s * alpha2_old * k12) \
                    + L * (y2 * (E2 + self.b) - s * alpha1_old * k12 - alpha2_old * k22) \
                    + 0.5 * a1_L * a1_L * k11 + 0.5 * L * L * k22 + s * a1_L * L * k12

            a1_H = alpha1_old + s * (alpha2_old - H)
            Hobj = a1_H * (y1 * (E1 + self.b) - alpha1_old * k11 - s * alpha2_old * k12) \
                    + H * (y2 * (E2 + self.b) - s * alpha1_old * k12 - alpha2_old * k22) \
                    + 0.5 * a1_H * a1_H * k11 + 0.5 * H * H * k22 + s * a1_H * H * k12

            if Lobj < Hobj - eps:
                a2 = L
            elif Lobj > Hobj + eps:
                a2 = H
            else:
                a2 = alpha2_old

        if abs(a2 - alpha2_old) < eps * (a2 + alpha2_old + eps):
            return 0

        # 更新 alpha1
        a1 = alpha1_old + s * (alpha2_old - a2)

        # 更新 b
        b1 = self.b - E1 - y1 * (a1 - alpha1_old) * k11 - y2 * (a2 - alpha2_old) * k12
        b2 = self.b - E2 - y1 * (a1 - alpha1_old) * k12 - y2 * (a2 - alpha2_old) * k22

        if 0 < a1 < self.C:
            self.b = b1
        elif 0 < a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        # 更新 alpha
        alpha[i1] = a1
        alpha[i2] = a2

        # 更新误差缓存
        errors[i1] = self._decision_f(i1, alpha, y, K) - y1
        errors[i2] = self._decision_f(i2, alpha, y, K) - y2

        return 1

    def _examine_example(self, i2, alpha, y, K, errors, eps):
        """检查第 i2 个样本是否需要优化"""
        y2 = y[i2]
        alpha2 = alpha[i2]
        E2 = self._decision_f(i2, alpha, y, K) - y2
        errors[i2] = E2
        r2 = E2 * y2

        # KKT条件检查:
        # α_i = 0    → y_i f(x_i) ≥ 1  → r2 ≥ -eps
        # 0<α_i<C   → y_i f(x_i) = 1  → |r2| < eps
        # α_i = C    → y_i f(x_i) ≤ 1  → r2 ≤ eps
        if (alpha2 < self.C - eps and r2 < -eps) or \
           (alpha2 > eps and r2 > eps):
            # 违反KKT条件，需要优化

            # 选择第一个变量 i1
            # 启发式1：在非零非C的alpha中找 |E2 - E1| 最大的
            non_bound = np.where((alpha > eps) & (alpha < self.C - eps))[0]
            if len(non_bound) > 0:
                if len(non_bound) > 1:
                    # 找与E2差异最大的
                    i1 = non_bound[np.argmax(np.abs(errors[non_bound] - E2))]
                    if self._take_step(i1, i2, alpha, y, K, errors, eps):
                        return 1
                # 遍历所有非边界alpha
                for i1 in non_bound:
                    if self._take_step(i1, i2, alpha, y, K, errors, eps):
                        return 1

            # 启发式2：遍历所有样本
            all_idx = np.arange(len(alpha))
            np.random.shuffle(all_idx)
            for i1 in all_idx:
                if self._take_step(i1, i2, alpha, y, K, errors, eps):
                    return 1

        return 0

    def fit(self, X, y, max_iter=3000, tol=1e-3, eps=1e-6, verbose=True):
        """
        使用SMO算法训练SVM

        参数:
            X: 训练数据 (n_samples, n_features)
            y: 标签 (+1 或 -1)
            max_iter: 最大迭代次数
            tol: KKT条件容忍度
            eps: 数值精度
            verbose: 是否打印训练信息
        """
        n = X.shape[0]
        self.X_train = X
        self.y_train = y

        # 初始化
        alpha = np.zeros(n)
        self.b = 0.0
        K = self._kernel_matrix(X)
        errors = np.zeros(n)

        # 初始化误差
        for i in range(n):
            errors[i] = self._decision_f(i, alpha, y, K) - y[i]

        num_changed = 0
        examine_all = True
        iter_count = 0

        while (num_changed > 0 or examine_all) and iter_count < max_iter:
            num_changed = 0
            iter_count += 1

            if examine_all:
                # 遍历所有样本
                for i in range(n):
                    num_changed += self._examine_example(i, alpha, y, K, errors, eps)
            else:
                # 只遍历非边界样本 (0 < α < C)
                non_bound = np.where((alpha > eps) & (alpha < self.C - eps))[0]
                for i in non_bound:
                    num_changed += self._examine_example(i, alpha, y, K, errors, eps)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            # 记录历史
            obj_val = 0.5 * np.sum((alpha * y)[:, None] * (alpha * y)[None, :] * K) - np.sum(alpha)
            constraint_violation = np.abs(np.sum(alpha * y))

            self.history.append({
                'iter': iter_count,
                'objective': obj_val,
                'violation': constraint_violation
            })

            if verbose and iter_count % 200 == 0:
                n_sv = np.sum(alpha > eps)
                n_bsv = np.sum((alpha > eps) & (alpha < self.C - eps))
                print(f"  iter {iter_count}: obj={obj_val:.6f}, "
                      f"violation={constraint_violation:.6f}, "
                      f"#SV={n_sv}, #BSV={n_bsv}")

        # 存储结果
        self.alpha = alpha

        # 提取支持向量
        threshold = 1e-5
        self.support_vec_idx = np.where(alpha > threshold)[0]

        # 计算 w = Σ α_i y_i x_i
        self.w = np.sum(alpha[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

        # 修正 b（使用边界支持向量的平均值）
        margin_sv = np.where((alpha > threshold) & (alpha < self.C - threshold))[0]
        if len(margin_sv) > 0:
            self.b = np.mean(y[margin_sv] - X[margin_sv] @ self.w)
        elif len(self.support_vec_idx) > 0:
            self.b = np.mean(y[self.support_vec_idx] - X[self.support_vec_idx] @ self.w)

        if verbose:
            n_sv = len(self.support_vec_idx)
            n_bsv = len(margin_sv)
            print(f"  SMO 完成: iter={iter_count}")
            print(f"  支持向量数量: {n_sv}")
            print(f"  边界上支持向量 (0<alpha<C): {n_bsv}")
            print(f"  间隔内支持向量 (alpha=C): {n_sv - n_bsv}")
            print(f"  w = [{self.w[0]:.4f}, {self.w[1]:.4f}], b = {self.b:.4f}")
            print(f"  ||w|| = {np.linalg.norm(self.w):.4f}, margin = {2.0/np.linalg.norm(self.w):.4f}")

        return self

    def predict(self, X):
        """预测类别"""
        return np.sign(X @ self.w + self.b)

    def decision_function(self, X):
        """决策函数值"""
        return X @ self.w + self.b


# ============================================================
# 第二部分（续）：增广拉格朗日法 SVM（用于对比分析）
# ============================================================

class AugmentedLagrangianSVM:
    """
    使用增广拉格朗日法求解SVM对偶问题

    将等式约束 y^T α = 0 作为惩罚项加入目标函数，
    内层使用 L-BFGS-B 处理盒子约束 [0, C]

    增广拉格朗日函数：
        L_A(α, λ; ρ) = f(α) + λ·(y^T α) + (ρ/2)·(y^T α)^2
        where f(α) = (1/2)α^T Q α - e^T α

    算法流程：
        1. 初始化 α₀, λ₀, ρ₀
        2. 固定 λ_k, ρ_k，求解 min L_A(α, λ_k; ρ_k) s.t. 0 ≤ α ≤ C
        3. 更新 λ_{k+1} = λ_k + ρ_k·(y^T α_{k+1})
        4. 更新 ρ_{k+1} = min(ρ_max, ρ_scale * ρ_k)
        5. 检查 ||y^T α_{k+1}|| < ε 则收敛，否则回到步骤2
    """

    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vec_idx = None
        self.history = []
        self.n_inner_iters = []  # 记录每次外层迭代的内层迭代次数

    def _kernel_matrix(self, X):
        if self.kernel == 'linear':
            return X @ X.T
        elif self.kernel == 'rbf':
            n = X.shape[0]
            K = np.zeros((n, n))
            for i in range(n):
                diff = X[i] - X
                K[i, :] = np.exp(-self.gamma * np.sum(diff ** 2, axis=1))
            return K
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y,
            rho_init=1.0,
            rho_max=1e6,
            rho_scale=10.0,
            lambda_init=0.0,
            max_outer=200,
            tol=1e-6,
            verbose=True):
        """
        使用增广拉格朗日法训练SVM

        参数:
            rho_init: 惩罚参数初始值
            rho_max: 惩罚参数上限
            rho_scale: 惩罚参数增长因子
            lambda_init: 拉格朗日乘子初始值
            max_outer: 最大外层迭代次数
            tol: 约束违反容忍度
        """
        from scipy.optimize import minimize

        n = X.shape[0]
        K = self._kernel_matrix(X)
        Q = np.outer(y, y) * K

        alpha = np.zeros(n)
        lambd = lambda_init
        rho = rho_init

        total_inner = 0

        for outer_iter in range(max_outer):
            # ---- 内层：固定 λ, ρ，用 L-BFGS-B 求解 ----
            def objective(a):
                return 0.5 * a @ Q @ a - np.sum(a) + lambd * (y @ a) + 0.5 * rho * (y @ a) ** 2

            def gradient(a):
                return Q @ a - np.ones(n) + lambd * y + rho * (y @ a) * y

            res = minimize(
                objective, alpha,
                method='L-BFGS-B',
                jac=gradient,
                bounds=[(0, self.C) for _ in range(n)],
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            alpha = res.x
            total_inner += res.nit

            # ---- 检查约束违反 ----
            violation = np.abs(np.sum(alpha * y))
            obj_val = 0.5 * alpha @ Q @ alpha - np.sum(alpha)

            self.history.append({
                'iter': outer_iter,
                'objective': obj_val,
                'violation': violation,
                'rho': rho,
                'n_inner': res.nit
            })
            self.n_inner_iters.append(res.nit)

            if verbose and outer_iter % 10 == 0:
                print(f"  AL outer {outer_iter}: obj={obj_val:.4f}, "
                      f"violation={violation:.2e}, rho={rho:.1f}, inner_iters={res.nit}")

            if violation < tol:
                if verbose:
                    print(f"  AL 收敛于 outer_iter={outer_iter}, violation={violation:.2e}")
                break

            # ---- 更新 λ 和 ρ ----
            lambd = lambd + rho * (y @ alpha)
            rho = min(rho * rho_scale, rho_max)

        if verbose:
            print(f"  总内层迭代次数: {total_inner}")

        # 存储结果
        self.alpha = alpha
        threshold = 1e-5
        self.support_vec_idx = np.where(alpha > threshold)[0]

        # 计算 w 和 b
        self.w = np.sum(alpha[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

        margin_sv = np.where((alpha > threshold) & (alpha < self.C - threshold))[0]
        if len(margin_sv) > 0:
            self.b = np.mean(y[margin_sv] - X[margin_sv] @ self.w)
        elif len(self.support_vec_idx) > 0:
            self.b = np.mean(y[self.support_vec_idx] - X[self.support_vec_idx] @ self.w)
        else:
            self.b = 0.0

        if verbose:
            print(f"  支持向量数量: {len(self.support_vec_idx)}")
            print(f"  w = [{self.w[0]:.4f}, {self.w[1]:.4f}], b = {self.b:.4f}")

        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def decision_function(self, X):
        return X @ self.w + self.b


# ============================================================
# 第三部分：可视化函数
# ============================================================

def plot_svm_result(svm, X, y, title="SVM Classification", ax=None,
                    show_sv=True, show_boundary_sv=True):
    """
    绘制SVM分类结果
    包括：数据点、分类面、间隔边界、支持向量
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    # 确定绘图范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 绘制数据点
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'go', linewidth=1, markersize=8, label='class 1')
    ax.plot(X[y == -1, 0], X[y == -1, 1], 'b*', linewidth=1, markersize=8, label='class 2')

    # 绘制决策边界和间隔
    w, b = svm.w, svm.b
    xx = np.linspace(x_min, x_max, 500)

    if abs(w[1]) > 1e-8:
        # 分类面: w1*x + w2*y + b = 0  =>  y = (-b - w1*x) / w2
        yy_decision = (-b - w[0] * xx) / w[1]
        # 上边界: w1*x + w2*y + b = +1  =>  y = (1 - b - w1*x) / w2
        yy_upper = (1 - b - w[0] * xx) / w[1]
        # 下边界: w1*x + w2*y + b = -1  =>  y = (-1 - b - w1*x) / w2
        yy_lower = (-1 - b - w[0] * xx) / w[1]

        # 只绘制在范围内的部分
        mask_decision = (yy_decision >= y_min) & (yy_decision <= y_max)
        mask_upper = (yy_upper >= y_min) & (yy_upper <= y_max)
        mask_lower = (yy_lower >= y_min) & (yy_lower <= y_max)

        ax.plot(xx[mask_decision], yy_decision[mask_decision], 'k-', linewidth=1.5,
                label='classification surface')
        ax.plot(xx[mask_upper], yy_upper[mask_upper], 'k--', linewidth=1,
                label='boundary')
        ax.plot(xx[mask_lower], yy_lower[mask_lower], 'k--', linewidth=1)
    else:
        # w[1] ≈ 0 的情况，用垂直线
        x_val = -b / w[0]
        ax.axvline(x=x_val, color='k', linewidth=1.5, label='classification surface')
        ax.axvline(x=x_val - 1 / w[0], color='k', linestyle='--', linewidth=1, label='boundary')
        ax.axvline(x=x_val + 1 / w[0], color='k', linestyle='--', linewidth=1)

    # 标记支持向量
    if show_sv:
        sv_idx = svm.support_vec_idx
        alpha = svm.alpha

        # 所有支持向量（红色方框）
        ax.plot(X[sv_idx, 0], X[sv_idx, 1], 'rs', linewidth=1.5, markersize=10,
                markerfacecolor='none', label='support vectors')

        # 边界上的支持向量（实心红色方框）: 0 < α < C
        if show_boundary_sv:
            C = svm.C
            threshold = 1e-5
            boundary_sv = np.where((alpha > threshold) & (alpha < C - threshold))[0]
            if len(boundary_sv) > 0:
                ax.plot(X[boundary_sv, 0], X[boundary_sv, 1], 'rs', linewidth=1.5,
                        markersize=10, markerfacecolor='r',
                        label='support vectors on boundary')

    ax.set_xlabel('x axis', fontsize=14)
    ax.set_ylabel('y axis', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return ax


# ============================================================
# 第四部分：主实验
# ============================================================

def experiment_separable():
    """实验一：线性可分数据"""
    print("=" * 60)
    print("实验一：线性可分 SVM")
    print("=" * 60)

    # 生成数据
    X, y = generate_data(n=100, center1=(1, 1), center2=(6, 6))

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1：原始数据
    plot_data(X, y, title="Linearly Separable Data", ax=axes[0])

    # 子图2：训练SVM并绘图
    # C取大值近似硬间隔
    svm = SVM(C=100.0, kernel='linear')
    svm.fit(X, y, verbose=True)
    plot_svm_result(svm, X, y, title=f"SVM (C={svm.C})", ax=axes[1])

    plt.tight_layout()
    plt.savefig("svm_separable.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为 svm_separable.png\n")

    return svm, X, y


def experiment_nonseparable():
    """实验二：线性不可分数据 + 不同C值对比"""
    print("\n" + "=" * 60)
    print("实验二：线性不可分 SVM（不同 C 值对比）")
    print("=" * 60)

    # 生成不可分数据
    X, y = generate_data(n=100, center1=(1, 1), center2=(3, 3))

    # 测试不同的C值
    C_values = [0.1, 1.0, 10.0, 100.0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 先画原始数据
    plot_data(X, y, title="Linearly Non-Separable Data", ax=axes[0])

    svms = []
    for i, C in enumerate(C_values):
        print(f"\n训练 C = {C}")
        svm = SVM(C=C, kernel='linear')
        svm.fit(X, y, verbose=True)
        plot_svm_result(svm, X, y, title=f"SVM (C = {C})", ax=axes[i + 1])
        svms.append(svm)

    # 最后一个子图不显示
    axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig("svm_nonseparable.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为 svm_nonseparable.png\n")

    return svms, X, y


def experiment_convergence():
    """实验三：SMO算法收敛性分析"""
    print("\n" + "=" * 60)
    print("实验三：SMO算法收敛性分析")
    print("=" * 60)

    X, y = generate_data(n=100, center1=(1, 1), center2=(3, 3))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 训练并记录收敛过程
    svm = SVM(C=1.0, kernel='linear')
    svm.fit(X, y, verbose=False)

    history = svm.history
    iters = [h['iter'] for h in history]
    obj_vals = [h['objective'] for h in history]
    violations = [h['violation'] for h in history]

    # 左图：目标函数收敛
    ax = axes[0]
    ax.plot(iters, obj_vals, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('SMO: Objective Function Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 右图：约束满足
    ax = axes[1]
    ax.plot(iters, violations, 'r-', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Constraint Violation |sum(alpha*y)|', fontsize=12)
    ax.set_title('SMO: Constraint Satisfaction', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("svm_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为 svm_convergence.png\n")


def experiment_augmented_lagrangian():
    """实验六：增广拉格朗日法 — 不同 rho 初始值对收敛速度的影响"""
    print("\n" + "=" * 60)
    print("实验六：增广拉格朗日法参数影响分析")
    print("=" * 60)

    X, y = generate_data(n=100, center1=(1, 1), center2=(3, 3))

    # 不同 rho 初始值
    rho_values = [0.1, 1.0, 10.0, 100.0]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, rho in enumerate(rho_values):
        ax = axes[idx // 2, idx % 2]
        print(f"\nrho_0 = {rho}")

        svm = AugmentedLagrangianSVM(C=1.0, kernel='linear')
        svm.fit(X, y, rho_init=rho, rho_scale=10.0, max_outer=50, verbose=False)

        # 左Y轴：目标函数值
        history = svm.history
        outers = [h['iter'] for h in history]
        obj_vals = [h['objective'] for h in history]

        # 右Y轴：约束违反度（对数坐标）
        violations = [max(h['violation'], 1e-16) for h in history]

        ax2 = ax.twinx()
        ax.plot(outers, obj_vals, '-o', color='#2196F3', linewidth=1, markersize=4, label='Objective')
        ax2.semilogy(outers, violations, '-s', color='#F44336', linewidth=1, markersize=4, label='Violation')

        ax.set_xlabel('Outer Iteration', fontsize=11)
        ax.set_ylabel('Objective Value', color='#2196F3', fontsize=11)
        ax2.set_ylabel('Constraint Violation (log)', color='#F44336', fontsize=11)
        ax.set_title(f'Augmented Lagrangian (rho_0={rho})', fontsize=13)
        ax.grid(True, alpha=0.2)

        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

        print(f"  外层迭代: {len(outers)}, 最终obj={obj_vals[-1]:.4f}, violation={violations[-1]:.2e}")

    plt.suptitle('Effect of rho_0 on Augmented Lagrangian Convergence', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig("svm_al_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n图片已保存为 svm_al_convergence.png")

    # 对比 SMO 和 AL 的结果
    print("\n" + "-" * 50)
    print("SMO vs 增广拉格朗日法对比 (C=1.0, 线性不可分数据):")
    print("-" * 50)

    # SMO
    print("\n[1] SMO算法:")
    svm_smo = SVM(C=1.0, kernel='linear')
    svm_smo.fit(X, y, verbose=True)

    # AL
    print("\n[2] 增广拉格朗日法 (rho_0=1.0):")
    svm_al = AugmentedLagrangianSVM(C=1.0, kernel='linear')
    svm_al.fit(X, y, rho_init=1.0, rho_scale=10.0, max_outer=50, verbose=True)

    # 画对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_svm_result(svm_smo, X, y, title="SMO Algorithm", ax=axes[0])
    plot_svm_result(svm_al, X, y, title="Augmented Lagrangian", ax=axes[1])
    plt.tight_layout()
    plt.savefig("svm_smo_vs_al.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n图片已保存为 svm_smo_vs_al.png\n")


def experiment_c_comparison():
    """实验四：C值对SVM影响的详细对比（线性不可分）"""
    print("\n" + "=" * 60)
    print("实验四：C值对分类效果的影响")
    print("=" * 60)

    X, y = generate_data(n=100, center1=(1, 1), center2=(3, 3))

    C_range = np.logspace(-2, 2, 8)  # 0.01 到 100

    n_sv = []
    n_boundary_sv = []
    margin_widths = []

    for C in C_range:
        svm = SVM(C=C, kernel='linear')
        svm.fit(X, y, verbose=False)

        n_sv.append(len(svm.support_vec_idx))
        # 边界上的支持向量
        threshold = 1e-5
        bsv = np.sum((svm.alpha > threshold) & (svm.alpha < C - threshold))
        n_boundary_sv.append(bsv)
        # 间隔宽度 = 2 / ||w||
        margin_widths.append(2.0 / np.linalg.norm(svm.w))

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].semilogx(C_range, n_sv, 'bo-', linewidth=1.5, markersize=6)
    axes[0].set_xlabel('C (log scale)', fontsize=12)
    axes[0].set_ylabel('Number of Support Vectors', fontsize=12)
    axes[0].set_title('Support Vectors vs C', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(C_range, n_boundary_sv, 'rs-', linewidth=1.5, markersize=6)
    axes[1].set_xlabel('C (log scale)', fontsize=12)
    axes[1].set_ylabel('Support Vectors on Margin Boundary', fontsize=12)
    axes[1].set_title('Boundary SVs vs C', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(C_range, margin_widths, 'g^-', linewidth=1.5, markersize=6)
    axes[2].set_xlabel('C (log scale)', fontsize=12)
    axes[2].set_ylabel('Margin Width (2/||w||)', fontsize=12)
    axes[2].set_title('Margin Width vs C', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("svm_c_effect.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为 svm_c_effect.png\n")

    # 打印C值对支持向量的影响
    print("\nC值对支持向量的影响:")
    print(f"{'C':>10} {'总SV数':>8} {'边界SV数':>8} {'间隔宽度':>10}")
    print("-" * 40)
    for C, sv, bsv, mw in zip(C_range, n_sv, n_boundary_sv, margin_widths):
        print(f"{C:10.3f} {sv:8d} {bsv:8d} {mw:10.4f}")


def experiment_soft_vs_hard():
    """实验五：软间隔 vs 硬间隔效果对比（针对线性可分数据）"""
    print("\n" + "=" * 60)
    print("实验五：线性可分数据上 C 值对比")
    print("=" * 60)

    X, y = generate_data(n=100, center1=(1, 1), center2=(6, 6))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 小C（软间隔）
    svm_small = SVM(C=0.1, kernel='linear')
    svm_small.fit(X, y, verbose=False)
    plot_svm_result(svm_small, X, y, title="SVM (C = 0.1, Soft Margin)", ax=axes[0])

    # 中C
    svm_medium = SVM(C=1.0, kernel='linear')
    svm_medium.fit(X, y, verbose=False)
    plot_svm_result(svm_medium, X, y, title="SVM (C = 1.0)", ax=axes[1])

    # 大C（近似硬间隔）
    svm_large = SVM(C=100.0, kernel='linear')
    svm_large.fit(X, y, verbose=False)
    plot_svm_result(svm_large, X, y, title="SVM (C = 100, ~Hard Margin)", ax=axes[2])

    plt.tight_layout()
    plt.savefig("svm_soft_vs_hard.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为 svm_soft_vs_hard.png\n")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("SVM 手动实现（SMO 算法）")
    print("=" * 60)
    print("SVM 对偶问题推导：")
    print("  原始问题（软间隔）：")
    print("    min_{w,b,xi}  (1/2)||w||^2 + C * sum(xi_i)")
    print("    s.t.  y_i(w^T x_i + b) >= 1 - xi_i,  xi_i >= 0")
    print()
    print("  对偶问题：")
    print("    min_alpha  (1/2) sum_i sum_j alpha_i alpha_j y_i y_j (x_i^T x_j) - sum_i alpha_i")
    print("    s.t.   sum_i alpha_i y_i = 0,  0 <= alpha_i <= C")
    print()
    print("  决策函数： f(x) = sum_i alpha_i y_i (x_i^T x) + b")
    print("  其中 b = mean_{sv}( y_sv - sum_i alpha_i y_i (x_i^T x_sv) )")
    print("=" * 60)

    # 运行所有实验
    experiment_separable()
    experiment_nonseparable()
    experiment_convergence()
    experiment_c_comparison()
    experiment_soft_vs_hard()
    experiment_augmented_lagrangian()

    print("\n所有实验完成！生成的图片：")
    print("  1. svm_separable.png       - 线性可分SVM结果")
    print("  2. svm_nonseparable.png    - 线性不可分SVM（不同C值）")
    print("  3. svm_convergence.png     - SMO收敛性分析")
    print("  4. svm_c_effect.png        - C值对分类效果的影响")
    print("  5. svm_soft_vs_hard.png    - 软间隔vs硬间隔对比")
    print("  6. svm_al_convergence.png  - 增广拉格朗日法收敛性")
    print("  7. svm_smo_vs_al.png       - SMO vs 增广拉格朗日法对比")
