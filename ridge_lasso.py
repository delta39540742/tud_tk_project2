# ============================================================
#  hồi quy ridge 
# ============================================================
from linalg_utils import (
    shape, mat_add, transpose, matmul, matvec, inv, mat_scale, eye
    )
import random 
import math 
import matplotlib.pyplot as plt

def ridge_fit(X, y, lam):
    """
    Ridge Regression cài từ đầu.

    Tham số:
        X   : list[list[float]]  – ma trận thiết kế n×p
        y   : list[float]        – vector mục tiêu độ dài n
        lam : float ≥ 0          – hệ số regularization λ

    Trả về:
        beta_hat : list[float] độ dài p,  β̂ = (X^T X + λI)^{-1} X^T y
    """
    assert lam >= 0, "lam phải không âm"
    n, p = shape(X)

    Xt   = transpose(X)              # p×n
    XtX  = matmul(Xt, X)             # p×p
    Xty  = matvec(Xt, y)             # p

    # X^T X + λI
    XtX_reg = mat_add(XtX, mat_scale(eye(p), lam))

    return matvec(inv(XtX_reg), Xty)

# ============================================================
#  ridge trace
# ============================================================
def ridge_trace(X, y, lambdas):
    """
    Tính β̂(λ) cho danh sách λ. Trả về list các vector hệ số,
    mỗi phần tử ứng với một λ.
    """
    return [ridge_fit(X, y, lam) for lam in lambdas]


# ============================================================
# (c) MINH HỌA BẰNG DỮ LIỆU GIẢ LẬP
# ============================================================



def standardize_columns(X):
    """
    Trừ trung bình rồi chia độ lệch chuẩn của từng cột.
    Ridge cần X đã chuẩn hóa, nếu không thì kết quả co không công bằng
    giữa các biến có thang đo khác nhau.
    """
    n, p = shape(X)
    means = [sum(X[i][j] for i  in range(n)) / n for j in range(p)]
    stds = []
    for j in range(p):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    X_std = [[(X[i][j] - means[j]) / stds[j] for j in range(p)]
             for i in range(n)]
    return X_std, means, stds


def center(y):
    """Trừ trung bình khỏi y (để khỏi cần intercept)."""
    m = sum(y) / len(y)
    return [yi - m for yi in y], m


def simulate_collinear(n=120, seed=42):
    """
    Sinh dữ liệu có đa cộng tuyến mạnh:
        x3 ≈ x1,  x4 ≈ x2,  x5 = 0.5 x1 + 0.5 x2 + nhiễu nhỏ.
    Đây chính là tình huống ridge phát huy tác dụng so với OLS.
    """
    rng = random.Random(seed)
    X = []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        x3 = x1 + 0.02 * rng.gauss(0, 1)
        x4 = x2 + 0.02 * rng.gauss(0, 1)
        x5 = 0.5 * x1 + 0.5 * x2 + 0.02 * rng.gauss(0, 1)
        X.append([x1, x2, x3, x4, x5])

    true_beta = [1.0, -1.0, 0.5, -0.5, 2.0]
    y = []
    for i in range(n):
        yi = sum(X[i][j] * true_beta[j] for j in range(5)) + rng.gauss(0, 1.0)
        y.append(yi)
    return X, y, true_beta


def plot_ridge_trace(lambdas, traces, true_beta=None,
                     savepath="ridge_trace.png"):
    """Vẽ ridge trace bằng matplotlib (matplotlib chỉ dùng để vẽ, không để tính)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Bỏ qua vẽ] matplotlib không có sẵn.")
        return

    p = len(traces[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(p):
        ys = [traces[t][j] for t in range(len(lambdas))]
        ax.plot(lambdas, ys, marker="o", markersize=3, label=f"β{j+1}")
        if true_beta is not None:
            ax.axhline(true_beta[j], linestyle=":", linewidth=0.7, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("λ (thang log)")
    ax.set_ylabel("β̂_j(λ)")
    ax.set_title("Ridge Trace")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=120)
    plt.close(fig)
    print(f"[OK] Đã lưu {savepath}")


# ============================================================
# (d) KIỂM CHỨNG VỚI SKLEARN
# ============================================================
def verify_with_sklearn(X_std, y_centered, lambdas_check):
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
    except ImportError:
        print("[Bỏ qua kiểm chứng] numpy/sklearn không có sẵn.")
        return

    X_np = np.array(X_std)
    y_np = np.array(y_centered)

    print("\n--- (d) Kiểm chứng với sklearn.Ridge ---")
    print(f"{'λ':>8} | {'sai khác max':>14}")
    print("-" * 28)
    for lam in lambdas_check:
        mine = ridge_fit(X_std, y_centered, lam)
        # sklearn Ridge tối thiểu hoá ||y - Xβ||² + α||β||², khớp định nghĩa của ta
        sk = Ridge(alpha=lam, fit_intercept=False, solver="cholesky")
        sk.fit(X_np, y_np)
        sk_beta = sk.coef_.tolist()
        diff = max(abs(mine[j] - sk_beta[j]) for j in range(len(mine)))
        print(f"{lam:>8.3f} | {diff:>14.3e}")