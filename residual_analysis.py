from linalg_utils import (shape, matvec, )
from ols_implementation import (hat_matrix)
import math
import random


# ============================================================
# (b) CÁC ĐẠI LƯỢNG NỀN — CÀI TỪ ĐẦU
# ============================================================
def fitted_and_residuals(X, y, beta_hat):
    n, _ = shape(X)
    y_hat = matvec(X, beta_hat)
    e = [y[i] - y_hat[i] for i in range(n)]
    return y_hat, e


def leverages(X):
    """Trả về list các h_ii."""
    H, _ = hat_matrix(X)
    return [H[i][i] for i in range(len(H))]


def standardized_residuals(e, h, sigma2):
    """r_i = e_i / (σ̂ √(1 - h_i))."""
    sig = math.sqrt(sigma2)
    out = []
    for i in range(len(e)):
        denom = sig * math.sqrt(max(1.0 - h[i], 1e-12))
        out.append(e[i] / denom)
    return out


def cooks_distance(r, h, p):
    """D_i = (r_i² / p) · h_i / (1 - h_i)."""
    out = []
    for i in range(len(r)):
        out.append((r[i] ** 2 / p) * h[i] / max(1.0 - h[i], 1e-12))
    return out


# ----- Inverse CDF của N(0, 1): Acklam's approximation -----
# Sai số tuyệt đối ~ 1.15e-9. Đủ chính xác cho Q-Q plot.
def norm_ppf(p):
    if not (0.0 < p < 1.0):
        raise ValueError("norm_ppf: p phải trong (0, 1)")
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)

    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1.0)


# ============================================================
# (b) HÀM CHÍNH — VẼ 4 BIỂU ĐỒ
# ============================================================
def residual_plots(X, y, beta_hat, savepath="residual_plots.png"):
    """
    Vẽ bộ 4 biểu đồ phân tích phần dư cho hồi quy tuyến tính.

    Mọi đại lượng (e, h, σ̂², r, Cook's D) đều tính từ đầu, không numpy.
    matplotlib chỉ làm nhiệm vụ hiển thị.
    """
    n, p = shape(X)

    # --- Tính các đại lượng ---
    y_hat, e = fitted_and_residuals(X, y, beta_hat)
    h = leverages(X)
    rss = sum(ei * ei for ei in e)
    sigma2 = rss / max(n - p, 1)
    r = standardized_residuals(e, h, sigma2)
    cook = cooks_distance(r, h, p)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Bỏ qua vẽ] matplotlib không có sẵn.")
        return {"residuals": e, "leverage": h, "sigma2": sigma2,
                "std_residuals": r, "cooks": cook}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # ----- 1. Residuals vs Fitted -----
    ax = axes[0, 0]
    ax.scatter(y_hat, e, alpha=0.65, s=22, edgecolor="none")
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Giá trị fitted ŷ")
    ax.set_ylabel("Phần dư e")
    ax.set_title("1. Residuals vs Fitted")
    ax.grid(True, alpha=0.3)

    # ----- 2. Normal Q–Q -----
    ax = axes[0, 1]
    e_sorted = sorted(r)                 # dùng phần dư chuẩn hóa cho Q-Q
    theo = [norm_ppf((i + 0.5) / n) for i in range(n)]
    ax.scatter(theo, e_sorted, alpha=0.65, s=22, edgecolor="none")
    # Đường tham chiếu y = x
    lo, hi = min(theo[0], e_sorted[0]), max(theo[-1], e_sorted[-1])
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Quantile lý thuyết (chuẩn)")
    ax.set_ylabel("Quantile của r (đã sort)")
    ax.set_title("2. Normal Q–Q")
    ax.grid(True, alpha=0.3)

    # ----- 3. Scale–Location -----
    ax = axes[1, 0]
    sqrt_abs_r = [math.sqrt(abs(ri)) for ri in r]
    ax.scatter(y_hat, sqrt_abs_r, alpha=0.65, s=22, edgecolor="none")
    ax.set_xlabel("Giá trị fitted ŷ")
    ax.set_ylabel("√|r| (chuẩn hóa)")
    ax.set_title("3. Scale–Location")
    ax.grid(True, alpha=0.3)

    # ----- 4. Residuals vs Leverage (kèm đường Cook's = 0.5, 1) -----
    ax = axes[1, 1]
    ax.scatter(h, r, alpha=0.65, s=22, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6)
    # Đường mức Cook's distance D = c:  r² = c · p · (1 - h) / h
    h_grid = [0.001 + 0.001 * k for k in range(int(0.999 / 0.001))]
    for D_level, ls in [(0.5, "--"), (1.0, ":")]:
        rs_plus = []
        rs_minus = []
        hs = []
        for hh in h_grid:
            if hh <= 0 or hh >= 1:
                continue
            val = D_level * p * (1.0 - hh) / hh
            if val < 0:
                continue
            rs_plus.append(math.sqrt(val))
            rs_minus.append(-math.sqrt(val))
            hs.append(hh)
        ax.plot(hs, rs_plus,  color="red", linestyle=ls, linewidth=0.8,
                label=f"Cook D = {D_level}")
        ax.plot(hs, rs_minus, color="red", linestyle=ls, linewidth=0.8)
    ax.set_xlabel("Leverage h_ii")
    ax.set_ylabel("Phần dư chuẩn hóa r")
    ax.set_title("4. Residuals vs Leverage")
    ax.set_xlim(0, max(h) * 1.1 + 1e-6)
    # Giới hạn y hợp lý
    rmax = max(abs(min(r)), abs(max(r))) * 1.2 + 1
    ax.set_ylim(-rmax, rmax)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=120)
    plt.close(fig)
    print(f"[OK] Đã lưu {savepath}")

    # hiển thị trực quan
    from IPython.display import Image
    display(Image(savepath))

    return {"residuals": e, "leverage": h, "sigma2": sigma2,
            "std_residuals": r, "cooks": cook}


# ============================================================
# (c) DỮ LIỆU GIẢ LẬP
# ============================================================
def simulate_with_outliers(n=80, seed=7):
    """Trộn vài điểm outlier & một điểm leverage cao để 4 biểu đồ có gì xem."""
    rng = random.Random(seed)
    X = []
    y = []
    true_beta = [2.0, -1.5, 0.8]      # cột 1 là intercept (đặt =1)

    for i in range(n - 2):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        # Heteroscedasticity nhẹ: sigma tăng theo |x1|
        noise = rng.gauss(0, 0.5 + 0.3 * abs(x1))
        yi = true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2 + noise
        X.append([1.0, x1, x2])
        y.append(yi)

    # Điểm outlier có leverage cao
    X.append([1.0,  6.0,  6.0]);  y.append(20.0)
    # Outlier về y (leverage thường, residual lớn)
    X.append([1.0,  0.5, -0.3]);  y.append(15.0)

    return X, y, true_beta


# ============================================================
# (d) KIỂM CHỨNG (so leverage & Cook's với statsmodels)
# ============================================================
def verify_with_statsmodels(X, y, mine):
    try:
        import numpy as np
        import statsmodels.api as sm
    except ImportError:
        print("[Bỏ qua kiểm chứng] statsmodels không có sẵn — "
              "có thể `pip install statsmodels` để chạy phần này.")
        return

    X_np = np.array(X)
    y_np = np.array(y)
    res = sm.OLS(y_np, X_np).fit()
    infl = res.get_influence()

    h_sm = infl.hat_matrix_diag.tolist()
    cook_sm = infl.cooks_distance[0].tolist()

    n = len(mine["leverage"])
    h_diff = max(abs(mine["leverage"][i] - h_sm[i]) for i in range(n))
    cook_diff = max(abs(mine["cooks"][i] - cook_sm[i]) for i in range(n))

    print("\n--- (d) Kiểm chứng leverage & Cook's với statsmodels ---")
    print(f"max |h_ii   - h_sm|     = {h_diff:.3e}")
    print(f"max |Cook_i - Cook_sm|  = {cook_diff:.3e}")

