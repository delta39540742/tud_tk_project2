import random
from linalg_utils import (shape, matvec)
from ols_implementation import (ols_fit)

## k-fold cross-validation


# ============================================================
# (b) CÀI ĐẶT TỪ ĐẦU
# ============================================================
def _shuffle_indices(n, seed=None):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx


def _split_into_folds(indices, k):
    """
    Chia danh sách chỉ số thành k phần cân bằng nhất có thể.
    Nếu n không chia hết cho k, một số fold đầu có thêm 1 phần tử
    (giống sklearn.KFold).
    """
    n = len(indices)
    base = n // k
    rem  = n %  k
    folds = []
    start = 0
    for j in range(k):
        size = base + (1 if j < rem else 0)
        folds.append(indices[start:start + size])
        start += size
    return folds


def kfold_cv(X, y, k, shuffle=True, seed=42):
    """
    k-fold cross-validation cho OLS regression.

    Trả về dict:
        cv_score   : trung bình MSE qua k folds
        fold_mses  : list MSE từng fold
        fold_betas : list β̂^(j)
    """
    n, p = shape(X)
    assert k >= 2 and k <= n, "k phải trong [2, n]"

    idx = _shuffle_indices(n, seed if shuffle else None)
    folds = _split_into_folds(idx, k)

    fold_mses = []
    fold_betas = []

    for j in range(k):
        test_idx = folds[j]
        # train_idx = mọi chỉ số trừ fold j
        train_idx = [i for jj in range(k) if jj != j for i in folds[jj]]

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]

        if len(X_train) <= len(X_train[0]):
            raise ValueError(
                "Train fold không đủ số dòng so với số biến. "
                "Hãy giảm k hoặc giảm số biến trong X."
            )

        beta_j, _ = ols_fit(X_train, y_train)
        y_pred = matvec(X_test, beta_j)
        mse_j = sum((y_test[i] - y_pred[i]) ** 2
                    for i in range(len(y_test))) / len(y_test)

        fold_mses.append(mse_j)
        fold_betas.append(beta_j)

    cv_score = sum(fold_mses) / k
    return {
        "cv_score":   cv_score,
        "fold_mses":  fold_mses,
        "fold_betas": fold_betas,
    }

# ============================================================
# (c) DỮ LIỆU GIẢ LẬP
# ============================================================
def simulate_regression(n=200, p=4, sigma=1.0, seed=0):
    """y = Xβ + ε, ε ~ N(0, σ²).  Cột đầu là intercept = 1."""
    rng = random.Random(seed)
    true_beta = [2.0, -1.5, 0.8, 0.3, -0.6][:p]
    X = []
    y = []
    for _ in range(n):
        row = [1.0] + [rng.gauss(0, 1) for _ in range(p - 1)]
        X.append(row)
        eps = rng.gauss(0, sigma)
        y.append(sum(row[j] * true_beta[j] for j in range(p)) + eps)
    return X, y, true_beta


# ============================================================
# (d) KIỂM CHỨNG VỚI SKLEARN
# ============================================================
def verify_with_sklearn(X, y, k, seed=42):
    try:
        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
    except ImportError:
        print("[Bỏ qua kiểm chứng] numpy/sklearn không có sẵn.")
        return

    # Để so sánh "chính xác" tới cùng phân chia fold:
    # mình sẽ tự tách bằng cùng thứ tự rồi ép sklearn dùng đúng các index đó.
    n = len(X)
    idx = _shuffle_indices(n, seed)
    folds = _split_into_folds(idx, k)

    X_np = np.array(X)
    y_np = np.array(y)

    sk_mses = []
    for j in range(k):
        test_idx  = folds[j]
        train_idx = [i for jj in range(k) if jj != j for i in folds[jj]]
        model = LinearRegression(fit_intercept=False)
        model.fit(X_np[train_idx], y_np[train_idx])
        y_pred = model.predict(X_np[test_idx])
        sk_mses.append(mean_squared_error(y_np[test_idx], y_pred))

    sk_cv = sum(sk_mses) / k
    print(f"\n--- (d) Kiểm chứng với sklearn (cùng phân chia fold) ---")
    print(f"CV sklearn = {sk_cv:.8f}")
    return sk_cv, sk_mses

