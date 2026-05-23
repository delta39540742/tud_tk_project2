# import tools_for_matrix_manipulation as tm
from linalg_utils import (transpose, mat_mul, invert_matrix, t_cdf, _get_t_crit)

# 1. ols_fit
def ols_fit(X, y):
    XT = transpose(X)
    XTX = mat_mul(XT, X)
    XTX_inv = invert_matrix(XTX)
    XTy = mat_mul(XT, [[yi] for yi in y])
    beta_hat = mat_mul(XTX_inv, XTy) # Kết quả dạng [[b0], [b1]...]
    
    n, p = len(X), len(X[0])
    y_hat = [sum(X[i][j] * beta_hat[j][0] for j in range(p)) for i in range(n)]
    rss = sum((y[i] - y_hat[i])**2 for i in range(n))
    sigma2 = rss / (n - p)
    return [b[0] for b in beta_hat], sigma2

# 2. hat_matrix
def hat_matrix(X):
    XT = transpose(X)
    XTX_inv = invert_matrix(mat_mul(XT, X))
    H = mat_mul(mat_mul(X, XTX_inv), XT)
    # Kiểm tra idempotent: H*H = H
    H2 = mat_mul(H, H)
    is_idempotent = all(abs(H2[i][j] - H[i][j]) < 1e-9 for i in range(len(H)) for j in range(len(H[0])))
    return H, is_idempotent

# 3. model_metrics
def model_metrics(y, y_hat, p):
    n = len(y)
    y_mean = sum(y) / n
    rss = sum((y[i] - y_hat[i])**2 for i in range(n))
    tss = sum((y[i] - y_mean)**2 for i in range(n))
    r2 = 1 - (rss / tss)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)
    f_stat = ((tss - rss) / (p - 1)) / (rss / (n - p))
    return {"RSS": rss, "TSS": tss, "R2": r2, "Adj_R2": adj_r2, "F": f_stat}

# 4. coef_inference
def coef_inference(X, y, beta_hat, sigma2, alpha=0.05):
    import math
    n, p = len(X), len(X[0])
    
    # 1. Tính Toán Standard Errors & T-statistics (Code gốc của bạn)
    XTX_inv = invert_matrix(mat_mul(transpose(X), X))
    var_beta = [[sigma2 * XTX_inv[i][j] for j in range(p)] for i in range(p)]
    se = [var_beta[i][i]**0.5 for i in range(p)]
    t_stats = [beta_hat[i] / se[i] for i in range(p)]
    
    # 2. Số bậc tự do
    dof = n - p
    
    # 3. Tính P-values thuần Python (Kiểm định 2 phía)
    p_values = []
    for t in t_stats:
        abs_t = abs(t)
        # P-value = 2 * (1 - CDF(abs_t))
        cdf_val = t_cdf(abs_t, dof)
        p_val = 2 * (1.0 - cdf_val)
        p_values.append(p_val)
        
    # 4. Tính Khoảng tin cậy thuần Python
    t_crit = _get_t_crit(alpha, dof)
    
    confidence_intervals = []
    for i in range(p):
        margin_of_error = t_crit * se[i]
        lower_bound = beta_hat[i] - margin_of_error
        upper_bound = beta_hat[i] + margin_of_error
        confidence_intervals.append((lower_bound, upper_bound))
        
    return se, t_stats, p_values, confidence_intervals

# 5. vif
def vif(X):
    X_sub = [row[1:] for row in X]
    p_sub = len(X_sub[0])
    vifs = []
    for i in range(p_sub):
        y_i = [row[i] for row in X_sub]
        X_others = [[1] + row[:i] + row[i+1:] for row in X_sub]
        b_i, _ = ols_fit(X_others, y_i)
        y_i_hat = [sum(X_others[k][j] * b_i[j] for j in range(len(b_i))) for k in range(len(X_sub))]
        rss_i = sum((y_i[k] - y_i_hat[k])**2 for k in range(len(y_i)))
        y_i_mean = sum(y_i) / len(y_i)
        tss_i = sum((y_i[k] - y_i_mean)**2 for k in range(len(y_i)))
        vifs.append(1 / (1 - (1 - rss_i/tss_i)))
    return vifs




