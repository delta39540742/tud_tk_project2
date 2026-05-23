import numpy as np
import math 


def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def mat_mul(A, B):
    result = [[sum(a * b for a, b in zip(row, col)) for col in transpose(B)] for row in A]
    return result

def get_matrix_minor(m, i, j):
    return [row[:j] + row[j+1:] for row in (m[:i] + m[i+1:])]

def get_matrix_det(m):
    if len(m) == 1: return m[0][0]
    if len(m) == 2: return m[0][0]*m[1][1] - m[0][1]*m[1][0]
    det = 0
    for c in range(len(m)):
        det += ((-1)**c) * m[0][c] * get_matrix_det(get_matrix_minor(m, 0, c))
    return det

def invert_matrix(m):
    det = get_matrix_det(m)
    if det == 0: raise ValueError("Ma trận không khả nghịch")
    # Ma trận nghịch đảo = (1/det) * Adj(M)
    n = len(m)
    if n == 1: return [[1/m[0][0]]]
    cofactors = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            minor = get_matrix_minor(m, r, c)
            cofactor_row.append(((-1)**(r+c)) * get_matrix_det(minor))
        cofactors.append(cofactor_row)
    adj = transpose(cofactors)
    return [[val / det for val in row] for row in adj]

def t_pdf(t, df):
    """Hàm mật độ xác suất của phân phối t-Student (xấp xỉ)."""
    return (1 + (t**2) / df)**(-(df + 1) / 2)

def t_cdf(t, df):
    """
    Xấp xỉ hàm phân phối tích lũy (CDF) của phân phối t-Student bằng chuỗi Taylor.
    Độ chính xác cao cho các bài toán thống kê thông thường.
    """
    import math # Chỉ dùng math có sẵn của Python để tính toán số học căn bản (pi, sin, cos)
    
    x = t / math.sqrt(df)
    theta = math.atan(x)
    
    if df == 1:
        return 0.5 + theta / math.pi
    
    # Tính tổng chuỗi dựa trên bậc tự do chẵn hay lẻ
    total = 0.0
    if df % 2 == 1: # dof lẻ
        total = math.sin(theta)
        term = math.sin(theta)
        for i in range(3, df, 2):
            term *= (cos_sq := math.cos(theta)**2) * (i - 1) / i
            total += term
        return 0.5 + (theta + total * math.cos(theta)) / math.pi
    else:           # dof chẵn
        total = math.sin(theta)
        term = math.sin(theta)
        for i in range(2, df, 2):
            term *= (cos_sq := math.cos(theta)**2) * (i - 1) / i
            total += term
        return 0.5 + total / 2.0

def _get_t_crit(alpha, df):
    """Tìm giá trị t tới hạn bằng phương pháp chia đôi (Bisection)."""
    low, high = 0, 10
    for _ in range(20):
        mid = (low + high) / 2
        # Diện tích hai đuôi: 1 - P(-mid < T < mid) = alpha
        p_val = 2 * (1 - t_cdf(mid, df))
        if p_val > alpha:
            low = mid
        else:
            high = mid
            
    # Tái tính toán mid sau khi đã tối ưu hóa low và high
    return (low + high) / 2

# ============================================================
# Truy vấn kích thước
# ============================================================
def shape(A):
    """Trả về (số hàng, số cột)."""
    return (len(A), len(A[0]) if A else 0)


# ============================================================
# Khởi tạo ma trận
# ============================================================
def zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def eye(n):
    """Ma trận đơn vị n×n."""
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


# ============================================================
# Phép toán cơ bản
# ============================================================

def mat_add(A, B):
    n, m = shape(A)
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def mat_sub(A, B):
    n, m = shape(A)
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]


def mat_scale(A, c):
    n, m = shape(A)
    return [[A[i][j] * c for j in range(m)] for i in range(n)]


def matmul(A, B):
    """C = A · B với A là n×m, B là m×p."""
    n, m = shape(A)
    m2, p = shape(B)
    assert m == m2, f"matmul: kích thước không khớp ({m} vs {m2})"
    C = zeros(n, p)
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(m):
            aik = Ai[k]
            if aik == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                Ci[j] += aik * Bk[j]
    return C


def matvec(A, x):
    """A · x."""
    n, m = shape(A)
    assert len(x) == m, f"matvec: kích thước không khớp ({m} vs {len(x)})"
    return [sum(A[i][j] * x[j] for j in range(m)) for i in range(n)]


# ============================================================
# Vector
# ============================================================
def vec_sub(x, y): return [x[i] - y[i] for i in range(len(x))]
def vec_add(x, y): return [x[i] + y[i] for i in range(len(x))]
def vec_scale(x, c): return [xi * c for xi in x]
def vec_dot(x, y): return sum(x[i] * y[i] for i in range(len(x)))
def vec_norm(x): return math.sqrt(sum(xi * xi for xi in x))


# ============================================================
# Nghịch đảo ma trận: Gauss–Jordan có partial pivoting
# ============================================================
def inv(A):
    """
    Tính A^{-1} bằng phương pháp Gauss–Jordan với pivoting riêng phần.
    Raise ValueError nếu ma trận suy biến.
    """
    n = len(A)
    # Ma trận mở rộng [A | I], copy để không sửa A
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]

    for col in range(n):
        # Tìm pivot có |giá trị| lớn nhất trong cột (ổn định số học)
        pivot_row = col
        pivot_val = abs(M[col][col])
        for k in range(col + 1, n):
            if abs(M[k][col]) > pivot_val:
                pivot_val = abs(M[k][col])
                pivot_row = k
        if pivot_val < 1e-14:
            raise ValueError(f"inv: ma trận suy biến (cột {col})")

        # Hoán đổi
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]

        # Chuẩn hóa hàng pivot
        piv = M[col][col]
        inv_piv = 1.0 / piv
        for j in range(2 * n):
            M[col][j] *= inv_piv

        # Khử các hàng còn lại
        for k in range(n):
            if k == col:
                continue
            factor = M[k][col]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                M[k][j] -= factor * M[col][j]

    return [[M[i][j + n] for j in range(n)] for i in range(n)]


def solve(A, b):
    """Giải Ax = b (A vuông, không suy biến)."""
    return matvec(inv(A), b)


# ============================================================
# Một vài thống kê cơ bản dùng cho mọi bài
# ============================================================
def mean(x):
    return sum(x) / len(x)


def variance(x, ddof=0):
    """Phương sai mẫu. ddof=0: chia n; ddof=1: chia n-1."""
    n = len(x)
    m = mean(x)
    return sum((xi - m) ** 2 for xi in x) / (n - ddof)


def std(x, ddof=0):
    return math.sqrt(variance(x, ddof=ddof))



