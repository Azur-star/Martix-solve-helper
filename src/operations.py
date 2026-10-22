"""矩阵运算：所有算法从零实现，展示数据结构与算法知识。"""

from fractions import Fraction
from typing import Tuple, List
from .matrix import Matrix


# ── 基本算术 ──────────────────────────────────────────────

def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    """矩阵加法 O(mn)"""
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError(f"维度不匹配: {a.rows}x{a.cols} 与 {b.rows}x{b.cols}")
    return Matrix([
        [a.data[i][j] + b.data[i][j] for j in range(a.cols)]
        for i in range(a.rows)
    ])


def matrix_sub(a: Matrix, b: Matrix) -> Matrix:
    """矩阵减法 O(mn)"""
    if a.rows != b.rows or a.cols != b.cols:
        raise ValueError(f"维度不匹配: {a.rows}x{a.cols} 与 {b.rows}x{b.cols}")
    return Matrix([
        [a.data[i][j] - b.data[i][j] for j in range(a.cols)]
        for i in range(a.rows)
    ])


def matrix_mul(a: Matrix, b: Matrix) -> Matrix:
    """矩阵乘法 O(m·n·p)，三重循环实现。"""
    if a.cols != b.rows:
        raise ValueError(f"维度不匹配: {a.rows}x{a.cols} 与 {b.rows}x{b.cols}")
    m, n, p = a.rows, a.cols, b.cols
    data = [[Fraction(0)] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            a_ik = a.data[i][k]
            if a_ik == 0:
                continue
            for j in range(p):
                data[i][j] += a_ik * b.data[k][j]
    return Matrix(data)


def scalar_mul(s, m: Matrix) -> Matrix:
    """标量乘法 O(mn)"""
    s = Fraction(s)
    return Matrix([[s * m.data[i][j] for j in range(m.cols)] for i in range(m.rows)])


def transpose(m: Matrix) -> Matrix:
    """矩阵转置 O(mn)"""
    return Matrix([[m.data[i][j] for i in range(m.rows)] for j in range(m.cols)])


# ── 行变换与行阶梯形 ──────────────────────────────────────

def row_echelon(m: Matrix) -> Tuple[Matrix, Fraction]:
    """
    高斯消元化为行阶梯形 (REF)。
    返回 (结果矩阵, 行列式符号因子)。
    使用部分主元选取 (partial pivoting) 提高数值稳定性。
    时间复杂度: O(n^3)
    """
    mat = m.copy()
    n_rows, n_cols = mat.rows, mat.cols
    sign = Fraction(1)
    pivot_row = 0

    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        # 部分主元选取：在当前列中找绝对值最大的元素
        max_row = pivot_row
        max_val = abs(mat.data[pivot_row][col])
        for r in range(pivot_row + 1, n_rows):
            val = abs(mat.data[r][col])
            if val > max_val:
                max_val = val
                max_row = r
        if max_val == 0:
            continue
        if max_row != pivot_row:
            mat.data[pivot_row], mat.data[max_row] = mat.data[max_row], mat.data[pivot_row]
            sign *= -1
        # 消去下方元素
        pivot_val = mat.data[pivot_row][col]
        for r in range(pivot_row + 1, n_rows):
            if mat.data[r][col] == 0:
                continue
            factor = mat.data[r][col] / pivot_val
            for c in range(col, n_cols):
                mat.data[r][c] -= factor * mat.data[pivot_row][c]
        pivot_row += 1

    return mat, sign


def reduced_row_echelon(m: Matrix) -> Matrix:
    """
    高斯-约当消元化为简化行阶梯形 (RREF)。
    用于求逆矩阵、解线性方程组、求秩等。
    时间复杂度: O(n^3)
    """
    mat = m.copy()
    n_rows, n_cols = mat.rows, mat.cols
    pivot_row = 0

    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        # 部分主元选取
        max_row = pivot_row
        max_val = abs(mat.data[pivot_row][col])
        for r in range(pivot_row + 1, n_rows):
            val = abs(mat.data[r][col])
            if val > max_val:
                max_val = val
                max_row = r
        if max_val == 0:
            continue
        if max_row != pivot_row:
            mat.data[pivot_row], mat.data[max_row] = mat.data[max_row], mat.data[pivot_row]
        # 归一化主元行
        pivot_val = mat.data[pivot_row][col]
        for c in range(n_cols):
            mat.data[pivot_row][c] /= pivot_val
        # 消去其他行
        for r in range(n_rows):
            if r == pivot_row:
                continue
            factor = mat.data[r][col]
            if factor == 0:
                continue
            for c in range(n_cols):
                mat.data[r][c] -= factor * mat.data[pivot_row][c]
        pivot_row += 1

    return mat


# ── 行列式 ──────────────────────────────────────────────

def determinant(m: Matrix) -> Fraction:
    """
    行列式计算：通过高斯消元化为上三角矩阵，对角线元素之积即为行列式。
    时间复杂度: O(n^3)，远优于按定义展开的 O(n!)。
    """
    if m.rows != m.cols:
        raise ValueError("只有方阵才能计算行列式")
    n = m.rows
    if n == 1:
        return m.data[0][0]
    if n == 2:
        return m.data[0][0] * m.data[1][1] - m.data[0][1] * m.data[1][0]
    if n == 3:
        # Sarrus 法则
        a = m.data
        return (a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1]
              - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1])

    ref, sign = row_echelon(m)
    det = sign
    for i in range(n):
        det *= ref.data[i][i]
    return det


# ── 余子式与伴随矩阵 ──────────────────────────────────────

def _minor_matrix(m: Matrix, skip_row: int, skip_col: int) -> Matrix:
    """删除第 skip_row 行和第 skip_col 列后的子矩阵。"""
    data = []
    for i in range(m.rows):
        if i == skip_row:
            continue
        row = []
        for j in range(m.cols):
            if j == skip_col:
                continue
            row.append(m.data[i][j])
        data.append(row)
    return Matrix(data)


def cofactor_matrix(m: Matrix) -> Matrix:
    """
    余子式矩阵。C[i][j] = (-1)^(i+j) * det(M_ij)
    用于伴随矩阵法求逆。
    时间复杂度: O(n^4)（每个元素需计算一个 (n-1)x(n-1) 行列式）
    """
    if m.rows != m.cols:
        raise ValueError("只有方阵才能计算余子式矩阵")
    n = m.rows
    data = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor_det = determinant(_minor_matrix(m, i, j))
            sign = Fraction(1) if (i + j) % 2 == 0 else Fraction(-1)
            data[i][j] = sign * minor_det
    return Matrix(data)


def adjugate(m: Matrix) -> Matrix:
    """伴随矩阵 (adjugate)：余子式矩阵的转置。"""
    return transpose(cofactor_matrix(m))


# ── 逆矩阵 ──────────────────────────────────────────────

def inverse(m: Matrix) -> Matrix:
    """
    逆矩阵：使用高斯-约当消元法 [A|I] → [I|A^(-1)]。
    时间复杂度: O(n^3)
    """
    if m.rows != m.cols:
        raise ValueError("只有方阵才能求逆")
    n = m.rows
    # 构造增广矩阵 [A | I]
    aug_data = [[Fraction(0)] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug_data[i][j] = m.data[i][j]
        aug_data[i][n + i] = Fraction(1)
    aug = Matrix(aug_data)

    # 高斯-约当消元
    rref = reduced_row_echelon(aug)

    # 检查左半部分是否为单位矩阵
    for i in range(n):
        for j in range(n):
            expected = Fraction(1) if i == j else Fraction(0)
            if rref.data[i][j] != expected:
                raise ValueError("矩阵不可逆 (奇异矩阵)")

    # 提取右半部分
    inv_data = [[rref.data[i][n + j] for j in range(n)] for i in range(n)]
    return Matrix(inv_data)


# ── 秩 ──────────────────────────────────────────────────

def rank(m: Matrix) -> int:
    """
    矩阵的秩：行阶梯形中非零行的数目。
    时间复杂度: O(n^3)
    """
    ref, _ = row_echelon(m)
    count = 0
    for i in range(ref.rows):
        if any(ref.data[i][j] != 0 for j in range(ref.cols)):
            count += 1
    return count


# ── 线性方程组求解 ──────────────────────────────────────────

def solve(a: Matrix, b: Matrix) -> Matrix:
    """
    求解线性方程组 Ax = b。
    构造增广矩阵 [A|b]，化为 RREF 后读取解。
    返回列向量 x。
    """
    if a.rows != a.cols:
        raise ValueError("目前仅支持方阵系数矩阵")
    if b.rows != a.rows or b.cols != 1:
        raise ValueError("b 必须是与 A 行数相同的列向量")
    n = a.rows
    aug_data = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug_data[i][j] = a.data[i][j]
        aug_data[i][n] = b.data[i][0]
    aug = Matrix(aug_data)
    rref = reduced_row_echelon(aug)

    # 检查是否有解
    for i in range(n):
        if all(rref.data[i][j] == 0 for j in range(n)) and rref.data[i][n] != 0:
            raise ValueError("方程组无解")

    x_data = [[rref.data[i][n]] for i in range(n)]
    return Matrix(x_data)


# ── LU 分解 ──────────────────────────────────────────────

def lu_decompose(m: Matrix) -> Tuple[Matrix, Matrix]:
    """
    LU 分解（不带主元选取）：A = L·U
    L 为下三角矩阵（对角线为 1），U 为上三角矩阵。
    时间复杂度: O(n^3)
    """
    if m.rows != m.cols:
        raise ValueError("只有方阵才能进行 LU 分解")
    n = m.rows
    L = Matrix.identity(n)
    U = m.copy()

    for k in range(n):
        if U.data[k][k] == 0:
            raise ValueError("主元为零，需要主元选取（此实现不支持）")
        for i in range(k + 1, n):
            factor = U.data[i][k] / U.data[k][k]
            L.data[i][k] = factor
            for j in range(k, n):
                U.data[i][j] -= factor * U.data[k][j]

    return L, U


# ── 2×2 矩阵特征值 ──────────────────────────────────────────

def eigenvalues_2x2(m: Matrix) -> Tuple[Fraction, Fraction]:
    """
    2×2 矩阵特征值：通过特征方程 λ² - tr(A)λ + det(A) = 0 求解。
    使用求根公式。
    """
    if m.rows != 2 or m.cols != 2:
        raise ValueError("仅支持 2×2 矩阵")
    a, b = m.data[0][0], m.data[0][1]
    c, d = m.data[1][0], m.data[1][1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4 * det
    if disc < 0:
        raise ValueError("特征值为复数，当前不支持")
    import math
    sqrt_disc = Fraction(math.isqrt(int(disc * 10**20)) if disc > 0 else 0)
    # 精确计算需要开方，这里用浮点近似
    disc_float = float(disc)
    if disc_float < 0:
        raise ValueError("特征值为复数")
    sqrt_val = disc_float ** 0.5
    tr_f = float(tr)
    lam1 = (tr_f + sqrt_val) / 2
    lam2 = (tr_f - sqrt_val) / 2
    return Fraction(lam1).limit_denominator(10**10), Fraction(lam2).limit_denominator(10**10)
