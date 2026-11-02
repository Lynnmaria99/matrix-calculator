"""矩阵计算器命令行界面。"""

from __future__ import annotations

import numpy as np

from .constants import EPS
from .formatting import print_matrix, show_two_results, signed_polynomial
from .io_utils import (
    mkMat,
    read_int,
    read_matrix,
    read_two_matrices,
    read_vector,
    read_vector_group,
)
from .linalg_topics import analyze_quadratic, rank_and_nullspace, solve_linear
from .operations import (
    MatrixOperationError,
    add_matrices,
    determinant,
    eigendecompose,
    hadamard_divide,
    hadamard_multiply,
    invert_matrix,
    matrix_multiply,
    singular_value_decompose,
    subtract_matrices,
    transpose,
)


def _try_binary_operation(operation, m1, m2):
    """执行双矩阵运算，并在 CLI 层展示参数错误。"""
    try:
        result = operation(m1, m2)
    except MatrixOperationError as exc:
        print(exc)
        return
    show_two_results(m1, m2, result)


def _try_unary_operation(operation, matrix):
    """执行单矩阵运算，并在 CLI 层展示参数错误。"""
    try:
        return operation(matrix)
    except MatrixOperationError as exc:
        print(exc)
        return None


# ==================== 菜单功能 (1~10 基础运算) ====================


def op_add():
    print("【矩阵相加】两个矩阵的 shape 需要相同")
    m1, m2 = read_two_matrices()
    _try_binary_operation(add_matrices, m1, m2)


def op_matmulti():
    print("【矩阵相乘】矩阵 1 的列数需要等于矩阵 2 的行数")
    m1, m2 = read_two_matrices()
    _try_binary_operation(matrix_multiply, m1, m2)


def op_det():
    print("【行列式计算】需要输入一个方阵")
    m = read_matrix()
    val = _try_unary_operation(determinant, m)
    if val is not None:
        print("矩阵:")
        print_matrix(m)
        print(f"的行列式值为: {val:.4f}")


def op_invert():
    print("【逆矩阵计算】需要输入一个可逆方阵")
    m = read_matrix()
    res = _try_unary_operation(invert_matrix, m)
    if res is not None:
        print("矩阵:")
        print_matrix(m)
        print("的逆矩阵为:")
        print_matrix(res)


def op_eigs():
    print("【特征分解】需要输入一个方阵")
    m = read_matrix()
    result = _try_unary_operation(eigendecompose, m)
    if result is not None:
        vals, vecs = result
        print("矩阵:")
        print_matrix(m)
        print("特征值为:")
        print_matrix(vals)
        print("特征向量 (每一列对应一个特征值) 为:")
        print_matrix(vecs)
        if np.iscomplexobj(vals) and np.any(np.abs(vals.imag) > EPS):
            print("提示: 该矩阵存在复数特征值/特征向量")


def op_trans():
    print("【矩阵转置】")
    m = read_matrix()
    print("矩阵:")
    print_matrix(m)
    print("的转置矩阵为:")
    print_matrix(transpose(m))


def op_sub():
    print("【矩阵相减】两个矩阵的 shape 需要相同")
    m1, m2 = read_two_matrices()
    _try_binary_operation(subtract_matrices, m1, m2)


def op_divide():
    print("【矩阵对应相除】两个矩阵 shape 需要相同，且矩阵 2 不能含 0")
    m1, m2 = read_two_matrices()
    _try_binary_operation(hadamard_divide, m1, m2)


def op_multi():
    print("【矩阵对应相乘】两个矩阵的 shape 需要相同")
    m1, m2 = read_two_matrices()
    _try_binary_operation(hadamard_multiply, m1, m2)


def op_svd():
    print("【奇异值分解】任意矩阵均可")
    m = read_matrix()
    u, s, vt = singular_value_decompose(m)
    print("矩阵:")
    print_matrix(m)
    print("左奇异向量矩阵 U:")
    print_matrix(u)
    print("奇异值:")
    print_matrix(s)
    print("右奇异向量矩阵 V^T (V 的转置):")
    print_matrix(vt)


# ==================== 菜单功能 (11~14 课程专题) ====================


def op_solve_linear():
    print("【解线性方程组 Ax = b】")
    m = read_int("请输入方程个数 (A 的行数 m): ")
    n = read_int("请输入未知数个数 (A 的列数 n): ")
    print(f"请输入系数矩阵 A ({m} 行 {n} 列):")
    a = mkMat(m, n)
    b = read_vector(m, "请输入常数项向量 b")

    status, info = solve_linear(a, b)
    print("系数矩阵 A 为:")
    print_matrix(a)
    print("增广矩阵 (A|b) 为:")
    print_matrix(info["aug"], sep=n)
    print(f"秩: r(A) = {info['rA']}，r(A|b) = {info['rAb']}，未知数个数 n = {n}")

    if status == "unique":
        print("→ r(A) = r(A|b) = n，方程组有【唯一解】:")
        print_matrix(info["x"].reshape(-1, 1))
    elif status == "infinite":
        print(f"→ r(A) = r(A|b) = {info['rA']} < n = {n}，方程组【有无穷多解】")
        print(f"  自由未知数个数为 n - r = {n - info['rA']}")
        print("一个特解为:")
        print_matrix(info["xp"].reshape(-1, 1))
        null = info["null"]
        print("基础解系 (每列为一个基础解向量 ξ):")
        print_matrix(null)
        terms = " + ".join(f"c{i + 1}·ξ{i + 1}" for i in range(null.shape[1]))
        print(f"通解为: x = 特解 + {terms}  (c 为任意常数)")
    else:
        print(f"→ r(A) = {info['rA']} < r(A|b) = {info['rAb']}，方程组【无解】(矛盾方程组)")
        print("作为参考，残差最小的最小二乘近似解为:")
        print_matrix(info["x_ls"].reshape(-1, 1))


def op_vector_group():
    print("【向量组分析：秩 / 线性相关性 / 极大无关组】")
    vecs = read_vector_group()
    k, n = vecs.shape
    b = vecs.T
    print("向量组按列排成的矩阵 (第 j 列 = α_j):")
    print_matrix(b)

    r = int(np.linalg.matrix_rank(b))
    print(f"向量组的秩 r = {r}，向量个数 = {k}")
    if r < k:
        print(f"→ r = {r} < 向量个数 {k}，向量组【线性相关】")
    else:
        print(f"→ r = {r} = 向量个数 {k}，向量组【线性无关】")

    chosen = []
    for j in range(k):
        trial = chosen + [j]
        if np.linalg.matrix_rank(b[:, trial]) == len(trial):
            chosen.append(j)
    names = ", ".join(f"α{j + 1}" for j in chosen)
    print(f"一个极大线性无关组为: {names}")

    basis = b[:, chosen]
    for j in range(k):
        if j in chosen:
            continue
        c, *_ = np.linalg.lstsq(basis, b[:, j], rcond=None)
        pairs = [(coef, f"α{chosen[t] + 1}") for t, coef in enumerate(c)]
        print(f"  线性表示: α{j + 1} = {signed_polynomial(pairs)}")


def op_vector_inner():
    print("【向量内积 / 长度 / 夹角】")
    n = read_int("请输入向量维数: ")
    a = read_vector(n, "请输入向量 a")
    b = read_vector(n, "请输入向量 b")
    print("向量 a =", )
    print_matrix(a)
    print("向量 b =")
    print_matrix(b)

    dot = float(a @ b)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    print(f"内积 (a, b) = {dot:.4f}")
    print(f"长度 ‖a‖ = {na:.4f}，‖b‖ = {nb:.4f}")
    if na < EPS or nb < EPS:
        print("含零向量，夹角无定义；零向量与任何向量正交")
    else:
        cos_theta = np.clip(dot / (na * nb), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        print(f"cos θ = {cos_theta:.4f}，夹角 θ = {angle:.2f}°")
        if abs(dot) < EPS:
            print("→ (a, b) = 0，两向量【正交】")


def op_quadratic():
    print("【二次型 f = xᵀAx：标准形与正定性】")
    n = read_int("请输入方阵阶数 n: ")
    a = mkMat(n, n)
    result = analyze_quadratic(a)
    if not result["symmetric"]:
        print("提示: 输入矩阵不对称；二次型只取决于对称部分，已自动对称化 A ← (A+Aᵀ)/2")

    print("二次型的矩阵 A 为:")
    print_matrix(result["matrix"])
    print("特征值 (升序) 为:")
    print_matrix(result["eigenvalues"])
    print("正交变换矩阵 Q (x = Qy，每一列是对应特征向量):")
    print_matrix(result["Q"])

    standard_form = signed_polynomial(result["standard_terms"])
    print(f"经正交变换 x = Qy，二次型化为标准形:\n  f = {standard_form}")
    print(
        f"正惯性指数 p = {result['pos']}，负惯性指数 q = {result['neg']}，"
        f"该二次型{result['kind']}"
    )

    shown = ", ".join(
        f"Δ{k} = {v:.4g}" for k, v in enumerate(result["minors"], start=1)
    )
    print(f"顺序主子式: {shown}")
    if all(v > EPS for v in result["minors"]):
        print("→ 各阶顺序主子式全大于 0，由 Sylvester 判据，A【正定】")


MENU = """请输入想要的操作对应序号：
  1. 矩阵相加
  2. 矩阵相乘
  3. 行列式计算
  4. 逆矩阵计算
  5. 特征分解
  6. 矩阵的转置
  7. 矩阵相减
  8. 矩阵对应相除
  9. 矩阵对应相乘
10. 奇异值分解
11. 解线性方程组 Ax=b
12. 向量组：秩 / 线性相关性 / 极大无关组
13. 向量内积 / 长度 / 夹角
14. 二次型：标准形与正定性
  0. 退出
请输入: """

ACTIONS = {
    "1": op_add,
    "2": op_matmulti,
    "3": op_det,
    "4": op_invert,
    "5": op_eigs,
    "6": op_trans,
    "7": op_sub,
    "8": op_divide,
    "9": op_multi,
    "10": op_svd,
    "11": op_solve_linear,
    "12": op_vector_group,
    "13": op_vector_inner,
    "14": op_quadratic,
}


def main():
    while True:
        try:
            choice = input(MENU).strip()
        except EOFError:
            print("\n输入结束，退出程序")
            break

        if choice == "0":
            print("再见！")
            break

        action = ACTIONS.get(choice)
        if action is None:
            print("输入错误，请输入 0~14 之间的序号")
        else:
            try:
                action()
            except Exception as e:
                print(f"运算过程中出现未预期的错误: {e}")

        try:
            input("\n按回车键继续...")
        except EOFError:
            break
