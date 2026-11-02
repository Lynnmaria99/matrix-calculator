# -*- coding: utf-8 -*-
"""
矩阵计算器 V1.2  (面向线性代数课程)
=============================================================
相对 V1.1 的新增内容：
[打印格式]
 * 教材风格矩阵输出：⎡ ⎤ / ⎢ ⎥ / ⎣ ⎦ 括号 + 列对齐
 * 增广矩阵支持 │ 分隔线；一维向量按行/列正确显示
 * 数字按有效位自适应显示 (1 显示为 1 而不是 1.00)
[新功能 —— 对应教材章节]
 * 11. 解线性方程组 Ax=b：用秩判定 r(A)、r(A|b)、n 的关系，
     覆盖 唯一解 / 无穷多解(特解+基础解系) / 无解(最小二乘近似)
 * 12. 向量组：秩、线性相关/无关、极大无关组、其余向量的线性表示
 * 13. 向量：内积、长度(范数)、夹角、正交判定
 * 14. 二次型：对称矩阵特征值 -> 标准形、正交矩阵 Q、
     正定/负定/半正定/不定判定、顺序主子式 (Sylvester 判据)
V1.1 已有的修复 (错误输入重试、奇异矩阵求逆不崩、除零拦截、
SVD 标签、复数结果显示等) 全部保留。
"""

import numpy as np


# ==================== 矩阵运算类 ====================

class MatCal:
    """两个矩阵之间的运算"""

    def __init__(self, mat1, mat2):
        self._mat1 = mat1
        self._mat2 = mat2

    def add(self):
        """矩阵相加：两个矩阵 shape 必须完全相同"""
        if self._mat1.shape != self._mat2.shape:
            print("两个待相加矩阵的 shape 不一致，无法操作")
            return None
        return self._mat1 + self._mat2

    def sub(self):
        """矩阵相减：两个矩阵 shape 必须完全相同"""
        if self._mat1.shape != self._mat2.shape:
            print("两个待相减矩阵的 shape 不一致，无法操作")
            return None
        return self._mat1 - self._mat2

    def multi(self):
        """矩阵对应元素相乘 (Hadamard 积)：两个矩阵 shape 必须完全相同"""
        if self._mat1.shape != self._mat2.shape:
            print("两个待对应相乘矩阵的 shape 不一致，无法操作")
            return None
        return self._mat1 * self._mat2

    def matmulti(self):
        """矩阵乘法：mat1 的列数必须等于 mat2 的行数"""
        if self._mat1.shape[1] != self._mat2.shape[0]:
            print("两个待相乘矩阵的 shape 不满足要求 (矩阵1的列数 != 矩阵2的行数)，无法操作")
            return None
        return self._mat1 @ self._mat2

    def divide(self):
        """矩阵对应元素相除：shape 必须相同，且除数矩阵不能含 0"""
        if self._mat1.shape != self._mat2.shape:
            print("两个待相除矩阵的 shape 不一致，无法操作")
            return None
        if np.any(self._mat2 == 0):
            print("除数矩阵中含有 0，对应相除会产生 inf/nan，已取消操作")
            return None
        return self._mat1 / self._mat2


class MatCom:
    """单个矩阵的运算"""

    def __init__(self, mat):
        self._mat = mat

    def det(self):
        """行列式：必须为方阵"""
        if self._mat.shape[0] != self._mat.shape[1]:
            print("矩阵行列数目应该相等 (方阵)，无法计算行列式")
            return None
        return np.linalg.det(self._mat)

    def invert(self):
        """逆矩阵：必须为方阵且可逆"""
        if self._mat.shape[0] != self._mat.shape[1]:
            print("只有方阵才可能存在逆矩阵")
            return None
        try:
            return np.linalg.inv(self._mat)
        except np.linalg.LinAlgError:
            print("矩阵不可逆 (行列式为 0 或矩阵接近奇异)")
            return None

    def eigs(self):
        """特征分解：必须为方阵，返回 (特征值, 特征向量)"""
        if self._mat.shape[0] != self._mat.shape[1]:
            print("矩阵行列数目应该相等 (方阵)，无法进行特征分解")
            return None
        return np.linalg.eig(self._mat)

    def svds(self):
        """奇异值分解：任意矩阵均可，返回 (U, S, Vt)"""
        return np.linalg.svd(self._mat)

    def trans(self):
        """矩阵转置"""
        return self._mat.T


# ==================== 输入相关 ====================

def read_int(prompt, min_value=1):
    """读取一个整数；非法或过小的输入会循环重试"""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("输入无效，请输入一个整数")
            continue
        if value < min_value:
            print(f"请输入不小于 {min_value} 的整数")
            continue
        return value


def _read_row_floats(n, prompt):
    """读取一行恰好 n 个数字，出错返回 None 由调用方重试"""
    parts = input(prompt).strip().split()
    if len(parts) != n:
        print(f"输入的数字个数不正确，长度应该为 {n}，请重新输入")
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        print("输入中包含无法识别为数字的内容，请重新输入")
        return None


def mkMat(row, col):
    """根据 row、col 逐行交互式构造矩阵；某行输入有误只重输该行"""
    print("----start----")
    arr = []
    for i in range(row):
        while True:
            line = _read_row_floats(col, f"请输入第 {i + 1} 行内容 (用空格分隔 {col} 个数字): ")
            if line is not None:
                arr.append(line)
                break
    print("----end----")
    return np.array(arr, dtype=np.float64)


def read_matrix():
    """读取一个矩阵：先问行列，再逐行输入"""
    row = read_int("请输入矩阵的行: ")
    col = read_int("请输入矩阵的列: ")
    return mkMat(row, col)


def read_two_matrices():
    """读取两个矩阵"""
    print("—— 矩阵 1 ——")
    mat1 = read_matrix()
    print("—— 矩阵 2 ——")
    mat2 = read_matrix()
    return mat1, mat2


def read_vector(n, prompt):
    """读取一个 n 维向量 (一维数组)"""
    while True:
        line = _read_row_floats(n, f"{prompt} (用空格分隔 {n} 个数): ")
        if line is not None:
            return np.array(line, dtype=np.float64)


def read_vector_group():
    """读取向量组：返回 shape=(k, n) 的数组，每一行是一个向量 α"""
    k = read_int("请输入向量的个数: ")
    n = read_int("请输入每个向量的维数: ")
    print(f"请逐行输入 {k} 个向量，每行 {n} 个数:")
    vecs = []
    for i in range(k):
        while True:
            line = _read_row_floats(n, f"请输入向量 α{i + 1}: ")
            if line is not None:
                vecs.append(line)
                break
    return np.array(vecs, dtype=np.float64)


# ==================== 输出相关 ====================

_EPS = 1e-9


def _fmt(x):
    """格式化单个数字，兼容复数；接近 0 的值显示为 0"""
    if np.iscomplexobj(x):
        r = 0.0 if abs(x.real) < _EPS else x.real
        i = 0.0 if abs(x.imag) < _EPS else x.imag
        if i == 0:
            return f"{r:.4g}"
        if r == 0:
            return f"{i:.4g}j"
        return f"{r:.4g}{i:+.4g}j"
    v = 0.0 if abs(x) < _EPS else float(x)
    return f"{v:.4g}"


def printMat(mat, sep=None):
    """教材风格打印矩阵。
    sep: 若给定整数 k，则在第 k 列之后画一条 │ 分隔线 (用于增广矩阵)。"""
    mat = np.atleast_2d(np.asarray(mat))
    nrows, ncols = mat.shape
    cells = [[_fmt(mat[i, j]) for j in range(ncols)] for i in range(nrows)]
    widths = [max(len(cells[i][j]) for i in range(nrows)) for j in range(ncols)]

    def render_row(i, left, right):
        parts = []
        for j in range(ncols):
            parts.append(cells[i][j].rjust(widths[j]))
            if sep is not None and j == sep - 1 and j != ncols - 1:
                parts.append("│")
        return f"{left} " + "  ".join(parts) + f" {right}"

    if nrows == 1:
        print(render_row(0, "(", ")"))
        return
    for i in range(nrows):
        if i == 0:
            print(render_row(i, "⎡", "⎤"))
        elif i == nrows - 1:
            print(render_row(i, "⎣", "⎦"))
        else:
            print(render_row(i, "⎢", "⎥"))


def show2Res(mat1, mat2, res):
    print("矩阵 1:")
    printMat(mat1)
    print("矩阵 2:")
    printMat(mat2)
    print("运算结果为:")
    printMat(res)


def _signed_poly(pairs):
    """把 [(系数, 变量名), ...] 拼成教材风格的线性组合，如 'a1 + 2*a2 - a3'"""
    parts = []
    for coef, var in pairs:
        c = 0.0 if abs(coef) < _EPS else float(coef)
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else f"{abs(c):g}·"
        body = f"{mag}{var}"
        if not parts:
            parts.append(f"-{body}" if c < 0 else body)
        else:
            parts.append(f"- {body}" if c < 0 else f"+ {body}")
    return " ".join(parts) if parts else "0"


# ==================== 线性代数工具 ====================

def rank_and_nullspace(A):
    """基于 SVD 计算秩与零空间基。
    返回 (秩 r, 零空间基矩阵 N(每列为一个基向量), 奇异值阈值)"""
    m, n = A.shape
    S = np.linalg.svd(A, compute_uv=False)
    tol = max(m, n) * np.finfo(float).eps * (S[0] if S.size else 0.0)
    r = int(np.sum(S > tol))
    if r < n:
        Vt = np.linalg.svd(A, full_matrices=True)[2]
        null = Vt[r:].T          # 每一列是 N(A) 的一个基向量
    else:
        null = np.zeros((n, 0))
    return r, null, tol


def solve_linear(A, b):
    """解 Ax=b，返回 (状态码, 解信息字典)。
    状态码: 'unique' 唯一解 / 'infinite' 无穷多解 / 'none' 无解"""
    m, n = A.shape
    rA, null, _ = rank_and_nullspace(A)
    aug = np.hstack([A, b.reshape(-1, 1)])
    rAb = np.linalg.matrix_rank(aug)
    info = {"rA": rA, "rAb": rAb, "n": n, "aug": aug}
    if rA < rAb:
        x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
        info["x_ls"] = x_ls
        return "none", info
    if rA == n:
        if m == n:
            x = np.linalg.solve(A, b)
        else:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        info["x"] = x
        return "unique", info
    xp, *_ = np.linalg.lstsq(A, b, rcond=None)
    info["xp"] = xp
    info["null"] = null
    return "infinite", info


# ==================== 菜单功能 (1~10 基础运算) ====================

def op_add():
    print("【矩阵相加】两个矩阵的 shape 需要相同")
    m1, m2 = read_two_matrices()
    res = MatCal(m1, m2).add()
    if res is not None:
        show2Res(m1, m2, res)


def op_matmulti():
    print("【矩阵相乘】矩阵 1 的列数需要等于矩阵 2 的行数")
    m1, m2 = read_two_matrices()
    res = MatCal(m1, m2).matmulti()
    if res is not None:
        show2Res(m1, m2, res)


def op_det():
    print("【行列式计算】需要输入一个方阵")
    m = read_matrix()
    val = MatCom(m).det()
    if val is not None:
        print("矩阵:")
        printMat(m)
        print(f"的行列式值为: {val:.4f}")


def op_invert():
    print("【逆矩阵计算】需要输入一个可逆方阵")
    m = read_matrix()
    res = MatCom(m).invert()
    if res is not None:
        print("矩阵:")
        printMat(m)
        print("的逆矩阵为:")
        printMat(res)


def op_eigs():
    print("【特征分解】需要输入一个方阵")
    m = read_matrix()
    result = MatCom(m).eigs()
    if result is not None:
        vals, vecs = result
        print("矩阵:")
        printMat(m)
        print("特征值为:")
        printMat(vals)
        print("特征向量 (每一列对应一个特征值) 为:")
        printMat(vecs)
        if np.iscomplexobj(vals) and np.any(np.abs(vals.imag) > _EPS):
            print("提示: 该矩阵存在复数特征值/特征向量")


def op_trans():
    print("【矩阵转置】")
    m = read_matrix()
    print("矩阵:")
    printMat(m)
    print("的转置矩阵为:")
    printMat(MatCom(m).trans())


def op_sub():
    print("【矩阵相减】两个矩阵的 shape 需要相同")
    m1, m2 = read_two_matrices()
    res = MatCal(m1, m2).sub()
    if res is not None:
        show2Res(m1, m2, res)


def op_divide():
    print("【矩阵对应相除】两个矩阵 shape 需要相同，且矩阵 2 不能含 0")
    m1, m2 = read_two_matrices()
    res = MatCal(m1, m2).divide()
    if res is not None:
        show2Res(m1, m2, res)


def op_multi():
    print("【矩阵对应相乘】两个矩阵的 shape 需要相同")
    m1, m2 = read_two_matrices()
    res = MatCal(m1, m2).multi()
    if res is not None:
        show2Res(m1, m2, res)


def op_svd():
    print("【奇异值分解】任意矩阵均可")
    m = read_matrix()
    U, S, Vt = MatCom(m).svds()
    print("矩阵:")
    printMat(m)
    print("左奇异向量矩阵 U:")
    printMat(U)
    print("奇异值:")
    printMat(S)
    print("右奇异向量矩阵 V^T (V 的转置):")
    printMat(Vt)


# ==================== 菜单功能 (11~14 课程专题) ====================

def op_solve_linear():
    print("【解线性方程组 Ax = b】")
    m = read_int("请输入方程个数 (A 的行数 m): ")
    n = read_int("请输入未知数个数 (A 的列数 n): ")
    print(f"请输入系数矩阵 A ({m} 行 {n} 列):")
    A = mkMat(m, n)
    b = read_vector(m, "请输入常数项向量 b")

    status, info = solve_linear(A, b)
    print("系数矩阵 A 为:")
    printMat(A)
    print("增广矩阵 (A|b) 为:")
    printMat(info["aug"], sep=n)
    print(f"秩: r(A) = {info['rA']}，r(A|b) = {info['rAb']}，未知数个数 n = {n}")

    if status == "unique":
        print("→ r(A) = r(A|b) = n，方程组有【唯一解】:")
        printMat(info["x"].reshape(-1, 1))
    elif status == "infinite":
        print(f"→ r(A) = r(A|b) = {info['rA']} < n = {n}，方程组【有无穷多解】")
        print(f"  自由未知数个数为 n - r = {n - info['rA']}")
        print("一个特解为:")
        printMat(info["xp"].reshape(-1, 1))
        null = info["null"]
        print("基础解系 (每列为一个基础解向量 ξ):")
        printMat(null)
        terms = " + ".join(f"c{i + 1}·ξ{i + 1}" for i in range(null.shape[1]))
        print(f"通解为: x = 特解 + {terms}  (c 为任意常数)")
    else:
        print(f"→ r(A) = {info['rA']} < r(A|b) = {info['rAb']}，方程组【无解】(矛盾方程组)")
        print("作为参考，残差最小的最小二乘近似解为:")
        printMat(info["x_ls"].reshape(-1, 1))


def op_vector_group():
    print("【向量组分析：秩 / 线性相关性 / 极大无关组】")
    vecs = read_vector_group()      # 每行一个向量，shape (k, n)
    k, n = vecs.shape
    B = vecs.T                      # 按教材习惯：矩阵的每一列为一个向量 α_j
    print("向量组按列排成的矩阵 (第 j 列 = α_j):")
    printMat(B)

    r = np.linalg.matrix_rank(B)
    print(f"向量组的秩 r = {r}，向量个数 = {k}")
    if r < k:
        print(f"→ r = {r} < 向量个数 {k}，向量组【线性相关】")
    else:
        print(f"→ r = {r} = 向量个数 {k}，向量组【线性无关】")

    # 贪心求极大线性无关组：逐个加入能使秩增大的向量
    chosen = []
    for j in range(k):
        trial = chosen + [j]
        if np.linalg.matrix_rank(B[:, trial]) == len(trial):
            chosen.append(j)
    names = ", ".join(f"α{j + 1}" for j in chosen)
    print(f"一个极大线性无关组为: {names}")

    # 其余向量用极大无关组线性表示
    Bb = B[:, chosen]
    for j in range(k):
        if j in chosen:
            continue
        c, *_ = np.linalg.lstsq(Bb, B[:, j], rcond=None)
        pairs = [(coef, f"α{chosen[t] + 1}") for t, coef in enumerate(c)]
        print(f"  线性表示: α{j + 1} = {_signed_poly(pairs)}")


def op_vector_inner():
    print("【向量内积 / 长度 / 夹角】")
    n = read_int("请输入向量维数: ")
    a = read_vector(n, "请输入向量 a")
    b = read_vector(n, "请输入向量 b")
    print("向量 a =", )
    printMat(a)
    print("向量 b =")
    printMat(b)

    dot = float(a @ b)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    print(f"内积 (a, b) = {dot:.4f}")
    print(f"长度 ‖a‖ = {na:.4f}，‖b‖ = {nb:.4f}")
    if na < _EPS or nb < _EPS:
        print("含零向量，夹角无定义；零向量与任何向量正交")
    else:
        cos_theta = np.clip(dot / (na * nb), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        print(f"cos θ = {cos_theta:.4f}，夹角 θ = {angle:.2f}°")
        if abs(dot) < _EPS:
            print("→ (a, b) = 0，两向量【正交】")


def op_quadratic():
    print("【二次型 f = xᵀAx：标准形与正定性】")
    n = read_int("请输入方阵阶数 n: ")
    A = mkMat(n, n)
    if not np.allclose(A, A.T, atol=1e-8):
        print("提示: 输入矩阵不对称；二次型只取决于对称部分，已自动对称化 A ← (A+Aᵀ)/2")
    S = (A + A.T) / 2.0

    print("二次型的矩阵 A 为:")
    printMat(S)
    w, Q = np.linalg.eigh(S)      # 对称矩阵特征分解，特征值升序
    print("特征值 (升序) 为:")
    printMat(w)
    print("正交变换矩阵 Q (x = Qy，每一列是对应特征向量):")
    printMat(Q)

    # 标准形
    pairs = [(lam, f"y{i + 1}²") for i, lam in enumerate(w)]
    print(f"经正交变换 x = Qy，二次型化为标准形:\n  f = {_signed_poly(pairs)}")

    # 正定性
    pos = int(np.sum(w > _EPS))
    neg = int(np.sum(w < -_EPS))
    zero = n - pos - neg
    if pos == n:
        kind = "【正定】(特征值全为正)"
    elif neg == n:
        kind = "【负定】(特征值全为负)"
    elif neg == 0 and zero > 0:
        # 非负且至少含一个 0：正数+0 或全 0，均为半正定
        kind = "【半正定】(特征值非负且含 0)"
    elif pos == 0 and zero > 0:
        # 非正且至少含一个 0：负数+0，为半负定
        kind = "【半负定】(特征值非正且含 0)"
    elif pos > 0 and neg > 0:
        kind = "【不定】(特征值有正有负)"
    else:
        kind = "判定需人工复核"
    print(f"正惯性指数 p = {pos}，负惯性指数 q = {neg}，该二次型{kind}")

    # 顺序主子式 (Sylvester)
    minors = [float(np.linalg.det(S[:k, :k])) for k in range(1, n + 1)]
    shown = ", ".join(f"Δ{k} = {v:.4g}" for k, v in enumerate(minors, start=1))
    print(f"顺序主子式: {shown}")
    if all(v > _EPS for v in minors):
        print("→ 各阶顺序主子式全大于 0，由 Sylvester 判据，A【正定】")


# ==================== 菜单与主循环 ====================

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


if __name__ == "__main__":
    main()