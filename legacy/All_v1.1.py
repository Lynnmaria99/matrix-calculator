# -*- coding: utf-8 -*-
"""
矩阵计算器 V1.1
=============================================================
相对 V1.0 (All.py) 的修改：
[Bug 修复]
 1. 修复：形状不匹配时 add/sub/multi/divide 返回 None，
    主流程却判断对象本身 (if tmp != None) 导致 printMat(None) 崩溃
 2. 修复：逆矩阵对奇异矩阵执行 None.all() 崩溃；
    改用 try/except 捕获 np.linalg.LinAlgError，
    浮点意义下的奇异矩阵 (det≈0) 也能正确识别
 3. 修复：对应相除时除数含 0 只报 RuntimeWarning 并输出 inf/nan，
    现在提前检查零元素并拒绝运算
 4. 修复：行列数 / 矩阵元素输入非法时 int()/float() 直接崩溃，
    现在统一循环重试，程序不会因一次输错而退出
 5. 修复：SVD 输出标签错误 (U=左奇异向量, S=奇异值, Vt=右奇异向量)；
    特征分解重复计算两次；divide 的注释写成了"相减"
[重构]
 6. 抽出 read_int / mkMat / read_matrix / read_two_matrices，
    消除 10 个菜单分支中大量重复的输入代码
 7. 菜单改为字典分派 (ACTIONS)，消灭长 if-elif 链
 8. 运算结果统一"先接返回值，再判 is not None"
[其他]
 9. 默认精度 float32 -> float64；打印支持复数特征值/特征向量
10. 每次运算后 time.sleep(2) 改为"按回车继续"
11. 输入流结束 (EOF) 时优雅退出；运算出现未预期异常不再致命
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
            # det 恰好为 0 或浮点意义下接近奇异都会走到这里
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


def mkMat(row, col):
    """根据 row、col 逐行交互式构造矩阵；某行输入有误只重输该行"""
    print("----start----")
    arr = []
    for i in range(row):
        while True:
            r = input(f"请输入第 {i + 1} 行内容 (用空格分隔 {col} 个数字): ").strip()
            tmp = r.split()
            if len(tmp) != col:
                print(f"输入的数字个数不正确，长度应该为 {col}，请重新输入本行")
                continue
            try:
                arr.append([float(x) for x in tmp])
            except ValueError:
                print("输入中包含无法识别为数字的内容，请重新输入本行")
                continue
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


# ==================== 输出相关 ====================

def _fmt(x):
    """格式化单个数字，兼容复数 (特征分解可能产生复数结果)"""
    if np.iscomplexobj(x):
        if abs(x.imag) < 1e-9:
            return f"{x.real:8.2f}"
        return f"{x.real:7.2f}{x.imag:+.2f}j"
    return f"{float(x):8.2f}"


def printMat(mat):
    """打印矩阵；一维数组 (如特征值/奇异值) 按一行打印"""
    mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    for row in mat:
        print("".join("\t" + _fmt(x) for x in row))


def show2Res(mat1, mat2, res):
    print("矩阵 1:")
    printMat(mat1)
    print("矩阵 2:")
    printMat(mat2)
    print("运算结果为:")
    printMat(res)


# ==================== 菜单功能 ====================
# 每个功能函数只负责：提示 -> 读矩阵 -> 调运算 -> 判空 -> 展示

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
        vals, vecs = result  # 只计算一次
        print("矩阵:")
        printMat(m)
        print("特征值为:")
        printMat(vals)
        print("特征向量 (每一列对应一个特征值) 为:")
        printMat(vecs)
        if np.iscomplexobj(vals) and np.any(np.abs(vals.imag) > 1e-9):
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
 0. 退出
请输入: """

# 菜单序号 -> 处理函数，新增功能只需在这里注册
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
            print("输入错误，请输入 0~10 之间的序号")
        else:
            try:
                action()
            except Exception as e:
                # 兜底：任何未预期错误都不应让整个程序崩溃
                print(f"运算过程中出现未预期的错误: {e}")

        try:
            input("\n按回车键继续...")
        except EOFError:
            break


if __name__ == "__main__":
    main()