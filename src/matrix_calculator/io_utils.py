"""命令行输入工具。"""

from __future__ import annotations

import numpy as np


def read_int(prompt, min_value=1):
    """读取一个整数；非法或过小的输入会循环重试。"""
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


def read_row_floats(n, prompt):
    """读取一行恰好 n 个数字，出错返回 None 由调用方重试。"""
    parts = input(prompt).strip().split()
    if len(parts) != n:
        print(f"输入的数字个数不正确，长度应该为 {n}，请重新输入")
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        print("输入中包含无法识别为数字的内容，请重新输入")
        return None


_read_row_floats = read_row_floats


def mkMat(row, col):
    """根据 row、col 逐行交互式构造矩阵。"""
    print("----start----")
    arr = []
    for i in range(row):
        while True:
            line = read_row_floats(col, f"请输入第 {i + 1} 行内容 (用空格分隔 {col} 个数字): ")
            if line is not None:
                arr.append(line)
                break
    print("----end----")
    return np.array(arr, dtype=np.float64)


def read_matrix():
    """读取一个矩阵：先问行列，再逐行输入。"""
    row = read_int("请输入矩阵的行: ")
    col = read_int("请输入矩阵的列: ")
    return mkMat(row, col)


def read_two_matrices():
    """读取两个矩阵。"""
    print("—— 矩阵 1 ——")
    mat1 = read_matrix()
    print("—— 矩阵 2 ——")
    mat2 = read_matrix()
    return mat1, mat2


def read_vector(n, prompt):
    """读取一个 n 维一维向量。"""
    while True:
        line = read_row_floats(n, f"{prompt} (用空格分隔 {n} 个数): ")
        if line is not None:
            return np.array(line, dtype=np.float64)


def read_vector_group():
    """读取向量组，返回 shape=(k, n)，每一行是一个向量。"""
    k = read_int("请输入向量的个数: ")
    n = read_int("请输入每个向量的维数: ")
    print(f"请逐行输入 {k} 个向量，每行 {n} 个数:")
    vecs = []
    for i in range(k):
        while True:
            line = read_row_floats(n, f"请输入向量 α{i + 1}: ")
            if line is not None:
                vecs.append(line)
                break
    return np.array(vecs, dtype=np.float64)
