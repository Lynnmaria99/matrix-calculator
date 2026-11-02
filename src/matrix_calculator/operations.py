"""矩阵基础运算。

这里的函数是“纯计算核心”：不读取 input，也不直接 print。
参数不合法时抛出 ``MatrixOperationError``，由 CLI、GUI 或 Notebook 自行展示。
``MatCal`` / ``MatCom`` 暂时保留为课程风格的兼容封装。
"""

from __future__ import annotations

import numpy as np


class MatrixOperationError(ValueError):
    """矩阵运算参数不合法，例如形状不匹配、非方阵、奇异矩阵等。"""


def _same_shape(a: np.ndarray, b: np.ndarray, operation: str) -> None:
    if a.shape != b.shape:
        raise MatrixOperationError(f"两个待{operation}矩阵的 shape 不一致，无法操作")


def add_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """矩阵加法，要求两个矩阵 shape 完全相同。"""
    _same_shape(a, b, "相加")
    return a + b


def subtract_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """矩阵减法，要求两个矩阵 shape 完全相同。"""
    _same_shape(a, b, "相减")
    return a - b


def hadamard_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """对应元素相乘（Hadamard 积），要求两个矩阵 shape 完全相同。"""
    _same_shape(a, b, "对应相乘")
    return a * b


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """矩阵乘法：a 的列数必须等于 b 的行数。"""
    if a.shape[1] != b.shape[0]:
        raise MatrixOperationError(
            "两个待相乘矩阵的 shape 不满足要求 "
            "(矩阵1的列数 != 矩阵2的行数)，无法操作"
        )
    return a @ b


def hadamard_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """对应元素相除，要求 shape 相同，且除数矩阵不能包含 0。"""
    _same_shape(a, b, "相除")
    if np.any(b == 0):
        raise MatrixOperationError("除数矩阵中含有 0，对应相除会产生 inf/nan，已取消操作")
    return a / b


def _require_square(matrix: np.ndarray, operation: str) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise MatrixOperationError(f"矩阵行列数目应该相等 (方阵)，无法进行{operation}")


def determinant(matrix: np.ndarray) -> float:
    """计算方阵行列式。"""
    _require_square(matrix, "行列式计算")
    return float(np.linalg.det(matrix))


def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    """计算方阵的逆矩阵；奇异或数值上不可逆时抛出异常。"""
    if matrix.shape[0] != matrix.shape[1]:
        raise MatrixOperationError("只有方阵才可能存在逆矩阵")
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise MatrixOperationError("矩阵不可逆 (行列式为 0 或矩阵接近奇异)") from exc


def eigendecompose(matrix: np.ndarray):
    """特征分解，返回 ``(特征值, 特征向量)``。"""
    _require_square(matrix, "特征分解")
    return np.linalg.eig(matrix)


def singular_value_decompose(matrix: np.ndarray):
    """奇异值分解，返回 ``(U, S, Vt)``。"""
    return np.linalg.svd(matrix)


def transpose(matrix: np.ndarray) -> np.ndarray:
    """矩阵转置。"""
    return matrix.T


class MatCal:
    """两个矩阵之间的运算（兼容旧接口）。

    旧代码通过返回 ``None`` 表示失败；新代码建议直接使用上面的纯函数，
    捕获 ``MatrixOperationError``。
    """

    def __init__(self, mat1, mat2):
        self._mat1 = np.asarray(mat1)
        self._mat2 = np.asarray(mat2)

    @staticmethod
    def _fail(exc):
        print(str(exc))
        return None

    def add(self):
        try:
            return add_matrices(self._mat1, self._mat2)
        except MatrixOperationError as exc:
            return self._fail(exc)

    def sub(self):
        try:
            return subtract_matrices(self._mat1, self._mat2)
        except MatrixOperationError as exc:
            return self._fail(exc)

    def multi(self):
        try:
            return hadamard_multiply(self._mat1, self._mat2)
        except MatrixOperationError as exc:
            return self._fail(exc)

    def matmulti(self):
        try:
            return matrix_multiply(self._mat1, self._mat2)
        except MatrixOperationError as exc:
            return self._fail(exc)

    def divide(self):
        try:
            return hadamard_divide(self._mat1, self._mat2)
        except MatrixOperationError as exc:
            return self._fail(exc)


class MatCom:
    """单个矩阵的运算（兼容旧接口）。"""

    def __init__(self, mat):
        self._mat = np.asarray(mat)

    @staticmethod
    def _fail(exc):
        print(str(exc))
        return None

    def det(self):
        try:
            return determinant(self._mat)
        except MatrixOperationError as exc:
            return self._fail(exc)

    def invert(self):
        try:
            return invert_matrix(self._mat)
        except MatrixOperationError as exc:
            return self._fail(exc)

    def eigs(self):
        try:
            return eigendecompose(self._mat)
        except MatrixOperationError as exc:
            return self._fail(exc)

    def svds(self):
        return singular_value_decompose(self._mat)

    def trans(self):
        return transpose(self._mat)
