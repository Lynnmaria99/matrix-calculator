"""矩阵和线性组合的格式化输出。"""

from __future__ import annotations

import io

import numpy as np

from .constants import EPS


def format_scalar(x) -> str:
    """格式化单个数字，兼容复数；接近 0 的值显示为 0。"""
    if np.iscomplexobj(x):
        r = 0.0 if abs(x.real) < EPS else x.real
        i = 0.0 if abs(x.imag) < EPS else x.imag
        if i == 0:
            return f"{r:.4g}"
        if r == 0:
            return f"{i:.4g}j"
        return f"{r:.4g}{i:+.4g}j"
    value = 0.0 if abs(x) < EPS else float(x)
    return f"{value:.4g}"


# 向后兼容旧函数名
_fmt = format_scalar


def format_matrix(mat, sep: int | None = None) -> str:
    """返回教材风格矩阵字符串。

    ``sep`` 表示在第几列后画 ``│`` 分隔线，常用于增广矩阵。
    """
    mat = np.atleast_2d(np.asarray(mat))
    nrows, ncols = mat.shape
    cells = [[format_scalar(mat[i, j]) for j in range(ncols)] for i in range(nrows)]
    widths = [max(len(cells[i][j]) for i in range(nrows)) for j in range(ncols)]

    def render_row(i, left, right):
        parts = []
        for j in range(ncols):
            parts.append(cells[i][j].rjust(widths[j]))
            if sep is not None and j == sep - 1 and j != ncols - 1:
                parts.append("│")
        return f"{left} " + "  ".join(parts) + f" {right}"

    if nrows == 1:
        return render_row(0, "(", ")")

    rows = []
    for i in range(nrows):
        if i == 0:
            left, right = "⎡", "⎤"
        elif i == nrows - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"
        rows.append(render_row(i, left, right))
    return "\n".join(rows)


def print_mat(mat, sep=None):
    """打印教材风格矩阵；保留旧函数名。"""
    print(format_matrix(mat, sep=sep))


def print_matrix(mat, sep=None):
    """打印教材风格矩阵。"""
    print(format_matrix(mat, sep=sep))


def show2Res(mat1, mat2, res):
    """旧接口：展示两个输入矩阵和运算结果。"""
    print("矩阵 1:")
    print_matrix(mat1)
    print("矩阵 2:")
    print_matrix(mat2)
    print("运算结果为:")
    print_matrix(res)


def show_two_results(mat1, mat2, res):
    """展示两个输入矩阵和运算结果。"""
    show2Res(mat1, mat2, res)


def signed_polynomial(pairs):
    """把 ``[(系数, 变量名), ...]`` 拼成教材风格线性组合。"""
    parts = []
    for coef, var in pairs:
        c = 0.0 if abs(coef) < EPS else float(coef)
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else f"{abs(c):g}·"
        body = f"{mag}{var}"
        if not parts:
            parts.append(f"-{body}" if c < 0 else body)
        else:
            parts.append(f"- {body}" if c < 0 else f"+ {body}")
    return " ".join(parts) if parts else "0"


_signed_poly = signed_polynomial


def matrix_to_csv_text(value) -> str:
    """把矩阵转换成 CSV 文本，便于粘贴到 Excel 或数据分析工具中。"""
    matrix = np.atleast_2d(np.asarray(value))
    buffer = io.StringIO()
    fmt = "%s" if np.iscomplexobj(matrix) else "%.12g"
    np.savetxt(buffer, matrix, delimiter=",", fmt=fmt)
    return buffer.getvalue()


def matrix_to_numpy_text(value) -> str:
    """把矩阵转换成可直接粘贴到 Python/Notebook 的 np.array 文本。"""
    matrix = np.atleast_2d(np.asarray(value))
    array_text = np.array2string(
        matrix,
        separator=", ",
        threshold=10_000,
        suppress_small=True,
        precision=10,
    )
    return f"np.array({array_text})\n"
