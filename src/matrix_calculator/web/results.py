"""网页端矩阵结果、导出、错误说明与后续分析组件。"""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import (
    determinant,
    eigendecompose,
    invert_matrix,
    singular_value_decompose,
    transpose,
)
from ..formatting import format_matrix, matrix_to_csv_text, matrix_to_numpy_text


_RESULT_SERIAL_KEY = "matrix-calculator:result-serial"


def _next_result_serial() -> int:
    """返回本次脚本运行内的下一个导出编号。"""
    serial = st.session_state.get(_RESULT_SERIAL_KEY, 0) + 1
    st.session_state[_RESULT_SERIAL_KEY] = serial
    return serial


def show_text_matrix(label: str, value) -> None:
    """展示教材风格矩阵；st.code 自带复制按钮，另提供 TXT/CSV/NumPy 导出。"""
    result_id = _next_result_serial()
    matrix_text = format_matrix(value)
    csv_text = matrix_to_csv_text(value)
    numpy_text = matrix_to_numpy_text(value)

    with st.container(border=True):
        title_col, export_col = st.columns([3, 1], vertical_alignment="center")
        title_col.markdown(f"**{label}**")
        with export_col:
            with st.popover("导出", width="stretch"):
                st.download_button(
                    "下载 TXT",
                    data=matrix_text,
                    file_name=f"matrix-{result_id}.txt",
                    mime="text/plain",
                    key=f"download-txt-{result_id}",
                    width="stretch",
                )
                st.download_button(
                    "下载 CSV",
                    data=csv_text,
                    file_name=f"matrix-{result_id}.csv",
                    mime="text/csv",
                    key=f"download-csv-{result_id}",
                    width="stretch",
                )
                st.download_button(
                    "NumPy 文本",
                    data=numpy_text,
                    file_name=f"matrix-{result_id}.py",
                    mime="text/x-python",
                    key=f"download-numpy-{result_id}",
                    width="stretch",
                )
        st.code(matrix_text, language="text", wrap_lines=True)


def show_error(exc: Exception) -> None:
    st.error(f"输入或计算有误：{exc}")


def shape_text(matrix: np.ndarray) -> str:
    return f"{matrix.shape[0]} × {matrix.shape[1]}"


def explain_binary_error(
    exc: Exception,
    operation_code: str,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
) -> str:
    """把核心库的通用异常翻译成网页上可操作的维度提示。"""
    if operation_code == "multiply" and mat_a.shape[1] != mat_b.shape[0]:
        return (
            f"A × B 要求 A 的列数等于 B 的行数。当前 A 是 {shape_text(mat_a)}，"
            f"有 {mat_a.shape[1]} 列；B 是 {shape_text(mat_b)}，有 {mat_b.shape[0]} 行。"
            "请调整矩阵维度，或改选 A + B / A ⊙ B。"
        )

    elementwise_operations = {
        "add": ("A + B", "相加"),
        "subtract": ("A − B", "相减"),
        "hadamard_multiply": ("A ⊙ B", "对应元素相乘"),
        "hadamard_divide": ("A ⊘ B", "对应元素相除"),
    }
    if operation_code in elementwise_operations and mat_a.shape != mat_b.shape:
        symbol, action = elementwise_operations[operation_code]
        return (
            f"{symbol} 要求两个矩阵形状完全相同，才能逐元素{action}。"
            f"当前 A 是 {shape_text(mat_a)}，B 是 {shape_text(mat_b)}。"
            "请用 +行 / +列 调整成相同形状。"
        )

    if operation_code == "hadamard_divide" and mat_b.size and np.any(mat_b == 0):
        return "A ⊘ B 是对应元素相除；B 中包含 0，会产生无穷大，因此已取消计算。请把 0 改成非零数。"

    return str(exc)


def explain_unary_error(exc: Exception, operation_code: str, mat: np.ndarray) -> str:
    square_only = {"determinant", "inverse", "eigendecomposition"}
    labels = {
        "determinant": "det(A)",
        "inverse": "A⁻¹",
        "eigendecomposition": "特征分解",
    }
    if operation_code in square_only and mat.shape[0] != mat.shape[1]:
        return (
            f"{labels[operation_code]} 只能用于方阵。当前矩阵是 {shape_text(mat)}，"
            "请调整成相同行数和列数；非方阵可以选择 SVD 或 Aᵀ。"
        )
    return str(exc)


def _matrix_signature(matrix: np.ndarray) -> tuple[tuple[int, int], tuple[float, ...]]:
    """生成可比较的矩阵输入签名，用于判断上次结果是否已经过期。"""
    array = np.asarray(matrix, dtype=float)
    return array.shape, tuple(array.ravel().tolist())


def render_matrix_analysis(
    matrix: np.ndarray,
    operation_code: str,
    symbol: str = "A",
) -> None:
    """渲染单矩阵分析结果，供单矩阵页和双矩阵结果复用。"""
    if operation_code == "determinant":
        with st.container(border=True):
            st.metric(f"行列式 det({symbol})", f"{determinant(matrix):.6g}")
    elif operation_code == "inverse":
        show_text_matrix(f"逆矩阵 {symbol}⁻¹", invert_matrix(matrix))
    elif operation_code == "transpose":
        show_text_matrix(f"转置 {symbol}ᵀ", transpose(matrix))
    elif operation_code == "eigendecomposition":
        eigenvalues, eigenvectors = eigendecompose(matrix)
        show_text_matrix("特征值", eigenvalues)
        show_text_matrix("特征向量（每列对应一个特征值）", eigenvectors)
        if np.iscomplexobj(eigenvalues) and np.any(np.abs(eigenvalues.imag) > 1e-9):
            st.info("该矩阵存在复数特征值或特征向量。")
    elif operation_code == "svd":
        u, singular_values, vt = singular_value_decompose(matrix)
        show_text_matrix("U", u)
        show_text_matrix("奇异值 S", singular_values)
        show_text_matrix("Vᵀ", vt)


def reset_result_serial() -> None:
    """每次脚本重跑时重置当前会话的导出控件编号。"""
    st.session_state[_RESULT_SERIAL_KEY] = 0
