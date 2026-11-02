# -*- coding: utf-8 -*-
"""矩阵计算器 Streamlit 网页界面。

运行方式：

    streamlit run streamlit_app.py

网页层只负责收集输入、调用计算核心、展示结果和错误信息。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from matrix_calculator import (  # noqa: E402
    MatrixOperationError,
    add_matrices,
    analyze_quadratic,
    determinant,
    eigendecompose,
    format_matrix,
    hadamard_divide,
    hadamard_multiply,
    invert_matrix,
    matrix_multiply,
    singular_value_decompose,
    solve_linear,
    subtract_matrices,
    transpose,
)
from matrix_calculator.formatting import signed_polynomial  # noqa: E402


st.set_page_config(page_title="矩阵计算器", page_icon="🧮", layout="wide")


# ==================== 通用输入输出 ====================


def matrix_editor(label: str, rows: int, cols: int, key: str,
                  default: np.ndarray | None = None) -> np.ndarray:
    """用可编辑表格输入矩阵。"""
    st.markdown(f"**{label}**")
    if default is None:
        data = np.zeros((rows, cols), dtype=float)
    else:
        data = np.asarray(default, dtype=float)
    df = pd.DataFrame(data, columns=[f"列{j + 1}" for j in range(cols)])
    column_config = {
        col: st.column_config.NumberColumn(col, step=0.1, format="%.6g")
        for col in df.columns
    }
    edited = st.data_editor(
        df,
        key=key,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config=column_config,
    )
    return edited.to_numpy(dtype=float)


def show_text_matrix(label: str, value) -> None:
    st.markdown(f"**{label}**")
    st.text(format_matrix(value))


def show_error(exc: Exception) -> None:
    st.error(f"输入或计算有误：{exc}")


# ==================== 页面：双矩阵运算 ====================


def page_binary() -> None:
    st.header("双矩阵运算")
    operation_name = st.selectbox(
        "选择运算",
        [
            "矩阵加法（A + B）",
            "矩阵减法（A - B）",
            "矩阵乘法（A × B）",
            "对应元素相乘（Hadamard）",
            "对应元素相除（A / B）",
        ],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("矩阵 A")
        rows_a = st.number_input("A 的行数", 1, 12, 2, key="rows_a")
        cols_a = st.number_input("A 的列数", 1, 12, 2, key="cols_a")
        mat_a = matrix_editor("编辑 A", rows_a, cols_a, key=f"A-{rows_a}-{cols_a}")

    with col_b:
        st.subheader("矩阵 B")
        rows_b = st.number_input("B 的行数", 1, 12, 2, key="rows_b")
        cols_b = st.number_input("B 的列数", 1, 12, 2, key="cols_b")
        mat_b = matrix_editor("编辑 B", rows_b, cols_b, key=f"B-{rows_b}-{cols_b}")

    if st.button("计算", type="primary", key="run_binary"):
        operations = {
            "矩阵加法（A + B）": add_matrices,
            "矩阵减法（A - B）": subtract_matrices,
            "矩阵乘法（A × B）": matrix_multiply,
            "对应元素相乘（Hadamard）": hadamard_multiply,
            "对应元素相除（A / B）": hadamard_divide,
        }
        try:
            result = operations[operation_name](mat_a, mat_b)
            show_text_matrix("运算结果", result)
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 页面：单矩阵运算 ====================


def page_unary() -> None:
    st.header("单矩阵运算")
    operation_name = st.selectbox(
        "选择运算",
        ["行列式", "逆矩阵", "特征分解", "奇异值分解 SVD", "转置"],
    )

    rows = st.number_input("行数", 1, 12, 2, key="unary_rows")
    cols = st.number_input("列数", 1, 12, 2, key="unary_cols")
    mat = matrix_editor("编辑矩阵", rows, cols, key=f"unary-{rows}-{cols}")

    if st.button("计算", type="primary", key="run_unary"):
        try:
            if operation_name == "行列式":
                st.metric("行列式 det(A)", f"{determinant(mat):.6g}")
            elif operation_name == "逆矩阵":
                show_text_matrix("逆矩阵 A⁻¹", invert_matrix(mat))
            elif operation_name == "转置":
                show_text_matrix("转置 Aᵀ", transpose(mat))
            elif operation_name == "特征分解":
                eigenvalues, eigenvectors = eigendecompose(mat)
                show_text_matrix("特征值", eigenvalues)
                show_text_matrix("特征向量（每列对应一个特征值）", eigenvectors)
                if np.iscomplexobj(eigenvalues) and np.any(np.abs(eigenvalues.imag) > 1e-9):
                    st.info("该矩阵存在复数特征值或特征向量。")
            elif operation_name == "奇异值分解 SVD":
                u, singular_values, vt = singular_value_decompose(mat)
                show_text_matrix("U", u)
                show_text_matrix("奇异值 S", singular_values)
                show_text_matrix("Vᵀ", vt)
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 页面：线性方程组 ====================


def page_solve() -> None:
    st.header("解线性方程组 Ax = b")
    m = st.number_input("方程个数 m", 1, 12, 2, key="solve_m")
    n = st.number_input("未知数个数 n", 1, 12, 2, key="solve_n")
    st.caption("下面一次输入增广矩阵 [A | b]：最后一列是常数项 b。")
    augmented_editor = matrix_editor(
        "增广矩阵 [A | b]", m, n + 1, key=f"aug-{m}-{n}"
    )

    if st.button("求解", type="primary", key="run_solve"):
        a = augmented_editor[:, :-1]
        b = augmented_editor[:, -1]
        try:
            status, info = solve_linear(a, b)
            col1, col2, col3 = st.columns(3)
            col1.metric("r(A)", info["rA"])
            col2.metric("r(A|b)", info["rAb"])
            col3.metric("未知数 n", info["n"])

            if status == "unique":
                st.success("方程组有唯一解。")
                show_text_matrix("唯一解 x", info["x"].reshape(-1, 1))
            elif status == "infinite":
                st.warning("方程组有无穷多解。")
                st.write(f"自由未知数个数：{n - info['rA']}")
                show_text_matrix("一个特解 xp", info["xp"].reshape(-1, 1))
                show_text_matrix("基础解系（每列一个向量）", info["null"])
            else:
                st.error("方程组无解（矛盾方程组）。下面是最小二乘近似解。")
                show_text_matrix("最小二乘近似解", info["x_ls"].reshape(-1, 1))
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 页面：二次型 ====================


def page_quadratic() -> None:
    st.header("二次型 f = xᵀAx")
    n = st.number_input("方阵阶数 n", 1, 12, 2, key="quad_n")
    mat = matrix_editor("输入矩阵 A", n, n, key=f"quad-{n}",
                        default=np.eye(n, dtype=float))

    if st.button("分析", type="primary", key="run_quad"):
        try:
            result = analyze_quadratic(mat)
            if not result["symmetric"]:
                st.info("输入矩阵不对称；二次型只取决于对称部分，已自动使用 S=(A+Aᵀ)/2。")

            show_text_matrix("用于分析的对称矩阵 S", result["matrix"])
            show_text_matrix("特征值（升序）", result["eigenvalues"])
            show_text_matrix("正交变换矩阵 Q", result["Q"])
            standard_form = signed_polynomial(result["standard_terms"])
            st.markdown(f"经正交变换 x = Qy，标准形为：**f = {standard_form}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("正惯性指数 p", result["pos"])
            c2.metric("负惯性指数 q", result["neg"])
            c3.metric("零特征值个数", result["zero"])
            st.success(f"判定结果：{result['kind']}")

            minors = result["minors"]
            minors_df = pd.DataFrame(
                {f"Δ{i}": [float(value)] for i, value in enumerate(minors, start=1)}
            )
            st.caption("顺序主子式（Sylvester 判据）")
            st.dataframe(minors_df, hide_index=True, width="stretch")
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 主导航 ====================


def main() -> None:
    st.title("🧮 矩阵计算器")
    st.caption("NumPy 计算核心 + Streamlit 网页界面")

    page = st.sidebar.radio(
        "功能",
        ["双矩阵运算", "单矩阵运算", "解线性方程组", "二次型分析"],
    )
    if page == "双矩阵运算":
        page_binary()
    elif page == "单矩阵运算":
        page_unary()
    elif page == "解线性方程组":
        page_solve()
    else:
        page_quadratic()


if __name__ == "__main__":
    main()