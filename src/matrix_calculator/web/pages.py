"""Streamlit 的四个线性代数页面与示例数据。"""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import (
    MatrixOperationError,
    add_matrices,
    analyze_quadratic,
    determinant,
    hadamard_divide,
    hadamard_multiply,
    invert_matrix,
    matrix_multiply,
    solve_linear,
    subtract_matrices,
)
from ..formatting import signed_polynomial
from .matrix_input import (
    _component_key,
    _raw_from_matrix,
    _state_keys,
    matrix_input,
)
from .results import (
    _matrix_signature,
    explain_binary_error,
    explain_unary_error,
    render_matrix_analysis,
    show_error,
    show_text_matrix,
)


# ==================== 页面：双矩阵运算 ====================


BINARY_OPERATIONS = [
    ("add", "A + B", add_matrices),
    ("subtract", "A − B", subtract_matrices),
    ("multiply", "A × B", matrix_multiply),
    ("hadamard_multiply", "A ⊙ B", hadamard_multiply),
    ("hadamard_divide", "A ⊘ B", hadamard_divide),
]
BINARY_OPTION_LABELS = {code: label for code, label, _ in BINARY_OPERATIONS}
BINARY_OPERATION_FUNCTIONS = {code: func for code, _, func in BINARY_OPERATIONS}


BINARY_RESULT_KEY = "binary:last_result"
BINARY_RESULT_ANALYSIS_KEY = "binary:result_analysis"
BINARY_RESULT_ANALYSIS_LABELS = {
    "eigendecomposition": "特征分解",
    "svd": "SVD",
    "transpose": "Cᵀ",
}


BINARY_EXAMPLES = [
    {
        "label": "I＋交换矩阵",
        "operation": "add",
        "a": np.eye(2),
        "b": np.array([[0, 1], [1, 0]], dtype=float),
    },
    {
        "label": "2×3 × 3×2",
        "operation": "multiply",
        "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
        "b": np.array([[7, 8], [9, 10], [11, 12]], dtype=float),
    },
    {
        "label": "非方阵相加",
        "operation": "add",
        "a": np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
        "b": np.array([[0, 1, 0], [1, 0, 0]], dtype=float),
    },
]

UNARY_EXAMPLES = [
    {
        "label": "旋转矩阵",
        "operation": "determinant",
        "matrix": np.array([[0, -1], [1, 0]], dtype=float),
    },
    {
        "label": "奇异矩阵",
        "operation": "determinant",
        "matrix": np.array([[1, 1], [1, 1]], dtype=float),
    },
    {
        "label": "可逆三角阵",
        "operation": "inverse",
        "matrix": np.array([[2, 1], [0, 3]], dtype=float),
    },
    {
        "label": "3×3 可逆",
        "operation": "determinant",
        "matrix": np.array([[2, 1, 0], [1, 3, 2], [0, 2, 4]], dtype=float),
    },
]

SOLVE_EXAMPLES = [
    {
        "label": "唯一解",
        "matrix": np.array([[1, 1, 3], [2, -1, 0]], dtype=float),
    },
    {
        "label": "无穷多解",
        "matrix": np.array([[1, 1, 1], [2, 2, 2]], dtype=float),
    },
    {
        "label": "无解",
        "matrix": np.array([[1, 1, 1], [1, 1, 2]], dtype=float),
    },
]

QUADRATIC_EXAMPLES = [
    {"label": "正定", "matrix": np.eye(2)},
    {"label": "负定", "matrix": -np.eye(2)},
    {"label": "不定", "matrix": np.diag([1.0, -1.0])},
    {"label": "半正定", "matrix": np.array([[1, 1], [1, 1]], dtype=float)},
]


def _set_matrix_example(key: str, matrix: np.ndarray) -> None:
    array = np.asarray(matrix, dtype=float)
    dims_key, raw_key = _state_keys(key)
    st.session_state[dims_key] = [int(array.shape[0]), int(array.shape[1])]
    st.session_state[raw_key] = _raw_from_matrix(array)
    st.session_state.pop(_component_key(key), None)


def render_example_buttons(examples: list[dict], apply_example, prefix: str) -> None:
    with st.container(border=True):
        st.caption("示例：点击即可载入，然后再点页面下方的计算按钮。")
        columns = st.columns(min(4, len(examples)), wrap=True)
        for index, (column, example) in enumerate(zip(columns, examples)):
            with column:
                if st.button(
                    example["label"],
                    key=f"{prefix}-example-{index}",
                    width="stretch",
                    wrap=True,
                ):
                    apply_example(example)
                    st.rerun()


def _apply_binary_example(example: dict) -> None:
    st.session_state["binary_operation"] = example["operation"]
    _set_matrix_example("binary-a", example["a"])
    _set_matrix_example("binary-b", example["b"])
    st.session_state.pop(BINARY_RESULT_KEY, None)
    st.session_state.pop(BINARY_RESULT_ANALYSIS_KEY, None)


def _apply_unary_example(example: dict) -> None:
    st.session_state["unary_operation"] = example["operation"]
    _set_matrix_example("unary-a", example["matrix"])


def _apply_solve_example(example: dict) -> None:
    _set_matrix_example("solve-ab", example["matrix"])


def _apply_quadratic_example(example: dict) -> None:
    _set_matrix_example("quadratic-a", example["matrix"])


def _binary_input_signature(
    operation_code: str,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
) -> tuple[str, tuple, tuple]:
    return (
        operation_code,
        _matrix_signature(mat_a),
        _matrix_signature(mat_b),
    )


def _render_binary_result(
    result: np.ndarray,
    operation_code: str,
    *,
    stale: bool,
) -> None:
    """展示 C=A*B，并把常用单矩阵分析直接接到结果下面。"""
    operation_label = BINARY_OPTION_LABELS[operation_code]
    if stale:
        st.warning("A、B 或运算方式已改变，下面是上一次结果；点击“计算”后可继续分析。")

    show_text_matrix(f"运算结果 C = {operation_label}", result)

    rows, cols = result.shape
    is_square = rows == cols
    determinant_value = None
    inverse_matrix = None

    if is_square:
        determinant_value = determinant(result)
        try:
            inverse_matrix = invert_matrix(result)
        except (MatrixOperationError, np.linalg.LinAlgError):
            inverse_matrix = None

        col_shape, col_determinant, col_invertible = st.columns(3)
        with col_shape:
            with st.container(border=True):
                st.metric("形状 C", f"{rows} × {cols}")
        with col_determinant:
            with st.container(border=True):
                st.metric("行列式 det(C)", f"{determinant_value:.6g}")
        with col_invertible:
            with st.container(border=True):
                st.metric("C 是否可逆", "可逆" if inverse_matrix is not None else "不可逆")

        if inverse_matrix is not None:
            show_text_matrix("逆矩阵 C⁻¹", inverse_matrix)
        else:
            st.info("det(C)=0（或数值上接近奇异），所以 C 不存在可用的逆矩阵。")
    else:
        col_shape, _, _ = st.columns(3)
        with col_shape:
            with st.container(border=True):
                st.metric("形状 C", f"{rows} × {cols}")
        st.caption("非方阵没有行列式和逆矩阵；仍可继续查看 SVD 或转置。")

    analysis_options = (
        ["eigendecomposition", "svd", "transpose"]
        if is_square
        else ["svd", "transpose"]
    )
    if st.session_state.get(BINARY_RESULT_ANALYSIS_KEY) not in analysis_options:
        st.session_state[BINARY_RESULT_ANALYSIS_KEY] = None

    with st.container(border=True):
        st.caption("继续分析结果 C，选择一项即可；det(C) 和可逆时的 C⁻¹ 已自动显示。")
        result_operation = st.segmented_control(
            "继续分析结果 C",
            options=analysis_options,
            format_func=BINARY_RESULT_ANALYSIS_LABELS.get,
            default=None,
            required=False,
            label_visibility="collapsed",
            width="stretch",
            key=BINARY_RESULT_ANALYSIS_KEY,
            disabled=stale,
        )

    if result_operation is not None and not stale:
        render_matrix_analysis(result, result_operation, symbol="C")


def page_binary() -> None:
    st.header("双矩阵运算")

    render_example_buttons(BINARY_EXAMPLES, _apply_binary_example, "binary")

    with st.container(border=True):
        operation_code = st.segmented_control(
            "选择双矩阵运算",
            options=[code for code, _, _ in BINARY_OPERATIONS],
            format_func=BINARY_OPTION_LABELS.get,
            default="add",
            required=True,
            label_visibility="collapsed",
            width="stretch",
            key="binary_operation",
        )
        st.caption("A ⊙ B 和 A ⊘ B 表示对应元素相乘 / 相除；A × B 是矩阵乘法。")

    with st.container(border=True):
        mat_a = matrix_input("矩阵 A", "binary-a", symbol="A")
        st.caption("对下面的矩阵 B 应用上面选择的运算。")
        mat_b = matrix_input("矩阵 B", "binary-b", symbol="B")

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_binary = st.button("计算", type="primary", key="run_binary")

    current_signature = _binary_input_signature(operation_code, mat_a, mat_b)
    if run_binary:
        try:
            result = BINARY_OPERATION_FUNCTIONS[operation_code](mat_a, mat_b)
            st.session_state[BINARY_RESULT_KEY] = {
                "operation": operation_code,
                "matrix": result,
                "signature": current_signature,
            }
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            st.session_state.pop(BINARY_RESULT_KEY, None)
            message = explain_binary_error(exc, operation_code, mat_a, mat_b)
            show_error(MatrixOperationError(message))

    cached_result = st.session_state.get(BINARY_RESULT_KEY)
    if cached_result is not None:
        _render_binary_result(
            cached_result["matrix"],
            cached_result["operation"],
            stale=cached_result["signature"] != current_signature,
        )


# ==================== 页面：单矩阵运算 ====================


UNARY_OPERATIONS = {
    "determinant": "det(A)",
    "inverse": "A⁻¹",
    "eigendecomposition": "特征分解",
    "svd": "SVD",
    "transpose": "Aᵀ",
}


def page_unary() -> None:
    st.header("单矩阵运算")

    render_example_buttons(UNARY_EXAMPLES, _apply_unary_example, "unary")

    with st.container(border=True):
        operation_code = st.segmented_control(
            "选择单矩阵运算",
            options=list(UNARY_OPERATIONS),
            format_func=UNARY_OPERATIONS.get,
            default="determinant",
            required=True,
            label_visibility="collapsed",
            width="stretch",
            key="unary_operation",
        )

    with st.container(border=True):
        mat = matrix_input("矩阵 A", "unary-a", symbol="A")

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_unary = st.button("计算", type="primary", key="run_unary")

    if run_unary:
        try:
            render_matrix_analysis(mat, operation_code, symbol="A")
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            message = explain_unary_error(exc, operation_code, mat)
            show_error(MatrixOperationError(message))


# ==================== 页面：线性方程组 ====================


def page_solve() -> None:
    st.header("解线性方程组 Ax = b")

    render_example_buttons(SOLVE_EXAMPLES, _apply_solve_example, "solve")

    with st.container(border=True):
        st.caption("输入增广矩阵 [A | b]：最后一列自动作为常数项 b，可通过 +列 增加未知数。")
        augmented = matrix_input(
            "增广矩阵 [A | b]",
            "solve-ab",
            default=np.zeros((2, 3)),
            min_cols=2,
            symbol="[A | b]",
        )

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_solve = st.button("求解", type="primary", key="run_solve")

    if run_solve:
        m, n_plus_one = augmented.shape
        n = n_plus_one - 1
        a = augmented[:, :n]
        b = augmented[:, -1]
        try:
            status, info = solve_linear(a, b)
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.metric("r(A)", info["rA"])
            with col2:
                with st.container(border=True):
                    st.metric("r(A|b)", info["rAb"])
            with col3:
                with st.container(border=True):
                    st.metric("未知数 n", info["n"])

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

    render_example_buttons(QUADRATIC_EXAMPLES, _apply_quadratic_example, "quadratic")

    with st.container(border=True):
        st.caption("输入方阵阶数可通过 −阶 / +阶 调整；默认是 2 阶单位矩阵。")
        mat = matrix_input(
            "矩阵 A",
            "quadratic-a",
            default=np.eye(2),
            square=True,
            min_rows=1,
            symbol="A",
        )

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_quad = st.button("分析", type="primary", key="run_quad")

    if run_quad:
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
            with c1:
                with st.container(border=True):
                    st.metric("正惯性指数 p", result["pos"])
            with c2:
                with st.container(border=True):
                    st.metric("负惯性指数 q", result["neg"])
            with c3:
                with st.container(border=True):
                    st.metric("零特征值个数", result["zero"])
            st.success(f"判定结果：{result['kind']}")

            minors = result["minors"]
            minors_text = "，".join(
                f"Δ{i} = {float(value):.4g}" for i, value in enumerate(minors, start=1)
            )
            st.caption(f"顺序主子式（Sylvester 判据）：{minors_text}")
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)
