"""桌面主窗口：操作切换、计算调用、结果与错误展示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..formatting import format_matrix, format_scalar
from ..linalg_topics import solve_linear
from ..operations import (
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
from .matrix_input import MatrixTableWidget


@dataclass(frozen=True)
class OperationSpec:
    """一种运算的界面配置。"""

    kind: str  # "binary" / "unary" / "solve"
    func: Callable | None = None
    description: str = ""


OPERATIONS: dict[str, OperationSpec] = {
    "矩阵加法 A+B": OperationSpec(
        "binary", add_matrices, "矩阵加法：对应位置元素相加，要求两个矩阵形状完全相同。"
    ),
    "矩阵减法 A−B": OperationSpec(
        "binary", subtract_matrices, "矩阵减法：对应位置元素相减，要求两个矩阵形状完全相同。"
    ),
    "矩阵乘法 A×B": OperationSpec(
        "binary", matrix_multiply, "矩阵乘法：A 的列数必须等于 B 的行数。"
    ),
    "Hadamard 乘法 A⊙B": OperationSpec(
        "binary", hadamard_multiply, "Hadamard 乘法：对应位置元素相乘，要求形状相同。"
    ),
    "Hadamard 除法 A⊘B": OperationSpec(
        "binary",
        hadamard_divide,
        "Hadamard 除法：对应位置元素相除，要求形状相同且 B 中没有 0。",
    ),
    "行列式 det(A)": OperationSpec(
        "unary", determinant, "行列式：方阵到标量的映射，反映矩阵对体积的缩放因子。"
    ),
    "逆矩阵 A⁻¹": OperationSpec(
        "unary",
        invert_matrix,
        "逆矩阵：满足 A·A⁻¹ = I；行列式为 0 的奇异矩阵不存在逆矩阵。",
    ),
    "特征分解": OperationSpec(
        "unary", eigendecompose, "特征分解：A·v = λ·v，特征值逐个列出，特征向量按列展示。"
    ),
    "奇异值分解 (SVD)": OperationSpec(
        "unary", singular_value_decompose, "SVD：A = U·Σ·Vᵀ，奇异值按从大到小排列。"
    ),
    "转置 Aᵀ": OperationSpec("unary", transpose, "转置：行列互换。"),
    "解线性方程组 Ax=b": OperationSpec("solve"),
}


class MainWindow(QWidget):
    """矩阵计算器桌面主窗口。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("矩阵计算器（桌面版）")
        self.setMinimumSize(720, 560)
        self._build_ui()
        self.op_combo.setCurrentText("矩阵加法 A+B")
        self._on_operation_changed("矩阵加法 A+B")

    def _build_ui(self) -> None:
        self.op_combo = QComboBox()
        self.op_combo.addItems(list(OPERATIONS.keys()))
        self.op_combo.currentTextChanged.connect(self._on_operation_changed)

        self.compute_button = QPushButton("计算")
        self.compute_button.clicked.connect(self._on_compute)
        self.copy_button = QPushButton("复制结果")
        self.copy_button.clicked.connect(self._on_copy)

        self.input_a = MatrixTableWidget(
            "矩阵 A",
            rows=2,
            cols=2,
            example=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        self.input_b = MatrixTableWidget(
            "矩阵 B",
            rows=2,
            cols=2,
            example=np.array([[5.0, 6.0], [7.0, 8.0]]),
        )

        self.result_label = QLabel("请选择运算并点击“计算”。")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.details_group = QGroupBox("中间结果 / 解释")
        self.details_group.setCheckable(True)
        self.details_group.setChecked(False)
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_layout = QVBoxLayout(self.details_group)
        details_layout.addWidget(self.details_label)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #d13438;")
        self.error_label.setWordWrap(True)
        self.error_label.setHidden(True)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.op_combo, stretch=1)
        top_bar.addWidget(self.compute_button)
        top_bar.addWidget(self.copy_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self.input_a)
        layout.addWidget(self.input_b)
        layout.addWidget(QLabel("计算结果"))
        layout.addWidget(self.result_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.details_group)
        layout.addStretch(1)

    def _on_operation_changed(self, name: str) -> None:
        spec = OPERATIONS[name]
        if spec.kind == "unary":
            self.input_b.setHidden(True)
        else:
            self.input_b.setHidden(False)
            if spec.kind == "solve":
                self.input_b.set_label("向量 b")
                self.input_b.set_column_resize_enabled(False)
                self.input_b.set_columns(1)
                self.input_b.set_matrix(np.zeros((2, 1)))
            else:
                self.input_b.set_label("矩阵 B")
                self.input_b.set_column_resize_enabled(True)
                self.input_b.set_columns(2)
                self.input_b.set_matrix(np.zeros((2, 2)))
        self._clear_output()

    def _clear_output(self) -> None:
        self.error_label.setHidden(True)
        self.error_label.setText("")
        self.result_label.setText("请选择运算并点击“计算”。")
        self.details_label.setText("")

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setHidden(False)

    def _on_compute(self) -> None:
        self._clear_output()
        name = self.op_combo.currentText()
        spec = OPERATIONS[name]
        try:
            a = self.input_a.get_matrix()
            if spec.kind == "binary":
                b = self.input_b.get_matrix()
                result_text = format_matrix(spec.func(a, b))
                details = spec.description
            elif spec.kind == "unary":
                result_text, details = self._format_unary(name, spec.func(a))
            else:
                b = self.input_b.get_matrix()
                if b.shape[1] != 1:
                    raise MatrixOperationError("向量 b 只能有一列，请用“− 列”删除多余列")
                result_text, details = self._format_solve(*solve_linear(a, b.reshape(-1)))
        except (MatrixOperationError, np.linalg.LinAlgError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.result_label.setText(result_text)
        self.details_label.setText(details)

    def _on_copy(self) -> None:
        QGuiApplication.clipboard().setText(self.result_label.text())

    def _format_unary(self, name: str, result) -> tuple[str, str]:
        spec = OPERATIONS[name]
        if name == "行列式 det(A)":
            return format_scalar(result), spec.description
        if name == "特征分解":
            values, vectors = result
            text = (
                "特征值（每行一个）：\n"
                + format_matrix(np.asarray(values).reshape(-1, 1))
                + "\n\n特征向量（每列一个）：\n"
                + format_matrix(vectors)
            )
            return text, spec.description
        if name == "奇异值分解 (SVD)":
            u, s, vt = result
            text = (
                "U：\n"
                + format_matrix(u)
                + "\n\n奇异值 Σ（每行一个）：\n"
                + format_matrix(np.asarray(s).reshape(-1, 1))
                + "\n\nVᵀ：\n"
                + format_matrix(vt)
            )
            return text, spec.description
        return format_matrix(result), spec.description

    def _format_solve(self, status: str, info: dict) -> tuple[str, str]:
        details = (
            f"r(A) = {info['rA']}\n"
            f"r([A|b]) = {info['rAb']}\n"
            f"未知数个数 n = {info['n']}"
        )
        if status == "unique":
            return "方程组有唯一解：\n" + format_matrix(info["x"].reshape(-1, 1)), details
        if status == "infinite":
            free = info["n"] - info["rA"]
            text = (
                f"方程组有无穷多解，自由未知数个数：{free}\n\n"
                "一个特解 xp：\n"
                + format_matrix(info["xp"].reshape(-1, 1))
                + "\n\n基础解系（每列一个向量）：\n"
                + format_matrix(info["null"])
            )
            return text, details
        return (
            "方程组无解（矛盾方程组）。最小二乘近似解：\n"
            + format_matrix(info["x_ls"].reshape(-1, 1)),
            details,
        )
