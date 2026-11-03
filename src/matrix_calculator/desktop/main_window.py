"""桌面主窗口：操作切换、计算调用、结果与错误展示。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..formatting import (
    format_matrix,
    format_scalar,
    matrix_to_csv_text,
    matrix_to_numpy_text,
    signed_polynomial,
)
from ..linalg_topics import analyze_quadratic, solve_linear
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
    "二次型分析 f=xᵀAx": OperationSpec(
        "unary",
        analyze_quadratic,
        "二次型：f = xᵀAx 只取决于对称部分 S=(A+Aᵀ)/2，用特征值判定正定性。",
    ),
}

EXAMPLES: dict[str, list[dict]] = {
    "矩阵加法 A+B": [
        {
            "label": "I＋交换矩阵",
            "operation": "矩阵加法 A+B",
            "a": np.eye(2),
            "b": np.array([[0, 1], [1, 0]], dtype=float),
        },
        {
            "label": "非方阵相加",
            "operation": "矩阵加法 A+B",
            "a": np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
            "b": np.array([[0, 1, 0], [1, 0, 0]], dtype=float),
        },
    ],
    "矩阵乘法 A×B": [
        {
            "label": "2×3 × 3×2",
            "operation": "矩阵乘法 A×B",
            "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
            "b": np.array([[7, 8], [9, 10], [11, 12]], dtype=float),
        },
    ],
    "行列式 det(A)": [
        {
            "label": "旋转矩阵",
            "operation": "行列式 det(A)",
            "matrix": np.array([[0, -1], [1, 0]], dtype=float),
        },
        {
            "label": "奇异矩阵",
            "operation": "行列式 det(A)",
            "matrix": np.array([[1, 1], [1, 1]], dtype=float),
        },
        {
            "label": "3×3 可逆",
            "operation": "行列式 det(A)",
            "matrix": np.array([[2, 1, 0], [1, 3, 2], [0, 2, 4]], dtype=float),
        },
    ],
    "逆矩阵 A⁻¹": [
        {
            "label": "可逆三角阵",
            "operation": "逆矩阵 A⁻¹",
            "matrix": np.array([[2, 1], [0, 3]], dtype=float),
        },
    ],
    "解线性方程组 Ax=b": [
        {"label": "唯一解", "augmented": np.array([[1, 1, 3], [2, -1, 0]], dtype=float)},
        {"label": "无穷多解", "augmented": np.array([[1, 1, 1], [2, 2, 2]], dtype=float)},
        {"label": "无解", "augmented": np.array([[1, 1, 1], [1, 1, 2]], dtype=float)},
    ],
    "二次型分析 f=xᵀAx": [
        {"label": "正定", "matrix": np.eye(2)},
        {"label": "负定", "matrix": -np.eye(2)},
        {"label": "不定", "matrix": np.diag([1.0, -1.0])},
        {"label": "半正定", "matrix": np.array([[1, 1], [1, 1]], dtype=float)},
    ],
}


class MainWindow(QWidget):
    """矩阵计算器桌面主窗口。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("矩阵计算器（桌面版）")
        self.setMinimumSize(760, 600)
        self._export_items: list[tuple[str, np.ndarray]] = []
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
        self.export_button = QPushButton("导出")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)

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

        self.example_group = QGroupBox("示例")
        self.example_layout = QHBoxLayout(self.example_group)

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
        top_bar.addWidget(self.export_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self.example_group)
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
        self._rebuild_example_buttons()
        self._clear_output()

    def _rebuild_example_buttons(self) -> None:
        while self.example_layout.count():
            item = self.example_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for example in EXAMPLES.get(self.op_combo.currentText(), []):
            button = QPushButton(example["label"])
            button.clicked.connect(
                lambda checked=False, ex=example: self._on_example_clicked(ex)
            )
            self.example_layout.addWidget(button)
        self.example_layout.addStretch(1)

    def _on_example_clicked(self, example: dict) -> None:
        if "operation" in example:
            self.op_combo.setCurrentText(example["operation"])
        augmented = example.get("augmented")
        if augmented is not None:
            self.input_a.set_matrix(np.asarray(augmented)[:, :-1])
            self.input_b.set_matrix(np.asarray(augmented)[:, -1:])
            return
        if "a" in example:
            self.input_a.set_matrix(example["a"])
        if "matrix" in example:
            self.input_a.set_matrix(example["matrix"])
        if "b" in example:
            self.input_b.set_matrix(example["b"])

    def _clear_output(self) -> None:
        self.error_label.setHidden(True)
        self.error_label.setText("")
        self.result_label.setText("请选择运算并点击“计算”。")
        self.details_label.setText("")
        self._export_items = []
        self.export_button.setEnabled(False)

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
                result = spec.func(a, b)
                result_text = format_matrix(result)
                details = spec.description
                self._export_items = [("结果", np.asarray(result))]
            elif spec.kind == "unary":
                result_text, details, self._export_items = self._format_unary(
                    name, spec.func(a)
                )
            else:
                b = self.input_b.get_matrix()
                if b.shape[1] != 1:
                    raise MatrixOperationError("向量 b 只能有一列，请用“− 列”删除多余列")
                result_text, details, self._export_items = self._format_solve(
                    *solve_linear(a, b.reshape(-1))
                )
        except (MatrixOperationError, np.linalg.LinAlgError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.result_label.setText(result_text)
        self.details_label.setText(details)
        self.export_button.setEnabled(True)

    def _on_copy(self) -> None:
        QGuiApplication.clipboard().setText(self.result_label.text())

    def _on_export(self) -> None:
        menu = QMenu(self)
        menu.addAction(
            "TXT（完整结果）",
            lambda: self._save_export_text(self.result_label.text(), "txt", "结果"),
        )
        for label, matrix in self._export_items:
            menu.addAction(
                f"CSV · {label}",
                lambda checked=False, m=matrix, l=label: self._save_export_text(
                    matrix_to_csv_text(m), "csv", l
                ),
            )
            menu.addAction(
                f"NumPy · {label}",
                lambda checked=False, m=matrix, l=label: self._save_export_text(
                    matrix_to_numpy_text(m), "py", l
                ),
            )
        menu.exec(self.export_button.mapToGlobal(self.export_button.rect().bottomLeft()))

    def _save_export_text(self, text: str, ext: str, label: str) -> None:
        safe_label = re.sub(r"[^\w\u4e00-\u9fff]+", "-", label).strip("-") or "matrix"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            f"{safe_label}.{ext}",
            "文本文件 (*.txt);;CSV 文件 (*.csv);;Python 文件 (*.py);;所有文件 (*)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _format_unary(self, name: str, result) -> tuple[str, str, list[tuple[str, np.ndarray]]]:
        spec = OPERATIONS[name]
        if name == "行列式 det(A)":
            value = np.array([[float(result)]])
            return format_scalar(result), spec.description, [("行列式", value)]
        if name == "二次型分析 f=xᵀAx":
            return self._format_quadratic(result)
        if name == "特征分解":
            values, vectors = result
            text = (
                "特征值（每行一个）：\n"
                + format_matrix(np.asarray(values).reshape(-1, 1))
                + "\n\n特征向量（每列一个）：\n"
                + format_matrix(vectors)
            )
            items = [
                ("特征值", np.asarray(values).reshape(-1, 1)),
                ("特征向量", np.asarray(vectors)),
            ]
            return text, spec.description, items
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
            items = [
                ("U", np.asarray(u)),
                ("奇异值", np.asarray(s).reshape(-1, 1)),
                ("Vᵀ", np.asarray(vt)),
            ]
            return text, spec.description, items
        return format_matrix(result), spec.description, [("结果", np.asarray(result))]

    def _format_quadratic(self, result: dict) -> tuple[str, str, list[tuple[str, np.ndarray]]]:
        note = ""
        if not result["symmetric"]:
            note = "输入矩阵不对称；二次型只取决于对称部分，已自动使用 S=(A+Aᵀ)/2。\n\n"
        standard_form = signed_polynomial(result["standard_terms"])
        minors_text = "，".join(
            f"Δ{i} = {float(v):.4g}" for i, v in enumerate(result["minors"], start=1)
        )
        text = (
            note
            + "用于分析的对称矩阵 S：\n"
            + format_matrix(result["matrix"])
            + "\n\n特征值（升序）：\n"
            + format_matrix(result["eigenvalues"].reshape(-1, 1))
            + "\n\n正交变换矩阵 Q：\n"
            + format_matrix(result["Q"])
            + f"\n\n经正交变换 x = Qy，标准形为：f = {standard_form}"
            + f"\n\n判定结果：{result['kind']}"
        )
        details = (
            f"正惯性指数 p = {result['pos']}\n"
            f"负惯性指数 q = {result['neg']}\n"
            f"零特征值个数 = {result['zero']}\n"
            f"顺序主子式（Sylvester 判据）：{minors_text}"
        )
        items = [
            ("对称矩阵S", result["matrix"]),
            ("特征值", result["eigenvalues"].reshape(-1, 1)),
            ("正交矩阵Q", result["Q"]),
        ]
        return text, details, items

    def _format_solve(self, status: str, info: dict) -> tuple[str, str, list[tuple[str, np.ndarray]]]:
        details = (
            f"r(A) = {info['rA']}\n"
            f"r([A|b]) = {info['rAb']}\n"
            f"未知数个数 n = {info['n']}"
        )
        if status == "unique":
            x = info["x"].reshape(-1, 1)
            return "方程组有唯一解：\n" + format_matrix(x), details, [("解x", x)]
        if status == "infinite":
            free = info["n"] - info["rA"]
            xp = info["xp"].reshape(-1, 1)
            text = (
                f"方程组有无穷多解，自由未知数个数：{free}\n\n"
                "一个特解 xp：\n"
                + format_matrix(xp)
                + "\n\n基础解系（每列一个向量）：\n"
                + format_matrix(info["null"])
            )
            return text, details, [("特解xp", xp), ("基础解系", info["null"])]
        x_ls = info["x_ls"].reshape(-1, 1)
        return (
            "方程组无解（矛盾方程组）。最小二乘近似解：\n" + format_matrix(x_ls),
            details,
            [("最小二乘近似解", x_ls)],
        )
