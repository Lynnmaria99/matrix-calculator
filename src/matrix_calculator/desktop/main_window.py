"""桌面主窗口：侧边栏导航、独立运算按钮、结果链式继续分析。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
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

    kind: str  # "binary" / "unary" / "solve" / "quadratic"
    func: Callable | None = None
    description: str = ""


OPERATIONS: dict[str, OperationSpec] = {
    "A + B": OperationSpec(
        "binary", add_matrices, "矩阵加法：对应位置元素相加，要求两个矩阵形状完全相同。"
    ),
    "A − B": OperationSpec(
        "binary", subtract_matrices, "矩阵减法：对应位置元素相减，要求两个矩阵形状完全相同。"
    ),
    "A × B": OperationSpec(
        "binary", matrix_multiply, "矩阵乘法：A 的列数必须等于 B 的行数。"
    ),
    "A ⊙ B": OperationSpec(
        "binary", hadamard_multiply, "Hadamard 乘法：对应位置元素相乘，要求形状相同。"
    ),
    "A ⊘ B": OperationSpec(
        "binary",
        hadamard_divide,
        "Hadamard 除法：对应位置元素相除，要求形状相同且 B 中没有 0。",
    ),
    "det(A)": OperationSpec(
        "unary", determinant, "行列式：方阵到标量的映射，反映矩阵对体积的缩放因子。"
    ),
    "A⁻¹": OperationSpec(
        "unary",
        invert_matrix,
        "逆矩阵：满足 A·A⁻¹ = I；行列式为 0 的奇异矩阵不存在逆矩阵。",
    ),
    "特征分解": OperationSpec(
        "unary", eigendecompose, "特征分解：A·v = λ·v，特征值逐个列出，特征向量按列展示。"
    ),
    "SVD": OperationSpec(
        "unary", singular_value_decompose, "SVD：A = U·Σ·Vᵀ，奇异值按从大到小排列。"
    ),
    "Aᵀ": OperationSpec("unary", transpose, "转置：行列互换。"),
    "二次型 f=xᵀAx": OperationSpec(
        "quadratic",
        analyze_quadratic,
        "二次型：f = xᵀAx 只取决于对称部分 S=(A+Aᵀ)/2，用特征值判定正定性。",
    ),
}

PAGES: dict[str, list[str]] = {
    "双矩阵运算": ["A + B", "A − B", "A × B", "A ⊙ B", "A ⊘ B"],
    "单矩阵运算": ["det(A)", "A⁻¹", "特征分解", "SVD", "Aᵀ"],
    "解线性方程组": [],
    "二次型分析": [],
}

EXAMPLES: dict[str, list[dict]] = {
    "双矩阵运算": [
        {
            "label": "I＋交换矩阵",
            "operation": "A + B",
            "a": np.eye(2),
            "b": np.array([[0, 1], [1, 0]], dtype=float),
        },
        {
            "label": "2×3 × 3×2",
            "operation": "A × B",
            "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
            "b": np.array([[7, 8], [9, 10], [11, 12]], dtype=float),
        },
        {
            "label": "非方阵相加",
            "operation": "A + B",
            "a": np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
            "b": np.array([[0, 1, 0], [1, 0, 0]], dtype=float),
        },
    ],
    "单矩阵运算": [
        {
            "label": "旋转矩阵",
            "operation": "det(A)",
            "matrix": np.array([[0, -1], [1, 0]], dtype=float),
        },
        {
            "label": "奇异矩阵",
            "operation": "det(A)",
            "matrix": np.array([[1, 1], [1, 1]], dtype=float),
        },
        {
            "label": "可逆三角阵",
            "operation": "A⁻¹",
            "matrix": np.array([[2, 1], [0, 3]], dtype=float),
        },
        {
            "label": "3×3 可逆",
            "operation": "det(A)",
            "matrix": np.array([[2, 1, 0], [1, 3, 2], [0, 2, 4]], dtype=float),
        },
    ],
    "解线性方程组": [
        {"label": "唯一解", "augmented": np.array([[1, 1, 3], [2, -1, 0]], dtype=float)},
        {"label": "无穷多解", "augmented": np.array([[1, 1, 1], [2, 2, 2]], dtype=float)},
        {"label": "无解", "augmented": np.array([[1, 1, 1], [1, 1, 2]], dtype=float)},
    ],
    "二次型分析": [
        {"label": "正定", "matrix": np.eye(2)},
        {"label": "负定", "matrix": -np.eye(2)},
        {"label": "不定", "matrix": np.diag([1.0, -1.0])},
        {"label": "半正定", "matrix": np.array([[1, 1], [1, 1]], dtype=float)},
    ],
}

# 继续分析结果 C 时，按钮文字 → 复用的单矩阵运算名。
ANALYSIS_OPS = {"特征分解": "特征分解", "SVD": "SVD", "Cᵀ": "Aᵀ"}


def _add_shadow(widget: QWidget) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(40)
    effect.setOffset(0, 8)
    effect.setColor(QColor(15, 23, 42, 38))
    widget.setGraphicsEffect(effect)


class MainWindow(QWidget):
    """矩阵计算器桌面主窗口。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Root")
        self.setWindowTitle("矩阵计算器（桌面版）")
        self.setMinimumSize(880, 640)
        self._current_page = "双矩阵运算"
        self._selected_ops = {
            page: (ops[0] if ops else "") for page, ops in PAGES.items()
        }
        self._primary_matrix: np.ndarray | None = None
        self._analysis_matrix: np.ndarray | None = None
        self._analysis_active: str | None = None
        self._export_items: list[tuple[str, np.ndarray]] = []
        self._build_ui()
        self._on_page_changed("双矩阵运算")

    # ---------------- UI 构建 ----------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addWidget(self._build_sidebar())
        root.addLayout(self._build_content(), stretch=1)

    def _build_sidebar(self) -> QFrame:
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(210)
        _add_shadow(self.sidebar)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(12, 18, 12, 18)
        side.setSpacing(4)

        title = QLabel("🧮 矩阵计算器")
        title.setObjectName("Title")
        caption = QLabel("NumPy 计算核心 · 桌面版")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        side.addWidget(title)
        side.addWidget(caption)
        side.addSpacing(10)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: dict[str, QPushButton] = {}
        for page in PAGES:
            button = QPushButton(page)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            self.nav_group.addButton(button)
            self.nav_buttons[page] = button
            button.clicked.connect(
                lambda checked=False, p=page: self._on_page_changed(p)
            )
            side.addWidget(button)
        side.addStretch(1)

        footer = QLabel("矩阵 A/B 与结果跨页面共享，\n可将结果一键设为 A 或 B\n继续分析。")
        footer.setObjectName("Caption")
        footer.setWordWrap(True)
        side.addWidget(footer)
        return self.sidebar

    def _build_content(self) -> QVBoxLayout:
        content = QVBoxLayout()
        content.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        self.sidebar_toggle = QPushButton("☰")
        self.sidebar_toggle.setObjectName("Ghost")
        self.sidebar_toggle.setFixedWidth(40)
        self.sidebar_toggle.setCursor(Qt.PointingHandCursor)
        self.sidebar_toggle.setToolTip("收起 / 展开侧边栏")
        self.sidebar_toggle.clicked.connect(self._toggle_sidebar)
        head.addWidget(self.sidebar_toggle)
        self.page_title = QLabel("双矩阵运算")
        self.page_title.setObjectName("Title")
        head.addWidget(self.page_title)
        head.addStretch(1)
        content.addLayout(head)

        self.op_row = QWidget()
        self.op_layout = QHBoxLayout(self.op_row)
        self.op_layout.setContentsMargins(0, 0, 0, 0)
        self.op_layout.setSpacing(8)
        self.op_group = QButtonGroup(self)
        self.op_group.setExclusive(True)
        content.addWidget(self.op_row)

        self.example_row = QWidget()
        self.example_layout = QHBoxLayout(self.example_row)
        self.example_layout.setContentsMargins(0, 0, 0, 0)
        self.example_layout.setSpacing(6)
        content.addWidget(self.example_row)

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
        content.addWidget(self.input_a)
        content.addWidget(self.input_b)

        compute_row = QHBoxLayout()
        compute_row.addStretch(1)
        self.compute_button = QPushButton("计算")
        self.compute_button.setObjectName("Accent")
        self.compute_button.setCursor(Qt.PointingHandCursor)
        self.compute_button.clicked.connect(self._on_compute)
        compute_row.addWidget(self.compute_button)
        content.addLayout(compute_row)

        self.error_label = QLabel("")
        self.error_label.setObjectName("Error")
        self.error_label.setWordWrap(True)
        self.error_label.setHidden(True)
        content.addWidget(self.error_label)

        content.addWidget(self._build_result_card())
        content.addWidget(self._build_details())
        content.addStretch(1)
        return content

    def _build_result_card(self) -> QFrame:
        self.result_card = QFrame()
        self.result_card.setObjectName("Card")
        _add_shadow(self.result_card)
        card = QVBoxLayout(self.result_card)
        card.setContentsMargins(16, 14, 16, 14)
        card.setSpacing(8)

        self.result_title = QLabel("计算结果")
        self.result_title.setObjectName("H2")
        self.result_label = QLabel("请在上方选择运算并点击“计算”。")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.metrics_label = QLabel("")
        self.metrics_label.setObjectName("Caption")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setHidden(True)

        self.analysis_row = QWidget()
        self.analysis_layout = QHBoxLayout(self.analysis_row)
        self.analysis_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_layout.setSpacing(6)
        self.analysis_group = QButtonGroup(self)
        self.analysis_group.setExclusive(False)
        self.analysis_buttons: dict[str, QPushButton] = {}
        for code in ANALYSIS_OPS:
            button = QPushButton(code)
            button.setObjectName("Pill")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(
                lambda checked=False, c=code, b=button: self._on_analysis_clicked(c, b)
            )
            self.analysis_group.addButton(button)
            self.analysis_buttons[code] = button
            self.analysis_layout.addWidget(button)
        self.analysis_layout.addStretch(1)
        self.analysis_row.setHidden(True)

        self.analysis_label = QLabel("")
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.analysis_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.analysis_label.setHidden(True)

        actions = QHBoxLayout()
        self.copy_button = QPushButton("复制结果")
        self.copy_button.setObjectName("Ghost")
        self.copy_button.clicked.connect(self._on_copy)
        self.export_button = QPushButton("导出")
        self.export_button.setObjectName("Ghost")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        self.to_a_button = QPushButton("结果 → A")
        self.to_a_button.setObjectName("Ghost")
        self.to_a_button.setEnabled(False)
        self.to_a_button.clicked.connect(self._result_to_a)
        self.to_b_button = QPushButton("结果 → B")
        self.to_b_button.setObjectName("Ghost")
        self.to_b_button.setEnabled(False)
        self.to_b_button.clicked.connect(self._result_to_b)
        for button in (self.copy_button, self.export_button, self.to_a_button, self.to_b_button):
            button.setCursor(Qt.PointingHandCursor)
            actions.addWidget(button)
        actions.addStretch(1)

        card.addWidget(self.result_title)
        card.addWidget(self.result_label)
        card.addWidget(self.metrics_label)
        card.addWidget(self.analysis_row)
        card.addWidget(self.analysis_label)
        card.addLayout(actions)
        return self.result_card

    def _build_details(self) -> QGroupBox:
        self.details_group = QGroupBox("中间结果 / 解释")
        self.details_group.setCheckable(True)
        self.details_group.setChecked(False)
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_layout = QVBoxLayout(self.details_group)
        details_layout.addWidget(self.details_label)
        return self.details_group

    # ---------------- 页面与按钮 ----------------

    def _toggle_sidebar(self) -> None:
        """收起或展开侧边栏（Shapr3D 式面板开关）。"""
        self.sidebar.setHidden(not self.sidebar.isHidden())
    def _on_page_changed(self, page: str) -> None:
        self._current_page = page
        self.nav_buttons[page].setChecked(True)
        self.page_title.setText(page)
        self._rebuild_op_buttons()
        self._rebuild_example_buttons()
        self._apply_op_layout()
        self._clear_output()

    def _rebuild_op_buttons(self) -> None:
        while self.op_layout.count():
            item = self.op_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        ops = PAGES[self._current_page]
        self.op_row.setHidden(not ops)
        for label in ops:
            button = QPushButton(label)
            button.setObjectName("Pill")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, l=label: self._select_op(l))
            self.op_group.addButton(button)
            button.setChecked(label == self._selected_ops.get(self._current_page))
            self.op_layout.addWidget(button)
        self.op_layout.addStretch(1)

    def _select_op(self, label: str) -> None:
        if self._selected_ops.get(self._current_page) == label:
            return
        self._selected_ops[self._current_page] = label
        for button in self.op_group.buttons():
            button.setChecked(button.text() == label)
        self._apply_op_layout()
        self._clear_output()

    def _rebuild_example_buttons(self) -> None:
        while self.example_layout.count():
            item = self.example_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for example in EXAMPLES.get(self._current_page, []):
            button = QPushButton(example["label"])
            button.setObjectName("Ghost")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(
                lambda checked=False, ex=example: self._on_example_clicked(ex)
            )
            self.example_layout.addWidget(button)
        self.example_layout.addStretch(1)

    def _on_example_clicked(self, example: dict) -> None:
        if "operation" in example:
            self._select_op(example["operation"])
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

    def _current_kind(self) -> str:
        page = self._current_page
        if page == "解线性方程组":
            return "solve"
        if page == "二次型分析":
            return "quadratic"
        return OPERATIONS[self._selected_ops[page]].kind

    def _apply_op_layout(self) -> None:
        kind = self._current_kind()
        if kind == "binary":
            self.input_b.setHidden(False)
            self.input_b.set_label("矩阵 B")
            self.input_b.set_column_resize_enabled(True)
            self.input_b.set_columns(2)
            self.input_b.set_matrix(np.zeros((2, 2)))
        elif kind == "solve":
            self.input_b.setHidden(False)
            self.input_b.set_label("向量 b")
            self.input_b.set_column_resize_enabled(False)
            self.input_b.set_columns(1)
            self.input_b.set_matrix(np.zeros((2, 1)))
        else:
            self.input_b.setHidden(True)
        self.to_b_button.setHidden(kind != "binary")

    def _clear_output(self) -> None:
        self.error_label.setHidden(True)
        self.error_label.setText("")
        self.result_title.setText("计算结果")
        self.result_label.setText("请在上方选择运算并点击“计算”。")
        self.metrics_label.setHidden(True)
        self.metrics_label.setText("")
        self.analysis_row.setHidden(True)
        self.analysis_label.setHidden(True)
        self.analysis_label.setText("")
        self._analysis_active = None
        self._analysis_matrix = None
        for button in self.analysis_buttons.values():
            button.setChecked(False)
        self._export_items = []
        self._primary_matrix = None
        self.export_button.setEnabled(False)
        self.to_a_button.setEnabled(False)
        self.to_b_button.setEnabled(False)

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setHidden(False)

    # ---------------- 计算 ----------------

    def _on_compute(self) -> None:
        self._clear_output()
        kind = self._current_kind()
        try:
            a = self.input_a.get_matrix()
            if kind == "binary":
                b = self.input_b.get_matrix()
                result = OPERATIONS[self._selected_ops[self._current_page]].func(a, b)
                details = self._render_binary_result(np.asarray(result))
            elif kind == "quadratic":
                result_text, details, items = self._format_quadratic(analyze_quadratic(a))
                self._show_result("二次型分析结果", result_text, items)
            elif kind == "solve":
                b = self.input_b.get_matrix()
                if b.shape[1] != 1:
                    raise MatrixOperationError("向量 b 只能有一列，请用“− 列”删除多余列")
                result_text, details, items = self._format_solve(
                    *solve_linear(a, b.reshape(-1))
                )
                self._show_result("方程组求解结果", result_text, items)
            else:
                name = self._selected_ops[self._current_page]
                result_text, details, items = self._format_unary(
                    name, OPERATIONS[name].func(a)
                )
                self._show_result(f"{name} 计算结果", result_text, items)
        except (MatrixOperationError, np.linalg.LinAlgError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.details_label.setText(details)

    def _show_result(
        self, title: str, text: str, items: list[tuple[str, np.ndarray]]
    ) -> None:
        self.result_title.setText(title)
        self.result_label.setText(text)
        self._export_items = items
        self._primary_matrix = items[0][1] if items else None
        self.export_button.setEnabled(True)
        self.to_a_button.setEnabled(self._primary_matrix is not None)
        self.to_b_button.setEnabled(self._primary_matrix is not None)

    def _render_binary_result(self, c: np.ndarray) -> str:
        op_label = self._selected_ops[self._current_page]
        rows, cols = c.shape
        is_square = rows == cols
        text = f"C = {op_label}\n" + format_matrix(c)
        if is_square:
            det_value = determinant(c)
            try:
                inv = invert_matrix(c)
            except (MatrixOperationError, np.linalg.LinAlgError):
                inv = None
            if inv is not None:
                text += "\n\n逆矩阵 C⁻¹：\n" + format_matrix(inv)
            else:
                text += "\n\ndet(C)=0（或数值上接近奇异），所以 C 不存在可用的逆矩阵。"
            metrics = (
                f"形状 C：{rows} × {cols} · det(C) = {det_value:.6g} · "
                + ("C 可逆" if inv is not None else "C 不可逆")
            )
        else:
            metrics = f"形状 C：{rows} × {cols} · 非方阵没有行列式和逆矩阵，仍可继续 SVD 或转置"
        self._show_result(f"运算结果 C = {op_label}", text, [("结果C", c)])
        self.metrics_label.setText(metrics)
        self.metrics_label.setHidden(False)
        self._analysis_matrix = c
        for code, button in self.analysis_buttons.items():
            button.setHidden(code == "特征分解" and not is_square)
        self.analysis_row.setHidden(False)
        self.to_b_button.setHidden(False)
        return OPERATIONS[op_label].description

    def _on_analysis_clicked(self, code: str, button: QPushButton) -> None:
        if self._analysis_matrix is None:
            return
        if self._analysis_active == code:
            self._analysis_active = None
            button.setChecked(False)
            self.analysis_label.setHidden(True)
            self.analysis_label.setText("")
            return
        self._analysis_active = code
        for other in self.analysis_buttons.values():
            other.setChecked(other is button)
        op_name = ANALYSIS_OPS[code]
        text, _desc, _items = self._format_unary(
            op_name, OPERATIONS[op_name].func(self._analysis_matrix)
        )
        self.analysis_label.setText(f"继续分析结果 C · {code}\n\n{text}")
        self.analysis_label.setHidden(False)

    def _result_to_a(self) -> None:
        if self._primary_matrix is not None:
            self.input_a.set_matrix(self._primary_matrix)

    def _result_to_b(self) -> None:
        if self._primary_matrix is not None:
            self.input_b.set_matrix(self._primary_matrix)

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

    # ---------------- 结果格式化 ----------------

    def _format_unary(
        self, name: str, result
    ) -> tuple[str, str, list[tuple[str, np.ndarray]]]:
        spec = OPERATIONS[name]
        if name == "det(A)":
            value = np.array([[float(result)]])
            return format_scalar(result), spec.description, [("行列式", value)]
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
        if name == "SVD":
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

    def _format_quadratic(
        self, result: dict
    ) -> tuple[str, str, list[tuple[str, np.ndarray]]]:
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

    def _format_solve(
        self, status: str, info: dict
    ) -> tuple[str, str, list[tuple[str, np.ndarray]]]:
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

