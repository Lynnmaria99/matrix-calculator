"""桌面端矩阵输入组件：逐格输入框、行列调整、粘贴与数值转换。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..formatting import format_scalar
from ..operations import MatrixOperationError

MAX_DIMENSION = 12


class _CellEdit(QLineEdit):
    """单个矩阵元素输入框，支持 Ctrl+V 粘贴整块数据。"""

    def __init__(self, wrapper: "MatrixTableWidget", parent=None):
        super().__init__(parent)
        self.wrapper = wrapper
        self.setObjectName("Cell")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedWidth(64)
        self.setPlaceholderText("0")

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt 命名约定)
        if event.matches(QKeySequence.Paste):
            self.wrapper.paste_from_clipboard()
            return
        super().keyPressEvent(event)


class _MatrixGrid(QWidget):
    """逐格输入网格。

    固定预建 ``MAX_DIMENSION × MAX_DIMENSION`` 个输入框，调整行列数时只
    切换可见性：避免动态创建/销毁控件，数值也天然保留。
    """

    def __init__(self, rows: int, cols: int, wrapper: "MatrixTableWidget", parent=None):
        super().__init__(parent)
        self.wrapper = wrapper
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._cells: list[list[_CellEdit]] = [
            [_CellEdit(wrapper) for _ in range(MAX_DIMENSION)]
            for _ in range(MAX_DIMENSION)
        ]
        for i in range(MAX_DIMENSION):
            for j in range(MAX_DIMENSION):
                self._layout.addWidget(self._cells[i][j], i, j)
        self._rows = 0
        self._cols = 0
        self.set_dimensions(rows, cols)

    def set_dimensions(self, rows: int, cols: int) -> None:
        rows = max(0, min(int(rows), MAX_DIMENSION))
        cols = max(0, min(int(cols), MAX_DIMENSION))
        self._rows = rows
        self._cols = cols
        for i in range(MAX_DIMENSION):
            for j in range(MAX_DIMENSION):
                self._cells[i][j].setVisible(i < rows and j < cols)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def cell(self, i: int, j: int) -> _CellEdit:
        return self._cells[i][j]

class MatrixTableWidget(QWidget):
    """带括号的逐格矩阵输入，负责把单元格内容转换为 NumPy 数组。"""

    def __init__(
        self,
        label: str,
        rows: int = 2,
        cols: int = 2,
        *,
        example: np.ndarray | None = None,
        resize_cols: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._example = example

        self.grid = _MatrixGrid(rows, cols, self)
        self.title = QLabel(label)
        self.title.setObjectName("H2")

        self.add_row_button = QPushButton("+ 行")
        self.add_row_button.setObjectName("Ghost")
        self.remove_row_button = QPushButton("− 行")
        self.remove_row_button.setObjectName("Ghost")
        self.add_col_button = QPushButton("+ 列")
        self.add_col_button.setObjectName("Ghost")
        self.remove_col_button = QPushButton("− 列")
        self.remove_col_button.setObjectName("Ghost")
        self.clear_button = QPushButton("清空")
        self.clear_button.setObjectName("Ghost")
        self.example_button = QPushButton("加载示例")
        self.example_button.setObjectName("Ghost")

        self.add_row_button.clicked.connect(self._add_row)
        self.remove_row_button.clicked.connect(self._remove_row)
        self.add_col_button.clicked.connect(self._add_col)
        self.remove_col_button.clicked.connect(self._remove_col)
        self.clear_button.clicked.connect(self.clear_cells)
        self.example_button.clicked.connect(self.load_example)

        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(6)
        for button in (
            self.add_row_button,
            self.remove_row_button,
            self.add_col_button,
            self.remove_col_button,
            self.clear_button,
            self.example_button,
        ):
            buttons_layout.addWidget(button)

        # 教材风格括号：左右两个竖框
        self.left_bracket = QFrame()
        self.left_bracket.setObjectName("BracketL")
        self.left_bracket.setFixedWidth(10)
        self.right_bracket = QFrame()
        self.right_bracket.setObjectName("BracketR")
        self.right_bracket.setFixedWidth(10)

        matrix_row = QHBoxLayout()
        matrix_row.setSpacing(2)
        matrix_row.addWidget(self.left_bracket)
        matrix_row.addWidget(self.grid, stretch=1)
        matrix_row.addWidget(self.right_bracket)
        matrix_row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(self.title)
        outer.addWidget(buttons)
        outer.addLayout(matrix_row)

        self._col_buttons = [self.add_col_button, self.remove_col_button]
        self.set_column_resize_enabled(resize_cols)
        if example is None:
            self.example_button.setHidden(True)

    def set_label(self, label: str) -> None:
        self._label = label
        self.title.setText(label)

    def set_column_resize_enabled(self, enabled: bool) -> None:
        for button in self._col_buttons:
            button.setVisible(enabled)

    def set_columns(self, cols: int) -> None:
        self.grid.set_dimensions(self.grid.rows, cols)

    def _add_row(self) -> None:
        if self.grid.rows >= MAX_DIMENSION:
            return
        self.grid.set_dimensions(self.grid.rows + 1, self.grid.cols)

    def _remove_row(self) -> None:
        if self.grid.rows <= 1:
            return
        self.grid.set_dimensions(self.grid.rows - 1, self.grid.cols)

    def _add_col(self) -> None:
        if self.grid.cols >= MAX_DIMENSION:
            return
        self.grid.set_dimensions(self.grid.rows, self.grid.cols + 1)

    def _remove_col(self) -> None:
        if self.grid.cols <= 1:
            return
        self.grid.set_dimensions(self.grid.rows, self.grid.cols - 1)

    def clear_cells(self) -> None:
        for i in range(self.grid.rows):
            for j in range(self.grid.cols):
                self.grid.cell(i, j).clear()

    def load_example(self) -> None:
        if self._example is not None:
            self.set_matrix(self._example)

    def set_cell(self, i: int, j: int, text: str) -> None:
        self.grid.cell(i, j).setText(text)

    def paste_from_clipboard(self) -> None:
        """把剪贴板中的制表符/空白分隔文本从焦点格开始填入。"""
        text = QApplication.clipboard().text()
        if not text.strip():
            return
        lines = [line for line in text.strip().splitlines() if line.strip()]
        rows: list[list[str]] = []
        for line in lines:
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t")]
            else:
                cells = line.replace(",", " ").split()
            rows.append(cells)

        row0, col0 = self._focused_cell()
        need_rows = min(
            MAX_DIMENSION, max(self.grid.rows, row0 + len(rows))
        )
        need_cols = min(
            MAX_DIMENSION,
            max(self.grid.cols, col0 + max(len(r) for r in rows)),
        )
        self.grid.set_dimensions(need_rows, need_cols)
        for i, cells in enumerate(rows):
            for j, cell_text in enumerate(cells):
                if row0 + i < need_rows and col0 + j < need_cols:
                    self.set_cell(row0 + i, col0 + j, cell_text)

    def _focused_cell(self) -> tuple[int, int]:
        focused = QApplication.focusWidget()
        for i, row in enumerate(self.grid._cells):
            for j, cell in enumerate(row):
                if cell is focused:
                    return i, j
        return 0, 0

    def set_matrix(self, mat) -> None:
        """用数组填充输入框并调整行列数。"""
        mat = np.atleast_2d(np.asarray(mat, dtype=float))
        self.grid.set_dimensions(mat.shape[0], mat.shape[1])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                self.set_cell(i, j, format_scalar(mat[i, j]))

    def get_matrix(self) -> np.ndarray:
        """校验并转换输入内容，非法输入抛出面向用户的 ``MatrixOperationError``。"""
        rows, cols = self.grid.rows, self.grid.cols
        if rows == 0 or cols == 0:
            raise MatrixOperationError(f"{self._label} 不能为空矩阵")
        values = []
        for i in range(rows):
            row = []
            for j in range(cols):
                text = self.grid.cell(i, j).text().strip()
                if not text:
                    raise MatrixOperationError(
                        f"{self._label} 第 {i + 1} 行第 {j + 1} 列为空，请填写数字"
                    )
                try:
                    row.append(float(text))
                except ValueError as exc:
                    raise MatrixOperationError(
                        f"{self._label} 第 {i + 1} 行第 {j + 1} 列 “{text}” 不是数字"
                    ) from exc
            values.append(row)
        return np.array(values, dtype=float)

