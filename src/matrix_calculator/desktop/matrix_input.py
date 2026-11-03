"""桌面端矩阵输入组件：表格输入、行列调整、粘贴与数值转换。"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..formatting import format_scalar
from ..operations import MatrixOperationError

MAX_DIMENSION = 12


class _MatrixTable(QTableWidget):
    """支持 Ctrl+V 粘贴文本数据的表格。"""

    def __init__(self, rows: int, cols: int, wrapper: "MatrixTableWidget", parent=None):
        super().__init__(rows, cols, parent)
        self.wrapper = wrapper

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt 命名约定)
        if event.matches(QKeySequence.Paste):
            self.wrapper.paste_from_clipboard()
            return
        super().keyPressEvent(event)


class MatrixTableWidget(QWidget):
    """带工具栏的表格矩阵输入，负责把单元格内容转换为 NumPy 数组。"""

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

        self.table = _MatrixTable(rows, cols, self)
        self.title = QLabel(label)

        self.add_row_button = QPushButton("+ 行")
        self.remove_row_button = QPushButton("− 行")
        self.add_col_button = QPushButton("+ 列")
        self.remove_col_button = QPushButton("− 列")
        self.clear_button = QPushButton("清空")
        self.example_button = QPushButton("加载示例")

        self.add_row_button.clicked.connect(self._add_row)
        self.remove_row_button.clicked.connect(self._remove_row)
        self.add_col_button.clicked.connect(self._add_col)
        self.remove_col_button.clicked.connect(self._remove_col)
        self.clear_button.clicked.connect(self.clear_cells)
        self.example_button.clicked.connect(self.load_example)

        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        for button in (
            self.add_row_button,
            self.remove_row_button,
            self.add_col_button,
            self.remove_col_button,
            self.clear_button,
            self.example_button,
        ):
            buttons_layout.addWidget(button)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.title)
        outer.addWidget(buttons)
        outer.addWidget(self.table)

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
        self.table.setColumnCount(cols)

    def _add_row(self) -> None:
        if self.table.rowCount() >= MAX_DIMENSION:
            return
        self.table.insertRow(self.table.rowCount())

    def _remove_row(self) -> None:
        if self.table.rowCount() <= 1:
            return
        self.table.removeRow(self.table.rowCount() - 1)

    def _add_col(self) -> None:
        if self.table.columnCount() >= MAX_DIMENSION:
            return
        self.table.insertColumn(self.table.columnCount())

    def _remove_col(self) -> None:
        if self.table.columnCount() <= 1:
            return
        self.table.removeColumn(self.table.columnCount() - 1)

    def clear_cells(self) -> None:
        for i in range(self.table.rowCount()):
            for j in range(self.table.columnCount()):
                self.table.setItem(i, j, QTableWidgetItem(""))

    def load_example(self) -> None:
        if self._example is not None:
            self.set_matrix(self._example)

    def paste_from_clipboard(self) -> None:
        """把剪贴板中的制表符/空白分隔文本填入表格。"""
        text = QApplication.clipboard().text()
        if not text.strip():
            return
        lines = [line for line in text.strip().splitlines() if line.strip()]
        rows = []
        for line in lines:
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t")]
            else:
                cells = line.replace(",", " ").split()
            rows.append(cells)

        row0 = max(self.table.currentRow(), 0)
        col0 = max(self.table.currentColumn(), 0)
        need_rows = min(MAX_DIMENSION, max(self.table.rowCount(), row0 + len(rows)))
        need_cols = min(
            MAX_DIMENSION,
            max(self.table.columnCount(), col0 + max(len(r) for r in rows)),
        )
        self.table.setRowCount(need_rows)
        self.table.setColumnCount(need_cols)
        for i, cells in enumerate(rows):
            for j, cell in enumerate(cells):
                if row0 + i < need_rows and col0 + j < need_cols:
                    self.table.setItem(row0 + i, col0 + j, QTableWidgetItem(cell))

    def set_matrix(self, mat) -> None:
        """用数组填充表格并调整行列数。"""
        mat = np.atleast_2d(np.asarray(mat, dtype=float))
        self.table.setRowCount(mat.shape[0])
        self.table.setColumnCount(mat.shape[1])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(format_scalar(mat[i, j])))

    def get_matrix(self) -> np.ndarray:
        """校验并转换表格内容，非法输入抛出面向用户的 ``MatrixOperationError``。"""
        rows, cols = self.table.rowCount(), self.table.columnCount()
        if rows == 0 or cols == 0:
            raise MatrixOperationError(f"{self._label} 不能为空矩阵")
        values = []
        for i in range(rows):
            row = []
            for j in range(cols):
                item = self.table.item(i, j)
                text = item.text().strip() if item is not None else ""
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
