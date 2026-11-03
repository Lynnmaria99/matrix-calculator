"""桌面 MVP 测试：表格转换、非法输入、核心调用与 offscreen 窗口行为。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from matrix_calculator.operations import MatrixOperationError


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(qapp):
    from matrix_calculator.desktop.main_window import MainWindow

    return MainWindow()


def _fill(widget, values) -> None:
    widget.set_matrix(np.asarray(values, dtype=float))


class TestMatrixTableWidget:
    def test_roundtrip(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A")
        _fill(widget, [[1, 2], [3, 4.5]])
        np.testing.assert_allclose(widget.get_matrix(), [[1, 2], [3, 4.5]])

    def test_empty_cell_rejected(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A", rows=2, cols=2)
        widget.table.setItem(0, 0, QTableWidgetItem("1"))
        widget.table.setItem(1, 1, QTableWidgetItem("2"))
        with pytest.raises(MatrixOperationError, match="第 1 行第 2 列为空"):
            widget.get_matrix()

    def test_non_numeric_rejected(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A")
        widget.table.setItem(0, 0, QTableWidgetItem("abc"))
        with pytest.raises(MatrixOperationError, match="不是数字"):
            widget.get_matrix()

    def test_empty_matrix_rejected(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A", rows=0, cols=2)
        with pytest.raises(MatrixOperationError, match="不能为空矩阵"):
            widget.get_matrix()


class TestMainWindow:
    def test_window_created(self, window):
        assert window.windowTitle() == "矩阵计算器（桌面版）"

    def test_binary_addition(self, window):
        window.op_combo.setCurrentText("矩阵加法 A+B")
        assert not window.input_b.isHidden()
        _fill(window.input_a, [[1, 2], [3, 4]])
        _fill(window.input_b, [[5, 6], [7, 8]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "6" in window.result_label.text()

    def test_shape_error_then_recovery(self, window):
        window.op_combo.setCurrentText("矩阵加法 A+B")
        _fill(window.input_a, [[1, 2]])
        _fill(window.input_b, [[1], [2]])
        window.compute_button.click()
        assert not window.error_label.isHidden()
        assert "shape 不一致" in window.error_label.text()

        _fill(window.input_b, [[3, 4]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "4" in window.result_label.text()

    def test_unary_hides_b_and_singular_inverse(self, window):
        window.op_combo.setCurrentText("逆矩阵 A⁻¹")
        assert window.input_b.isHidden()
        _fill(window.input_a, [[1, 2], [2, 4]])
        window.compute_button.click()
        assert not window.error_label.isHidden()
        assert "不可逆" in window.error_label.text()

        _fill(window.input_a, [[4, 2], [1, 3]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "0.2" in window.result_label.text()

    def test_eigendecompose(self, window):
        window.op_combo.setCurrentText("特征分解")
        _fill(window.input_a, [[2, 0], [0, 3]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "特征值" in window.result_label.text()
        assert "特征向量" in window.result_label.text()

    def test_solve_unique_with_collapsible_details(self, window):
        window.op_combo.setCurrentText("解线性方程组 Ax=b")
        assert not window.input_b.isHidden()
        _fill(window.input_a, [[2, 1], [1, 3]])
        _fill(window.input_b, [[4], [5]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "唯一解" in window.result_label.text()

        assert window.details_group.isChecked() is False  # 默认折叠
        window.details_group.setChecked(True)
        assert "r(A)" in window.details_label.text()
        window.details_group.setChecked(False)

    def test_solve_infinite_and_none(self, window):
        window.op_combo.setCurrentText("解线性方程组 Ax=b")
        _fill(window.input_a, [[1, 1], [2, 2]])
        _fill(window.input_b, [[1], [2]])
        window.compute_button.click()
        assert "无穷多解" in window.result_label.text()

        _fill(window.input_b, [[1], [3]])
        window.compute_button.click()
        assert "无解" in window.result_label.text()

    def test_solve_rejects_multi_column_b(self, window):
        window.op_combo.setCurrentText("解线性方程组 Ax=b")
        window.input_b.set_columns(2)
        _fill(window.input_a, [[1, 0], [0, 1]])
        _fill(window.input_b, [[1, 2], [3, 4]])
        window.compute_button.click()
        assert "只能有一列" in window.error_label.text()
        window.input_b.set_columns(1)

    def test_non_numeric_shows_error(self, window):
        window.op_combo.setCurrentText("矩阵加法 A+B")
        window.input_a.table.setItem(0, 0, QTableWidgetItem("oops"))
        window.input_b.table.setItem(0, 0, QTableWidgetItem("1"))
        window.compute_button.click()
        assert "不是数字" in window.error_label.text()
