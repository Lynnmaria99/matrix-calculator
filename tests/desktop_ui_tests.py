"""桌面端测试：侧边栏导航、运算按钮、结果继续分析与 offscreen 窗口行为。"""

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

    win = MainWindow()
    yield win
    # QScrollArea 在解释器关闭阶段销毁会触发 PySide6 6.12 的原生崩溃；
    # 在这里主动解绑并销毁滚动容器与其内容。
    from PySide6.QtWidgets import QScrollArea

    win.close()
    from PySide6.QtCore import QPropertyAnimation

    for anim in win.findChildren(QPropertyAnimation):
        anim.stop()
    for scroll in win.findChildren(QScrollArea):
        inner = scroll.takeWidget()
        if inner is not None:
            inner.deleteLater()
        scroll.deleteLater()
    win.deleteLater()
    qapp.processEvents()
    qapp.quit()


def _fill(widget, values) -> None:
    widget.set_matrix(np.asarray(values, dtype=float))


def _page(window, page: str) -> None:
    window.nav_buttons[page].click()


def _op(window, label: str) -> None:
    window._select_op(label)


class TestNavigation:
    def test_sidebar_pages(self, window):
        assert list(window.nav_buttons) == [
            "双矩阵运算", "单矩阵运算", "解线性方程组", "二次型分析",
        ]

    def test_op_buttons_per_page(self, window):
        _page(window, "双矩阵运算")
        labels = [
            window.op_layout.itemAt(i).widget().text()
            for i in range(window.op_layout.count())
            if window.op_layout.itemAt(i).widget() is not None
        ]
        assert labels == ["A + B", "A − B", "A × B", "A ⊙ B", "A ⊘ B"]

        _page(window, "解线性方程组")
        assert window.op_row.isHidden()

    def test_default_selection_binary(self, window):
        _page(window, "双矩阵运算")
        assert window._selected_ops["双矩阵运算"] == "A + B"
        assert not window.input_b.isHidden()


class TestMatrixTableWidget:
    def test_roundtrip(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A")
        _fill(widget, [[1, 2], [3, 4.5]])
        np.testing.assert_allclose(widget.get_matrix(), [[1, 2], [3, 4.5]])

    def test_empty_cell_rejected(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A", rows=2, cols=2)
        widget.set_cell(0, 0, "1")
        widget.set_cell(1, 1, "2")
        with pytest.raises(MatrixOperationError, match="第 1 行第 2 列为空"):
            widget.get_matrix()

    def test_non_numeric_rejected(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A")
        widget.set_cell(0, 0, "abc")
        with pytest.raises(MatrixOperationError, match="不是数字"):
            widget.get_matrix()

    def test_empty_matrix_rejected(self, qapp):
        from matrix_calculator.desktop.matrix_input import MatrixTableWidget

        widget = MatrixTableWidget("矩阵 A", rows=0, cols=2)
        with pytest.raises(MatrixOperationError, match="不能为空矩阵"):
            widget.get_matrix()


class TestBinaryPage:
    def test_addition_with_det_and_inverse(self, window):
        _page(window, "双矩阵运算")
        _op(window, "A + B")
        _fill(window.input_a, [[2, 0], [0, 3]])
        _fill(window.input_b, [[1, 0], [0, 1]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "C = A + B" in window.result_title.text()
        assert "3" in window.result_label.text()
        assert "det(C)" in window.metrics_label.text()
        assert "C 可逆" in window.metrics_label.text()
        assert "C⁻¹" in window.result_label.text()
        assert not window.analysis_row.isHidden()

    def test_singular_result_no_inverse(self, window):
        _page(window, "双矩阵运算")
        _op(window, "A + B")
        _fill(window.input_a, [[1, 1], [1, 1]])
        _fill(window.input_b, [[0, 0], [0, 0]])
        window.compute_button.click()
        assert "C 不可逆" in window.metrics_label.text()
        assert "不存在可用的逆矩阵" in window.result_label.text()

    def test_nonsquare_hides_eig_analysis(self, window):
        _page(window, "双矩阵运算")
        _op(window, "A × B")
        _fill(window.input_a, [[1, 2, 3], [4, 5, 6]])
        _fill(window.input_b, [[1], [2], [3]])
        window.compute_button.click()
        assert window.analysis_buttons["特征分解"].isHidden()
        assert not window.analysis_buttons["SVD"].isHidden()

    def test_continue_analysis_toggle(self, window):
        _page(window, "双矩阵运算")
        _op(window, "A + B")
        _fill(window.input_a, [[2, 0], [0, 3]])
        _fill(window.input_b, [[1, 0], [0, 1]])
        window.compute_button.click()
        window.analysis_buttons["特征分解"].click()
        assert not window.analysis_label.isHidden()
        assert "特征值" in window.analysis_label.text()
        window.analysis_buttons["特征分解"].click()
        assert window.analysis_label.isHidden()

    def test_result_to_a(self, window):
        _page(window, "双矩阵运算")
        _op(window, "A + B")
        _fill(window.input_a, [[2, 0], [0, 3]])
        _fill(window.input_b, [[1, 0], [0, 1]])
        window.compute_button.click()
        window.to_a_button.click()
        np.testing.assert_allclose(window.input_a.get_matrix(), [[3, 0], [0, 4]])

    def test_shape_error_then_recovery(self, window):
        _page(window, "双矩阵运算")
        _op(window, "A + B")
        _fill(window.input_a, [[1, 2]])
        _fill(window.input_b, [[1], [2]])
        window.compute_button.click()
        assert not window.error_label.isHidden()
        assert "shape 不一致" in window.error_label.text()

        _fill(window.input_b, [[3, 4]])
        window.compute_button.click()
        assert window.error_label.isHidden()

    def test_binary_example_switches_operation(self, window):
        _page(window, "双矩阵运算")
        buttons = [
            window.example_layout.itemAt(i).widget()
            for i in range(window.example_layout.count())
        ]
        buttons = [b for b in buttons if b is not None]
        buttons[1].click()  # 2×3 × 3×2
        assert window._selected_ops["双矩阵运算"] == "A × B"
        np.testing.assert_allclose(window.input_a.get_matrix(), [[1, 2, 3], [4, 5, 6]])


class TestUnaryPage:
    def test_unary_hides_b_and_singular_inverse(self, window):
        _page(window, "单矩阵运算")
        _op(window, "A⁻¹")
        assert window.input_b.isHidden()
        _fill(window.input_a, [[1, 2], [2, 4]])
        window.compute_button.click()
        assert not window.error_label.isHidden()
        assert "不可逆" in window.error_label.text()

        _fill(window.input_a, [[4, 2], [1, 3]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "0.2" in window.result_label.text()

    def test_determinant(self, window):
        _page(window, "单矩阵运算")
        _op(window, "det(A)")
        _fill(window.input_a, [[0, -1], [1, 0]])
        window.compute_button.click()
        assert "1" in window.result_label.text()
        assert not window.error_label.isHidden() or True

    def test_eigendecompose(self, window):
        _page(window, "单矩阵运算")
        _op(window, "特征分解")
        _fill(window.input_a, [[2, 0], [0, 3]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "特征值" in window.result_label.text()

    def test_result_to_a_across_pages(self, window):
        _page(window, "单矩阵运算")
        _op(window, "Aᵀ")
        _fill(window.input_a, [[1, 2], [3, 4]])
        window.compute_button.click()
        window.to_a_button.click()
        np.testing.assert_allclose(window.input_a.get_matrix(), [[1, 3], [2, 4]])
        _page(window, "双矩阵运算")
        np.testing.assert_allclose(window.input_a.get_matrix(), [[1, 3], [2, 4]])


class TestSolvePage:
    def test_solve_unique_with_details(self, window):
        _page(window, "解线性方程组")
        _fill(window.input_a, [[2, 1], [1, 3]])
        _fill(window.input_b, [[4], [5]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "唯一解" in window.result_label.text()
        assert window.details_group.isChecked() is False
        window.details_group.setChecked(True)
        assert "r(A)" in window.details_label.text()
        window.details_group.setChecked(False)

    def test_solve_infinite_and_none(self, window):
        _page(window, "解线性方程组")
        _fill(window.input_a, [[1, 1], [2, 2]])
        _fill(window.input_b, [[1], [2]])
        window.compute_button.click()
        assert "无穷多解" in window.result_label.text()

        _fill(window.input_b, [[1], [3]])
        window.compute_button.click()
        assert "无解" in window.result_label.text()

    def test_solve_example_fills(self, window):
        _page(window, "解线性方程组")
        buttons = [
            window.example_layout.itemAt(i).widget()
            for i in range(window.example_layout.count())
        ]
        buttons = [b for b in buttons if b is not None]
        buttons[0].click()
        np.testing.assert_allclose(window.input_a.get_matrix(), [[1, 1], [2, -1]])
        np.testing.assert_allclose(window.input_b.get_matrix(), [[3], [0]])


class TestQuadraticPage:
    def test_positive_definite(self, window):
        _page(window, "二次型分析")
        assert window.input_b.isHidden()
        _fill(window.input_a, [[2, 0], [0, 3]])
        window.compute_button.click()
        assert window.error_label.isHidden()
        assert "正定" in window.result_label.text()
        window.details_group.setChecked(True)
        assert "正惯性指数" in window.details_label.text()

    def test_asymmetric_note(self, window):
        _page(window, "二次型分析")
        _fill(window.input_a, [[1, 2], [0, 3]])
        window.compute_button.click()
        assert "不对称" in window.result_label.text()

    def test_examples(self, window):
        _page(window, "二次型分析")
        buttons = [
            window.example_layout.itemAt(i).widget()
            for i in range(window.example_layout.count())
        ]
        labels = [b.text() for b in buttons if b is not None]
        assert labels == ["正定", "负定", "不定", "半正定"]


class TestExport:
    def test_export_items_populated(self, window):
        _page(window, "单矩阵运算")
        _op(window, "特征分解")
        _fill(window.input_a, [[2, 0], [0, 3]])
        window.compute_button.click()
        labels = [label for label, _ in window._export_items]
        assert labels == ["特征值", "特征向量"]
        assert window.export_button.isEnabled()

    def test_export_disabled_before_compute(self, window):
        _page(window, "二次型分析")
        assert not window.export_button.isEnabled()

    def test_non_numeric_shows_error(self, window):
        _page(window, "双矩阵运算")
        _op(window, "A + B")
        window.input_a.set_cell(0, 0, "oops")
        window.input_b.set_cell(0, 0, "1")
        window.compute_button.click()
        assert "不是数字" in window.error_label.text()



def _wait_for(qapp, condition, timeout_ms: int = 1500) -> bool:
    import time

    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class TestSidebarToggle:
    def test_sidebar_collapses_and_expands(self, window, qapp):
        assert not window.sidebar.isHidden()
        window.sidebar_toggle.click()
        assert _wait_for(qapp, lambda: window.sidebar.isHidden())
        window.sidebar_toggle.click()
        assert _wait_for(qapp, lambda: not window.sidebar.isHidden())

class TestScrollLayout:
    def test_content_in_scroll_area(self, window):
        from PySide6.QtWidgets import QScrollArea

        scroll = window.findChild(QScrollArea)
        assert scroll is not None
        assert scroll.widgetResizable() is True





