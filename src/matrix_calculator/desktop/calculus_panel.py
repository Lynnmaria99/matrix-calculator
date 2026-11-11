"""桌面端微积分面板：输入控件、数值计算与图示。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..calculus import (
    ExpressionError,
    derivative_value,
    parse_expression,
    riemann_midpoint,
    simpson_integral,
)

if TYPE_CHECKING:
    from .main_window import MainWindow


def build_calculus_panel(owner: MainWindow) -> QWidget:
    """微积分页：表达式、切线滑块、黎曼和滑块与两张图。"""
    from .plots import PlotCanvas

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    expr_row = QHBoxLayout()
    expr_row.addWidget(QLabel("f(x) ="))
    owner.calc_expr = QLineEdit()
    owner.calc_expr.setText("sin(x)")
    owner.calc_expr.setPlaceholderText("输入表达式，例如 x^2 * sin(x)（支持 ^、sin、cos、exp、log、sqrt、pi、e）")
    expr_row.addWidget(owner.calc_expr, stretch=1)
    layout.addLayout(expr_row)

    range_row = QHBoxLayout()
    for label_text, attr, default in (
        ("图示范围从", "calc_xmin", -5.0),
        ("到", "calc_xmax", 5.0),
        ("积分下限 a", "calc_a", 0.0),
        ("上限 b", "calc_b", 3.0),
    ):
        range_row.addWidget(QLabel(label_text))
        spin = QDoubleSpinBox()
        spin.setRange(-100.0, 100.0)
        spin.setSingleStep(0.5)
        spin.setValue(default)
        spin.setDecimals(1)
        setattr(owner, attr, spin)
        range_row.addWidget(spin)
    range_row.addStretch(1)
    layout.addLayout(range_row)

    owner.calc_tangent_label = QLabel("")
    owner.calc_tangent_label.setObjectName("Caption")
    owner.calc_tangent_label.setWordWrap(True)
    owner.calc_func_canvas = PlotCanvas()
    layout.addWidget(owner.calc_tangent_label)
    layout.addWidget(owner.calc_func_canvas)

    n_row = QHBoxLayout()
    n_row.addWidget(QLabel("分割数 n"))
    owner.calc_n = QSlider(Qt.Horizontal)
    owner.calc_n.setRange(4, 200)
    owner.calc_n.setValue(20)
    n_row.addWidget(owner.calc_n, stretch=1)
    layout.addLayout(n_row)

    owner.calc_integral_label = QLabel("")
    owner.calc_integral_label.setObjectName("Caption")
    owner.calc_integral_label.setWordWrap(True)
    owner.calc_int_canvas = PlotCanvas()
    layout.addWidget(owner.calc_integral_label)
    layout.addWidget(owner.calc_int_canvas)

    owner.calc_error_label = QLabel("")
    owner.calc_error_label.setObjectName("Error")
    owner.calc_error_label.setWordWrap(True)
    owner.calc_error_label.setHidden(True)
    layout.addWidget(owner.calc_error_label)

    owner.calc_expr.returnPressed.connect(owner._on_compute)
    owner.calc_expr.editingFinished.connect(owner._on_compute)
    for spin in (owner.calc_xmin, owner.calc_xmax, owner.calc_a, owner.calc_b):
        spin.valueChanged.connect(owner._on_compute)
    owner.calc_x0 = QSlider(Qt.Horizontal)
    owner.calc_x0.setRange(0, 1000)
    owner.calc_x0.setValue(500)
    owner.calc_x0.valueChanged.connect(owner._on_compute)
    owner.calc_n.valueChanged.connect(owner._on_compute)

    x0_row = QHBoxLayout()
    x0_row.addWidget(QLabel("切线位置 x₀"))
    x0_row.addWidget(owner.calc_x0, stretch=1)
    layout.insertLayout(2, x0_row)
    return panel


def run_calculus(owner: MainWindow) -> None:
    try:
        f = parse_expression(owner.calc_expr.text())
    except ExpressionError as exc:
        owner.calc_error_label.setText(str(exc))
        owner.calc_error_label.setHidden(False)
        return
    owner.calc_error_label.setHidden(True)
    try:
        owner._render_calculus_plots(f)
    except (ArithmeticError, TypeError, ValueError) as exc:
        owner.calc_error_label.setText(f"表达式无法求值：{exc}")
        owner.calc_error_label.setHidden(False)


def render_calculus_plots(owner: MainWindow, f) -> None:
    from ..visualization import _setup_plot_style

    _setup_plot_style()
    xmin, xmax = sorted((owner.calc_xmin.value(), owner.calc_xmax.value()))
    if xmax - xmin < 1e-9:
        xmax = xmin + 1.0
    xs = np.linspace(xmin, xmax, 400)
    with np.errstate(all="ignore"):
        ys = f(xs)

    # 函数曲线 + 切线
    frac = owner.calc_x0.value() / owner.calc_x0.maximum()
    x0 = xmin + frac * (xmax - xmin)
    slope = derivative_value(f, x0)
    y0 = float(f(x0))

    fig = owner.calc_func_canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, color="#1f77b4", lw=1.8, label="f(x)")
    if np.isfinite(y0) and np.isfinite(slope):
        half = (xmax - xmin) * 0.18
        tx = np.linspace(x0 - half, x0 + half, 2)
        ax.plot(tx, y0 + slope * (tx - x0), color="#ff9500", lw=1.6, label="切线")
        ax.plot([x0], [y0], "o", color="#d13438", markersize=6)
        owner.calc_tangent_label.setText(
            f"x₀ = {x0:.3g}    f(x₀) = {y0:.4g}    切线斜率 f'(x₀) ≈ {slope:.4g}"
        )
    else:
        owner.calc_tangent_label.setText(f"x₀ = {x0:.3g}（该点无定义或值无效）")
    ax.set_title("函数曲线与切线", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    owner.calc_func_canvas.adjust_layout()
    owner.calc_func_canvas.draw()

    # 黎曼和 + 辛普森对比
    a, b = owner.calc_a.value(), owner.calc_b.value()
    n = owner.calc_n.value()
    total, lefts, heights, widths = riemann_midpoint(f, a, b, n)
    simpson = simpson_integral(f, a, b, n=max(n, 200))
    fig2 = owner.calc_int_canvas.figure
    fig2.clear()
    ax2 = fig2.add_subplot(111)
    ax2.bar(lefts, heights, width=widths, align="edge", alpha=0.55,
            color="#0071e3", edgecolor="white", label=f"黎曼和（n={n}）")
    xs2 = np.linspace(min(a, b), max(a, b), 300)
    with np.errstate(all="ignore"):
        ax2.plot(xs2, f(xs2), color="#d62728", lw=1.8, label="f(x)")
    ax2.set_title("定积分 = 曲线下面积", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=9)
    owner.calc_int_canvas.adjust_layout()
    owner.calc_int_canvas.draw()

    owner.calc_integral_label.setText(
        f"区间 [{min(a, b):.3g}, {max(a, b):.3g}]    黎曼和 ≈ {total:.5g}    辛普森 ≈ {simpson:.5g}"
    )
