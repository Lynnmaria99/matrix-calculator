"""桌面端结果可视化：Matplotlib 图嵌入结果卡片（可选依赖 matplotlib）。"""

from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..operations import MatrixOperationError
from ..visualization import _setup_plot_style, plot_linear_transform, plot_singular_values


def _circle_points(n: int = 144) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def _style_axes(ax, title: str) -> None:
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)


def plot_linear_transform_2d(ax, matrix) -> None:
    """复用核心的二维线性变换图，只画标准基的变换前后。"""
    plot_linear_transform(
        matrix, vectors=np.array([[1.0, 0.0], [0.0, 1.0]]), ax=ax
    )


def plot_eig_geometry(ax, a, values, vectors) -> None:
    """2×2 实矩阵特征分解几何：单位圆、变换后椭圆与实特征向量方向。"""
    matrix = np.asarray(a, dtype=float)
    if matrix.shape != (2, 2):
        raise MatrixOperationError("特征分解图示只支持 2×2 矩阵")
    _setup_plot_style()
    circle = _circle_points()
    ellipse = circle @ matrix.T
    ax.plot(circle[:, 0], circle[:, 1], color="#1f77b4", lw=1.6, label="单位圆")
    ax.plot(ellipse[:, 0], ellipse[:, 1], color="#d62728", lw=1.8,
            label=r"$A\circ$单位圆")
    values = np.asarray(values)
    vectors = np.asarray(vectors)
    for j in range(2):
        v = vectors[:, j]
        if np.iscomplexobj(v):
            continue
        v = np.real(v)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            continue
        unit = v / norm
        ax.quiver(0, 0, unit[0], unit[1], angles="xy", scale_units="xy",
                  scale=1, color="#2ca02c", width=0.006)
        ax.annotate(
            rf"$\lambda_{j + 1}={np.real(values[j]):.3g}$",
            (unit[0] * 1.18, unit[1] * 1.18),
            fontsize=9,
            color="#2ca02c",
        )
    _style_axes(ax, "特征方向：单位圆 → A∘单位圆")


def plot_svd_geometry(ax, u, s, vt) -> None:
    """2×2 SVD 几何：单位圆变为椭圆，主轴长度为奇异值。"""
    u = np.asarray(u, dtype=float)
    s = np.asarray(s, dtype=float)
    vt = np.asarray(vt, dtype=float)
    if u.shape != (2, 2):
        raise MatrixOperationError("SVD 几何图示只支持 2×2 矩阵")
    _setup_plot_style()
    circle = _circle_points()
    ellipse = circle @ (u @ np.diag(s) @ vt).T
    ax.plot(circle[:, 0], circle[:, 1], color="#1f77b4", lw=1.6, label="单位圆")
    ax.plot(ellipse[:, 0], ellipse[:, 1], color="#d62728", lw=1.8,
            label=r"$A\circ$单位圆")
    for i in range(2):
        vec = u[:, i] * s[i]
        ax.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy",
                  scale=1, color="#ff9500", width=0.006)
        ax.annotate(
            rf"$\sigma_{i + 1}={s[i]:.3g}$",
            (vec[0] * 1.1, vec[1] * 1.1),
            fontsize=9,
            color="#ff9500",
        )
    _style_axes(ax, "SVD 几何：单位圆 → 椭圆")


def plot_svd_bar(ax, a) -> None:
    """复用核心的奇异值柱状图。"""
    plot_singular_values(a, ax=ax)


def plot_solve_lines(ax, a, b, status, info) -> None:
    """2 元方程组的直线图像：相交 / 平行 / 重合。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape[1] != 2:
        raise MatrixOperationError("方程组图示只支持两个未知数")
    _setup_plot_style()
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    xs = np.linspace(-10.0, 10.0, 200)
    for i in range(min(a.shape[0], 3)):
        c0, c1 = a[i]
        if abs(c1) > 1e-12:
            ax.plot(xs, (b[i] - c0 * xs) / c1, color=colors[i], lw=1.8,
                    label=f"方程 {i + 1}")
        elif abs(c0) > 1e-12:
            ax.axvline(b[i] / c0, color=colors[i], lw=1.8, label=f"方程 {i + 1}")
    if status == "unique":
        x = info["x"]
        ax.plot([x[0]], [x[1]], "o", color="#ff9500", markersize=8,
                label="唯一解")
    titles = {
        "unique": "直线相交 → 唯一解",
        "infinite": "直线重合 → 无穷多解",
        "none": "直线平行 → 无解",
    }
    ax.set_title(titles.get(status, "Ax = b"), fontsize=11)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)


def plot_quadratic_contours(ax, result) -> None:
    """2×2 二次型等高线与主轴方向。"""
    s = np.asarray(result["matrix"], dtype=float)
    if s.shape != (2, 2):
        raise MatrixOperationError("二次型等高线图示只支持 2×2 矩阵")
    _setup_plot_style()
    grid = np.linspace(-3.0, 3.0, 240)
    x, y = np.meshgrid(grid, grid)
    f = s[0, 0] * x ** 2 + 2 * s[0, 1] * x * y + s[1, 1] * y ** 2
    ax.contourf(x, y, f, levels=24, cmap="Blues", alpha=0.75)
    q = np.asarray(result["Q"], dtype=float)
    for j in range(2):
        v = q[:, j]
        ax.plot([-3 * v[0], 3 * v[0]], [-3 * v[1], 3 * v[1]],
                color="#ff9500", lw=1.8, label="主轴" if j == 0 else None)
    ax.set_title(f"二次型等高线：{result['kind']}", fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=9)


BUILDERS = {
    "linear_transform": plot_linear_transform_2d,
    "eig_geometry": plot_eig_geometry,
    "svd_geometry": plot_svd_geometry,
    "svd_bar": plot_svd_bar,
    "solve_lines": plot_solve_lines,
    "quadratic_contours": plot_quadratic_contours,
}


class PlotCanvas(FigureCanvasQTAgg):
    """嵌入结果卡片的 Matplotlib 画布。"""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(5.8, 3.4), dpi=100)
        super().__init__(self.figure)
        self.setMinimumHeight(250)
