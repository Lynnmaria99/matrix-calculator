"""线性代数可视化工具（可选依赖 matplotlib）。

用法::

    import matplotlib.pyplot as plt
    from matrix_calculator.visualization import plot_linear_transform

为避免给纯命令行/计算核心强加 matplotlib，本模块不会在包的
``__init__`` 中自动导入，需要时显式导入即可。
"""

from __future__ import annotations

import numpy as np

from .constants import EPS
from .operations import MatrixOperationError

_CJK_FONTS = [
    "Microsoft YaHei",   # Windows
    "PingFang SC",       # macOS
    "Noto Sans CJK SC",  # Linux 常见
    "SimHei",
]
_FONT_READY = False


def _setup_plot_style():
    """尽力挑选一个能显示中文的字体；找不到就退回默认英文字体。"""
    global _FONT_READY
    if _FONT_READY:
        return
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def plot_linear_transform(a, vectors=None, ax=None):
    """把二维向量在线性变换 A 作用前后画在同一张图上。

    参数
    ----
    a: array_like, shape (2, 2)
        线性变换矩阵。
    vectors: array_like, shape (k, 2)，可选
        原始向量；默认为标准基 e1、e2 和向量 (1, 1)。
    ax: matplotlib.axes.Axes，可选
        画到指定坐标轴上；不传则新建一张图。

    返回
    ----
    ax: matplotlib.axes.Axes
    """
    matrix = np.asarray(a, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise MatrixOperationError("线性变换可视化只支持 2×2 矩阵")

    if vectors is None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    vectors = np.asarray(vectors, dtype=np.float64)
    transformed = vectors @ matrix.T

    _setup_plot_style()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.quiver(
        np.zeros(len(vectors)),
        np.zeros(len(vectors)),
        vectors[:, 0],
        vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#1f77b4",
        width=0.006,
        label="x（变换前）",
    )
    ax.quiver(
        np.zeros(len(transformed)),
        np.zeros(len(transformed)),
        transformed[:, 0],
        transformed[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#d62728",
        width=0.006,
        alpha=0.8,
        label="Ax（变换后）",
    )

    for x, y in vectors:
        ax.annotate(f"({x:g}, {y:g})", (x, y), textcoords="offset points",
                    xytext=(4, 4), color="#1f77b4", fontsize=9)
    for x, y in transformed:
        ax.annotate(f"({x:g}, {y:g})", (x, y), textcoords="offset points",
                    xytext=(4, -10), color="#d62728", fontsize=9)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(r"二维线性变换：$x \rightarrow Ax$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    return ax


def plot_singular_values(a, ax=None):
    """画矩阵的奇异值柱状图（数据分析中 PCA/降维常见的“碎石图”雏形）。

    返回 ``(ax, singular_values)``。
    """
    matrix = np.asarray(a, dtype=np.float64)
    s = np.linalg.svd(matrix, compute_uv=False)
    indices = np.arange(1, s.size + 1)

    _setup_plot_style()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    colors = ["#2ca02c" if value > EPS else "#bbbbbb" for value in s]
    ax.bar(indices, s, color=colors, alpha=0.85)
    for x, value in zip(indices, s):
        ax.text(x, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(indices)
    ax.set_xlabel("奇异值序号")
    ax.set_ylabel(r"奇异值 $\sigma_i$")
    ax.set_title(r"奇异值分解 SVD：$A = U \cdot diag(\sigma) \cdot V^{T}$")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return ax, s
