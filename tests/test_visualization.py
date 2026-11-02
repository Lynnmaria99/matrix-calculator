import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

import matplotlib.pyplot as plt

from matrix_calculator.operations import MatrixOperationError
from matrix_calculator.visualization import (
    plot_linear_transform,
    plot_singular_values,
)


def test_linear_transform_plot_renders_2d():
    fig, ax = plt.subplots()
    returned = plot_linear_transform(np.array([[2.0, 1.0], [0.0, 1.0]]), ax=ax)
    assert returned is ax
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_linear_transform_rejects_non_2x2():
    with pytest.raises(MatrixOperationError, match="2×2"):
        plot_linear_transform(np.eye(3))


def test_singular_value_plot_values():
    fig, ax = plt.subplots()
    _, s = plot_singular_values(np.array([[1.0, 2.0], [2.0, 4.0]]), ax=ax)
    assert s[0] == pytest.approx(5.0, abs=1e-8)
    assert s[1] == pytest.approx(0.0, abs=1e-8)
    plt.close(fig)