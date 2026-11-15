# Matrix Calculator

**English** · [简体中文](#简体中文)

A Python learning tool for introductory linear algebra and numerical calculus. It includes a PySide6 desktop application, a command-line interface, and a NumPy-powered computation library for scripts and Jupyter notebooks. The Streamlit web interface is maintained for existing functionality; new features are focused on the desktop app.

## Features

- Matrix addition, subtraction, multiplication, and element-wise operations
- Determinants, inverses, eigendecomposition, singular value decomposition (SVD), and transpose
- Linear systems `Ax = b`, including unique solutions, infinitely many solutions, inconsistent systems, and least-squares approximations
- Vector sets: rank, linear dependence, maximal linearly independent subsets, and linear representations
- Vector dot products, norms, angles, and orthogonality checks
- Quadratic forms: eigenvalues, canonical form, orthogonal matrices, and definiteness
- Introductory numerical calculus on desktop: function plots, numerical tangent slopes, midpoint Riemann sums, and Simpson's rule
- Standalone Windows desktop package available from [GitHub Releases](https://github.com/Lynnmaria99/matrix-calculator/releases)

## Quick start

Install the package in editable mode:

```bash
python -m pip install -e .
```

Launch the desktop app (install desktop dependencies first):

```bash
python -m pip install -e ".[desktop]"
matrix-calculator-desktop
```

Or launch the command-line interface:

```bash
matrix-calculator
```

## Use as a Python library

```python
import numpy as np
from matrix_calculator import classify_quadratic, solve_linear

A = np.array([[1.0, 1.0], [2.0, -1.0]])
b = np.array([3.0, 0.0])

status, info = solve_linear(A, b)
print(status, info["x"])

kind, positive, negative, zero = classify_quadratic(np.array([1.0, 2.0, 0.0]))
print(kind)
```

## Windows standalone app

Download `MatrixCalculator-Windows-x64.zip` from [Releases](https://github.com/Lynnmaria99/matrix-calculator/releases), extract the complete folder, and run `MatrixCalculator.exe`. Python and project dependencies do not need to be installed. Keep the extracted `MatrixCalculator` folder together; the executable depends on files beside it.

See the [Windows packaging guide](packaging/WINDOWS.md) for local builds and release details.

## Jupyter notebook and visualizations

The [linear algebra demo notebook](examples/linear_algebra_demo.ipynb) shows core operations, linear-system classification, 2D linear transformations, and SVD singular-value plots.

```bash
python -m pip install -e ".[notebook]"
jupyter notebook examples/linear_algebra_demo.ipynb
```

## Streamlit web interface

The Streamlit interface is in maintenance mode. It supports binary and unary matrix operations, linear-system solving, and quadratic-form analysis.

```bash
python -m pip install -e ".[web]"
streamlit run streamlit_app.py
```

## Project structure

```text
src/matrix_calculator/
├── operations.py       # Core matrix operations
├── linalg_topics.py    # Linear systems, vector spaces, and quadratic forms
├── calculus.py         # Safe expression parsing and numerical calculus
├── desktop/            # PySide6 desktop interface
├── web/                # Streamlit interface
├── computation_*.py    # Shared computation and result models
├── input_validator.py  # Input validation
├── operation_registry.py
├── result_*.py         # Result formatting and caching
└── __init__.py         # Public API

tests/                  # Automated test suite
plans/PROJECT_PLAN.md   # Project roadmap and phase notes
```

The computation core does not read user input or print results. Invalid matrix arguments raise `MatrixOperationError`; invalid calculus expressions raise `ExpressionError`.

## Development

```bash
python -m pip install -e ".[dev,web,desktop]"
python -m pytest
```

## Notes on numerical calculus

Calculus results are numerical approximations, not symbolic derivatives or exact integrals. The current implementation does not automatically detect every discontinuity, singularity, or domain issue; inspect results and adjust the interval when needed.

## Roadmap

- Explore optional SymPy support for exact fractions, symbolic expressions, and elementary row-operation steps
- Add desktop matrix input from screenshots, with manual review before calculation

---

<a id="简体中文"></a>
## 简体中文

[回到 English](#matrix-calculator)

这是一个面向线性代数与数值微积分入门的 Python 学习工具，提供 PySide6 桌面应用、命令行界面，以及可供脚本和 Jupyter Notebook 复用的 NumPy 计算库。Streamlit 网页版处于维护状态，新功能优先面向桌面端。

### 功能

- 矩阵加减、乘法和对应元素运算
- 行列式、逆矩阵、特征分解、奇异值分解（SVD）与转置
- 线性方程组 `Ax = b`：唯一解、无穷多解、无解和最小二乘近似解
- 向量组的秩、线性相关性、极大线性无关组和线性表示
- 向量内积、长度、夹角与正交判断
- 二次型的特征值、标准形、正交矩阵与正定性判定
- 桌面端数值微积分入门：函数图像、数值切线斜率、中点黎曼和与辛普森积分
- 可从 [GitHub Releases](https://github.com/Lynnmaria99/matrix-calculator/releases) 下载 Windows 独立桌面版

### 快速开始

```bash
python -m pip install -e ".[desktop]"
matrix-calculator-desktop
```

安装并运行命令行版本：

```bash
python -m pip install -e .
matrix-calculator
```

下载 Windows 独立版后，解压完整的 `MatrixCalculator` 文件夹并运行 `MatrixCalculator.exe`，无需安装 Python。请保留文件夹内的全部文件。

### Notebook 演示

`examples/linear_algebra_demo.ipynb` 展示基础运算、线性方程组分类、二维线性变换图和 SVD 奇异值图。安装与启动方式见上方 [Notebook 说明](#jupyter-notebook-and-visualizations)。

### Streamlit 网页版

网页端目前仅维护已有功能。安装依赖并启动：

```bash
python -m pip install -e ".[web]"
streamlit run streamlit_app.py
```

### 开发与测试

```bash
python -m pip install -e ".[dev,web,desktop]"
python -m pytest
```

数学计算核心不直接读取用户输入或打印结果。矩阵参数不合法时会抛出 `MatrixOperationError`；微积分表达式不合法时会抛出 `ExpressionError`。微积分结果为数值近似，并不会自动识别所有奇点、不连续点或定义域问题。

项目结构与阶段记录见 [`plans/PROJECT_PLAN.md`](plans/PROJECT_PLAN.md)。Windows 打包说明见 [`packaging/WINDOWS.md`](packaging/WINDOWS.md)。
