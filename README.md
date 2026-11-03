# Matrix Calculator / 矩阵计算器

一个用于学习线性代数的 Python 矩阵计算器。它既是一个命令行工具，也是一组可以在 Jupyter Notebook、脚本或未来 GUI 中复用的计算函数。

## 功能

- 矩阵加法、减法、矩阵乘法、对应元素乘除
- 行列式、逆矩阵、特征分解、奇异值分解、转置
- 线性方程组 `Ax=b`：唯一解、无穷多解、无解与最小二乘近似解
- 向量组：秩、线性相关性、极大线性无关组和线性表示
- 向量内积、长度、夹角与正交判断
- 二次型：特征值、标准形、正交矩阵和正定性判定

## 项目结构

```text
src/matrix_calculator/
├── operations.py      # 矩阵基础运算（纯计算，不做 input/print）
├── linalg_topics.py   # 方程组、零空间、二次型判定等专题算法
├── formatting.py      # 教材风格矩阵格式化
├── io_utils.py        # 命令行输入工具
├── cli.py             # 命令行菜单
├── constants.py       # 共享常量
├── __main__.py        # python -m matrix_calculator 入口
└── __init__.py        # 对外暴露的主要 API

src/matrix_calculator.py # 兼容旧运行方式的薄入口
tests/                   # pytest 测试
streamlit_app.py        # Streamlit 网页界面（初版）
```

核心设计原则：**数学计算函数不直接读取输入，也不直接打印结果**。参数不合法时抛出 `MatrixOperationError`，由命令行、Notebook 或未来的 GUI 决定如何展示。

## 运行命令行版本

不安装也可以直接运行旧入口：

```bash
python src/matrix_calculator.py
```

如果当前位于 `src/` 目录，也可以运行：

```bash
python -m matrix_calculator
```

## 以可编辑模式安装（推荐）

```bash
python -m pip install -e .
```

安装后会得到一个命令：

```bash
matrix-calculator
```

以后修改 `src/` 中的代码，无需重新安装即可生效。

## 在代码或 Notebook 中使用

```python
import numpy as np
from matrix_calculator import (
    add_matrices,
    matrix_multiply,
    solve_linear,
    classify_quadratic,
)

A = np.array([[2.0, 0.0], [0.0, 3.0]])
b = np.array([4.0, 9.0])

status, info = solve_linear(A, b)
print(status, info["x"])

kind, pos, neg, zero = classify_quadratic(np.array([1.0, 2.0, 0.0]))
print(kind)  # 半正定
```

## 运行测试

```bash
python -m pip install -e ".[dev,web]"
python -m pytest
```

当前测试覆盖基础矩阵运算、线性方程组分类、零空间、二次型判定和格式化输出。

## Notebook 演示与可视化

`examples/linear_algebra_demo.ipynb` 演示如何把计算核心当作函数库使用，
包含基础运算、线性方程组分类、二维线性变换图和 SVD 奇异值图。

安装可视化依赖：

```bash
python -m pip install -e ".[notebook]"
```

启动 Notebook：

```bash
jupyter notebook examples/linear_algebra_demo.ipynb
```

只在代码里需要画图时，额外导入可视化模块（核心 CLI 不依赖 matplotlib）：

```python
import matplotlib.pyplot as plt
from matrix_calculator.visualization import plot_linear_transform

plot_linear_transform([[2, 1], [0, 1]])
plt.show()
```

## Streamlit 网页界面

项目提供一个初版 Streamlit 界面，支持双矩阵运算、单矩阵运算、
线性方程组求解和二次型分析。

安装 Web 依赖：

```bash
python -m pip install -e ".[web]"
```

启动：

```bash
streamlit run streamlit_app.py
```

浏览器会自动打开本地页面；也可以访问终端中显示的本地地址。

## PySide6 桌面界面

项目提供一个 PySide6 桌面窗口，支持双矩阵运算、单矩阵运算（行列式、逆、
特征分解、SVD、转置）、线性方程组 `Ax=b` 和二次型分析，并复用与网页版
相同的 NumPy 计算核心与教材风格结果展示。

安装桌面依赖：

```bash
python -m pip install -e ".[desktop]"
```

启动：

```bash
matrix-calculator-desktop
```

界面说明：

- 表格输入矩阵 A/B，可增删行列、粘贴数据、清空和加载示例；每个运算配有与网页版一致的示例按钮。
- 单矩阵运算时自动隐藏矩阵 B；方程组模式将 B 切换为向量输入。
- 输入错误和计算异常在窗口内直接提示，不显示 traceback。
- "中间结果 / 解释"默认折叠，需要时勾选展开。
- 计算结果可一键复制为文本，并可导出 TXT（完整结果）、CSV 和 NumPy 文本（与网页版相同的三种格式）。

## 后续计划

- 可选引入 SymPy，支持分数、符号和初等行变换过程

