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
python -m pytest
```

当前测试覆盖基础矩阵运算、线性方程组分类、零空间、二次型判定和格式化输出。

## 后续计划

- 增加 Jupyter Notebook 示例和线性代数可视化
- 增加 Streamlit 交互界面
- 可选引入 SymPy，支持分数、符号和初等行变换过程