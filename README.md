# Matrix Calculator / 矩阵计算器

一个面向线性代数与数值微积分入门的 Python 学习工具。项目提供 PySide6 桌面界面、命令行工具，以及可在 Jupyter Notebook 和脚本中复用的 NumPy 计算核心；Streamlit 网页版处于维护状态。

桌面端是后续新功能的主要方向。阶段路线、验收条件和收尾记录见 [项目计划](plans/PROJECT_PLAN.md)。

## 功能

- 矩阵加法、减法、矩阵乘法、对应元素乘除
- 行列式、逆矩阵、特征分解、奇异值分解、转置
- 线性方程组 `Ax=b`：唯一解、无穷多解、无解与最小二乘近似解
- 向量组：秩、线性相关性、极大线性无关组和线性表示
- 向量内积、长度、夹角与正交判断
- 二次型：特征值、标准形、正交矩阵和正定性判定
- 桌面端微积分入门：初等函数表达式、切线斜率（数值导数）、中点黎曼和、辛普森积分与图像

## 项目结构

```text
src/matrix_calculator/
├── operations.py      # 矩阵基础运算（纯计算，不做 input/print）
├── linalg_topics.py   # 方程组、零空间、二次型判定等专题算法
├── calculus.py        # 安全表达式解析、数值导数与数值积分
├── desktop/           # PySide6 桌面界面与 Matplotlib 嵌入图
├── formatting.py      # 教材风格矩阵格式化
├── io_utils.py        # 命令行输入工具
├── cli.py             # 命令行菜单
├── constants.py       # 共享常量
├── __main__.py        # python -m matrix_calculator 入口
└── __init__.py        # 对外暴露的主要 API

src/matrix_calculator.py # 兼容旧运行方式的薄入口
tests/                   # pytest 测试
streamlit_app.py        # Streamlit 网页界面（维护状态）
plans/PROJECT_PLAN.md   # 阶段路线、验收条件与收尾记录
```

核心设计原则：**数学计算函数不直接读取输入，也不直接打印结果**。矩阵参数不合法时抛出 `MatrixOperationError`；微积分表达式解析失败时抛出 `ExpressionError`，求值错误由桌面界面转换为用户可读提示。

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
python -m pip install -e ".[dev,web,desktop]"
python -m pytest
```

当前测试覆盖基础矩阵运算、微积分表达式与数值积分、线性方程组分类、零空间、二次型判定、格式化输出和 PySide6 桌面 UI。

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

## Streamlit 网页界面（维护状态）

桌面版发布后，网页版进入维护状态：只修复问题，不再增加新功能，新功能优先在 PySide6 桌面端实现。

项目仍保留一个初版 Streamlit 界面，支持双矩阵运算、单矩阵运算、线性方程组求解和二次型分析，适合没有本地 Python 环境时在浏览器中快速使用。

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
特征分解、SVD、转置）、线性方程组 `Ax=b`、二次型分析和微积分入门
（函数曲线/切线、黎曼和与辛普森积分），并复用 NumPy 计算核心与教材风格结果展示。

安装桌面依赖：

```bash
python -m pip install -e ".[desktop]"
```

启动：

```bash
matrix-calculator-desktop
```

界面说明：

- 左侧边栏切换五个页面：双矩阵运算、单矩阵运算、解线性方程组、二次型分析、微积分；侧边栏可一键收起/展开。
- 每个运算都是独立按钮（A + B、A − B、A × B 等），不再使用下拉框。
- “微积分”页支持 f(x) 表达式、图示范围/积分区间、切线位置与分割数滑块，并同步显示函数切线和黎曼和柱状图。
- 双矩阵结果 C 自动展示形状、det(C) 和可逆性，可逆时直接给出 C⁻¹，
  并可通过"继续分析结果 C"按钮接着做特征分解、SVD 或转置。
- "结果 → A / 结果 → B"把结果一键填回输入矩阵，配合页面切换即可实现
  A+B=C 再 det(C) 的链式计算。
- 矩阵 A/B 采用教材风格的逐格输入框（带括号），可增删行列、粘贴数据、清空和加载示例；每个页面配有与网页版一致的示例按钮。
- 输入错误和计算异常在窗口内直接提示，不显示 traceback。
- "中间结果 / 解释"默认折叠，需要时展开；支持图示的运算（SVD、特征分解、
  双矩阵结果、2 元方程组、二次型）会在结果下方自动给出 Matplotlib 图示。
- 计算结果可一键复制为文本，并可导出 TXT（完整结果）、CSV 和 NumPy 文本（与网页版相同的三种格式）。
- 界面采用 Apple 风格的玻璃质感主题。

微积分功能的使用边界：

- 表达式支持 `x`、数字、括号、`+`、`-`、`*`、`/`、`**`（或 `^`），以及 `sin`、`cos`、`tan`、反三角函数、`exp`、`log`、`log10`、`sqrt`、`abs`、`pi` 和 `e`。
- 切线斜率使用中心差分，积分使用中点黎曼和与辛普森法；结果均为**数值近似**，不是符号求导或精确积分。
- 当前版本适合连续且能在采样点有限求值的函数，不会自动识别奇点、不连续点或所有定义域问题；结果出现异常时应调整区间并人工核查。
- 积分上下限会按大小排序，反向区间不按有向积分取负；更完整的误差与方法教学列入后续阶段。

## 后续计划

- 可选引入 SymPy，支持分数、符号和初等行变换过程
- 桌面端 OCR 输入：截图识别单个矩阵并填入逐格输入框供人工确认；运算符仍由按钮选择，不做整表达式识别

