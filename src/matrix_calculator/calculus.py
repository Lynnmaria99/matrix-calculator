"""微积分计算核心：表达式解析、数值导数、切线与定积分。

纯计算模块，不做任何输入/输出；解析失败抛出 ``ExpressionError``。
表达式只允许 ``x``、白名单数学函数与常量，保证 ``eval`` 安全。
"""

from __future__ import annotations

import ast

import numpy as np


class ExpressionError(ValueError):
    """表达式解析或求值失败。"""


_ALLOWED_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
}
_ALLOWED_CONSTS = {"pi": np.pi, "e": np.e}


def parse_expression(expr: str):
    """把字符串解析为 ``f(x)`` 可调用对象。

    支持 ``^`` 代替 ``**``；出现未知符号、属性访问等一律拒绝。
    """
    if not expr or not expr.strip():
        raise ExpressionError("表达式不能为空")
    code = expr.strip().replace("^", "**")
    try:
        tree = ast.parse(code, mode="eval")
        compiled = compile(tree, "<expression>", "eval")
    except SyntaxError as exc:
        raise ExpressionError(f"表达式语法有误：{expr}") from exc

    allowed_nodes = (
        ast.Expression,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )
    only_basic = "表达式只允许数字、x、括号、算术运算和白名单数学函数"
    numeric_only = "表达式中只允许使用数字常量"
    function_only = "只能调用白名单数学函数"
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ExpressionError(only_basic)
        if isinstance(node, ast.Constant) and (
            not isinstance(node.value, (int, float)) or isinstance(node.value, bool)
        ):
            raise ExpressionError(numeric_only)
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_FUNCS
                or len(node.args) != 1
                or node.keywords
            ):
                raise ExpressionError(function_only)
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_FUNCS and node.id not in _ALLOWED_CONSTS and node.id != "x":
                raise ExpressionError(
                    f"表达式包含不支持的符号：{node.id}（可用：x, sin, cos, tan, exp, log, sqrt, abs, pi, e 等）"
                )

    def f(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            value = eval(compiled, {"__builtins__": {}}, {**_ALLOWED_FUNCS, **_ALLOWED_CONSTS, "x": x})
        return np.asarray(value, dtype=float)

    return f


def derivative_value(f, x0: float, h: float = 1e-6) -> float:
    """中心差分近似 ``f'(x0)``。"""
    return float((f(x0 + h) - f(x0 - h)) / (2 * h))


def simpson_integral(f, a: float, b: float, n: int = 1000) -> float:
    """辛普森法数值积分；``n`` 为子区间数，自动取偶。"""
    lo, hi = (a, b) if a <= b else (b, a)
    if hi - lo < 1e-15:
        return 0.0
    n = max(int(n), 2)
    if n % 2:
        n += 1
    xs = np.linspace(lo, hi, n + 1)
    ys = np.asarray(f(xs), dtype=float)
    h = (hi - lo) / n
    return float(h / 3 * (ys[0] + ys[-1] + 4 * ys[1:-1:2].sum() + 2 * ys[2:-2:2].sum()))


def riemann_midpoint(f, a: float, b: float, n: int):
    """中点黎曼和。

    返回 ``(和, 左边缘数组, 高度数组, 宽度数组)``，供绘图直接使用。
    """
    lo, hi = (a, b) if a <= b else (b, a)
    n = max(int(n), 1)
    edges = np.linspace(lo, hi, n + 1)
    widths = np.diff(edges)
    mids = (edges[:-1] + edges[1:]) / 2
    heights = np.asarray(f(mids), dtype=float)
    total = float(np.sum(widths * heights))
    return total, edges[:-1], heights, widths

