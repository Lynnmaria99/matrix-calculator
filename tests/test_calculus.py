"""微积分核心测试。"""

from __future__ import annotations

import math

import pytest

from matrix_calculator.calculus import (
    ExpressionError,
    derivative_value,
    parse_expression,
    riemann_midpoint,
    simpson_integral,
)


class TestParseExpression:
    def test_basic(self):
        f = parse_expression("x**2")
        assert f(3.0) == 9.0

    def test_caret_power(self):
        f = parse_expression("x^2")
        assert f(3.0) == 9.0

    def test_functions_and_consts(self):
        f = parse_expression("sin(pi * x)")
        assert abs(f(0.5) - 1.0) < 1e-12

    def test_empty_rejected(self):
        with pytest.raises(ExpressionError):
            parse_expression("   ")

    def test_unknown_symbol_rejected(self):
        with pytest.raises(ExpressionError, match="不支持的符号"):
            parse_expression("y + 1")

    def test_attribute_access_rejected(self):
        with pytest.raises(ExpressionError):
            parse_expression("x.__class__")

    def test_lambda_rejected(self):
        with pytest.raises(ExpressionError):
            parse_expression("lambda x: x")

    def test_non_numeric_constant_rejected(self):
        with pytest.raises(ExpressionError):
            parse_expression("1 + 2j")

    def test_bad_call_rejected(self):
        with pytest.raises(ExpressionError):
            parse_expression("pi()")
        with pytest.raises(ExpressionError):
            parse_expression("sin()")

    def test_vectorized(self):
        f = parse_expression("x^2")
        result = f([1.0, 2.0, 3.0])
        assert result[2] == 9.0


class TestDerivative:
    def test_quadratic(self):
        f = parse_expression("x^2")
        assert derivative_value(f, 2.0) == pytest.approx(4.0, abs=1e-4)

    def test_sin(self):
        f = parse_expression("sin(x)")
        assert derivative_value(f, 0.0) == pytest.approx(1.0, abs=1e-5)


class TestIntegral:
    def test_simpson_quadratic(self):
        f = parse_expression("x^2")
        assert simpson_integral(f, 0.0, 1.0, n=200) == pytest.approx(1 / 3, abs=1e-10)

    def test_simpson_sin(self):
        f = parse_expression("sin(x)")
        assert simpson_integral(f, 0.0, math.pi, n=200) == pytest.approx(2.0, abs=1e-8)

    def test_riemann_close_to_simpson(self):
        f = parse_expression("x^2")
        total, *_ = riemann_midpoint(f, 0.0, 1.0, n=50)
        assert total == pytest.approx(1 / 3, abs=1e-3)

    def test_reversed_bounds(self):
        f = parse_expression("x^2")
        assert simpson_integral(f, 1.0, 0.0, n=100) == pytest.approx(1 / 3, abs=1e-10)
