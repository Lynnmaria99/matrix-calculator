import numpy as np
import pytest

from matrix_calculator.linalg_topics import (
    analyze_quadratic,
    classify_quadratic,
    rank_and_nullspace,
    solve_linear,
)
from matrix_calculator.operations import MatrixOperationError


def test_solve_unique_square_system():
    a = np.array([[2.0, 0.0], [0.0, 3.0]])
    b = np.array([4.0, 9.0])

    status, info = solve_linear(a, b)

    assert status == "unique"
    np.testing.assert_allclose(info["x"], [2.0, 3.0])


def test_solve_infinite_system_reports_particular_and_null_solution():
    a = np.array([[1.0, 1.0], [2.0, 2.0]])
    b = np.array([1.0, 2.0])

    status, info = solve_linear(a, b)
    null = info["null"]

    assert status == "infinite"
    assert info["rA"] == 1
    assert null.shape == (2, 1)
    np.testing.assert_allclose(a @ null, np.zeros((2, 1)), atol=1e-10)
    np.testing.assert_allclose(a @ info["xp"], b, atol=1e-10)


def test_solve_inconsistent_system_reports_none_and_lstsq():
    a = np.array([[1.0, 1.0], [2.0, 2.0]])
    b = np.array([1.0, 3.0])

    status, info = solve_linear(a, b)

    assert status == "none"
    assert info["rA"] == 1
    assert info["rAb"] == 2
    assert "x_ls" in info


def test_rank_and_nullspace():
    a = np.array([[1.0, 1.0], [2.0, 2.0]])
    rank, null, _ = rank_and_nullspace(a)

    assert rank == 1
    assert null.shape == (2, 1)
    np.testing.assert_allclose(a @ null, np.zeros((2, 1)), atol=1e-10)


def test_quadratic_classification_cases():
    cases = [
        (np.array([1.0, 2.0]), "正定"),
        (np.array([-1.0, -2.0]), "负定"),
        (np.array([1.0, 2.0, 0.0]), "半正定"),
        (np.array([0.0, 0.0]), "半正定"),
        (np.array([-1.0, -2.0, 0.0]), "半负定"),
        (np.array([1.0, -1.0, 0.0]), "不定"),
    ]

    for eigenvalues, expected in cases:
        kind, pos, neg, zero = classify_quadratic(eigenvalues)
        assert expected in kind
        assert pos + neg + zero == eigenvalues.size


def test_analyze_quadratic_positive_definite():
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    result = analyze_quadratic(a)

    assert result["symmetric"] is True
    np.testing.assert_allclose(result["eigenvalues"], [1.0, 3.0])
    np.testing.assert_allclose(result["minors"], [2.0, 3.0])
    assert result["pos"] == 2
    assert result["neg"] == 0
    assert "正定" in result["kind"]


def test_analyze_quadratic_uses_symmetric_part():
    a = np.array([[2.0, 2.0], [0.0, 2.0]])
    result = analyze_quadratic(a)

    assert result["symmetric"] is False
    np.testing.assert_allclose(result["matrix"], [[2.0, 1.0], [1.0, 2.0]])
    assert "正定" in result["kind"]


def test_analyze_quadratic_requires_square_matrix():
    with pytest.raises(MatrixOperationError, match="方阵"):
        analyze_quadratic(np.zeros((2, 3)))