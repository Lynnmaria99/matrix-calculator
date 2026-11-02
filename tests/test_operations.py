import numpy as np
import pytest

from matrix_calculator.operations import (
    MatrixOperationError,
    add_matrices,
    determinant,
    hadamard_divide,
    hadamard_multiply,
    invert_matrix,
    matrix_multiply,
    subtract_matrices,
)


def test_add_and_subtract_matrices():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])

    np.testing.assert_array_equal(add_matrices(a, b), [[6, 8], [10, 12]])
    np.testing.assert_array_equal(subtract_matrices(b, a), [[4, 4], [4, 4]])


def test_shape_mismatch_raises_clear_error():
    a = np.zeros((2, 2))
    b = np.zeros((2, 3))

    with pytest.raises(MatrixOperationError, match="shape 不一致"):
        add_matrices(a, b)


def test_matrix_multiplication_checks_inner_dimension():
    a = np.zeros((2, 3))
    b = np.zeros((2, 2))

    with pytest.raises(MatrixOperationError, match="列数"):
        matrix_multiply(a, b)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_array_equal(matrix_multiply(x, y), [[19, 22], [43, 50]])


def test_hadamard_operations():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 2.0], [2.0, 2.0]])

    np.testing.assert_array_equal(hadamard_multiply(a, b), [[2, 4], [6, 8]])
    np.testing.assert_array_equal(hadamard_divide(a, b), [[0.5, 1.0], [1.5, 2.0]])

    with pytest.raises(MatrixOperationError, match="含有 0"):
        hadamard_divide(a, np.zeros_like(a))


def test_determinant_and_inverse():
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    invertible = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(MatrixOperationError, match="方阵"):
        determinant(np.zeros((2, 3)))

    assert determinant(invertible) == pytest.approx(-2.0)

    with pytest.raises(MatrixOperationError, match="不可逆"):
        invert_matrix(singular)

    identity = invertible @ invert_matrix(invertible)
    np.testing.assert_allclose(identity, np.eye(2), atol=1e-10)