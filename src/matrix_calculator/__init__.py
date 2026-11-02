"""矩阵计算器：可复用的计算核心与命令行界面。"""

from .constants import EPS
from .operations import (
    MatrixOperationError,
    MatCal,
    MatCom,
    add_matrices,
    subtract_matrices,
    hadamard_multiply,
    matrix_multiply,
    hadamard_divide,
    determinant,
    invert_matrix,
    eigendecompose,
    singular_value_decompose,
    transpose,
)
from .linalg_topics import rank_and_nullspace, solve_linear, classify_quadratic
from .formatting import format_matrix, print_matrix, show_two_results, signed_polynomial

__all__ = [
    "EPS",
    "MatrixOperationError",
    "MatCal",
    "MatCom",
    "add_matrices",
    "subtract_matrices",
    "hadamard_multiply",
    "matrix_multiply",
    "hadamard_divide",
    "determinant",
    "invert_matrix",
    "eigendecompose",
    "singular_value_decompose",
    "transpose",
    "rank_and_nullspace",
    "solve_linear",
    "classify_quadratic",
    "format_matrix",
    "print_matrix",
    "show_two_results",
    "signed_polynomial",
]
