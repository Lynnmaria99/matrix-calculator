"""Tests for the computation adapter layer (shared interface for Web/Desktop/CLI).

This module tests all adapter functions with both success and error cases.
"""

import numpy as np
import pytest

from matrix_calculator.computation_adapter import (
    execute_binary_operation,
    execute_unary_operation,
    execute_special_operation,
    format_result_for_display,
    _compute_input_signature,
)
from matrix_calculator.computation_result import ComputationResult
from matrix_calculator.operations import MatrixOperationError


class TestComputeInputSignature:
    """Test input signature computation for caching."""

    def test_signature_is_tuple_with_three_elements(self):
        """Input signature should be (operation_code, shapes, data_hash)."""
        a = np.array([[1, 2], [3, 4]])
        sig = _compute_input_signature("add", a)
        
        assert isinstance(sig, tuple)
        assert len(sig) == 3
        assert sig[0] == "add"
        assert isinstance(sig[1], tuple)
        assert isinstance(sig[2], str)

    def test_same_matrix_produces_same_signature(self):
        """Identical matrices should produce identical signatures."""
        a = np.array([[1, 2], [3, 4]])
        sig1 = _compute_input_signature("add", a)
        sig2 = _compute_input_signature("add", a.copy())
        
        assert sig1 == sig2

    def test_different_values_produce_different_signature(self):
        """Different matrix values should produce different signatures."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 5]])
        
        sig_a = _compute_input_signature("add", a)
        sig_b = _compute_input_signature("add", b)
        
        assert sig_a != sig_b

    def test_different_shapes_produce_different_signature(self):
        """Different matrix shapes should produce different signatures."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2, 3], [4, 5, 6]])
        
        sig_a = _compute_input_signature("add", a)
        sig_b = _compute_input_signature("add", b)
        
        assert sig_a != sig_b

    def test_multiple_matrices_in_signature(self):
        """Signature should handle multiple matrices."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        sig = _compute_input_signature("multiply", a, b)
        
        assert sig[1] == ((2, 2), (2, 2))  # shapes
        assert sig[0] == "multiply"


class TestExecuteBinaryOperation:
    """Test binary operation execution."""

    def test_add_matrices_success(self):
        """Binary add should wrap result in ComputationResult."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("add", a, b)
        
        assert isinstance(result, ComputationResult)
        assert result.status == "success"
        np.testing.assert_array_equal(result.data, [[6, 8], [10, 12]])
        assert result.error_message is None
        assert result.operation_code == "add"
        assert len(result.input_signature) == 3

    def test_subtract_matrices_success(self):
        """Binary subtract should work correctly."""
        a = np.array([[5, 6], [7, 8]])
        b = np.array([[1, 2], [3, 4]])
        
        result = execute_binary_operation("subtract", a, b)
        
        assert result.status == "success"
        np.testing.assert_array_equal(result.data, [[4, 4], [4, 4]])

    def test_multiply_matrices_success(self):
        """Matrix multiplication should work."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("multiply", a, b)
        
        assert result.status == "success"
        np.testing.assert_array_equal(result.data, [[19, 22], [43, 50]])

    def test_hadamard_multiply_success(self):
        """Element-wise multiplication should work."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[2, 2], [2, 2]])
        
        result = execute_binary_operation("hadamard_multiply", a, b)
        
        assert result.status == "success"
        np.testing.assert_array_equal(result.data, [[2, 4], [6, 8]])

    def test_hadamard_divide_success(self):
        """Element-wise division should work."""
        a = np.array([[2.0, 4.0], [6.0, 8.0]])
        b = np.array([[2.0, 2.0], [2.0, 2.0]])
        
        result = execute_binary_operation("hadamard_divide", a, b)
        
        assert result.status == "success"
        np.testing.assert_array_almost_equal(result.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_dimension_mismatch_error(self):
        """Dimension mismatch should return error result."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        
        result = execute_binary_operation("add", a, b)
        
        assert result.status == "error"
        assert result.data is None
        assert result.error_message is not None
        assert "shape" in result.error_message.lower() or "尺寸" in result.error_message

    def test_multiply_dimension_mismatch_error(self):
        """Matrix multiplication dimension mismatch should be caught."""
        a = np.array([[1, 2], [3, 4]])  # 2x2
        b = np.array([[5, 6, 7]])  # 1x3
        
        result = execute_binary_operation("multiply", a, b)
        
        assert result.status == "error"
        assert result.data is None

    def test_invalid_operation_code_error(self):
        """Invalid operation code should return error."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("invalid_op", a, b)
        
        assert result.status == "error"
        assert result.data is None

    def test_non_numeric_input_handled(self):
        """Non-numeric inputs should be caught gracefully."""
        a = np.array([[1, 2], [3, 4]])
        
        # This will fail when trying to compute the operation
        # The adapter should catch and wrap the error
        result = execute_binary_operation("add", a, "not_a_matrix")
        
        assert result.status == "error"
        assert result.data is None

    def test_result_not_cached_initially(self):
        """Freshly computed results should not be marked as cached."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("add", a, b)
        
        assert result.cached is False
        assert result.stale is False

    def test_division_by_zero_error(self):
        """Division by zero should be caught and reported."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[0.0, 2.0], [2.0, 2.0]])
        
        result = execute_binary_operation("hadamard_divide", a, b)
        
        assert result.status == "error"
        assert result.data is None


class TestExecuteUnaryOperation:
    """Test unary operation execution."""

    def test_determinant_success(self):
        """Determinant computation should work."""
        a = np.array([[1, 2], [3, 4]])
        
        result = execute_unary_operation("determinant", a)
        
        assert result.status == "success"
        assert result.data.size == 1
        assert result.data.item() == pytest.approx(-2.0)

    def test_transpose_success(self):
        """Transpose should work."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        
        result = execute_unary_operation("transpose", a)
        
        assert result.status == "success"
        np.testing.assert_array_equal(result.data, [[1, 4], [2, 5], [3, 6]])

    def test_inverse_success(self):
        """Matrix inversion should work."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        
        result = execute_unary_operation("inverse", a)
        
        assert result.status == "success"
        # Verify A * A^-1 = I
        identity = a @ result.data
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-10)

    def test_determinant_on_non_square_error(self):
        """Determinant on non-square matrix should error."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        
        result = execute_unary_operation("determinant", a)
        
        assert result.status == "error"
        assert result.data is None

    def test_inverse_singular_matrix_error(self):
        """Inverse of singular matrix should error."""
        a = np.array([[1, 2], [2, 4]], dtype=float)
        
        result = execute_unary_operation("inverse", a)
        
        assert result.status == "error"
        assert result.data is None

    def test_eigendecompose_success(self):
        """Eigendecomposition should return tuple of eigenvalues and eigenvectors."""
        a = np.array([[1, 0], [0, 2]], dtype=float)
        
        result = execute_unary_operation("eigendecompose", a)
        
        assert result.status == "success"
        # eigendecompose returns (eigenvalues, eigenvectors)
        assert isinstance(result.data, tuple)
        assert len(result.data) == 2

    def test_singular_value_decompose_success(self):
        """SVD should work on any matrix."""
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        
        result = execute_unary_operation("singular_value_decompose", a)
        
        assert result.status == "success"
        # SVD returns (U, S, Vt)
        assert isinstance(result.data, tuple)
        assert len(result.data) == 3

    def test_invalid_operation_code_error(self):
        """Invalid operation code should return error."""
        a = np.array([[1, 2], [3, 4]])
        
        result = execute_unary_operation("nonexistent_op", a)
        
        assert result.status == "error"


class TestExecuteSpecialOperation:
    """Test special operations (solve_linear, analyze_quadratic)."""

    def test_solve_linear_unique_solution(self):
        """Solve Ax=b with unique solution."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([5, 11], dtype=float)
        
        result = execute_special_operation("solve_linear", a, b)
        
        assert result.status == "success"
        assert isinstance(result.data, dict)
        assert "rA" in result.data
        assert "rAb" in result.data
        assert "n" in result.data

    def test_solve_linear_no_solution(self):
        """Solve Ax=b with no solution (inconsistent system)."""
        # Inconsistent system: x + y = 1, x + y = 2
        a = np.array([[1, 1], [1, 1]], dtype=float)
        b = np.array([1, 2], dtype=float)
        
        result = execute_special_operation("solve_linear", a, b)
        
        assert result.status == "success"
        assert isinstance(result.data, dict)

    def test_solve_linear_infinite_solutions(self):
        """Solve Ax=b with infinite solutions."""
        # Underdetermined: x + y = 1 (only one equation, two unknowns)
        a = np.array([[1, 1]], dtype=float)
        b = np.array([1], dtype=float)
        
        result = execute_special_operation("solve_linear", a, b)
        
        assert result.status == "success"
        assert isinstance(result.data, dict)

    def test_analyze_quadratic_success(self):
        """Analyze quadratic form should return analysis dict."""
        # Simple 2D positive definite matrix
        a = np.array([[2, -1], [-1, 2]], dtype=float)
        
        result = execute_special_operation("analyze_quadratic", a)
        
        assert result.status == "success"
        assert isinstance(result.data, dict)
        assert "eigenvalues" in result.data
        assert "Q" in result.data
        assert "kind" in result.data
        assert "pos" in result.data
        assert "neg" in result.data
        assert "zero" in result.data

    def test_analyze_quadratic_non_square_error(self):
        """Analyze quadratic on non-square matrix should error."""
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        result = execute_special_operation("analyze_quadratic", a)
        
        assert result.status == "error"

    def test_invalid_special_operation_error(self):
        """Invalid special operation should error."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        
        result = execute_special_operation("invalid_special", a)
        
        assert result.status == "error"


class TestFormatResultForDisplay:
    """Test result formatting for display."""

    def test_format_successful_binary_operation(self):
        """Format successful add result."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("add", a, b)
        display_str = format_result_for_display(result, format_style="text")
        
        assert isinstance(display_str, str)
        assert len(display_str) > 0
        # Should contain matrix representation
        assert "6" in display_str
        assert "8" in display_str

    def test_format_error_result(self):
        """Format error result should show error message."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7]])
        
        result = execute_binary_operation("add", a, b)
        display_str = format_result_for_display(result)
        
        assert isinstance(display_str, str)
        assert "Error" in display_str or "error" in display_str.lower()

    def test_format_determinant_result(self):
        """Format scalar result (determinant)."""
        a = np.array([[1, 2], [3, 4]])
        
        result = execute_unary_operation("determinant", a)
        display_str = format_result_for_display(result)
        
        assert isinstance(display_str, str)
        assert "-2" in display_str

    def test_format_transpose_result(self):
        """Format matrix result (transpose)."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        
        result = execute_unary_operation("transpose", a)
        display_str = format_result_for_display(result, format_style="text")
        
        assert isinstance(display_str, str)
        assert len(display_str) > 0

    def test_format_solve_linear_dict_result(self):
        """Format dict result (solve_linear)."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([5, 11], dtype=float)
        
        result = execute_special_operation("solve_linear", a, b)
        display_str = format_result_for_display(result)
        
        assert isinstance(display_str, str)
        assert "solve_linear" in display_str

    def test_format_latex_style(self):
        """Format result with LaTeX style."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("add", a, b)
        display_str = format_result_for_display(result, format_style="latex")
        
        assert isinstance(display_str, str)
        assert "array" in display_str or "6" in display_str

    def test_format_plain_style(self):
        """Format result with plain style."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("add", a, b)
        display_str = format_result_for_display(result, format_style="plain")
        
        assert isinstance(display_str, str)
        # Plain style uses space-separated values
        assert "6" in display_str

    def test_format_invalid_style_raises_error(self):
        """Invalid format style should raise ValueError."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result = execute_binary_operation("add", a, b)
        
        with pytest.raises(ValueError, match="Unknown format style"):
            format_result_for_display(result, format_style="invalid")

    def test_format_non_computation_result_raises_error(self):
        """Non-ComputationResult should raise TypeError."""
        with pytest.raises(TypeError):
            format_result_for_display("not a result")

    def test_format_eigendecompose_tuple_result(self):
        """Format tuple result (eigendecompose)."""
        a = np.array([[1, 0], [0, 2]], dtype=float)
        
        result = execute_unary_operation("eigendecompose", a)
        display_str = format_result_for_display(result)
        
        assert isinstance(display_str, str)
        assert "eigendecompose" in display_str


class TestAdapterIntegration:
    """Integration tests combining multiple adapter functions."""

    def test_workflow_compute_then_format(self):
        """Test typical workflow: compute, then format."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        # Compute
        result = execute_binary_operation("add", a, b)
        
        # Format
        display_str = format_result_for_display(result)
        
        assert result.status == "success"
        assert isinstance(display_str, str)
        assert len(display_str) > 0

    def test_cache_validation_workflow(self):
        """Test that input signatures match across calls."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        result1 = execute_binary_operation("add", a, b)
        result2 = execute_binary_operation("add", a, b)
        
        # Signatures should be identical for identical inputs
        assert result1.input_signature == result2.input_signature

    def test_error_handling_chain(self):
        """Test error handling through the entire chain."""
        a = np.array([[1, 2]])
        b = np.array([[3, 4, 5]])
        
        # Compute (will fail)
        result = execute_binary_operation("add", a, b)
        
        # Format (should show error)
        display_str = format_result_for_display(result)
        
        assert result.status == "error"
        assert "Error" in display_str or "error" in display_str.lower()

    def test_array_like_inputs(self):
        """Adapter should accept array-like inputs (lists, tuples)."""
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        
        result = execute_binary_operation("add", a, b)
        
        assert result.status == "success"
        np.testing.assert_array_equal(result.data, [[6, 8], [10, 12]])
