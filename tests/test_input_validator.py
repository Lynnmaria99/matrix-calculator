"""Tests for input_validator module.

Tests cover all parsing formats (decimals, fractions, scientific notation), dimension
clamping with various configurations, and input signature generation for caching.
"""

import numpy as np
import pytest

from matrix_calculator.input_validator import (
    DimensionConfig,
    ParseResult,
    clamp_dimensions,
    get_input_signature,
    parse_cell,
    parse_matrix_cells,
    validate_dimensions,
)


# ==================== parse_cell Tests ====================


class TestParseCell:
    """Test suite for parse_cell() function."""

    # --- Integers ---

    def test_parse_cell_positive_integer(self) -> None:
        result = parse_cell("5")
        assert result.ok is True
        assert result.value == 5.0
        assert result.message is None

    def test_parse_cell_negative_integer(self) -> None:
        result = parse_cell("-3")
        assert result.ok is True
        assert result.value == -3.0

    def test_parse_cell_zero(self) -> None:
        result = parse_cell("0")
        assert result.ok is True
        assert result.value == 0.0

    # --- Decimals ---

    def test_parse_cell_positive_decimal(self) -> None:
        result = parse_cell("2.5")
        assert result.ok is True
        assert abs(result.value - 2.5) < 1e-10

    def test_parse_cell_negative_decimal(self) -> None:
        result = parse_cell("-0.001")
        assert result.ok is True
        assert abs(result.value - (-0.001)) < 1e-10

    def test_parse_cell_leading_dot(self) -> None:
        result = parse_cell(".5")
        assert result.ok is True
        assert abs(result.value - 0.5) < 1e-10

    def test_parse_cell_trailing_dot(self) -> None:
        result = parse_cell("5.")
        assert result.ok is True
        assert result.value == 5.0

    # --- Scientific Notation ---

    def test_parse_cell_scientific_positive_exponent(self) -> None:
        result = parse_cell("1e+3")
        assert result.ok is True
        assert result.value == 1000.0

    def test_parse_cell_scientific_negative_exponent(self) -> None:
        result = parse_cell("1e-3")
        assert result.ok is True
        assert abs(result.value - 0.001) < 1e-10

    def test_parse_cell_scientific_no_sign(self) -> None:
        result = parse_cell("2e5")
        assert result.ok is True
        assert result.value == 200000.0

    def test_parse_cell_scientific_decimal_mantissa(self) -> None:
        result = parse_cell("2.5e+10")
        assert result.ok is True
        assert result.value == 2.5e10

    def test_parse_cell_scientific_negative_mantissa(self) -> None:
        result = parse_cell("-1.5e-3")
        assert result.ok is True
        assert abs(result.value - (-0.0015)) < 1e-10

    def test_parse_cell_scientific_extreme_exponent_max(self) -> None:
        result = parse_cell("1e+99")
        assert result.ok is True
        assert result.value == 1e99

    def test_parse_cell_scientific_extreme_exponent_min(self) -> None:
        result = parse_cell("1e-99")
        assert result.ok is True
        assert result.value == 1e-99

    def test_parse_cell_scientific_exponent_out_of_range_high(self) -> None:
        result = parse_cell("1e+100")
        assert result.ok is False
        assert result.message == "exponent_out_of_range"

    def test_parse_cell_scientific_exponent_out_of_range_low(self) -> None:
        result = parse_cell("1e-100")
        assert result.ok is False
        assert result.message == "exponent_out_of_range"

    # --- Fractions ---

    def test_parse_cell_simple_fraction(self) -> None:
        result = parse_cell("1/2")
        assert result.ok is True
        assert abs(result.value - 0.5) < 1e-10

    def test_parse_cell_fraction_negative_numerator(self) -> None:
        result = parse_cell("-3/4")
        assert result.ok is True
        assert abs(result.value - (-0.75)) < 1e-10

    def test_parse_cell_fraction_negative_denominator(self) -> None:
        result = parse_cell("1/-2")
        assert result.ok is True
        assert abs(result.value - (-0.5)) < 1e-10

    def test_parse_cell_fraction_both_negative(self) -> None:
        result = parse_cell("-1/-2")
        assert result.ok is True
        assert abs(result.value - 0.5) < 1e-10

    def test_parse_cell_fraction_improper(self) -> None:
        result = parse_cell("7/2")
        assert result.ok is True
        assert abs(result.value - 3.5) < 1e-10

    def test_parse_cell_fraction_large_denominator_valid(self) -> None:
        result = parse_cell("1/999999")
        assert result.ok is True
        assert abs(result.value - (1.0 / 999999)) < 1e-10

    def test_parse_cell_fraction_denominator_out_of_range(self) -> None:
        result = parse_cell("1/1000000")
        assert result.ok is False
        assert result.message == "denominator_out_of_range"

    def test_parse_cell_fraction_division_by_zero(self) -> None:
        result = parse_cell("1/0")
        assert result.ok is False
        assert result.message == "division_by_zero"

    def test_parse_cell_fraction_zero_denominator_negative(self) -> None:
        result = parse_cell("1/-0")
        assert result.ok is False
        assert result.message == "division_by_zero"

    # --- Multiple Operators (Error Cases) ---

    def test_parse_cell_multiple_division_operators(self) -> None:
        result = parse_cell("1/2/3")
        assert result.ok is False
        assert result.message == "multiple_division_operators"

    # --- Empty / Whitespace ---

    def test_parse_cell_empty_string(self) -> None:
        result = parse_cell("")
        assert result.ok is False
        assert result.message == "empty_cell"

    def test_parse_cell_whitespace_only(self) -> None:
        result = parse_cell("   ")
        assert result.ok is False
        assert result.message == "empty_cell"

    def test_parse_cell_whitespace_around_content(self) -> None:
        result = parse_cell("  3.14  ")
        assert result.ok is True
        assert abs(result.value - 3.14) < 1e-10

    # --- Invalid Formats ---

    def test_parse_cell_invalid_letters(self) -> None:
        result = parse_cell("abc")
        assert result.ok is False
        assert result.message == "invalid_number_format"

    def test_parse_cell_invalid_operator_in_middle(self) -> None:
        result = parse_cell("1+2")
        assert result.ok is False
        assert result.message == "invalid_number_format"

    def test_parse_cell_multiple_dots_in_decimal(self) -> None:
        result = parse_cell("1.2.3")
        assert result.ok is False
        assert result.message == "invalid_number_format"

    def test_parse_cell_fraction_with_non_numeric_numerator(self) -> None:
        result = parse_cell("a/2")
        assert result.ok is False
        assert result.message == "invalid_number_format"

    def test_parse_cell_fraction_with_non_numeric_denominator(self) -> None:
        result = parse_cell("1/b")
        assert result.ok is False
        assert result.message == "invalid_number_format"


# ==================== clamp_dimensions Tests ====================


class TestClampDimensions:
    """Test suite for clamp_dimensions() function."""

    def test_clamp_dimensions_within_bounds(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        result = clamp_dimensions(3, 4, config)
        assert result.ok is True
        assert result.value == (3, 4)

    def test_clamp_dimensions_below_min(self) -> None:
        config = DimensionConfig(min_rows=2, max_rows=10, min_cols=2, max_cols=10)
        result = clamp_dimensions(1, 1, config)
        assert result.ok is True
        assert result.value == (2, 2)

    def test_clamp_dimensions_above_max(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=5, min_cols=1, max_cols=5)
        result = clamp_dimensions(10, 10, config)
        assert result.ok is True
        assert result.value == (5, 5)

    def test_clamp_dimensions_mixed_out_of_bounds(self) -> None:
        config = DimensionConfig(min_rows=2, max_rows=10, min_cols=1, max_cols=5)
        result = clamp_dimensions(1, 15, config)
        assert result.ok is True
        assert result.value == (2, 5)

    def test_clamp_dimensions_square_only_forces_equal(self) -> None:
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square_only=True
        )
        result = clamp_dimensions(3, 5, config)
        assert result.ok is True
        assert result.value == (3, 3)

    def test_clamp_dimensions_square_only_with_out_of_bounds(self) -> None:
        config = DimensionConfig(
            min_rows=2, max_rows=5, min_cols=2, max_cols=5, square_only=True
        )
        result = clamp_dimensions(1, 10, config)
        assert result.ok is True
        assert result.value == (2, 2)

    def test_clamp_dimensions_truncates_not_rounds(self) -> None:
        # Verify truncation by clamping 7 to max of 5 should give 5, not 6
        config = DimensionConfig(min_rows=1, max_rows=5, min_cols=1, max_cols=5)
        result = clamp_dimensions(7, 7, config)
        assert result.value == (5, 5)

    def test_clamp_dimensions_zero_rows_clamps_to_min(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        result = clamp_dimensions(0, 5, config)
        assert result.ok is True
        assert result.value == (1, 5)

    def test_clamp_dimensions_negative_rows_clamps_to_min(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        result = clamp_dimensions(-5, 3, config)
        assert result.ok is True
        assert result.value == (1, 3)


# ==================== validate_dimensions Tests ====================


class TestValidateDimensions:
    """Test suite for validate_dimensions() function."""

    def test_validate_dimensions_valid(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        result = validate_dimensions(5, 5, config)
        assert result.ok is True
        assert result.value == (5, 5)

    def test_validate_dimensions_out_of_range_rows(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=5, min_cols=1, max_cols=10)
        result = validate_dimensions(10, 3, config)
        assert result.ok is False
        assert "out of range" in result.message.lower() or "clamped" in result.message.lower()

    def test_validate_dimensions_out_of_range_cols(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=5)
        result = validate_dimensions(3, 10, config)
        assert result.ok is False

    def test_validate_dimensions_square_only_not_square(self) -> None:
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square_only=True
        )
        result = validate_dimensions(3, 4, config)
        assert result.ok is False


# ==================== parse_matrix_cells Tests ====================


class TestParseMatrixCells:
    """Test suite for parse_matrix_cells() function."""

    def test_parse_matrix_cells_valid_integers(self) -> None:
        cells = [["1", "2"], ["3", "4"]]
        result = parse_matrix_cells(cells)
        assert result.ok is True
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result.value, expected)

    def test_parse_matrix_cells_mixed_formats(self) -> None:
        cells = [["1", "0.5"], ["-1/2", "1e-1"]]
        result = parse_matrix_cells(cells)
        assert result.ok is True
        expected = np.array([[1.0, 0.5], [-0.5, 0.1]])
        np.testing.assert_array_almost_equal(result.value, expected)

    def test_parse_matrix_cells_invalid_cell(self) -> None:
        cells = [["1", "2"], ["invalid", "4"]]
        result = parse_matrix_cells(cells)
        assert result.ok is False
        assert "Cell" in result.message and "[2, 1]" in result.message

    def test_parse_matrix_cells_empty_cell_error(self) -> None:
        cells = [["1", ""], ["3", "4"]]
        result = parse_matrix_cells(cells)
        assert result.ok is False
        assert "Cell [1, 2]" in result.message

    def test_parse_matrix_cells_with_config_valid(self) -> None:
        cells = [["1", "2"], ["3", "4"]]
        config = DimensionConfig(min_rows=2, max_rows=4, min_cols=2, max_cols=4)
        result = parse_matrix_cells(cells, config)
        assert result.ok is True

    def test_parse_matrix_cells_with_config_out_of_bounds(self) -> None:
        cells = [["1", "2", "3"], ["4", "5", "6"]]
        config = DimensionConfig(min_rows=1, max_rows=2, min_cols=1, max_cols=2)
        result = parse_matrix_cells(cells, config)
        assert result.ok is False

    def test_parse_matrix_cells_empty_input(self) -> None:
        result = parse_matrix_cells([])
        assert result.ok is False
        assert "empty" in result.message.lower()

    def test_parse_matrix_cells_rectangular(self) -> None:
        cells = [["1", "2", "3"], ["4", "5", "6"]]
        result = parse_matrix_cells(cells)
        assert result.ok is True
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(result.value, expected)

    def test_parse_matrix_cells_single_element(self) -> None:
        cells = [["42"]]
        result = parse_matrix_cells(cells)
        assert result.ok is True
        np.testing.assert_array_equal(result.value, np.array([[42.0]]))


# ==================== DimensionConfig Tests ====================


class TestDimensionConfig:
    """Test suite for DimensionConfig validation."""

    def test_dimension_config_valid(self) -> None:
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        assert config.min_rows == 1
        assert config.max_rows == 10

    def test_dimension_config_invalid_min_rows_zero(self) -> None:
        with pytest.raises(ValueError, match="min_rows"):
            DimensionConfig(min_rows=0)

    def test_dimension_config_invalid_min_rows_negative(self) -> None:
        with pytest.raises(ValueError, match="min_rows"):
            DimensionConfig(min_rows=-1)

    def test_dimension_config_invalid_max_less_than_min_rows(self) -> None:
        with pytest.raises(ValueError, match="max_rows"):
            DimensionConfig(min_rows=5, max_rows=3)

    def test_dimension_config_invalid_min_cols_zero(self) -> None:
        with pytest.raises(ValueError, match="min_cols"):
            DimensionConfig(min_cols=0)

    def test_dimension_config_invalid_max_less_than_min_cols(self) -> None:
        with pytest.raises(ValueError, match="max_cols"):
            DimensionConfig(min_cols=5, max_cols=3)

    def test_dimension_config_square_only_true(self) -> None:
        config = DimensionConfig(square_only=True)
        assert config.square_only is True

    def test_dimension_config_resizable_false(self) -> None:
        config = DimensionConfig(resizable=False)
        assert config.resizable is False


# ==================== get_input_signature Tests ====================


class TestGetInputSignature:
    """Test suite for get_input_signature() function."""

    def test_get_input_signature_single_matrix(self) -> None:
        matrices = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        sig = get_input_signature(matrices, "test_op")
        assert sig[0] == "test_op"
        assert len(sig[1]) == 1
        assert sig[1][0][0] == (2, 2)  # shape

    def test_get_input_signature_two_matrices(self) -> None:
        matrices = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0], [4.0]]),
        ]
        sig = get_input_signature(matrices, "multiply")
        assert sig[0] == "multiply"
        assert len(sig[1]) == 2
        assert sig[1][0][0] == (1, 2)
        assert sig[1][1][0] == (2, 1)

    def test_get_input_signature_identical_matrices_same_signature(self) -> None:
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        sig1 = get_input_signature([mat], "op1")
        sig2 = get_input_signature([mat.copy()], "op1")
        # Same shape and content should produce same hash
        assert sig1[1][0][0] == sig2[1][0][0]  # Same shape
        assert sig1[1][0][1] == sig2[1][0][1]  # Same content hash

    def test_get_input_signature_different_matrices_different_signature(self) -> None:
        mat1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        mat2 = np.array([[1.0, 2.0], [3.0, 5.0]])  # Different value
        sig1 = get_input_signature([mat1], "op")
        sig2 = get_input_signature([mat2], "op")
        # Different content should produce different hash
        assert sig1[1][0][1] != sig2[1][0][1]

    def test_get_input_signature_different_shapes_different_signature(self) -> None:
        mat1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        mat2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sig1 = get_input_signature([mat1], "op")
        sig2 = get_input_signature([mat2], "op")
        # Different shapes should be obvious
        assert sig1[1][0][0] != sig2[1][0][0]

    def test_get_input_signature_different_operation_codes(self) -> None:
        matrices = [np.array([[1.0, 2.0]])]
        sig1 = get_input_signature(matrices, "add")
        sig2 = get_input_signature(matrices, "multiply")
        assert sig1[0] != sig2[0]

    def test_get_input_signature_empty_matrix(self) -> None:
        matrices = [np.array([])]
        sig = get_input_signature(matrices, "op")
        assert sig[0] == "op"
        # Empty array has shape (0,), verify it's handled
        assert len(sig[1]) == 1

    def test_get_input_signature_complex_matrix(self) -> None:
        # Large matrix should still compute signature
        large_mat = np.random.randn(10, 10)
        sig = get_input_signature([large_mat], "large_op")
        assert sig[0] == "large_op"
        assert sig[1][0][0] == (10, 10)

    def test_get_input_signature_hashable(self) -> None:
        # Signature should be hashable (for use as dict key)
        matrices = [np.array([[1.0, 2.0]])]
        sig = get_input_signature(matrices, "op")
        # Try to use as dict key
        d = {sig: "result"}
        assert d[sig] == "result"


# ==================== Integration Tests ====================


class TestInputValidatorIntegration:
    """Integration tests combining multiple functions."""

    def test_workflow_parse_validate_sign_matrix(self) -> None:
        # Simulate: user enters cells → parse → validate dims → create signature
        cell_strings = [["1", "2", "3"], ["4", "5", "6"]]

        # Parse cells
        parse_result = parse_matrix_cells(cell_strings)
        assert parse_result.ok is True
        matrix = parse_result.value

        # Validate dimensions
        config = DimensionConfig(min_rows=2, max_rows=4, min_cols=2, max_cols=5)
        dim_result = validate_dimensions(matrix.shape[0], matrix.shape[1], config)
        assert dim_result.ok is True

        # Get signature
        sig = get_input_signature([matrix], "add")
        assert sig[0] == "add"
        assert sig[1][0][0] == (2, 3)

    def test_workflow_clamp_and_parse_after_dimension_change(self) -> None:
        # User changes dims to out-of-spec, then tries to parse
        config = DimensionConfig(min_rows=1, max_rows=5, min_cols=1, max_cols=5)

        # User wants 10x10 matrix
        clamp_result = clamp_dimensions(10, 10, config)
        assert clamp_result.ok is True
        rows, cols = clamp_result.value
        assert (rows, cols) == (5, 5)

        # Create appropriately-sized cell input
        cell_strings = [[str(i * cols + j) for j in range(cols)] for i in range(rows)]
        parse_result = parse_matrix_cells(cell_strings, config)
        assert parse_result.ok is True
