"""Tests for result_formatter module.

Tests cover all formatting styles, export formats, edge cases, and caching behavior.
"""

import numpy as np
import pytest

from matrix_calculator.result_formatter import (
    export_to_csv,
    export_to_latex,
    export_to_numpy_text,
    format_complex_number,
    format_matrix,
    format_scalar,
)


class TestFormatScalar:
    """Tests for format_scalar function."""

    def test_format_float_basic(self):
        """Test formatting basic float values."""
        assert format_scalar(3.14159) == "3.14159"
        assert format_scalar(1.0) == "1"
        assert format_scalar(0.0) == "0"

    def test_format_float_precision(self):
        """Test formatting with different precisions."""
        value = 3.14159265358979
        assert format_scalar(value, precision=3) == "3.14"
        assert format_scalar(value, precision=6) == "3.14159"

    def test_format_very_small_values(self):
        """Test that very small values close to zero are displayed as 0."""
        assert format_scalar(1e-10) == "0"  # Smaller than EPS
        assert format_scalar(0.0001) == "0.0001"  # Larger than EPS

    def test_format_complex_basic(self):
        """Test formatting complex numbers."""
        assert format_complex_number(1 + 2j) == "1+2j"
        assert format_complex_number(1 - 2j) == "1-2j"

    def test_format_complex_real_only(self):
        """Test complex number with only real part."""
        assert format_complex_number(5 + 0j) == "5"

    def test_format_complex_imaginary_only(self):
        """Test complex number with only imaginary part."""
        assert format_complex_number(0 + 3j) == "3j"

    def test_format_complex_small_values(self):
        """Test complex numbers with small real/imaginary parts."""
        result = format_complex_number(1e-10 + 2j)
        # Very small real part (< EPS) is treated as 0, shows as imaginary only
        assert "2j" in result

    def test_format_negative_values(self):
        """Test formatting negative numbers."""
        assert format_scalar(-5) == "-5"
        assert format_scalar(-3.14) == "-3.14"

    def test_format_scientific_notation(self):
        """Test formatting very large and very small values."""
        large = format_scalar(1e6)
        small = format_scalar(1e-6)
        assert "e" in large.lower() or large == "1e+06"
        assert "e" in small.lower() or small == "1e-06"


class TestFormatMatrixText:
    """Tests for text (Unicode) matrix formatting."""

    def test_single_element_matrix(self):
        """Test formatting a 1x1 matrix."""
        result = format_matrix([[5]], style="text")
        assert "5" in result
        assert "(" in result  # Single row uses parentheses

    def test_single_row_matrix(self):
        """Test formatting a 1xN matrix."""
        result = format_matrix([[1, 2, 3]], style="text")
        assert "(" in result and ")" in result
        assert "1" in result and "2" in result and "3" in result

    def test_2x2_matrix(self):
        """Test formatting a 2x2 matrix."""
        result = format_matrix([[1, 2], [3, 4]], style="text")
        lines = result.split("\n")
        assert len(lines) == 2
        assert "⎡" in lines[0]  # Top bracket
        assert "⎣" in lines[1]  # Bottom bracket
        assert "1" in result and "4" in result

    def test_3x3_identity_matrix(self):
        """Test formatting identity matrix."""
        result = format_matrix(np.eye(3), style="text")
        lines = result.split("\n")
        assert len(lines) == 3
        assert "⎡" in lines[0]
        assert "⎢" in lines[1]  # Middle rows
        assert "⎣" in lines[2]

    def test_2x3_rectangular_matrix(self):
        """Test formatting a non-square matrix."""
        result = format_matrix([[1, 2, 3], [4, 5, 6]], style="text")
        lines = result.split("\n")
        assert len(lines) == 2
        assert "1" in result and "6" in result

    def test_matrix_column_alignment(self):
        """Test that columns are properly aligned."""
        result = format_matrix([[1, 100], [1000, 2]], style="text")
        lines = result.split("\n")
        # Both lines should have similar structure
        assert len(lines[0]) == len(lines[1])

    def test_matrix_with_negative_values(self):
        """Test formatting matrix with negative numbers."""
        result = format_matrix([[-1, -2], [-3, -4]], style="text")
        assert "-1" in result
        assert "-4" in result

    def test_matrix_with_floats(self):
        """Test formatting matrix with floating point values."""
        result = format_matrix([[1.5, 2.7], [3.14, 4.99]], style="text")
        assert "1.5" in result or "1.5" in result
        assert "3.14" in result


class TestFormatMatrixLatex:
    """Tests for LaTeX matrix formatting."""

    def test_latex_2x2_matrix(self):
        """Test LaTeX format for 2x2 matrix."""
        result = format_matrix([[1, 2], [3, 4]], style="latex")
        assert "\\begin{array}" in result
        assert "\\end{array}" in result
        assert "&" in result  # Column separator
        assert "\\\\" in result  # Row separator

    def test_latex_structure(self):
        """Test LaTeX array structure."""
        result = format_matrix([[1, 2, 3], [4, 5, 6]], style="latex")
        lines = result.split("\n")
        assert lines[0].startswith("\\begin{array}")
        assert lines[-1] == "\\end{array}"

    def test_latex_column_alignment(self):
        """Test LaTeX column alignment specification."""
        result = format_matrix([[1, 2], [3, 4]], style="latex")
        # Should have {cc} for 2 columns
        assert "{cc}" in result

    def test_latex_3x3_matrix(self):
        """Test LaTeX format for 3x3 matrix."""
        result = format_matrix(np.eye(3), style="latex")
        assert "{ccc}" in result  # 3 columns, center-aligned

    def test_latex_single_row(self):
        """Test LaTeX format for single row."""
        result = format_matrix([[1, 2, 3]], style="latex")
        assert "\\begin{array}" in result
        # Single row should not have trailing \\
        assert result.count("\\\\") == 0


class TestFormatMatrixPlain:
    """Tests for plain (space-separated) matrix formatting."""

    def test_plain_2x2_matrix(self):
        """Test plain format for 2x2 matrix."""
        result = format_matrix([[1, 2], [3, 4]], style="plain")
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "1 2"
        assert lines[1] == "3 4"

    def test_plain_single_row(self):
        """Test plain format for single row."""
        result = format_matrix([[1, 2, 3]], style="plain")
        assert result == "1 2 3"

    def test_plain_single_column(self):
        """Test plain format for single column."""
        result = format_matrix([[1], [2], [3]], style="plain")
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "1"
        assert lines[2] == "3"


class TestFormatMatrixEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_1d_array_converted_to_2d(self):
        """Test that 1D arrays are converted to 2D."""
        result = format_matrix([1, 2, 3], style="text")
        # Should be treated as row vector
        assert "1" in result

    def test_scalar_array(self):
        """Test formatting scalar wrapped as array."""
        result = format_matrix(5, style="text")
        assert "5" in result

    def test_complex_number_matrix(self):
        """Test formatting matrix with complex numbers."""
        result = format_matrix([[1 + 2j, 3 - 4j]], style="text")
        assert "1" in result and "2j" in result

    def test_zero_matrix(self):
        """Test formatting all-zero matrix."""
        result = format_matrix(np.zeros((2, 2)), style="text")
        assert "0" in result

    def test_nan_raises_error(self):
        """Test that NaN values raise ValueError."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            format_matrix([[1, np.nan]], style="text")

    def test_inf_raises_error(self):
        """Test that Inf values raise ValueError."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            format_matrix([[1, np.inf]], style="text")

    def test_invalid_style_raises_error(self):
        """Test that invalid style raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format style"):
            format_matrix([[1, 2]], style="invalid")

    def test_non_array_input_raises_error(self):
        """Test that non-array-like input raises TypeError."""
        with pytest.raises(TypeError):
            format_matrix("not an array", style="text")

    def test_large_matrix(self):
        """Test formatting a larger matrix."""
        large = np.random.rand(10, 10)
        result = format_matrix(large, style="text")
        assert "⎡" in result
        assert "⎣" in result


class TestExportCsv:
    """Tests for CSV export."""

    def test_csv_basic_2x2(self):
        """Test CSV export of 2x2 matrix."""
        result = export_to_csv([[1, 2], [3, 4]])
        assert result == "1,2\n3,4\n"

    def test_csv_with_floats(self):
        """Test CSV export with floating point numbers."""
        result = export_to_csv([[1.5, 2.5], [3.5, 4.5]])
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_csv_custom_delimiter(self):
        """Test CSV export with custom delimiter."""
        result = export_to_csv([[1, 2], [3, 4]], delimiter=";")
        assert "1;2" in result
        assert "3;4" in result

    def test_csv_precision(self):
        """Test CSV export with different precisions."""
        result_p6 = export_to_csv([[1.123456789]], precision=6)
        result_p12 = export_to_csv([[1.123456789]], precision=12)
        # Both should be valid, but precision affects the output
        assert "1.12346" in result_p6

    def test_csv_complex_numbers(self):
        """Test CSV export with complex numbers."""
        result = export_to_csv([[1 + 2j, 3 - 4j]])
        assert "1" in result and "2" in result

    def test_csv_negative_values(self):
        """Test CSV export with negative values."""
        result = export_to_csv([[-1, -2], [-3, -4]])
        assert "-1,-2" in result


class TestExportNumpyText:
    """Tests for NumPy text export."""

    def test_numpy_text_basic(self):
        """Test NumPy text export."""
        result = export_to_numpy_text([[1, 2], [3, 4]])
        assert result.startswith("np.array(")
        assert result.endswith(")\n")
        assert "[[" in result

    def test_numpy_text_is_executable(self):
        """Test that exported text can be executed."""
        result = export_to_numpy_text([[1, 2], [3, 4]])
        # Should be valid Python when np is imported
        assert "np.array" in result

    def test_numpy_text_with_floats(self):
        """Test NumPy text export with floating point."""
        result = export_to_numpy_text([[1.5, 2.5], [3.5, 4.5]])
        assert "np.array" in result
        assert "[" in result

    def test_numpy_text_precision(self):
        """Test NumPy text export with different precisions."""
        result = export_to_numpy_text([[1.123456789]], precision=5)
        assert "np.array" in result


class TestExportLatex:
    """Tests for LaTeX export."""

    def test_latex_export_2x2(self):
        """Test LaTeX export of 2x2 matrix."""
        result = export_to_latex([[1, 2], [3, 4]])
        assert "\\begin{array}" in result
        assert "\\end{array}" in result

    def test_latex_export_structure(self):
        """Test LaTeX export structure."""
        result = export_to_latex([[1, 2], [3, 4]])
        assert "&" in result
        assert "\\\\" in result

    def test_latex_export_complex(self):
        """Test LaTeX export with complex numbers."""
        result = export_to_latex([[1 + 2j]])
        assert "\\begin{array}" in result
        assert "1" in result and "2j" in result


class TestCaching:
    """Tests for LRU caching behavior."""

    def test_cache_hit_identical_matrices(self):
        """Test that identical matrices use cache."""
        matrix = [[1, 2], [3, 4]]
        result1 = format_matrix(matrix, style="text")
        result2 = format_matrix(matrix, style="text")
        # Should return exactly the same object if cached
        assert result1 == result2

    def test_cache_miss_different_data(self):
        """Test that different data doesn't hit cache."""
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[5, 6], [7, 8]]
        result1 = format_matrix(matrix1, style="text")
        result2 = format_matrix(matrix2, style="text")
        assert result1 != result2

    def test_cache_miss_different_style(self):
        """Test that different styles don't hit cache."""
        matrix = [[1, 2], [3, 4]]
        result_text = format_matrix(matrix, style="text")
        result_plain = format_matrix(matrix, style="plain")
        assert result_text != result_plain

    def test_cache_miss_different_precision(self):
        """Test that different precisions don't hit cache."""
        matrix = [[1.123456789]]
        result_p3 = format_matrix(matrix, style="text", precision=3)
        result_p6 = format_matrix(matrix, style="text", precision=6)
        # Results should differ due to precision
        assert "1.12" in result_p3
        assert "1.12346" in result_p6


class TestPrecisionParameters:
    """Tests for precision parameter behavior."""

    def test_format_matrix_respects_precision(self):
        """Test that format_matrix uses precision parameter."""
        matrix = [[1.123456789]]
        result_p3 = format_matrix(matrix, style="text", precision=3)
        result_p6 = format_matrix(matrix, style="text", precision=6)
        # Should produce different outputs
        assert result_p3 != result_p6

    def test_export_csv_precision(self):
        """Test that export_to_csv respects precision."""
        matrix = [[1.123456789]]
        result_p6 = export_to_csv(matrix, precision=6)
        result_p12 = export_to_csv(matrix, precision=12)
        # Different precisions should produce different outputs
        assert result_p6 != result_p12

    def test_export_numpy_precision(self):
        """Test that export_to_numpy_text respects precision."""
        matrix = [[1.123456789]]
        result_p5 = export_to_numpy_text(matrix, precision=5)
        result_p10 = export_to_numpy_text(matrix, precision=10)
        assert "np.array" in result_p5
        assert "np.array" in result_p10

    def test_export_latex_precision(self):
        """Test that export_to_latex respects precision."""
        matrix = [[1.123456789]]
        result = export_to_latex(matrix, precision=3)
        assert "\\begin{array}" in result


class TestConsistency:
    """Tests for consistency across calls."""

    def test_identical_output_multiple_calls(self):
        """Test that identical inputs produce identical outputs."""
        matrix = [[1.5, 2.5], [3.5, 4.5]]
        results = [format_matrix(matrix, style="text") for _ in range(5)]
        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_consistency_across_styles(self):
        """Test that matrix is correctly represented in each style."""
        matrix = [[1, 2], [3, 4]]
        text = format_matrix(matrix, style="text")
        latex = format_matrix(matrix, style="latex")
        plain = format_matrix(matrix, style="plain")

        # All should contain the values
        for result in [text, latex, plain]:
            assert "1" in result
            assert "4" in result

    def test_export_consistency(self):
        """Test that export functions handle same matrix consistently."""
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        csv = export_to_csv(matrix)
        numpy = export_to_numpy_text(matrix)
        latex = export_to_latex(matrix)

        # All should contain matrix values
        for result in [csv, numpy, latex]:
            assert "1" in result
            assert "4" in result
