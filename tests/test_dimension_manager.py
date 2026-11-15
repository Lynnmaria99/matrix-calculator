"""Unit tests for dimension_manager module.

Tests dimension configuration, constraint application,
and validation logic for matrix operations.
"""

import pytest

from matrix_calculator.dimension_manager import (
    BINARY_ELEMENTWISE_CONFIG,
    BINARY_OPS_CONFIG,
    MATRIX_MULTIPLY_CONFIG,
    QUADRATIC_CONFIG,
    SOLVE_CONFIG,
    UNARY_RECT_CONFIG,
    UNARY_SQUARE_CONFIG,
    DimensionConfig,
    clamp_to_config,
    get_config,
    get_help_text,
    validate_against_config,
)


class TestDimensionConfig:
    """Tests for DimensionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DimensionConfig()
        assert config.min_rows == 1
        assert config.max_rows == 10
        assert config.min_cols == 1
        assert config.max_cols == 10
        assert config.square is False
        assert config.resizable is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DimensionConfig(
            min_rows=2,
            max_rows=20,
            min_cols=3,
            max_cols=30,
            square=True,
            resizable=False,
        )
        assert config.min_rows == 2
        assert config.max_rows == 20
        assert config.min_cols == 3
        assert config.max_cols == 30
        assert config.square is True
        assert config.resizable is False
    
    def test_validation_min_rows_positive(self):
        """Test that min_rows must be positive."""
        with pytest.raises(ValueError, match="min_rows must be >= 1"):
            DimensionConfig(min_rows=0)
    
    def test_validation_min_cols_positive(self):
        """Test that min_cols must be positive."""
        with pytest.raises(ValueError, match="min_cols must be >= 1"):
            DimensionConfig(min_cols=-1)
    
    def test_validation_min_max_rows(self):
        """Test that min_rows must be <= max_rows."""
        with pytest.raises(ValueError, match="min_rows must be <= max_rows"):
            DimensionConfig(min_rows=10, max_rows=5)
    
    def test_validation_min_max_cols(self):
        """Test that min_cols must be <= max_cols."""
        with pytest.raises(ValueError, match="min_cols must be <= max_cols"):
            DimensionConfig(min_cols=10, max_cols=5)


class TestPresetConfigs:
    """Tests for predefined operation-specific configurations."""
    
    def test_binary_ops_config(self):
        """Test binary operations config."""
        assert BINARY_OPS_CONFIG.square is False
        assert BINARY_OPS_CONFIG.resizable is True
    
    def test_binary_elementwise_config(self):
        """Test element-wise binary operations config."""
        assert BINARY_ELEMENTWISE_CONFIG.square is False
        assert BINARY_ELEMENTWISE_CONFIG.resizable is True
    
    def test_matrix_multiply_config(self):
        """Test matrix multiplication config."""
        assert MATRIX_MULTIPLY_CONFIG.square is False
        assert MATRIX_MULTIPLY_CONFIG.resizable is True
    
    def test_unary_square_config(self):
        """Test unary square matrix config."""
        assert UNARY_SQUARE_CONFIG.square is True
        assert UNARY_SQUARE_CONFIG.resizable is True
    
    def test_unary_rect_config(self):
        """Test unary rectangular config."""
        assert UNARY_RECT_CONFIG.square is False
        assert UNARY_RECT_CONFIG.resizable is True
    
    def test_solve_config(self):
        """Test solve operation config."""
        assert SOLVE_CONFIG.square is False
        assert SOLVE_CONFIG.resizable is True
    
    def test_quadratic_config(self):
        """Test quadratic form config."""
        assert QUADRATIC_CONFIG.square is True
        assert QUADRATIC_CONFIG.resizable is True


class TestGetConfig:
    """Tests for get_config function."""
    
    def test_get_config_for_binary_operations(self):
        """Test retrieving configs for binary operations."""
        assert get_config("add") == BINARY_ELEMENTWISE_CONFIG
        assert get_config("subtract") == BINARY_ELEMENTWISE_CONFIG
        assert get_config("hadamard_multiply") == BINARY_ELEMENTWISE_CONFIG
        assert get_config("hadamard_divide") == BINARY_ELEMENTWISE_CONFIG
    
    def test_get_config_for_matrix_multiply(self):
        """Test retrieving config for matrix multiplication."""
        assert get_config("multiply") == MATRIX_MULTIPLY_CONFIG
    
    def test_get_config_for_unary_square_operations(self):
        """Test retrieving configs for square-requiring unary operations."""
        assert get_config("determinant") == UNARY_SQUARE_CONFIG
        assert get_config("det") == UNARY_SQUARE_CONFIG
        assert get_config("inverse") == UNARY_SQUARE_CONFIG
        assert get_config("invert") == UNARY_SQUARE_CONFIG
        assert get_config("eigendecomposition") == UNARY_SQUARE_CONFIG
        assert get_config("eigendecompose") == UNARY_SQUARE_CONFIG
        assert get_config("eigs") == UNARY_SQUARE_CONFIG
    
    def test_get_config_for_unary_rect_operations(self):
        """Test retrieving configs for rectangular unary operations."""
        assert get_config("transpose") == UNARY_RECT_CONFIG
        assert get_config("trans") == UNARY_RECT_CONFIG
        assert get_config("svd") == UNARY_RECT_CONFIG
        assert get_config("svds") == UNARY_RECT_CONFIG
        assert get_config("singular_value_decompose") == UNARY_RECT_CONFIG
    
    def test_get_config_for_solve_operations(self):
        """Test retrieving configs for solve operations."""
        assert get_config("solve") == SOLVE_CONFIG
        assert get_config("solve_linear") == SOLVE_CONFIG
    
    def test_get_config_for_quadratic_operations(self):
        """Test retrieving configs for quadratic operations."""
        assert get_config("quadratic") == QUADRATIC_CONFIG
        assert get_config("analyze_quadratic") == QUADRATIC_CONFIG
    
    def test_get_config_unknown_operation(self):
        """Test that unknown operation raises KeyError."""
        with pytest.raises(KeyError, match="Unknown operation"):
            get_config("unknown_op")
    
    def test_get_config_unknown_operation_message(self):
        """Test error message for unknown operation."""
        with pytest.raises(KeyError) as exc_info:
            get_config("nonexistent")
        assert "nonexistent" in str(exc_info.value)


class TestClampToConfig:
    """Tests for clamp_to_config function."""
    
    def test_clamp_within_bounds(self):
        """Test clamping when already within bounds."""
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        rows, cols = clamp_to_config(5, 5, config)
        assert rows == 5
        assert cols == 5
    
    def test_clamp_below_min(self):
        """Test clamping when below minimum."""
        config = DimensionConfig(min_rows=2, max_rows=10, min_cols=2, max_cols=10)
        rows, cols = clamp_to_config(1, 1, config)
        assert rows == 2
        assert cols == 2
    
    def test_clamp_above_max(self):
        """Test clamping when above maximum."""
        config = DimensionConfig(min_rows=1, max_rows=8, min_cols=1, max_cols=8)
        rows, cols = clamp_to_config(20, 20, config)
        assert rows == 8
        assert cols == 8
    
    def test_clamp_rows_only(self):
        """Test clamping only rows."""
        config = DimensionConfig(min_rows=2, max_rows=5, min_cols=1, max_cols=10)
        rows, cols = clamp_to_config(10, 3, config)
        assert rows == 5
        assert cols == 3
    
    def test_clamp_cols_only(self):
        """Test clamping only columns."""
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=2, max_cols=8)
        rows, cols = clamp_to_config(3, 1, config)
        assert rows == 3
        assert cols == 2
    
    def test_clamp_to_square_symmetric(self):
        """Test clamping with square constraint when input is symmetric."""
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square=True
        )
        rows, cols = clamp_to_config(5, 5, config)
        assert rows == 5
        assert cols == 5
    
    def test_clamp_to_square_uneven_rows_larger(self):
        """Test clamping to square when rows > cols."""
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square=True
        )
        rows, cols = clamp_to_config(8, 3, config)
        assert rows == 3
        assert cols == 3
    
    def test_clamp_to_square_cols_larger(self):
        """Test clamping to square when cols > rows."""
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square=True
        )
        rows, cols = clamp_to_config(3, 8, config)
        assert rows == 3
        assert cols == 3
    
    def test_clamp_to_square_with_bounds(self):
        """Test clamping to square while respecting bounds."""
        config = DimensionConfig(
            min_rows=2, max_rows=8, min_cols=2, max_cols=8, square=True
        )
        rows, cols = clamp_to_config(1, 20, config)
        # min(min clamped to 2, 20 clamped to 8) = 2
        assert rows == 2
        assert cols == 2


class TestValidateAgainstConfig:
    """Tests for validate_against_config function."""
    
    def test_validate_within_bounds(self):
        """Test validation for dimensions within bounds."""
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        assert validate_against_config(5, 5, config) is True
    
    def test_validate_at_min_bounds(self):
        """Test validation at minimum boundaries."""
        config = DimensionConfig(min_rows=2, max_rows=10, min_cols=3, max_cols=10)
        assert validate_against_config(2, 3, config) is True
    
    def test_validate_at_max_bounds(self):
        """Test validation at maximum boundaries."""
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        assert validate_against_config(10, 10, config) is True
    
    def test_validate_below_min_rows(self):
        """Test validation when below minimum rows."""
        config = DimensionConfig(min_rows=2, max_rows=10, min_cols=1, max_cols=10)
        assert validate_against_config(1, 5, config) is False
    
    def test_validate_above_max_rows(self):
        """Test validation when above maximum rows."""
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        assert validate_against_config(11, 5, config) is False
    
    def test_validate_below_min_cols(self):
        """Test validation when below minimum columns."""
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=2, max_cols=10)
        assert validate_against_config(5, 1, config) is False
    
    def test_validate_above_max_cols(self):
        """Test validation when above maximum columns."""
        config = DimensionConfig(min_rows=1, max_rows=10, min_cols=1, max_cols=10)
        assert validate_against_config(5, 11, config) is False
    
    def test_validate_square_valid(self):
        """Test validation with square constraint when valid."""
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square=True
        )
        assert validate_against_config(5, 5, config) is True
    
    def test_validate_square_invalid(self):
        """Test validation with square constraint when invalid."""
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square=True
        )
        assert validate_against_config(5, 3, config) is False
        assert validate_against_config(3, 5, config) is False
    
    def test_validate_square_nonsquare_config(self):
        """Test validation when square constraint not enforced."""
        config = DimensionConfig(
            min_rows=1, max_rows=10, min_cols=1, max_cols=10, square=False
        )
        assert validate_against_config(5, 3, config) is True
        assert validate_against_config(3, 5, config) is True


class TestGetHelpText:
    """Tests for get_help_text function."""
    
    def test_help_text_for_add(self):
        """Test help text for addition operation."""
        help_text = get_help_text("add")
        assert "addition" in help_text.lower()
        assert "identical" in help_text.lower()
    
    def test_help_text_for_multiply(self):
        """Test help text for matrix multiplication."""
        help_text = get_help_text("multiply")
        assert "multiplication" in help_text.lower()
    
    def test_help_text_for_determinant(self):
        """Test help text for determinant."""
        help_text = get_help_text("determinant")
        assert "square" in help_text.lower()
    
    def test_help_text_for_determinant_alias(self):
        """Test help text for determinant alias."""
        help_text = get_help_text("det")
        assert "square" in help_text.lower()
    
    def test_help_text_for_inverse(self):
        """Test help text for matrix inverse."""
        help_text = get_help_text("inverse")
        assert "square" in help_text.lower()
        assert "inverse" in help_text.lower()
    
    def test_help_text_for_transpose(self):
        """Test help text for transpose."""
        help_text = get_help_text("transpose")
        assert "transpose" in help_text.lower()
    
    def test_help_text_for_svd(self):
        """Test help text for SVD."""
        help_text = get_help_text("svd")
        assert "singular" in help_text.lower() or "svd" in help_text.lower()
    
    def test_help_text_for_eigendecomposition(self):
        """Test help text for eigendecomposition."""
        help_text = get_help_text("eigendecomposition")
        assert "square" in help_text.lower() or "eigen" in help_text.lower()
    
    def test_help_text_for_hadamard_multiply(self):
        """Test help text for Hadamard multiplication."""
        help_text = get_help_text("hadamard_multiply")
        assert "hadamard" in help_text.lower() or "element" in help_text.lower()
    
    def test_help_text_for_hadamard_divide(self):
        """Test help text for Hadamard division."""
        help_text = get_help_text("hadamard_divide")
        assert "hadamard" in help_text.lower() or "element" in help_text.lower()
    
    def test_help_text_for_solve(self):
        """Test help text for solve operation."""
        help_text = get_help_text("solve")
        assert "solve" in help_text.lower()
    
    def test_help_text_for_quadratic(self):
        """Test help text for quadratic operation."""
        help_text = get_help_text("quadratic")
        assert "square" in help_text.lower() or "quadratic" in help_text.lower()
    
    def test_help_text_unknown_operation(self):
        """Test help text for unknown operation."""
        with pytest.raises(KeyError):
            get_help_text("nonexistent_op")
    
    def test_help_text_includes_dimensions(self):
        """Test that help text includes dimension information."""
        help_text = get_help_text("add")
        # Should contain min-max rows/cols info
        assert any(str(i) in help_text for i in range(1, 11))


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_square_operation_workflow(self):
        """Test complete workflow for square matrix operation."""
        config = get_config("determinant")
        assert config.square is True
        
        # Valid: 5x5 matrix
        assert validate_against_config(5, 5, config) is True
        clamped = clamp_to_config(5, 5, config)
        assert clamped == (5, 5)
        
        # Invalid: 5x3 matrix
        assert validate_against_config(5, 3, config) is False
        clamped = clamp_to_config(5, 3, config)
        assert clamped[0] == clamped[1]  # Should be square after clamping
    
    def test_rectangular_operation_workflow(self):
        """Test complete workflow for rectangular matrix operation."""
        config = get_config("transpose")
        assert config.square is False
        
        # Valid: various shapes
        assert validate_against_config(5, 3, config) is True
        assert validate_against_config(3, 5, config) is True
        assert validate_against_config(5, 5, config) is True
        
        clamped = clamp_to_config(5, 3, config)
        assert clamped == (5, 3)
    
    def test_operation_specific_configs_are_different(self):
        """Test that different operations have appropriate different configs."""
        det_config = get_config("determinant")
        trans_config = get_config("transpose")
        
        assert det_config.square is True
        assert trans_config.square is False
    
    def test_all_operations_have_valid_configs(self):
        """Test that all known operations can retrieve valid configs."""
        operations = [
            "add", "subtract", "hadamard_multiply", "hadamard_divide",
            "multiply",
            "determinant", "det", "inverse", "invert",
            "eigendecomposition", "eigendecompose", "eigs",
            "transpose", "trans", "svd", "svds", "singular_value_decompose",
            "solve", "solve_linear",
            "quadratic", "analyze_quadratic",
        ]
        
        for op in operations:
            config = get_config(op)
            assert isinstance(config, DimensionConfig)
            assert config.min_rows >= 1
            assert config.max_rows >= config.min_rows
            assert config.min_cols >= 1
            assert config.max_cols >= config.min_cols
    
    def test_all_operations_have_help_text(self):
        """Test that all known operations have non-empty help text."""
        operations = [
            "add", "subtract", "hadamard_multiply", "hadamard_divide",
            "multiply",
            "determinant", "det", "inverse", "invert",
            "eigendecomposition", "eigendecompose", "eigs",
            "transpose", "trans", "svd", "svds", "singular_value_decompose",
            "solve", "solve_linear",
            "quadratic", "analyze_quadratic",
        ]
        
        for op in operations:
            help_text = get_help_text(op)
            assert len(help_text) > 0
            assert isinstance(help_text, str)
