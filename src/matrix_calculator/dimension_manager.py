"""Dimension constraint management for matrix operations.

This module provides centralized management of dimension constraints
for matrix operations across desktop and CLI interfaces.

It defines constraints (min/max rows/columns, square requirements, resizability)
and provides utilities to apply and validate those constraints consistently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DimensionConfig:
    """Configuration for matrix dimension constraints.
    
    Attributes:
        min_rows: Minimum number of rows allowed (default: 1)
        max_rows: Maximum number of rows allowed (default: 10)
        min_cols: Minimum number of columns allowed (default: 1)
        max_cols: Maximum number of columns allowed (default: 10)
        square: If True, rows must equal columns (default: False)
        resizable: If True, user can add/remove rows/cols (default: True)
    """
    
    min_rows: int = 1
    max_rows: int = 10
    min_cols: int = 1
    max_cols: int = 10
    square: bool = False
    resizable: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.min_rows < 1:
            raise ValueError("min_rows must be >= 1")
        if self.min_cols < 1:
            raise ValueError("min_cols must be >= 1")
        if self.min_rows > self.max_rows:
            raise ValueError("min_rows must be <= max_rows")
        if self.min_cols > self.max_cols:
            raise ValueError("min_cols must be <= max_cols")


# ============================================================================
# Operation-specific preset configurations
# ============================================================================

# Binary operations: Most operations accept rectangular matrices
BINARY_OPS_CONFIG = DimensionConfig(
    min_rows=1,
    max_rows=10,
    min_cols=1,
    max_cols=10,
    square=False,
    resizable=True,
)

# Element-wise operations (add, subtract, hadamard multiply/divide): 
# Require same shape, but we don't enforce it here (validation layer handles it)
BINARY_ELEMENTWISE_CONFIG = DimensionConfig(
    min_rows=1,
    max_rows=10,
    min_cols=1,
    max_cols=10,
    square=False,
    resizable=True,
)

# Matrix multiplication: a @ b requires a.cols == b.rows
# Constraint applies to operation, not individual matrices
MATRIX_MULTIPLY_CONFIG = DimensionConfig(
    min_rows=1,
    max_rows=10,
    min_cols=1,
    max_cols=10,
    square=False,
    resizable=True,
)

# Unary operations requiring square matrix
UNARY_SQUARE_CONFIG = DimensionConfig(
    min_rows=1,
    max_rows=10,
    min_cols=1,
    max_cols=10,
    square=True,
    resizable=True,
)

# Unary operations accepting rectangular matrices
UNARY_RECT_CONFIG = DimensionConfig(
    min_rows=1,
    max_rows=10,
    min_cols=1,
    max_cols=10,
    square=False,
    resizable=True,
)

# Solve Ax=b: A is m×n (m equations, n unknowns), b is m×1
# Both share some constraints but are handled separately by input validators
SOLVE_CONFIG = DimensionConfig(
    min_rows=1,
    max_rows=10,
    min_cols=1,
    max_cols=10,
    square=False,
    resizable=True,
)

# Quadratic form analysis: requires square matrix
QUADRATIC_CONFIG = DimensionConfig(
    min_rows=1,
    max_rows=10,
    min_cols=1,
    max_cols=10,
    square=True,
    resizable=True,
)


def get_config(operation_code: str) -> DimensionConfig:
    """Get dimension configuration for a specific operation.
    
    Args:
        operation_code: Operation identifier (e.g., "add", "multiply", "determinant")
    
    Returns:
        DimensionConfig object for the operation
    
    Raises:
        KeyError: If operation_code is not recognized
    """
    operation_configs = {
        # Binary operations (element-wise)
        "add": BINARY_ELEMENTWISE_CONFIG,
        "subtract": BINARY_ELEMENTWISE_CONFIG,
        "hadamard_multiply": BINARY_ELEMENTWISE_CONFIG,
        "hadamard_divide": BINARY_ELEMENTWISE_CONFIG,
        
        # Binary operations (matrix)
        "multiply": MATRIX_MULTIPLY_CONFIG,
        
        # Unary operations (square required)
        "determinant": UNARY_SQUARE_CONFIG,
        "det": UNARY_SQUARE_CONFIG,
        "inverse": UNARY_SQUARE_CONFIG,
        "invert": UNARY_SQUARE_CONFIG,
        "eigendecomposition": UNARY_SQUARE_CONFIG,
        "eigendecompose": UNARY_SQUARE_CONFIG,
        "eigs": UNARY_SQUARE_CONFIG,
        
        # Unary operations (rectangular allowed)
        "transpose": UNARY_RECT_CONFIG,
        "trans": UNARY_RECT_CONFIG,
        "svd": UNARY_RECT_CONFIG,
        "svds": UNARY_RECT_CONFIG,
        "singular_value_decompose": UNARY_RECT_CONFIG,
        
        # Solve operations
        "solve": SOLVE_CONFIG,
        "solve_linear": SOLVE_CONFIG,
        
        # Quadratic form
        "quadratic": QUADRATIC_CONFIG,
        "analyze_quadratic": QUADRATIC_CONFIG,
    }
    
    if operation_code not in operation_configs:
        raise KeyError(f"Unknown operation: {operation_code}")
    
    return operation_configs[operation_code]


def clamp_to_config(rows: int, cols: int, config: DimensionConfig) -> tuple[int, int]:
    """Apply dimension constraints and return clamped (rows, cols).
    
    Clamps the given dimensions to the min/max bounds specified in the config.
    If config.square is True, enforces rows == cols after clamping.
    
    Args:
        rows: Number of rows to clamp
        cols: Number of columns to clamp
        config: DimensionConfig with constraints
    
    Returns:
        Tuple of (clamped_rows, clamped_cols)
    """
    # Clamp to min/max bounds
    rows = max(config.min_rows, min(rows, config.max_rows))
    cols = max(config.min_cols, min(cols, config.max_cols))
    
    # Enforce square constraint if needed
    if config.square:
        # Make it square by taking the minimum of rows and cols
        # This ensures we stay within bounds
        square_size = min(rows, cols)
        rows = square_size
        cols = square_size
    
    return rows, cols


def validate_against_config(rows: int, cols: int, config: DimensionConfig) -> bool:
    """Check if dimensions conform to the configuration.
    
    Args:
        rows: Number of rows to validate
        cols: Number of columns to validate
        config: DimensionConfig with constraints
    
    Returns:
        True if dimensions satisfy the config, False otherwise
    """
    # Check bounds
    if rows < config.min_rows or rows > config.max_rows:
        return False
    if cols < config.min_cols or cols > config.max_cols:
        return False
    
    # Check square constraint
    if config.square and rows != cols:
        return False
    
    return True


def get_help_text(operation_code: str) -> str:
    """Get user-friendly help text describing dimension requirements.
    
    Args:
        operation_code: Operation identifier
    
    Returns:
        Human-readable text describing the dimension constraints
    
    Raises:
        KeyError: If operation_code is not recognized
    """
    config = get_config(operation_code)
    
    # Build constraint description
    parts = []
    
    if config.square:
        parts.append(f"{config.min_rows}-{config.max_rows} rows and columns (square matrix)")
    else:
        parts.append(f"{config.min_rows}-{config.max_rows} rows")
        parts.append(f"{config.min_cols}-{config.max_cols} columns")
    
    if not config.resizable:
        parts.append("(fixed size, cannot add/remove rows or columns)")
    
    # Build help text specific to operation type
    operation_help = {
        # Binary element-wise
        "add": f"Matrix addition: requires both matrices to have identical dimensions. {' '.join(parts)}",
        "subtract": f"Matrix subtraction: requires both matrices to have identical dimensions. {' '.join(parts)}",
        "hadamard_multiply": f"Hadamard (element-wise) multiplication: requires identical dimensions. {' '.join(parts)}",
        "hadamard_divide": f"Hadamard (element-wise) division: requires identical dimensions and no zeros in divisor. {' '.join(parts)}",
        
        # Matrix multiplication
        "multiply": f"Matrix multiplication: first matrix columns must equal second matrix rows. {' '.join(parts)}",
        
        # Unary square
        "determinant": f"Determinant: only defined for square matrices. {' '.join(parts)}",
        "det": f"Determinant: only defined for square matrices. {' '.join(parts)}",
        "inverse": f"Matrix inverse: only defined for square matrices. {' '.join(parts)}",
        "invert": f"Matrix inverse: only defined for square matrices. {' '.join(parts)}",
        "eigendecomposition": f"Eigendecomposition: only defined for square matrices. {' '.join(parts)}",
        "eigendecompose": f"Eigendecomposition: only defined for square matrices. {' '.join(parts)}",
        "eigs": f"Eigendecomposition: only defined for square matrices. {' '.join(parts)}",
        
        # Unary rectangular
        "transpose": f"Transpose: works with any matrix. {' '.join(parts)}",
        "trans": f"Transpose: works with any matrix. {' '.join(parts)}",
        "svd": f"Singular Value Decomposition: works with any matrix. {' '.join(parts)}",
        "svds": f"Singular Value Decomposition: works with any matrix. {' '.join(parts)}",
        "singular_value_decompose": f"Singular Value Decomposition: works with any matrix. {' '.join(parts)}",
        
        # Solve
        "solve": f"Solve linear system Ax=b. {' '.join(parts)}",
        "solve_linear": f"Solve linear system Ax=b. {' '.join(parts)}",
        
        # Quadratic
        "quadratic": f"Quadratic form analysis: only for square matrices. {' '.join(parts)}",
        "analyze_quadratic": f"Quadratic form analysis: only for square matrices. {' '.join(parts)}",
    }
    
    return operation_help.get(
        operation_code,
        f"Operation {operation_code} requires: {' '.join(parts)}"
    )
