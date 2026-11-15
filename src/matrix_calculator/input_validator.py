"""Unified matrix input validation and parsing module.

This module consolidates input parsing and validation logic from web and desktop
matrix input components. It provides framework-agnostic functions for:
- Parsing individual cell values (decimals, fractions, scientific notation)
- Validating and clamping matrix dimensions
- Parsing entire matrix grids from raw text input
- Computing input signatures for caching

Design Principles:
- Framework-agnostic: no Streamlit, Qt, or other UI framework dependencies
- Type-safe: returns structured error/success objects instead of raising exceptions
- Composable: functions can be used independently or together
- Efficient: minimal allocations, suitable for interactive UI updates
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


# ==================== Type Definitions ====================


@dataclass
class ParseResult:
    """Result of parsing a cell, dimension, or matrix.

    Attributes:
        ok: True if parsing succeeded; False otherwise.
        value: The parsed value (float for parse_cell, ndarray for parse_matrix_cells, tuple for validate_*).
               None if parsing failed.
        message: Human-readable error message if ok=False; None otherwise.
    """

    ok: bool
    value: Any | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.ok:
            if self.value is None:
                raise ValueError("ParseResult with ok=True must have a non-None value")
            if self.message is not None:
                raise ValueError("ParseResult with ok=True must have message=None")
        else:
            if self.message is None:
                raise ValueError("ParseResult with ok=False must have a non-None message")


@dataclass(frozen=True)
class DimensionConfig:
    """Configuration for matrix dimension constraints.

    Attributes:
        min_rows: Minimum allowed number of rows (>= 1).
        max_rows: Maximum allowed number of rows.
        min_cols: Minimum allowed number of columns (>= 1).
        max_cols: Maximum allowed number of columns.
        square_only: If True, enforce rows == cols after clamping.
        resizable: If False, dimensions are fixed (informational only).
    """

    min_rows: int = 1
    max_rows: int = 10
    min_cols: int = 1
    max_cols: int = 10
    square_only: bool = False
    resizable: bool = True

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if not isinstance(self.min_rows, int) or self.min_rows < 1:
            raise ValueError("min_rows must be an integer >= 1")
        if not isinstance(self.max_rows, int) or self.max_rows < self.min_rows:
            raise ValueError("max_rows must be an integer >= min_rows")
        if not isinstance(self.min_cols, int) or self.min_cols < 1:
            raise ValueError("min_cols must be an integer >= 1")
        if not isinstance(self.max_cols, int) or self.max_cols < self.min_cols:
            raise ValueError("max_cols must be an integer >= min_cols")


# ==================== Constants ====================

# Regex for valid numeric strings (decimal, scientific notation)
# Matches: integers, decimals, scientific notation with optional sign
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

# Exponent bounds for scientific notation
_MIN_EXPONENT = -99
_MAX_EXPONENT = 99

# Fraction denominator bounds
_MIN_DENOMINATOR = 1
_MAX_DENOMINATOR = 999999


# ==================== Cell Parsing ====================


def parse_cell(text: str) -> ParseResult:
    """Parse a single cell value from raw text.

    Supports the following number formats:
    - Integers: "5", "-3"
    - Decimals: "2.5", "-0.001", ".5" (leading dot)
    - Fractions: "1/2", "-3/4" (denominator in [1, 999999])
    - Scientific: "1e-3", "2.5e+10" (exponent in [-99, +99])
    - Combinations: "-1.5e-3", "1/2e5" (as mathematically parsed)
    - Whitespace: stripped before processing

    Args:
        text: Raw cell input string (may contain whitespace, signs, etc.)

    Returns:
        ParseResult with:
        - ok=True, value=float if parsing succeeded
        - ok=False, message describing error if parsing failed

    Error Types:
        "empty_cell": Input is empty or whitespace-only
        "invalid_number_format": Input does not match any supported format
        "multiple_division_operators": More than one "/" in fraction form
        "division_by_zero": Denominator is 0 in fraction form
        "exponent_out_of_range": Exponent < -99 or > +99
        "denominator_out_of_range": Denominator > 999999
    """
    # Trim and validate non-empty
    text = str(text).strip()
    if not text:
        return ParseResult(ok=False, message="empty_cell")

    # Count division operators (for fractions)
    division_count = text.count("/")
    if division_count > 1:
        return ParseResult(ok=False, message="multiple_division_operators")

    # If no division, parse as decimal/scientific
    if division_count == 0:
        return _parse_decimal_or_scientific(text)

    # Parse fraction
    return _parse_fraction(text)


def _parse_decimal_or_scientific(text: str) -> ParseResult:
    """Parse a decimal or scientific notation number.

    Args:
        text: Trimmed string (no whitespace around operators)

    Returns:
        ParseResult with float value or error message
    """
    if not _NUMBER_PATTERN.match(text):
        return ParseResult(ok=False, message="invalid_number_format")

    try:
        value = float(text)
    except ValueError:
        return ParseResult(ok=False, message="invalid_number_format")

    # Validate exponent bounds if scientific notation
    if "e" in text.lower():
        try:
            _exponent = int(text.lower().split("e")[1])
            if _exponent < _MIN_EXPONENT or _exponent > _MAX_EXPONENT:
                return ParseResult(
                    ok=False, message="exponent_out_of_range"
                )
        except (ValueError, IndexError):
            return ParseResult(ok=False, message="invalid_number_format")

    return ParseResult(ok=True, value=value)


def _parse_fraction(text: str) -> ParseResult:
    """Parse a fraction like '1/2' or '-3/4'.

    Args:
        text: String with exactly one '/' separator

    Returns:
        ParseResult with float value (numerator/denominator) or error
    """
    parts = text.split("/")
    numerator_str, denominator_str = parts[0].strip(), parts[1].strip()

    # Parse numerator (supports scientific notation)
    if not _NUMBER_PATTERN.match(numerator_str):
        return ParseResult(ok=False, message="invalid_number_format")
    try:
        numerator = float(numerator_str)
    except ValueError:
        return ParseResult(ok=False, message="invalid_number_format")

    # Parse denominator (should be simple positive number)
    if not _NUMBER_PATTERN.match(denominator_str):
        return ParseResult(ok=False, message="invalid_number_format")
    try:
        denominator = float(denominator_str)
    except ValueError:
        return ParseResult(ok=False, message="invalid_number_format")

    # Check for zero denominator
    if denominator == 0:
        return ParseResult(ok=False, message="division_by_zero")

    # Check denominator bounds
    if abs(denominator) > _MAX_DENOMINATOR:
        return ParseResult(ok=False, message="denominator_out_of_range")

    return ParseResult(ok=True, value=numerator / denominator)


# ==================== Dimension Management ====================


def clamp_dimensions(
    rows: int, cols: int, config: DimensionConfig
) -> ParseResult:
    """Clamp matrix dimensions to configuration bounds.

    Truncates (not rounds) dimensions to the allowed range. If square_only=True,
    forces cols=rows after clamping.

    Args:
        rows: Current/desired number of rows
        cols: Current/desired number of columns
        config: DimensionConfig with bounds and constraints

    Returns:
        ParseResult with value=(clamped_rows, clamped_cols) or error message

    Raises:
        ValueError: If config validation fails (e.g., min > max)
    """
    # Config is validated in DimensionConfig.__post_init__
    # (may raise ValueError if invalid)

    # Clamp to bounds (truncate, not round)
    clamped_rows = int(np.clip(rows, config.min_rows, config.max_rows))
    clamped_cols = int(np.clip(cols, config.min_cols, config.max_cols))

    # Enforce square if required
    if config.square_only:
        clamped_cols = clamped_rows

    return ParseResult(ok=True, value=(clamped_rows, clamped_cols))


def validate_dimensions(rows: int, cols: int, config: DimensionConfig) -> ParseResult:
    """Check if dimensions satisfy configuration constraints.

    Args:
        rows: Number of rows to validate
        cols: Number of columns to validate
        config: DimensionConfig with constraints

    Returns:
        ParseResult with ok=True if valid, ok=False with error message otherwise

    Raises:
        ValueError: If config validation fails
    """
    # Perform clamping; if result differs, dimensions are out of bounds
    clamp_result = clamp_dimensions(rows, cols, config)
    if not clamp_result.ok:
        return clamp_result

    clamped_rows, clamped_cols = clamp_result.value
    if clamped_rows != rows or clamped_cols != cols:
        if config.square_only:
            return ParseResult(
                ok=False,
                message=f"Dimensions must be square. Clamped to {clamped_rows}x{clamped_rows}.",
            )
        return ParseResult(
            ok=False,
            message=f"Dimensions out of range. Clamped to {clamped_rows}x{clamped_cols}.",
        )

    return ParseResult(ok=True, value=(rows, cols))


def parse_matrix_cells(
    cell_values: list[list[str]], config: DimensionConfig | None = None
) -> ParseResult:
    """Parse all cells of a matrix input grid.

    Args:
        cell_values: 2D list of raw cell strings (each row is a list)
        config: Optional DimensionConfig to validate dimensions

    Returns:
        ParseResult with:
        - ok=True, value=np.ndarray if all cells parsed successfully
        - ok=False, message with error location and detail if any cell failed

    Notes:
        - Validates dimensions against config if provided
        - Returns detailed error message with row/col indices (1-indexed for users)
    """
    if not cell_values:
        return ParseResult(ok=False, message="Matrix is empty")

    rows = len(cell_values)
    cols = len(cell_values[0]) if cell_values else 0

    # Validate dimensions if config provided
    if config is not None:
        dim_result = validate_dimensions(rows, cols, config)
        if not dim_result.ok:
            return dim_result

    # Parse all cells
    matrix_data = []
    for i, row in enumerate(cell_values):
        row_data = []
        for j, cell_text in enumerate(row):
            parse_result = parse_cell(cell_text)
            if not parse_result.ok:
                error_msg = f"Cell [{i + 1}, {j + 1}]: {parse_result.message}"
                return ParseResult(ok=False, message=error_msg)
            row_data.append(parse_result.value)
        matrix_data.append(row_data)

    return ParseResult(ok=True, value=np.array(matrix_data, dtype=float))


# ==================== Input Signature / Caching ====================


def get_input_signature(matrices: list[np.ndarray], operation_code: str) -> tuple:
    """Generate a deterministic signature for input matrices and operation.

    Used by result_cache to detect when inputs change and cached results become stale.

    Args:
        matrices: List of input matrices (usually 1 or 2)
        operation_code: String code for the operation (e.g., "add", "determinant")

    Returns:
        Tuple of (operation_code, tuple_of_matrix_signatures) where each matrix
        signature is (shape, content_hash). Deterministic and hashable.

    Notes:
        - Signature is independent of memory address; identical content → identical signature
        - Suitable for cache key generation and equality testing
        - Content hash is SHA256 of flattened matrix bytes (robust for very large matrices)
    """
    matrix_sigs = []
    for mat in matrices:
        mat_array = np.asarray(mat, dtype=float)
        shape_sig = tuple(mat_array.shape)
        # Hash the serialized array bytes for content comparison
        content_bytes = mat_array.tobytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]  # Short hash
        matrix_sigs.append((shape_sig, content_hash))

    return (operation_code, tuple(matrix_sigs))
