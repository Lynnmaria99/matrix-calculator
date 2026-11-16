"""Result formatting and export functions for matrices.

This module consolidates matrix formatting and export logic that was previously
shared by desktop (PySide6) and CLI layers. It provides
framework-agnostic functions for converting numpy arrays to human-readable or
exportable formats.

Key functions:
- format_matrix(): Format matrix in multiple styles (text, latex, plain)
- export_to_csv(): Export as comma-separated values
- export_to_numpy_text(): Export as Python-executable numpy code
- export_to_latex(): Export as LaTeX notation
"""

from __future__ import annotations

import functools
import hashlib
import io
from typing import Literal

import numpy as np

from .constants import EPS


# Type alias for formatting style
FormatStyle = Literal["text", "latex", "plain"]


def _compute_signature(matrix: np.ndarray) -> str:
    """Compute SHA256 signature of matrix content for cache detection.

    Args:
        matrix: Input numpy array

    Returns:
        Hex string of SHA256 hash of matrix data and shape
    """
    matrix = np.asarray(matrix)
    data_bytes = matrix.tobytes()
    shape_bytes = str(matrix.shape).encode("utf-8")
    combined = data_bytes + shape_bytes
    return hashlib.sha256(combined).hexdigest()


def format_scalar(value: float | complex, precision: int = 6) -> str:
    """Format a single scalar value (real or complex).

    Values very close to zero are displayed as 0. For complex numbers,
    the format is "r+ij" or "r" or "ij" as appropriate.

    Args:
        value: Scalar value (float or complex)
        precision: Significant figures for display (default 6)

    Returns:
        Formatted string representation
    """
    if np.isnan(value) or np.isinf(value):
        return str(value)

    if np.iscomplexobj(value):
        r = 0.0 if abs(value.real) < EPS else value.real
        i = 0.0 if abs(value.imag) < EPS else value.imag
        if i == 0:
            return f"{r:.{precision}g}"
        if r == 0:
            return f"{i:.{precision}g}j"
        return f"{r:.{precision}g}{i:+.{precision}g}j"

    val = 0.0 if abs(value) < EPS else float(value)
    return f"{val:.{precision}g}"


def format_complex_number(value: complex, precision: int = 6) -> str:
    """Format a complex number with specified precision.

    Args:
        value: Complex number
        precision: Significant figures for display

    Returns:
        Formatted string like "a+bj"
    """
    return format_scalar(value, precision=precision)


def _format_matrix_text(matrix: np.ndarray, precision: int = 6) -> str:
    """Format matrix with Unicode box characters (textbook style).

    Uses Unicode brackets ⎡ ⎤ ⎢ ⎥ ⎣ ⎦ to create a visual matrix representation.
    Columns are aligned with right-justification.

    Args:
        matrix: Input numpy array
        precision: Significant figures for scalars

    Returns:
        Multi-line string with formatted matrix
    """
    matrix = np.atleast_2d(np.asarray(matrix))
    nrows, ncols = matrix.shape

    # Format each cell
    cells = [
        [format_scalar(matrix[i, j], precision=precision) for j in range(ncols)]
        for i in range(nrows)
    ]

    # Calculate column widths
    widths = [
        max(len(cells[i][j]) for i in range(nrows)) for j in range(ncols)
    ]

    def render_row(i: int, left: str, right: str) -> str:
        """Render a single row with brackets and aligned columns."""
        parts = []
        for j in range(ncols):
            parts.append(cells[i][j].rjust(widths[j]))
        return f"{left} " + "  ".join(parts) + f" {right}"

    if nrows == 1:
        return render_row(0, "(", ")")

    rows = []
    for i in range(nrows):
        if i == 0:
            left, right = "⎡", "⎤"
        elif i == nrows - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"
        rows.append(render_row(i, left, right))

    return "\n".join(rows)


def _format_matrix_latex(matrix: np.ndarray, precision: int = 4) -> str:
    """Format matrix as LaTeX array environment.

    Generates \\begin{array}{ccc}...\\end{array} with appropriate
    column separators and line breaks.

    Args:
        matrix: Input numpy array
        precision: Significant figures for scalars

    Returns:
        LaTeX-formatted string
    """
    matrix = np.atleast_2d(np.asarray(matrix))
    nrows, ncols = matrix.shape

    # Format cells
    cells = [
        [format_scalar(matrix[i, j], precision=precision) for j in range(ncols)]
        for i in range(nrows)
    ]

    # Build LaTeX array
    align = "c" * ncols  # Center-aligned columns
    lines = [f"\\begin{{array}}{{{align}}}"]

    for i in range(nrows):
        row_str = " & ".join(cells[i])
        if i < nrows - 1:
            row_str += " \\\\"
        lines.append(row_str)

    lines.append("\\end{array}")

    return "\n".join(lines)


def _format_matrix_plain(matrix: np.ndarray, precision: int = 6) -> str:
    """Format matrix as space-separated values.

    Args:
        matrix: Input numpy array
        precision: Significant figures for scalars

    Returns:
        Space-separated string (values on one line, rows separated by newlines)
    """
    matrix = np.atleast_2d(np.asarray(matrix))
    lines = []
    for row in matrix:
        row_str = " ".join(format_scalar(val, precision=precision) for val in row)
        lines.append(row_str)
    return "\n".join(lines)


@functools.lru_cache(maxsize=100)
def _cached_format_matrix(
    data_bytes: bytes,
    shape: tuple,
    dtype_str: str,
    style: FormatStyle,
    precision: int,
) -> str:
    """Internal cached formatter (LRU cache with 100 entries).

    This function is cached by functools.lru_cache. Since numpy arrays are
    not hashable, we convert them to bytes + shape before caching.

    Args:
        data_bytes: Array data as bytes
        shape: Array shape tuple
        dtype_str: Data type string
        style: Format style
        precision: Significant figures

    Returns:
        Formatted string
    """
    dtype = np.dtype(dtype_str)
    matrix = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

    if style == "text":
        return _format_matrix_text(matrix, precision=precision)
    elif style == "latex":
        return _format_matrix_latex(matrix, precision=precision)
    elif style == "plain":
        return _format_matrix_plain(matrix, precision=precision)
    else:
        raise ValueError(f"Unknown format style: {style}")


def format_matrix(matrix: np.ndarray, style: FormatStyle = "text", precision: int = 6) -> str:
    """Format a matrix in the specified style.

    Supports multiple output formats with caching for identical matrices.

    Args:
        matrix: Input numpy array (or array-like)
        style: Format style: "text" (Unicode), "latex", or "plain" (space-separated)
        precision: Significant figures for scalar values (default 6)

    Returns:
        Formatted string representation of matrix

    Raises:
        TypeError: If matrix is not array-like
        ValueError: If matrix contains NaN or Inf
    """
    try:
        matrix = np.asarray(matrix)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be array-like, got {type(matrix)}") from e

    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains NaN or Inf values")

    if style not in ("text", "latex", "plain"):
        raise ValueError(f"Unknown format style: {style}")

    # Make 2D
    matrix = np.atleast_2d(matrix)

    # Use cached formatter with hashable parameters
    return _cached_format_matrix(
        matrix.tobytes(),
        matrix.shape,
        str(matrix.dtype),
        style,
        precision,
    )


def export_to_csv(matrix: np.ndarray, delimiter: str = ",", precision: int = 12) -> str:
    """Export matrix as comma-separated values.

    Args:
        matrix: Input numpy array
        delimiter: Separator character (default ",")
        precision: Significant figures (default 12, uses "%.12g" format)

    Returns:
        CSV-formatted string

    Raises:
        TypeError: If matrix is not array-like
    """
    try:
        matrix = np.atleast_2d(np.asarray(matrix))
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be array-like, got {type(matrix)}") from e

    buffer = io.StringIO()
    fmt = "%s" if np.iscomplexobj(matrix) else f"%.{precision}g"
    np.savetxt(buffer, matrix, delimiter=delimiter, fmt=fmt)
    return buffer.getvalue()


def export_to_numpy_text(matrix: np.ndarray, precision: int = 10) -> str:
    """Export matrix as Python-executable numpy code.

    Returns a string like "np.array([[1, 2], [3, 4]])" that can be
    directly pasted into Python or a Jupyter notebook.

    Args:
        matrix: Input numpy array
        precision: Significant figures (default 10)

    Returns:
        String containing np.array() call

    Raises:
        TypeError: If matrix is not array-like
    """
    try:
        matrix = np.atleast_2d(np.asarray(matrix))
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be array-like, got {type(matrix)}") from e

    array_str = np.array2string(
        matrix,
        separator=", ",
        threshold=10_000,
        suppress_small=True,
        precision=precision,
    )
    return f"np.array({array_str})\n"


def export_to_latex(matrix: np.ndarray, precision: int = 4) -> str:
    """Export matrix as LaTeX notation.

    Generates \\begin{array}...\\end{array} with proper formatting
    for complex numbers: (a+bj).

    Args:
        matrix: Input numpy array
        precision: Significant figures (default 4)

    Returns:
        LaTeX-formatted string

    Raises:
        TypeError: If matrix is not array-like
    """
    try:
        matrix = np.atleast_2d(np.asarray(matrix))
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input must be array-like, got {type(matrix)}") from e

    return _format_matrix_latex(matrix, precision=precision)


# Backward compatibility: export as public function
__all__ = [
    "format_matrix",
    "format_scalar",
    "format_complex_number",
    "export_to_csv",
    "export_to_numpy_text",
    "export_to_latex",
]
