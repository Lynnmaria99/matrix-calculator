"""Unified computation adapter layer for Web/Desktop/CLI.

This module provides a shared adapter layer that wraps computation calls with
exception handling, result wrapping, and integration with shared modules
(computation_result, error_translator, result_formatter).

The adapter functions serve as the primary interface for Web, Desktop, and CLI
layers. They handle:
- Validating operation codes and matrix inputs
- Calling appropriate computation functions
- Catching and translating exceptions to user-friendly errors
- Wrapping results in ComputationResult objects
- Computing input signatures for cache invalidation
- Formatting results for display

All functions return ComputationResult objects, never raising exceptions to caller.
"""

from __future__ import annotations

from typing import Any, Optional
import hashlib

import numpy as np

from . import linalg_topics, operations
from .computation_result import ComputationResult
from .error_translator import translate_error
from .operation_registry import get_operation, get_all_operations
from .result_formatter import format_matrix


def _compute_input_signature(
    operation_code: str,
    *matrices: np.ndarray,
) -> tuple:
    """Compute a cache-friendly signature of input matrices and operation.

    Creates a tuple (operation_code, shapes_tuple, data_hash) that uniquely
    identifies the computation inputs for cache detection.

    Args:
        operation_code: Code of the operation (e.g., "add", "multiply")
        matrices: Variable number of input matrices (numpy arrays)

    Returns:
        Tuple (operation_code, shapes_tuple, data_hash) suitable as dict key
    """
    shapes = tuple(m.shape for m in matrices)
    
    # Combine all matrix data for hash
    combined = b""
    for matrix in matrices:
        combined += matrix.tobytes()
    
    data_hash = hashlib.sha256(combined).hexdigest()
    
    return (operation_code, shapes, data_hash)


def execute_binary_operation(
    operation_code: str,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> ComputationResult:
    """Execute a binary operation (operation on two matrices).

    Wraps the computation with error handling and result encapsulation.
    Never raises exceptions to caller; all errors are captured and wrapped
    in the ComputationResult's error state.

    Args:
        operation_code: Code identifier for operation (e.g., "add", "multiply")
        matrix_a: First input matrix (numpy array or array-like)
        matrix_b: Second input matrix (numpy array or array-like)

    Returns:
        ComputationResult with status "success" or "error"

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> b = np.array([[5, 6], [7, 8]])
        >>> result = execute_binary_operation("add", a, b)
        >>> result.status
        'success'
        >>> result.data
        array([[ 6,  8],
               [10, 12]])
    """
    try:
        # Convert inputs to numpy arrays
        matrix_a = np.asarray(matrix_a)
        matrix_b = np.asarray(matrix_b)

        # Validate operation code exists
        operation = get_operation(operation_code)
        if operation is None:
            raise ValueError(f"Operation '{operation_code}' not found in registry")

        # Get the operation function
        func = operation.get("function")
        if func is None or not callable(func):
            raise ValueError(f"Operation '{operation_code}' has no callable function")

        # Execute the operation
        result_data = func(matrix_a, matrix_b)

        # Compute input signature for caching
        input_sig = _compute_input_signature(operation_code, matrix_a, matrix_b)

        # Return success result
        return ComputationResult(
            status="success",
            data=np.asarray(result_data),
            error_message=None,
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
        )

    except operations.MatrixOperationError as exc:
        # Translate matrix operation errors
        # Ensure 2D shapes for context
        try:
            a_2d = np.atleast_2d(matrix_a)
            b_2d = np.atleast_2d(matrix_b)
            matrices_context = {
                "a_shape": a_2d.shape if isinstance(a_2d, np.ndarray) else None,
                "b_shape": b_2d.shape if isinstance(b_2d, np.ndarray) else None,
            }
        except Exception:
            matrices_context = {}
        
        translated = translate_error(exc, operation_code, matrices_context, language="zh")

        try:
            input_sig = _compute_input_signature(operation_code, 
                                                 np.atleast_1d(matrix_a), 
                                                 np.atleast_1d(matrix_b))
        except Exception:
            input_sig = (operation_code,)

        return ComputationResult(
            status="error",
            data=None,
            error_message=translated["message"],
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
            error_type=translated.get("error_type", "computation_failed"),
        )

    except Exception as exc:
        # Catch all other exceptions and wrap in generic error
        translated = translate_error(exc, operation_code, language="zh")

        try:
            # Safely convert to arrays for signature
            try:
                matrix_a_arr = np.asarray(matrix_a) if matrix_a is not None else None
                matrix_b_arr = np.asarray(matrix_b) if matrix_b is not None else None
            except Exception:
                matrix_a_arr = None
                matrix_b_arr = None
            
            if matrix_a_arr is not None and matrix_b_arr is not None:
                input_sig = _compute_input_signature(operation_code, matrix_a_arr, matrix_b_arr)
            else:
                input_sig = (operation_code,)
        except Exception:
            # If signature computation fails, use empty tuple
            input_sig = (operation_code,)

        return ComputationResult(
            status="error",
            data=None,
            error_message=translated["message"],
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
            error_type="computation_failed",
        )


def execute_unary_operation(
    operation_code: str,
    matrix: np.ndarray,
) -> ComputationResult:
    """Execute a unary operation (operation on one matrix).

    Wraps the computation with error handling and result encapsulation.
    Never raises exceptions to caller; all errors are captured and wrapped
    in the ComputationResult's error state.

    Args:
        operation_code: Code identifier for operation (e.g., "determinant", "transpose")
        matrix: Input matrix (numpy array or array-like)

    Returns:
        ComputationResult with status "success" or "error"

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> result = execute_unary_operation("determinant", a)
        >>> result.status
        'success'
        >>> result.data
        -2.0
    """
    try:
        # Convert input to numpy array
        matrix = np.asarray(matrix)

        # Validate operation code exists
        operation = get_operation(operation_code)
        if operation is None:
            raise ValueError(f"Operation '{operation_code}' not found in registry")

        # Get the operation function
        func = operation.get("function")
        if func is None or not callable(func):
            raise ValueError(f"Operation '{operation_code}' has no callable function")

        # Execute the operation
        result_data = func(matrix)

        # Compute input signature for caching
        input_sig = _compute_input_signature(operation_code, matrix)

        # Handle different result types
        if np.isscalar(result_data):
            # Scalar results (determinant) - wrap in array for consistency
            result_data = np.array([result_data])
        elif isinstance(result_data, tuple):
            # Tuple results (eigendecompose, SVD) - keep as tuple
            # Don't wrap, store as-is for format_result_for_display to handle
            pass
        else:
            # Ensure numpy array
            result_data = np.asarray(result_data)

        # Return success result
        return ComputationResult(
            status="success",
            data=result_data,
            error_message=None,
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
        )

    except operations.MatrixOperationError as exc:
        # Translate matrix operation errors
        matrices_context = {
            "a_shape": matrix.shape if isinstance(matrix, np.ndarray) else None,
        }
        translated = translate_error(exc, operation_code, matrices_context, language="zh")

        input_sig = _compute_input_signature(operation_code, matrix)

        return ComputationResult(
            status="error",
            data=None,
            error_message=translated["message"],
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
            error_type=translated.get("error_type", "computation_failed"),
        )

    except Exception as exc:
        # Catch all other exceptions and wrap in generic error
        translated = translate_error(exc, operation_code, language="zh")

        try:
            # Safely convert to array for signature
            try:
                matrix_arr = np.asarray(matrix) if matrix is not None else None
            except Exception:
                matrix_arr = None
            
            if matrix_arr is not None:
                input_sig = _compute_input_signature(operation_code, matrix_arr)
            else:
                input_sig = (operation_code,)
        except Exception:
            # If signature computation fails, use empty tuple
            input_sig = (operation_code,)

        return ComputationResult(
            status="error",
            data=None,
            error_message=translated["message"],
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
            error_type="computation_failed",
        )


def execute_special_operation(
    operation_code: str,
    *args: Any,
    **kwargs: Any,
) -> ComputationResult:
    """Execute special operations with variable arguments (solve_linear, analyze_quadratic).

    These operations have custom signatures beyond binary/unary patterns.
    Never raises exceptions to caller; all errors are captured and wrapped
    in the ComputationResult's error state.

    Args:
        operation_code: Code identifier (e.g., "solve_linear", "analyze_quadratic")
        *args: Positional arguments passed to the operation function
        **kwargs: Keyword arguments passed to the operation function

    Returns:
        ComputationResult with status "success" or "error"

    Example:
        For solve_linear:
        >>> a = np.array([[1, 2], [3, 4]], dtype=float)
        >>> b = np.array([1, 2], dtype=float)
        >>> result = execute_special_operation("solve_linear", a, b)
        >>> result.status
        'success'

        For analyze_quadratic:
        >>> q = np.array([[2, -1], [-1, 2]], dtype=float)
        >>> result = execute_special_operation("analyze_quadratic", q)
        >>> result.status
        'success'
    """
    try:
        # Validate operation code exists
        operation = get_operation(operation_code)
        if operation is None:
            raise ValueError(f"Operation '{operation_code}' not found in registry")

        # Get the operation function
        func = operation.get("function")
        if func is None or not callable(func):
            raise ValueError(f"Operation '{operation_code}' has no callable function")

        # Execute the operation
        result_data = func(*args, **kwargs)

        # Handle solve_linear which returns (status, info_dict)
        if operation_code == "solve_linear" and isinstance(result_data, tuple) and len(result_data) == 2:
            status, info_dict = result_data
            # Merge status into the info dict
            result_data = {"status": status, **info_dict}

        # Extract matrices from args for signature computation
        # Only numpy arrays count toward signature
        matrices = tuple(
            np.asarray(arg) for arg in args
            if isinstance(arg, (np.ndarray, list, tuple))
        )
        input_sig = _compute_input_signature(operation_code, *matrices) if matrices else (operation_code,)

        # Return success result
        return ComputationResult(
            status="success",
            data=result_data,  # May be dict, tuple, or array depending on operation
            error_message=None,
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
        )

    except operations.MatrixOperationError as exc:
        # Translate matrix operation errors
        translated = translate_error(exc, operation_code, language="zh")

        try:
            matrices = tuple(
                np.asarray(arg) for arg in args
                if isinstance(arg, (np.ndarray, list, tuple))
            )
            input_sig = _compute_input_signature(operation_code, *matrices) if matrices else (operation_code,)
        except Exception:
            input_sig = (operation_code,)

        return ComputationResult(
            status="error",
            data=None,
            error_message=translated["message"],
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
            error_type=translated.get("error_type", "computation_failed"),
        )

    except Exception as exc:
        # Catch all other exceptions and wrap in generic error
        translated = translate_error(exc, operation_code, language="zh")

        try:
            matrices = tuple(
                np.asarray(arg) for arg in args
                if isinstance(arg, (np.ndarray, list, tuple))
            )
            input_sig = _compute_input_signature(operation_code, *matrices) if matrices else (operation_code,)
        except Exception:
            input_sig = (operation_code,)

        return ComputationResult(
            status="error",
            data=None,
            error_message=translated["message"],
            operation_code=operation_code,
            input_signature=input_sig,
            cached=False,
            stale=False,
            error_type="computation_failed",
        )


def format_result_for_display(
    computation_result: ComputationResult,
    format_style: str = "text",
    precision: int = 6,
) -> str:
    """Format a ComputationResult for display in UI or CLI.

    Takes a computation result and converts it to a human-readable string
    suitable for desktop or command-line output.

    For success results with numpy array data, formats the matrix using
    result_formatter.format_matrix(). For dict/tuple data (from special
    operations), provides a formatted representation.

    For error results, returns the user-friendly error message.

    Args:
        computation_result: ComputationResult object to format
        format_style: Style for matrix formatting: "text" (Unicode), "latex", or "plain"
        precision: Significant figures for scalar values (default 6)

    Returns:
        Display-ready string formatted for end users

    Raises:
        ValueError: If format_style is not recognized
        TypeError: If computation_result is not a ComputationResult

    Example:
        For a successful computation:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> b = np.array([[5, 6], [7, 8]])
        >>> result = execute_binary_operation("add", a, b)
        >>> display_str = format_result_for_display(result, format_style="text")
        >>> print(display_str)
        ⎡  6   8 ⎤
        ⎣ 10  12 ⎦

        For an error result:
        >>> result = execute_binary_operation("add", a, np.zeros((3, 3)))
        >>> display_str = format_result_for_display(result)
        >>> print(display_str)
        Error: 两个待相加矩阵的 shape 不一致，无法操作
    """
    if not isinstance(computation_result, ComputationResult):
        raise TypeError(
            f"Expected ComputationResult, got {type(computation_result).__name__}"
        )

    if format_style not in ("text", "latex", "plain"):
        raise ValueError(f"Unknown format style: {format_style}")

    # For error results, return the error message
    if computation_result.status == "error":
        return computation_result.get_user_message()

    # For success results, format the data
    if computation_result.data is None:
        return "No data"

    # Handle dict results (from special operations like solve_linear, analyze_quadratic)
    if isinstance(computation_result.data, dict):
        # Format dictionary-based results
        lines = [f"Operation: {computation_result.operation_code}"]
        for key, value in computation_result.data.items():
            if isinstance(value, np.ndarray):
                lines.append(f"{key}:")
                lines.append("  " + format_matrix(value, style=format_style, precision=precision))
            elif isinstance(value, tuple):
                lines.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {str(value)[:100]}")
        return "\n".join(lines)

    # Handle tuple results (e.g., eigendecompose returns (eigenvalues, eigenvectors))
    if isinstance(computation_result.data, tuple):
        lines = [f"Operation: {computation_result.operation_code}"]
        for i, item in enumerate(computation_result.data):
            if isinstance(item, np.ndarray):
                # Convert tuple of arrays to list of arrays for display
                if item.ndim == 1:
                    lines.append(f"Result {i}: {format_matrix(item.reshape(-1, 1), style=format_style, precision=precision)}")
                else:
                    lines.append(f"Result {i}:")
                    lines.append("  " + format_matrix(item, style=format_style, precision=precision))
            else:
                lines.append(f"Result {i}: {item}")
        return "\n".join(lines)

    # Handle scalar results (wrapped in single-element array by execute_unary_operation)
    if isinstance(computation_result.data, np.ndarray) and computation_result.data.size == 1:
        value = computation_result.data.item()
        if np.isscalar(value):
            return f"{value:.{precision}g}"

    # Handle regular numpy array results
    if isinstance(computation_result.data, np.ndarray):
        try:
            return format_matrix(
                computation_result.data,
                style=format_style,
                precision=precision,
            )
        except (ValueError, TypeError) as exc:
            # If formatting fails (e.g., NaN/Inf values), return array representation
            return f"Result:\n{computation_result.data}"

    # Fallback for other data types
    return f"Result: {computation_result.data}"


__all__ = [
    "execute_binary_operation",
    "execute_unary_operation",
    "execute_special_operation",
    "format_result_for_display",
]
