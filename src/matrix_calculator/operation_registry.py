"""Unified Operation Registry — centralized operation definitions for all UI layers.

This module provides a single source of truth for all matrix operations available
in the calculator. Each operation is defined once with its metadata (name, category,
UI label, computation function, help text), enabling web, desktop, and CLI to share
consistent operation definitions without duplication.

Operations are organized by category:
- binary: operations requiring two matrices (add, subtract, multiply, etc.)
- unary: operations on a single matrix (determinant, transpose, etc.)
- solve: solving linear systems
- quadratic: quadratic form analysis
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import numpy as np

from . import linalg_topics, operations


# Operation metadata structure
OperationCategory = Literal["binary", "unary", "solve", "quadratic"]


def _get_operation_function(module_path: str, func_name: str) -> Callable | None:
    """Safely retrieve a function from operations or linalg_topics modules.
    
    Args:
        module_path: Either "operations" or "linalg_topics"
        func_name: Name of the function to retrieve
        
    Returns:
        The callable function, or None if not found
    """
    if module_path == "operations":
        return getattr(operations, func_name, None)
    elif module_path == "linalg_topics":
        return getattr(linalg_topics, func_name, None)
    return None


# Central operation registry
OPERATIONS: dict[str, dict[str, Any]] = {
    # Binary Operations
    "add": {
        "code": "add",
        "category": "binary",
        "ui_label": "A + B",
        "function": operations.add_matrices,
        "help_text": "Element-wise addition of two matrices with identical dimensions.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    "subtract": {
        "code": "subtract",
        "category": "binary",
        "ui_label": "A − B",
        "function": operations.subtract_matrices,
        "help_text": "Element-wise subtraction of two matrices with identical dimensions.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    "multiply": {
        "code": "multiply",
        "category": "binary",
        "ui_label": "A × B",
        "function": operations.matrix_multiply,
        "help_text": "Matrix multiplication: columns of A must equal rows of B.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    "hadamard_multiply": {
        "code": "hadamard_multiply",
        "category": "binary",
        "ui_label": "A ⊙ B",
        "function": operations.hadamard_multiply,
        "help_text": "Element-wise (Hadamard) multiplication of two matrices with identical dimensions.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    "hadamard_divide": {
        "code": "hadamard_divide",
        "category": "binary",
        "ui_label": "A ⊘ B",
        "function": operations.hadamard_divide,
        "help_text": "Element-wise division of two matrices; divisor matrix must not contain zeros.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    # Unary Operations
    "determinant": {
        "code": "determinant",
        "category": "unary",
        "ui_label": "det(A)",
        "function": operations.determinant,
        "help_text": "Determinant of a square matrix; measures invertibility and scaling factor.",
        "requires_square_matrix": True,
        "visible_in": ["web", "desktop", "cli"],
    },
    "inverse": {
        "code": "inverse",
        "category": "unary",
        "ui_label": "A⁻¹",
        "function": operations.invert_matrix,
        "help_text": "Inverse of a square non-singular matrix; raises error if matrix is singular.",
        "requires_square_matrix": True,
        "visible_in": ["web", "desktop", "cli"],
    },
    "transpose": {
        "code": "transpose",
        "category": "unary",
        "ui_label": "Aᵀ",
        "function": operations.transpose,
        "help_text": "Transpose of a matrix: rows become columns and columns become rows.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    "eigendecompose": {
        "code": "eigendecompose",
        "category": "unary",
        "ui_label": "λ",
        "function": operations.eigendecompose,
        "help_text": "Eigendecomposition of a square matrix; returns eigenvalues and eigenvectors.",
        "requires_square_matrix": True,
        "visible_in": ["web", "desktop", "cli"],
    },
    "singular_value_decompose": {
        "code": "singular_value_decompose",
        "category": "unary",
        "ui_label": "SVD",
        "function": operations.singular_value_decompose,
        "help_text": "Singular Value Decomposition; decomposes any matrix into orthogonal and diagonal components.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    # Solve Category
    "solve_linear": {
        "code": "solve_linear",
        "category": "solve",
        "ui_label": "Solve Ax=b",
        "function": linalg_topics.solve_linear,
        "help_text": "Solve the linear system Ax=b; detects unique, infinite, or no solutions.",
        "requires_square_matrix": False,
        "visible_in": ["web", "desktop", "cli"],
    },
    # Quadratic Category
    "analyze_quadratic": {
        "code": "analyze_quadratic",
        "category": "quadratic",
        "ui_label": "Quadratic Form",
        "function": linalg_topics.analyze_quadratic,
        "help_text": "Analyze quadratic form xᵀAx: eigenvalues, standard form, and definiteness classification.",
        "requires_square_matrix": True,
        "visible_in": ["web", "desktop", "cli"],
    },
}


def get_operation(code: str) -> dict[str, Any] | None:
    """Retrieve a single operation by its code.
    
    Args:
        code: Operation code (e.g., "add", "determinant")
        
    Returns:
        Operation metadata dict, or None if not found
    """
    return OPERATIONS.get(code)


def get_operations(category: Optional[OperationCategory] = None) -> list[dict[str, Any]]:
    """Retrieve all operations, optionally filtered by category.
    
    Args:
        category: Optional category to filter by ("binary", "unary", "solve", "quadratic")
                 If None, returns all operations.
        
    Returns:
        List of operation metadata dicts
    """
    if category is None:
        return list(OPERATIONS.values())
    return [op for op in OPERATIONS.values() if op.get("category") == category]


def get_all_operations() -> dict[str, dict[str, Any]]:
    """Return the entire operation registry.
    
    Returns:
        Dict mapping operation codes to operation metadata
    """
    return OPERATIONS.copy()


def validate_registry() -> bool:
    """Validate the registry structure and integrity.
    
    Checks:
    - All operations have required fields (code, category, ui_label, function, help_text)
    - Operation codes are valid (snake_case, 1-32 chars)
    - Categories are valid
    - UI labels are ≤100 chars
    - Help texts are ≤200 chars
    - All function references are callable
    - No duplicate UI labels within a category
    
    Returns:
        True if registry is valid, raises ValueError otherwise
    """
    required_fields = {"code", "category", "ui_label", "function", "help_text"}
    valid_categories = {"binary", "unary", "solve", "quadratic"}
    
    for code, operation in OPERATIONS.items():
        # Check required fields
        missing = required_fields - set(operation.keys())
        if missing:
            raise ValueError(
                f"Operation '{code}' missing required fields: {missing}"
            )
        
        # Validate code format (snake_case, 1-32 chars)
        if not (1 <= len(code) <= 32):
            raise ValueError(f"Operation code '{code}' must be 1-32 characters")
        if not code.replace("_", "").isalnum():
            raise ValueError(f"Operation code '{code}' must be alphanumeric with underscores")
        if code != code.lower():
            raise ValueError(f"Operation code '{code}' must be lowercase")
        
        # Validate category
        if operation["category"] not in valid_categories:
            raise ValueError(
                f"Operation '{code}' has invalid category '{operation['category']}'. "
                f"Must be one of: {valid_categories}"
            )
        
        # Validate UI label length
        if len(operation["ui_label"]) > 100:
            raise ValueError(
                f"Operation '{code}' ui_label exceeds 100 chars: "
                f"{len(operation['ui_label'])} chars"
            )
        
        # Validate help text length
        if len(operation["help_text"]) > 200:
            raise ValueError(
                f"Operation '{code}' help_text exceeds 200 chars: "
                f"{len(operation['help_text'])} chars"
            )
        
        # Validate function is callable
        if operation["function"] is not None and not callable(operation["function"]):
            raise ValueError(
                f"Operation '{code}' function is not callable: {type(operation['function'])}"
            )
    
    # Check for duplicate UI labels within categories
    category_labels: dict[str, set[str]] = {}
    for operation in OPERATIONS.values():
        cat = operation["category"]
        label = operation["ui_label"]
        if cat not in category_labels:
            category_labels[cat] = set()
        if label in category_labels[cat]:
            raise ValueError(
                f"Duplicate UI label '{label}' in category '{cat}'"
            )
        category_labels[cat].add(label)
    
    return True


def get_operations_by_ui_layer(layer: str) -> list[dict[str, Any]]:
    """Retrieve operations visible in a specific UI layer.
    
    Args:
        layer: UI layer name ("web", "desktop", or "cli")
        
    Returns:
        List of operation metadata dicts visible in the specified layer
    """
    return [
        op for op in OPERATIONS.values()
        if layer in op.get("visible_in", ["web", "desktop", "cli"])
    ]
