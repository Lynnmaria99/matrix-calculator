"""矩阵计算器：可复用的计算核心与命令行界面。

Package structure overview:
- Computation Core: Pure functions for matrix operations (add, multiply, etc.)
  These are located in `operations.py` and `linalg_topics.py`.
- Shared Adapter Modules (Phase 1): Web/Desktop/CLI coordination layer
  These modules extract common patterns and minimize cross-layer coupling:
  * computation_result: Unified result representation across layers
  * result_formatter: Consistent formatting (text, CSV, LaTeX, numpy)
  * input_validator: Cell/matrix parsing and validation
  * dimension_manager: Dimension constraint management
  * operation_registry: Centralized operation definitions
  * error_translator: Unified error message translation
  * result_cache: Session-local caching with LRU eviction
  * computation_adapter: Coordinated computation execution and formatting
- Legacy Compatibility: MatCal/MatCom wrapper classes (deprecated in Phase 2)
  Used only for backward compatibility during transition.

Recommended import patterns for new code:
  from matrix_calculator import execute_binary_operation, format_result_for_display
  from matrix_calculator import ResultCache, compute_input_signature
  from matrix_calculator import translate_error
"""

from .constants import EPS
from .dimension_manager import (
    DimensionConfig,
    clamp_to_config,
    get_config,
    get_help_text,
    validate_against_config,
)
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
from .linalg_topics import (
    analyze_quadratic,
    classify_quadratic,
    rank_and_nullspace,
    solve_linear,
)
from .formatting import format_matrix, print_matrix, show_two_results, signed_polynomial
from .calculus import (
    ExpressionError,
    parse_expression,
    derivative_value,
    simpson_integral,
    riemann_midpoint,
)

# Phase 1: Shared adapter modules
from .computation_adapter import (
    execute_binary_operation,
    execute_unary_operation,
    execute_special_operation,
    format_result_for_display,
)
from .computation_result import ComputationResult
from .error_translator import translate_error
from .input_validator import (
    parse_cell,
    clamp_dimensions,
    validate_dimensions,
    DimensionConfig as ValidatorDimensionConfig,
)
from .operation_registry import (
    get_operations,
    get_operation,
    OPERATIONS,
)
from .result_cache import ResultCache, compute_input_signature
from .result_formatter import (
    export_to_csv,
    export_to_numpy_text,
    export_to_latex,
)

# Note: DimensionConfig is imported from both dimension_manager and input_validator
# They are the same class, so we use the one from dimension_manager (already above)

__all__ = [
    # Constants
    "EPS",
    # Dimension management (Phase 1)
    "DimensionConfig",
    "clamp_to_config",
    "get_config",
    "get_help_text",
    "validate_against_config",
    # Computation core (legacy)
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
    # Linear algebra topics (legacy)
    "rank_and_nullspace",
    "solve_linear",
    "classify_quadratic",
    "analyze_quadratic",
    # Formatting (legacy)
    "format_matrix",
    "print_matrix",
    "show_two_results",
    "signed_polynomial",
    # Calculus (legacy)
    "ExpressionError",
    "parse_expression",
    "derivative_value",
    "simpson_integral",
    "riemann_midpoint",
    # Shared adapter modules (Phase 1)
    "execute_binary_operation",
    "execute_unary_operation",
    "execute_special_operation",
    "format_result_for_display",
    "ComputationResult",
    "translate_error",
    "parse_cell",
    "clamp_dimensions",
    "validate_dimensions",
    "get_operations",
    "get_operation",
    "OPERATIONS",
    "ResultCache",
    "compute_input_signature",
    "export_to_csv",
    "export_to_numpy_text",
    "export_to_latex",
]
