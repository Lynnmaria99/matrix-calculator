"""Unified computation result data structure for all computations.

This module defines the ComputationResult dataclass that encapsulates the
outcome of any computation (success or failure), along with metadata for
caching, staleness detection, and error context. It serves as the common
interface returned by computation functions across web, desktop, and CLI layers.

The ComputationResult is framework-agnostic and serializable, making it
suitable for logging, debugging, and cross-layer communication.

Example:
    For a successful matrix addition:
    >>> from numpy import array
    >>> result = ComputationResult(
    ...     status="success",
    ...     data=array([[2, 3], [5, 7]]),
    ...     error_message=None,
    ...     operation_code="add",
    ...     input_signature=("add", (1, 2), "abc123"),
    ...     cached=False,
    ...     stale=False,
    ... )
    >>> result.get_user_message()
    'Operation successful'

    For a failed computation:
    >>> error_result = ComputationResult(
    ...     status="error",
    ...     data=None,
    ...     error_message="Matrix dimensions must match for element-wise addition",
    ...     operation_code="add",
    ...     input_signature=("add", (1, 2), "abc123"),
    ...     cached=False,
    ...     stale=False,
    ...     error_type="dimension_mismatch",
    ... )
    >>> error_result.get_user_message()
    'Error: Matrix dimensions must match for element-wise addition'
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Any
import numpy as np
import json
import time


@dataclass
class ComputationResult:
    """Unified result object for all computations.

    Encapsulates the outcome of a computation (success or failure) with
    metadata for caching, staleness detection, and error context.

    Attributes:
        status: Computation outcome, either "success" or "error".
        data: The result matrix/array if status is "success", None otherwise.
        error_message: Human-readable error message if status is "error", None otherwise.
        operation_code: Code identifier for the operation (e.g., "add", "multiply", "determinant").
        input_signature: Tuple containing (operation_code, input_ids, data_hash) for cache detection.
        cached: Whether this result was retrieved from cache.
        stale: Whether this cached result is outdated due to input changes.
        error_type: Structured error type if status is "error". One of:
            - "dimension_mismatch": Matrices have different shapes
            - "singular_matrix": Matrix has zero determinant, not invertible
            - "invalid_input": Non-numeric or malformed input
            - "computation_failed": Computation raised an unexpected error
            - "validation_error": Input validation failed
            Or None if status is "success".
        timestamp: Creation time (seconds since epoch) for cache expiry logic.
    """

    status: Literal["success", "error"]
    data: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    operation_code: str = ""
    input_signature: tuple = field(default_factory=tuple)
    cached: bool = False
    stale: bool = False
    error_type: Optional[
        Literal[
            "dimension_mismatch",
            "singular_matrix",
            "invalid_input",
            "computation_failed",
            "validation_error",
        ]
    ] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate ComputationResult constraints.

        Enforces that:
        - If status == "success": error_message and error_type must be None, data must be set
        - If status == "error": data must be None
        - operation_code and input_signature are non-empty
        - timestamp is positive

        Raises:
            ValueError: If constraints are violated.
        """
        if self.status == "success":
            if self.error_message is not None:
                raise ValueError(
                    "Success result cannot have error_message; must be None"
                )
            if self.error_type is not None:
                raise ValueError(
                    "Success result cannot have error_type; must be None"
                )
            if self.data is None:
                raise ValueError(
                    "Success result must have data (numpy array); cannot be None"
                )
        elif self.status == "error":
            if self.data is not None:
                raise ValueError(
                    "Error result cannot have data; must be None"
                )
            if self.error_message is None:
                raise ValueError(
                    "Error result must have error_message"
                )
        else:
            raise ValueError(f"Invalid status: {self.status}; must be 'success' or 'error'")

        if not isinstance(self.operation_code, str) or len(self.operation_code) == 0:
            raise ValueError(
                f"operation_code must be a non-empty string; got {self.operation_code!r}"
            )

        if not isinstance(self.input_signature, tuple) or len(self.input_signature) == 0:
            raise ValueError(
                f"input_signature must be a non-empty tuple; got {self.input_signature!r}"
            )

        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative; got {self.timestamp}")

    def to_json(
        self,
        include_data: bool = False,
        include_input_signature: bool = False,
    ) -> dict[str, Any]:
        """Serialize ComputationResult to JSON-compatible dictionary.

        Converts the dataclass to a dictionary suitable for JSON serialization.
        By default, excludes large numpy arrays and input signatures to keep
        JSON size manageable for logging or debugging.

        Args:
            include_data: If True, attempt to serialize the numpy array as a nested list.
                If False (default), omit the data field.
            include_input_signature: If True, include the input_signature tuple.
                If False (default), omit for brevity.

        Returns:
            Dictionary with keys: status, error_message, operation_code, cached, stale,
            error_type, timestamp, and optionally data and input_signature.

        Example:
            >>> result = ComputationResult(
            ...     status="success",
            ...     data=np.array([[1, 2]]),
            ...     error_message=None,
            ...     operation_code="add",
            ...     input_signature=("add", (1, 2), "hash"),
            ...     cached=False,
            ...     stale=False,
            ... )
            >>> result.to_json(include_data=False)
            {
                'status': 'success',
                'error_message': None,
                'operation_code': 'add',
                'cached': False,
                'stale': False,
                'error_type': None,
                'timestamp': 1234567890.5
            }
        """
        json_dict = {
            "status": self.status,
            "error_message": self.error_message,
            "operation_code": self.operation_code,
            "cached": self.cached,
            "stale": self.stale,
            "error_type": self.error_type,
            "timestamp": self.timestamp,
        }

        if include_data and self.data is not None:
            # Convert numpy array to nested list for JSON compatibility
            json_dict["data"] = self.data.tolist()

        if include_input_signature:
            json_dict["input_signature"] = self.input_signature

        return json_dict

    def is_fresh(self, max_age_seconds: float = 300.0) -> bool:
        """Check if a cached result is still fresh (not expired).

        Used to determine whether a cached ComputationResult should be reused
        or if recomputation is needed. Results older than max_age_seconds are
        considered stale.

        Args:
            max_age_seconds: Maximum age in seconds before a result is considered stale.
                Defaults to 300 seconds (5 minutes).

        Returns:
            True if the result is cached and still fresh, False otherwise.
            Returns False immediately if not cached or already marked stale.

        Example:
            >>> import time
            >>> result = ComputationResult(
            ...     status="success",
            ...     data=np.array([[1]]),
            ...     operation_code="add",
            ...     input_signature=("add", (1, 1), "hash"),
            ...     cached=True,
            ...     stale=False,
            ...     timestamp=time.time() - 60,  # 60 seconds old
            ... )
            >>> result.is_fresh()
            True
            >>> result.is_fresh(max_age_seconds=30)
            False
        """
        if not self.cached or self.stale:
            return False

        age = time.time() - self.timestamp
        return age <= max_age_seconds

    def get_user_message(self) -> str:
        """Generate a user-friendly message from the computation result.

        Formats the result into a concise, readable message suitable for
        display in UI layers (Streamlit, PySide6, CLI, etc.). For success,
        indicates "Operation successful". For errors, prefixes the error
        message with "Error: ".

        Returns:
            A formatted string message for end-user display.

        Example:
            >>> result = ComputationResult(
            ...     status="success",
            ...     data=np.array([[1]]),
            ...     operation_code="add",
            ...     input_signature=("add", (1, 1), "hash"),
            ... )
            >>> result.get_user_message()
            'Operation successful'

            >>> error_result = ComputationResult(
            ...     status="error",
            ...     error_message="Dimension mismatch",
            ...     operation_code="add",
            ...     input_signature=("add", (1, 2), "hash"),
            ...     error_type="dimension_mismatch",
            ... )
            >>> error_result.get_user_message()
            'Error: Dimension mismatch'
        """
        if self.status == "success":
            return "Operation successful"
        else:
            return f"Error: {self.error_message}"

    def __repr__(self) -> str:
        """Return a detailed string representation of ComputationResult.

        Provides a programmer-friendly representation showing all fields.
        For large numpy arrays, shows shape instead of full content.

        Returns:
            A detailed string representation for debugging.
        """
        data_repr = (
            f"ndarray(shape={self.data.shape}, dtype={self.data.dtype})"
            if self.data is not None
            else "None"
        )
        return (
            f"ComputationResult(status={self.status!r}, "
            f"data={data_repr}, "
            f"error_message={self.error_message!r}, "
            f"operation_code={self.operation_code!r}, "
            f"input_signature={self.input_signature!r}, "
            f"cached={self.cached}, "
            f"stale={self.stale}, "
            f"error_type={self.error_type!r}, "
            f"timestamp={self.timestamp})"
        )

    def __eq__(self, other: Any) -> bool:
        """Compare two ComputationResult objects for equality.

        Two ComputationResult objects are equal if all their fields are equal.
        For numpy arrays, uses np.array_equal() for comparison.

        Args:
            other: Another object to compare with.

        Returns:
            True if all fields are equal, False otherwise.

        Example:
            >>> r1 = ComputationResult(
            ...     status="success",
            ...     data=np.array([[1, 2]]),
            ...     operation_code="add",
            ...     input_signature=("add", (1, 2), "hash"),
            ... )
            >>> r2 = ComputationResult(
            ...     status="success",
            ...     data=np.array([[1, 2]]),
            ...     operation_code="add",
            ...     input_signature=("add", (1, 2), "hash"),
            ... )
            >>> r1 == r2
            True
        """
        if not isinstance(other, ComputationResult):
            return False

        # Compare all fields except data (which needs special numpy comparison)
        data_equal = (
            (self.data is None and other.data is None) or
            (self.data is not None and other.data is not None and np.array_equal(self.data, other.data))
        )

        return (
            self.status == other.status
            and data_equal
            and self.error_message == other.error_message
            and self.operation_code == other.operation_code
            and self.input_signature == other.input_signature
            and self.cached == other.cached
            and self.stale == other.stale
            and self.error_type == other.error_type
        )
