"""Unit tests for the ComputationResult dataclass.

Tests cover:
- Successful computation result creation and properties
- Error computation result creation and properties
- Validation constraints in __post_init__
- JSON serialization with various options
- Cache freshness detection
- User-friendly message generation
- Equality comparison with numpy arrays
- String representation
"""

import pytest
import numpy as np
import time
from matrix_calculator.computation_result import ComputationResult


class TestComputationResultSuccess:
    """Tests for successful computation results."""

    def test_create_success_result_with_defaults(self):
        """Test creating a basic successful result."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="add",
            input_signature=("add", (2, 2), "hash123"),
        )

        assert result.status == "success"
        assert np.array_equal(result.data, data)
        assert result.error_message is None
        assert result.error_type is None
        assert result.cached is False
        assert result.stale is False
        assert result.operation_code == "add"
        assert result.input_signature == ("add", (2, 2), "hash123")

    def test_create_success_result_with_all_fields(self):
        """Test creating a successful result with all fields specified."""
        data = np.array([[5.0, 6.0]])
        timestamp = time.time() - 100
        result = ComputationResult(
            status="success",
            data=data,
            error_message=None,
            operation_code="multiply",
            input_signature=("multiply", (2, 3, 3, 2), "abc"),
            cached=True,
            stale=False,
            error_type=None,
            timestamp=timestamp,
        )

        assert result.status == "success"
        assert np.array_equal(result.data, data)
        assert result.cached is True
        assert result.stale is False
        assert result.timestamp == timestamp

    def test_success_result_with_complex_matrix(self):
        """Test success result with complex-valued matrix."""
        data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="eigendecomposition",
            input_signature=("eigendecomposition", (2, 2), "xyz"),
        )

        assert result.status == "success"
        assert np.array_equal(result.data, data)
        assert result.data.dtype == np.complex128

    def test_success_result_with_scalar(self):
        """Test success result with scalar (1x1 array)."""
        data = np.array([[42.0]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="determinant",
            input_signature=("determinant", (3, 3), "det_hash"),
        )

        assert result.status == "success"
        assert result.data.shape == (1, 1)

    def test_success_result_with_large_matrix(self):
        """Test success result with large matrix."""
        data = np.random.rand(100, 100)
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="svd",
            input_signature=("svd", (100, 100), "large_hash"),
        )

        assert result.status == "success"
        assert result.data.shape == (100, 100)


class TestComputationResultError:
    """Tests for error computation results."""

    def test_create_error_result(self):
        """Test creating a basic error result."""
        result = ComputationResult(
            status="error",
            error_message="Matrix dimensions must match",
            operation_code="add",
            input_signature=("add", (2, 2), "hash123"),
            error_type="dimension_mismatch",
        )

        assert result.status == "error"
        assert result.data is None
        assert result.error_message == "Matrix dimensions must match"
        assert result.error_type == "dimension_mismatch"
        assert result.cached is False
        assert result.stale is False

    def test_create_error_result_with_all_error_types(self):
        """Test error result with each valid error_type."""
        error_types = [
            "dimension_mismatch",
            "singular_matrix",
            "invalid_input",
            "computation_failed",
            "validation_error",
        ]

        for error_type in error_types:
            result = ComputationResult(
                status="error",
                error_message=f"Error of type {error_type}",
                operation_code="test",
                input_signature=("test", (), "hash"),
                error_type=error_type,
            )
            assert result.error_type == error_type

    def test_error_result_without_error_type(self):
        """Test error result with error_type=None (optional)."""
        result = ComputationResult(
            status="error",
            error_message="Some error occurred",
            operation_code="test",
            input_signature=("test", (), "hash"),
            error_type=None,
        )

        assert result.status == "error"
        assert result.error_type is None
        assert result.error_message == "Some error occurred"


class TestComputationResultValidation:
    """Tests for __post_init__ validation constraints."""

    def test_success_with_error_message_raises_error(self):
        """Test that success status with error_message raises ValueError."""
        with pytest.raises(ValueError, match="Success result cannot have error_message"):
            ComputationResult(
                status="success",
                data=np.array([[1]]),
                error_message="Should not have this",
                operation_code="add",
                input_signature=("add", (), "hash"),
            )

    def test_success_with_error_type_raises_error(self):
        """Test that success status with error_type raises ValueError."""
        with pytest.raises(ValueError, match="Success result cannot have error_type"):
            ComputationResult(
                status="success",
                data=np.array([[1]]),
                error_type="dimension_mismatch",
                operation_code="add",
                input_signature=("add", (), "hash"),
            )

    def test_success_without_data_raises_error(self):
        """Test that success status requires data."""
        with pytest.raises(ValueError, match="Success result must have data"):
            ComputationResult(
                status="success",
                data=None,
                operation_code="add",
                input_signature=("add", (), "hash"),
            )

    def test_error_with_data_raises_error(self):
        """Test that error status cannot have data."""
        with pytest.raises(ValueError, match="Error result cannot have data"):
            ComputationResult(
                status="error",
                data=np.array([[1]]),
                error_message="Error",
                operation_code="add",
                input_signature=("add", (), "hash"),
            )

    def test_error_without_error_message_raises_error(self):
        """Test that error status requires error_message."""
        with pytest.raises(ValueError, match="Error result must have error_message"):
            ComputationResult(
                status="error",
                error_message=None,
                operation_code="add",
                input_signature=("add", (), "hash"),
            )

    def test_invalid_status_raises_error(self):
        """Test that invalid status value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            ComputationResult(
                status="unknown",
                data=np.array([[1]]),
                operation_code="add",
                input_signature=("add", (), "hash"),
            )

    def test_empty_operation_code_raises_error(self):
        """Test that empty operation_code raises ValueError."""
        with pytest.raises(ValueError, match="operation_code must be a non-empty string"):
            ComputationResult(
                status="success",
                data=np.array([[1]]),
                operation_code="",
                input_signature=("add", (), "hash"),
            )

    def test_non_string_operation_code_raises_error(self):
        """Test that non-string operation_code raises ValueError."""
        with pytest.raises(ValueError, match="operation_code must be a non-empty string"):
            ComputationResult(
                status="success",
                data=np.array([[1]]),
                operation_code=123,
                input_signature=("add", (), "hash"),
            )

    def test_empty_input_signature_raises_error(self):
        """Test that empty input_signature raises ValueError."""
        with pytest.raises(ValueError, match="input_signature must be a non-empty tuple"):
            ComputationResult(
                status="success",
                data=np.array([[1]]),
                operation_code="add",
                input_signature=(),
            )

    def test_non_tuple_input_signature_raises_error(self):
        """Test that non-tuple input_signature raises ValueError."""
        with pytest.raises(ValueError, match="input_signature must be a non-empty tuple"):
            ComputationResult(
                status="success",
                data=np.array([[1]]),
                operation_code="add",
                input_signature=["add", (), "hash"],
            )

    def test_negative_timestamp_raises_error(self):
        """Test that negative timestamp raises ValueError."""
        with pytest.raises(ValueError, match="timestamp must be non-negative"):
            ComputationResult(
                status="success",
                data=np.array([[1]]),
                operation_code="add",
                input_signature=("add", (), "hash"),
                timestamp=-1.0,
            )


class TestToJson:
    """Tests for to_json() method."""

    def test_to_json_without_data(self):
        """Test to_json() excludes data by default."""
        result = ComputationResult(
            status="success",
            data=np.array([[1, 2], [3, 4]]),
            operation_code="add",
            input_signature=("add", (2, 2), "hash"),
        )
        json_dict = result.to_json(include_data=False)

        assert json_dict["status"] == "success"
        assert "data" not in json_dict
        assert json_dict["operation_code"] == "add"
        assert json_dict["cached"] is False

    def test_to_json_with_data(self):
        """Test to_json() includes data as nested list when requested."""
        data = np.array([[1, 2], [3, 4]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="add",
            input_signature=("add", (2, 2), "hash"),
        )
        json_dict = result.to_json(include_data=True)

        assert json_dict["status"] == "success"
        assert "data" in json_dict
        assert json_dict["data"] == [[1, 2], [3, 4]]

    def test_to_json_with_input_signature(self):
        """Test to_json() includes input_signature when requested."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (2, 2), "hash123"),
        )
        json_dict = result.to_json(include_input_signature=True)

        assert "input_signature" in json_dict
        assert json_dict["input_signature"] == ("add", (2, 2), "hash123")

    def test_to_json_without_input_signature(self):
        """Test to_json() excludes input_signature by default."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (2, 2), "hash123"),
        )
        json_dict = result.to_json(include_input_signature=False)

        assert "input_signature" not in json_dict

    def test_to_json_error_result(self):
        """Test to_json() with error result."""
        result = ComputationResult(
            status="error",
            error_message="Dimension mismatch",
            operation_code="add",
            input_signature=("add", (2, 3), "hash"),
            error_type="dimension_mismatch",
        )
        json_dict = result.to_json()

        assert json_dict["status"] == "error"
        assert json_dict["error_message"] == "Dimension mismatch"
        assert json_dict["error_type"] == "dimension_mismatch"
        assert "data" not in json_dict  # By default, data is excluded

    def test_to_json_with_complex_data(self):
        """Test to_json() converts complex arrays to nested lists."""
        data = np.array([[1 + 2j, 3 + 4j]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="eigendecomposition",
            input_signature=("eigendecomposition", (2, 2), "hash"),
        )
        json_dict = result.to_json(include_data=True)

        # Complex numbers are converted to lists in JSON
        assert json_dict["data"] == [[(1 + 2j), (3 + 4j)]]

    def test_to_json_all_fields(self):
        """Test to_json() includes all relevant fields."""
        result = ComputationResult(
            status="success",
            data=np.array([[1, 2]]),
            operation_code="add",
            input_signature=("add", (1, 2), "hash"),
            cached=True,
            stale=False,
        )
        json_dict = result.to_json(include_data=True, include_input_signature=True)

        required_fields = {
            "status", "error_message", "operation_code", "cached", 
            "stale", "error_type", "timestamp", "data", "input_signature"
        }
        assert set(json_dict.keys()) == required_fields


class TestIsFresh:
    """Tests for is_fresh() cache freshness detection."""

    def test_fresh_cached_result(self):
        """Test that recently cached result is fresh."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=True,
            stale=False,
            timestamp=time.time() - 30,  # 30 seconds old
        )

        assert result.is_fresh(max_age_seconds=60)

    def test_stale_cached_result(self):
        """Test that old cached result is not fresh."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=True,
            stale=False,
            timestamp=time.time() - 400,  # 400 seconds old
        )

        assert not result.is_fresh(max_age_seconds=300)

    def test_not_fresh_if_not_cached(self):
        """Test that non-cached result is never fresh."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=False,
            stale=False,
            timestamp=time.time(),
        )

        assert not result.is_fresh()

    def test_not_fresh_if_stale_flag_set(self):
        """Test that result with stale=True is not fresh."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=True,
            stale=True,
            timestamp=time.time() - 10,
        )

        assert not result.is_fresh()

    def test_boundary_exactly_at_max_age(self):
        """Test result that is exactly at or slightly less than max age threshold."""
        max_age = 300.0
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=True,
            stale=False,
            timestamp=time.time() - (max_age - 0.5),  # Slightly less than max_age
        )

        # Should be fresh because age is slightly less than max_age
        assert result.is_fresh(max_age_seconds=max_age)

    def test_boundary_just_past_max_age(self):
        """Test result that is just past max age threshold."""
        max_age = 300.0
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=True,
            stale=False,
            timestamp=time.time() - (max_age + 0.1),
        )

        assert not result.is_fresh(max_age_seconds=max_age)

    def test_is_fresh_with_custom_max_age(self):
        """Test is_fresh() with custom max_age_seconds parameter."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=True,
            stale=False,
            timestamp=time.time() - 100,
        )

        assert result.is_fresh(max_age_seconds=200)
        assert not result.is_fresh(max_age_seconds=50)


class TestGetUserMessage:
    """Tests for get_user_message() user-friendly formatting."""

    def test_success_message(self):
        """Test user message for successful result."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
        )

        assert result.get_user_message() == "Operation successful"

    def test_error_message_with_context(self):
        """Test user message for error result."""
        result = ComputationResult(
            status="error",
            error_message="Matrix dimensions do not match for addition",
            operation_code="add",
            input_signature=("add", (2, 3), "hash"),
            error_type="dimension_mismatch",
        )

        assert result.get_user_message() == "Error: Matrix dimensions do not match for addition"

    def test_error_message_formatting(self):
        """Test that error message is prefixed with 'Error: '."""
        error_msg = "Custom error description"
        result = ComputationResult(
            status="error",
            error_message=error_msg,
            operation_code="test",
            input_signature=("test", (), "hash"),
        )

        assert result.get_user_message() == f"Error: {error_msg}"

    def test_error_message_with_special_characters(self):
        """Test user message with special characters in error."""
        error_msg = "Error: A (2×3) × B (2×2) - dimensions mismatch!"
        result = ComputationResult(
            status="error",
            error_message=error_msg,
            operation_code="multiply",
            input_signature=("multiply", (), "hash"),
        )

        assert error_msg in result.get_user_message()


class TestEquality:
    """Tests for __eq__() comparison."""

    def test_equal_success_results(self):
        """Test that two identical success results are equal."""
        data = np.array([[1, 2], [3, 4]])
        result1 = ComputationResult(
            status="success",
            data=data,
            operation_code="add",
            input_signature=("add", (2, 2), "hash"),
        )
        result2 = ComputationResult(
            status="success",
            data=data.copy(),
            operation_code="add",
            input_signature=("add", (2, 2), "hash"),
        )

        assert result1 == result2

    def test_not_equal_different_status(self):
        """Test that results with different status are not equal."""
        result1 = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
        )
        result2 = ComputationResult(
            status="error",
            error_message="Error",
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
        )

        assert result1 != result2

    def test_not_equal_different_data(self):
        """Test that results with different data are not equal."""
        result1 = ComputationResult(
            status="success",
            data=np.array([[1, 2]]),
            operation_code="add",
            input_signature=("add", (1, 2), "hash"),
        )
        result2 = ComputationResult(
            status="success",
            data=np.array([[1, 3]]),
            operation_code="add",
            input_signature=("add", (1, 2), "hash"),
        )

        assert result1 != result2

    def test_equal_error_results(self):
        """Test that two identical error results are equal."""
        result1 = ComputationResult(
            status="error",
            error_message="Dimension mismatch",
            operation_code="add",
            input_signature=("add", (2, 3), "hash"),
            error_type="dimension_mismatch",
        )
        result2 = ComputationResult(
            status="error",
            error_message="Dimension mismatch",
            operation_code="add",
            input_signature=("add", (2, 3), "hash"),
            error_type="dimension_mismatch",
        )

        assert result1 == result2

    def test_not_equal_different_cached_flag(self):
        """Test that results with different cached flag are not equal."""
        result1 = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=True,
        )
        result2 = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
            cached=False,
        )

        assert result1 != result2

    def test_not_equal_to_non_computation_result(self):
        """Test that ComputationResult is not equal to other types."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="add",
            input_signature=("add", (1, 1), "hash"),
        )

        assert result != "not a result"
        assert result != 42
        assert result != None
        assert result != {"status": "success"}


class TestRepr:
    """Tests for __repr__() string representation."""

    def test_repr_success_result(self):
        """Test repr() for success result."""
        result = ComputationResult(
            status="success",
            data=np.array([[1, 2]]),
            operation_code="add",
            input_signature=("add", (1, 2), "hash"),
        )
        repr_str = repr(result)

        assert "ComputationResult" in repr_str
        assert "status='success'" in repr_str
        assert "operation_code='add'" in repr_str
        assert "ndarray" in repr_str
        assert "shape=(1, 2)" in repr_str

    def test_repr_error_result(self):
        """Test repr() for error result."""
        result = ComputationResult(
            status="error",
            error_message="Test error",
            operation_code="multiply",
            input_signature=("multiply", (), "hash"),
            error_type="dimension_mismatch",
        )
        repr_str = repr(result)

        assert "ComputationResult" in repr_str
        assert "status='error'" in repr_str
        assert "data=None" in repr_str
        assert "error_message='Test error'" in repr_str

    def test_repr_contains_all_fields(self):
        """Test that repr() includes all important fields."""
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code="test",
            input_signature=("test", (1, 1), "hash"),
            cached=True,
            stale=False,
        )
        repr_str = repr(result)

        assert "cached=True" in repr_str
        assert "stale=False" in repr_str
        assert "timestamp=" in repr_str

    def test_repr_with_large_array_shows_shape_not_values(self):
        """Test that repr() shows shape for large arrays, not values."""
        large_array = np.random.rand(100, 100)
        result = ComputationResult(
            status="success",
            data=large_array,
            operation_code="svd",
            input_signature=("svd", (100, 100), "hash"),
        )
        repr_str = repr(result)

        assert "shape=(100, 100)" in repr_str
        # Should not contain actual numerical values
        assert "[[" not in repr_str or "0." not in repr_str[:200]


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_result_with_nan_values(self):
        """Test result containing NaN values."""
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="eigendecomposition",
            input_signature=("eigendecomposition", (2, 2), "hash"),
        )

        assert result.status == "success"
        assert np.isnan(result.data[0, 1])

    def test_result_with_inf_values(self):
        """Test result containing infinity values."""
        data = np.array([[1.0, np.inf], [-np.inf, 4.0]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="invert",
            input_signature=("invert", (2, 2), "hash"),
        )

        assert result.status == "success"
        assert np.isinf(result.data[0, 1])

    def test_result_with_very_small_values(self):
        """Test result with very small floating point values."""
        data = np.array([[1e-300, 1e-310]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="test",
            input_signature=("test", (1, 2), "hash"),
        )

        assert result.status == "success"
        assert result.data[0, 0] == 1e-300

    def test_result_with_empty_error_message(self):
        """Test error result with empty error message."""
        result = ComputationResult(
            status="error",
            error_message="",
            operation_code="test",
            input_signature=("test", (), "hash"),
        )

        assert result.status == "error"
        assert result.error_message == ""
        assert result.get_user_message() == "Error: "

    def test_result_with_very_long_error_message(self):
        """Test error result with very long error message."""
        long_msg = "A" * 10000
        result = ComputationResult(
            status="error",
            error_message=long_msg,
            operation_code="test",
            input_signature=("test", (), "hash"),
        )

        assert result.status == "error"
        assert result.error_message == long_msg

    def test_result_with_zero_matrix(self):
        """Test result with zero matrix."""
        data = np.zeros((5, 5))
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="determinant",
            input_signature=("determinant", (5, 5), "hash"),
        )

        assert result.status == "success"
        assert np.allclose(result.data, 0)

    def test_result_with_identity_matrix(self):
        """Test result with identity matrix."""
        data = np.eye(4)
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="invert",
            input_signature=("invert", (4, 4), "hash"),
        )

        assert result.status == "success"
        assert np.allclose(result.data, np.eye(4))

    def test_to_json_with_nan_in_data(self):
        """Test to_json() with NaN values (should convert to None or NaN in JSON)."""
        data = np.array([[1.0, np.nan]])
        result = ComputationResult(
            status="success",
            data=data,
            operation_code="test",
            input_signature=("test", (1, 2), "hash"),
        )
        json_dict = result.to_json(include_data=True)

        # NaN should be converted to null (None) in JSON
        assert "data" in json_dict

    def test_long_operation_code(self):
        """Test result with very long operation code."""
        long_code = "operation_" + "x" * 1000
        result = ComputationResult(
            status="success",
            data=np.array([[1]]),
            operation_code=long_code,
            input_signature=(long_code, (), "hash"),
        )

        assert result.operation_code == long_code

    def test_complex_input_signature(self):
        """Test result with complex multi-level input_signature."""
        sig = ("multiply", ((2, 3), (3, 4)), "complex_hash_with_special_chars_!@#$%")
        result = ComputationResult(
            status="success",
            data=np.array([[1, 2, 3, 4]]),
            operation_code="multiply",
            input_signature=sig,
        )

        assert result.input_signature == sig
        json_dict = result.to_json(include_input_signature=True)
        assert json_dict["input_signature"] == sig
