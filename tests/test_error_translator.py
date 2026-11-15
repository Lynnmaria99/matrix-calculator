"""单元测试：error_translator 模块。

测试错误翻译、消息生成和上下文提取的正确性。
"""

import pytest
import numpy as np

from matrix_calculator.error_translator import (
    translate_error,
    TranslatedError,
    ERROR_TEMPLATES,
    _detect_error_type,
    _format_message,
)
from matrix_calculator.operations import MatrixOperationError


# ============================================================================
# 测试错误类型检测
# ============================================================================

class TestDetectErrorType:
    """测试 _detect_error_type 函数。"""
    
    def test_detect_dimension_mismatch(self):
        """检测维度不匹配错误。"""
        error_msg = "两个待相加矩阵的 shape 不一致，无法操作"
        error_type = _detect_error_type(error_msg, (2, 3), (3, 2))
        assert error_type == "dimension_mismatch"
    
    def test_detect_cols_rows_mismatch(self):
        """检测矩阵乘法列行不匹配错误。"""
        error_msg = "两个待相乘矩阵的 shape 不满足要求 (矩阵1的列数 != 矩阵2的行数)，无法操作"
        error_type = _detect_error_type(error_msg, (2, 3), (4, 2))
        assert error_type == "cols_rows_mismatch"
    
    def test_detect_not_square(self):
        """检测非方阵错误。"""
        error_msg = "矩阵行列数目应该相等 (方阵)，无法进行行列式计算"
        error_type = _detect_error_type(error_msg, (2, 3), None)
        assert error_type == "not_square"
    
    def test_detect_singular_matrix(self):
        """检测奇异矩阵错误。"""
        error_msg = "矩阵不可逆 (行列式为 0 或矩阵接近奇异)"
        error_type = _detect_error_type(error_msg, (2, 2), None)
        assert error_type == "singular_matrix"
    
    def test_detect_division_by_zero(self):
        """检测除零错误。"""
        error_msg = "除数矩阵中含有 0，对应相除会产生 inf/nan，已取消操作"
        error_type = _detect_error_type(error_msg, (2, 2), (2, 2))
        assert error_type == "division_by_zero"


# ============================================================================
# 测试消息格式化
# ============================================================================

class TestFormatMessage:
    """测试 _format_message 函数。"""
    
    def test_format_dimension_mismatch_zh(self):
        """格式化维度不匹配消息（中文）。"""
        template = ERROR_TEMPLATES["zh"]["dimension_mismatch"]["message_template"]
        message = _format_message(
            template,
            "dimension_mismatch",
            "add",
            (2, 3),
            (3, 2),
            ""
        )
        assert "(2, 3)" in message
        assert "(3, 2)" in message
        assert "add" in message
    
    def test_format_cols_rows_mismatch_en(self):
        """格式化矩阵乘法错误消息（英文）。"""
        template = ERROR_TEMPLATES["en"]["cols_rows_mismatch"]["message_template"]
        message = _format_message(
            template,
            "cols_rows_mismatch",
            "multiply",
            (2, 3),
            (4, 2),
            ""
        )
        assert "3" in message  # a_cols
        assert "4" in message  # b_rows
        assert "(2, 3)" in message
        assert "(4, 2)" in message


# ============================================================================
# 测试主翻译函数 - 中文
# ============================================================================

class TestTranslateErrorChinese:
    """测试 translate_error 函数（中文）。"""
    
    def test_dimension_mismatch_error_zh(self):
        """翻译维度不匹配错误（中文）。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="zh"
        )
        
        assert result["status"] == "error"
        assert result["error_type"] == "dimension_mismatch"
        assert "(2, 3)" in result["message"]
        assert "(3, 2)" in result["message"]
        assert "矩阵尺寸不匹配" in result["visual_context"]["problem_description"]
        assert len(result["context"]["suggestion"]) > 0
        assert result["context"]["operation"] == "add"
    
    def test_cols_rows_mismatch_error_zh(self):
        """翻译矩阵乘法列行不匹配错误（中文）。"""
        exc = MatrixOperationError(
            "两个待相乘矩阵的 shape 不满足要求 (矩阵1的列数 != 矩阵2的行数)，无法操作"
        )
        result = translate_error(
            exc,
            operation_code="multiply",
            matrices_context={"a_shape": (2, 3), "b_shape": (4, 2)},
            language="zh"
        )
        
        assert result["error_type"] == "cols_rows_mismatch"
        assert "矩阵乘法" in result["visual_context"]["problem_description"]
        assert result["context"]["a_cols"] == 3
        assert result["context"]["b_rows"] == 4
    
    def test_singular_matrix_error_zh(self):
        """翻译奇异矩阵错误（中文）。"""
        exc = MatrixOperationError("矩阵不可逆 (行列式为 0 或矩阵接近奇异)")
        result = translate_error(
            exc,
            operation_code="inverse",
            matrices_context={"a_shape": (2, 2)},
            language="zh"
        )
        
        assert result["error_type"] == "singular_matrix"
        assert "奇异" in result["visual_context"]["problem_description"]
        assert "(2, 2)" in result["message"]
    
    def test_not_square_error_zh(self):
        """翻译非方阵错误（中文）。"""
        exc = MatrixOperationError("矩阵行列数目应该相等 (方阵)，无法进行行列式计算")
        result = translate_error(
            exc,
            operation_code="determinant",
            matrices_context={"a_shape": (2, 3)},
            language="zh"
        )
        
        assert result["error_type"] == "not_square"
        assert "方阵" in result["visual_context"]["problem_description"]
        assert "2" in result["message"] and "3" in result["message"]
    
    def test_division_by_zero_error_zh(self):
        """翻译除零错误（中文）。"""
        exc = MatrixOperationError("除数矩阵中含有 0，对应相除会产生 inf/nan，已取消操作")
        result = translate_error(
            exc,
            operation_code="hadamard_divide",
            matrices_context={"a_shape": (2, 2), "b_shape": (2, 2)},
            language="zh"
        )
        
        assert result["error_type"] == "division_by_zero"
        assert "除" in result["visual_context"]["problem_description"]
    
    def test_none_exception_zh(self):
        """处理 None 异常（中文）。"""
        result = translate_error(None, language="zh")
        
        assert result["status"] == "error"
        assert result["error_type"] == "unknown_error"
        assert "未知" in result["message"]
    
    def test_error_without_matrices_context_zh(self):
        """处理没有矩阵上下文的错误（中文）。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context=None,
            language="zh"
        )
        
        assert result["status"] == "error"
        assert result["error_type"] == "dimension_mismatch"
        assert "message" in result


# ============================================================================
# 测试主翻译函数 - 英文
# ============================================================================

class TestTranslateErrorEnglish:
    """测试 translate_error 函数（英文）。"""
    
    def test_dimension_mismatch_error_en(self):
        """翻译维度不匹配错误（英文）。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="en"
        )
        
        assert result["status"] == "error"
        assert result["error_type"] == "dimension_mismatch"
        assert "Matrix" in result["visual_context"]["problem_description"]
        assert "cannot" in result["message"].lower() or "cannot" in result["message"]
    
    def test_cols_rows_mismatch_error_en(self):
        """翻译矩阵乘法列行不匹配错误（英文）。"""
        exc = MatrixOperationError(
            "两个待相乘矩阵的 shape 不满足要求 (矩阵1的列数 != 矩阵2的行数)，无法操作"
        )
        result = translate_error(
            exc,
            operation_code="multiply",
            matrices_context={"a_shape": (2, 3), "b_shape": (4, 2)},
            language="en"
        )
        
        assert result["error_type"] == "cols_rows_mismatch"
        assert "multiplication" in result["visual_context"]["problem_description"].lower()
    
    def test_singular_matrix_error_en(self):
        """翻译奇异矩阵错误（英文）。"""
        exc = MatrixOperationError("矩阵不可逆 (行列式为 0 或矩阵接近奇异)")
        result = translate_error(
            exc,
            operation_code="inverse",
            matrices_context={"a_shape": (2, 2)},
            language="en"
        )
        
        assert result["error_type"] == "singular_matrix"
        assert "Singular" in result["visual_context"]["problem_description"]
    
    def test_not_square_error_en(self):
        """翻译非方阵错误（英文）。"""
        exc = MatrixOperationError("矩阵行列数目应该相等 (方阵)，无法进行行列式计算")
        result = translate_error(
            exc,
            operation_code="determinant",
            matrices_context={"a_shape": (2, 3)},
            language="en"
        )
        
        assert result["error_type"] == "not_square"
        assert "square" in result["visual_context"]["problem_description"].lower()
    
    def test_none_exception_en(self):
        """处理 None 异常（英文）。"""
        result = translate_error(None, language="en")
        
        assert result["status"] == "error"
        assert result["error_type"] == "unknown_error"
        assert "Unknown" in result["message"] or "unknown" in result["message"].lower()


# ============================================================================
# 测试返回结构与序列化
# ============================================================================

class TestReturnStructure:
    """测试返回结构的完整性和 JSON 序列化能力。"""
    
    def test_return_has_all_required_fields(self):
        """检查返回对象包含所有必需字段。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="zh"
        )
        
        required_fields = ["status", "error_type", "message", "context", "visual_context"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
    
    def test_status_always_error(self):
        """检查 status 字段总是 'error'。"""
        exc = MatrixOperationError("某个错误")
        result = translate_error(exc, language="zh")
        assert result["status"] == "error"
    
    def test_error_type_valid(self):
        """检查 error_type 在已知错误类型中。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="zh"
        )
        
        valid_types = list(ERROR_TEMPLATES["zh"].keys())
        assert result["error_type"] in valid_types
    
    def test_context_has_suggestion(self):
        """检查 context 包含建议。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="zh"
        )
        
        assert "suggestion" in result["context"]
        assert isinstance(result["context"]["suggestion"], str)
        assert len(result["context"]["suggestion"]) > 0
    
    def test_visual_context_has_problem_description(self):
        """检查 visual_context 包含问题描述。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="zh"
        )
        
        assert "problem_description" in result["visual_context"]
        assert isinstance(result["visual_context"]["problem_description"], str)
    
    def test_visual_context_matrix_labels_list(self):
        """检查 visual_context 中 matrix_labels 是列表。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="zh"
        )
        
        assert "matrix_labels" in result["visual_context"]
        assert isinstance(result["visual_context"]["matrix_labels"], list)


# ============================================================================
# 测试边界情况
# ============================================================================

class TestEdgeCases:
    """测试边界情况和异常输入。"""
    
    def test_none_exception(self):
        """测试 None 异常。"""
        result = translate_error(None, language="zh")
        assert result["status"] == "error"
        assert result["error_type"] == "unknown_error"
    
    def test_empty_operation_code(self):
        """测试空操作代码。"""
        exc = MatrixOperationError("某个错误")
        result = translate_error(
            exc,
            operation_code=None,
            matrices_context={},
            language="zh"
        )
        assert result["context"]["operation"] == "unknown"
    
    def test_empty_matrices_context(self):
        """测试空矩阵上下文。"""
        exc = MatrixOperationError("某个错误")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={},
            language="zh"
        )
        assert result["status"] == "error"
    
    def test_invalid_language_defaults_to_zh(self):
        """测试无效语言代码默认为中文。"""
        exc = MatrixOperationError("两个待相加矩阵的 shape 不一致，无法操作")
        result = translate_error(
            exc,
            operation_code="add",
            matrices_context={"a_shape": (2, 3), "b_shape": (3, 2)},
            language="invalid"
        )
        # 应该能够成功生成中文消息
        assert "shape" in result["message"].lower() or "矩" in result["message"]
    
    def test_single_matrix_operation_context(self):
        """测试只有一个矩阵的上下文（如行列式）。"""
        exc = MatrixOperationError("矩阵行列数目应该相等 (方阵)，无法进行行列式计算")
        result = translate_error(
            exc,
            operation_code="determinant",
            matrices_context={"a_shape": (2, 3)},
            language="zh"
        )
        
        assert result["context"]["a_shape"] == (2, 3)
        assert result["context"]["a_rows"] == 2
        assert result["context"]["a_cols"] == 3



# ============================================================================
# 测试 TranslatedError 数据类
# ============================================================================

class TestTranslatedErrorDataclass:
    """测试 TranslatedError 数据类。"""
    
    def test_dataclass_creation(self):
        """测试数据类创建。"""
        error = TranslatedError(
            status="error",
            error_type="dimension_mismatch",
            message="Test message",
            context={"a_shape": (2, 3)},
            visual_context={"problem_description": "Test problem"}
        )
        
        assert error.status == "error"
        assert error.error_type == "dimension_mismatch"
        assert error.message == "Test message"
    
    def test_dataclass_default_values(self):
        """测试数据类默认值。"""
        error = TranslatedError()
        
        assert error.status == "error"
        assert error.error_type == ""
        assert error.message == ""
        assert error.context == {}
        assert error.visual_context == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
