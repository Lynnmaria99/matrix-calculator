"""错误消息翻译与结构化转换模块。

此模块将计算异常（MatrixOperationError 等）映射到用户友好的上下文感知消息。
这是框架无关的模块，所有 UI 层（Web/Desktop/CLI）都可以使用。
"""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

import numpy as np

from .operations import MatrixOperationError


# ============================================================================
# 错误消息模板（中文和英文）
# ============================================================================

ERROR_TEMPLATES = {
    "zh": {
        "dimension_mismatch": {
            "problem": "矩阵尺寸不匹配",
            "message_template": "无法进行{operation}：矩阵 A 的尺寸为 {a_shape}，矩阵 B 的尺寸为 {b_shape}，两个矩阵必须具有相同的尺寸。",
            "suggestion": "请调整矩阵 A 或矩阵 B 的尺寸，使它们相同。",
        },
        "cols_rows_mismatch": {
            "problem": "矩阵乘法尺寸不符",
            "message_template": "无法进行矩阵乘法：矩阵 A 的列数 ({a_cols}) 必须等于矩阵 B 的行数 ({b_rows})。矩阵 A 尺寸：{a_shape}，矩阵 B 尺寸：{b_shape}。",
            "suggestion": "调整矩阵 B 的行数，使其等于矩阵 A 的列数。或者尝试 A + B（矩阵加法）。",
        },
        "singular_matrix": {
            "problem": "矩阵奇异（不可逆）",
            "message_template": "无法进行{operation}：矩阵 A（尺寸 {a_shape}）是奇异矩阵（行列式为 0），不可逆。",
            "suggestion": "奇异矩阵无法求逆。可以尝试使用 SVD（奇异值分解）来分析矩阵的秩。或者修改矩阵元素使其非奇异。",
        },
        "not_square": {
            "problem": "矩阵非方阵",
            "message_template": "无法进行{operation}：{operation}要求矩阵必须是方阵（行数 = 列数）。矩阵 A 的尺寸为 {a_shape}（{a_rows} 行 × {a_cols} 列）。",
            "suggestion": "调整矩阵的行数和列数使其相等，或者选择其他支持非方阵的操作（如 SVD、转置）。",
        },
        "division_by_zero": {
            "problem": "除以零",
            "message_template": "无法进行对应元素相除：除数矩阵中含有 0，会产生无穷大或非数（inf/nan）。",
            "suggestion": "将除数矩阵中的 0 替换为非零值。或者检查你的输入数据。",
        },
        "invalid_input": {
            "problem": "非法输入",
            "message_template": "矩阵输入包含非数字值：{error_detail}。请确保每个单元格都是有效的数字。",
            "suggestion": "输入支持的格式：小数（3.14）、分数（1/2）、科学记数法（1e-3）。",
        },
        "validation_error": {
            "problem": "验证失败",
            "message_template": "输入验证失败：{error_detail}。",
            "suggestion": "检查输入数据并重试。",
        },
        "computation_failed": {
            "problem": "计算失败",
            "message_template": "计算过程中发生错误：{error_detail}。",
            "suggestion": "请检查你的输入，或稍后重试。",
        },
    },
    "en": {
        "dimension_mismatch": {
            "problem": "Matrix Dimension Mismatch",
            "message_template": "Cannot perform {operation}: Matrix A has shape {a_shape}, but matrix B has shape {b_shape}. Both matrices must have the same shape.",
            "suggestion": "Adjust matrix A or matrix B to have the same dimensions.",
        },
        "cols_rows_mismatch": {
            "problem": "Matrix Multiplication Dimension Error",
            "message_template": "Cannot perform matrix multiplication: The number of columns in matrix A ({a_cols}) must equal the number of rows in matrix B ({b_rows}). Matrix A shape: {a_shape}, Matrix B shape: {b_shape}.",
            "suggestion": "Adjust the number of rows in matrix B to match the number of columns in matrix A. Or try A + B (matrix addition).",
        },
        "singular_matrix": {
            "problem": "Singular Matrix (Not Invertible)",
            "message_template": "Cannot perform {operation}: Matrix A (shape {a_shape}) is singular (determinant = 0) and cannot be inverted.",
            "suggestion": "A singular matrix cannot be inverted. Try SVD (Singular Value Decomposition) to analyze the matrix rank. Or modify matrix elements to make it non-singular.",
        },
        "not_square": {
            "problem": "Matrix is Not Square",
            "message_template": "Cannot perform {operation}: {operation} requires a square matrix (rows = columns). Matrix A has shape {a_shape} ({a_rows} rows × {a_cols} columns).",
            "suggestion": "Adjust the matrix so that rows = columns, or choose another operation that supports non-square matrices (e.g., SVD, transpose).",
        },
        "division_by_zero": {
            "problem": "Division by Zero",
            "message_template": "Cannot perform element-wise division: The divisor matrix contains 0, which would result in inf/nan.",
            "suggestion": "Replace the 0 values in the divisor matrix with non-zero values. Or review your input data.",
        },
        "invalid_input": {
            "problem": "Invalid Input",
            "message_template": "Matrix input contains non-numeric values: {error_detail}. Please ensure each cell is a valid number.",
            "suggestion": "Supported formats: decimal (3.14), fraction (1/2), scientific notation (1e-3).",
        },
        "validation_error": {
            "problem": "Validation Failed",
            "message_template": "Input validation failed: {error_detail}.",
            "suggestion": "Check your input data and try again.",
        },
        "computation_failed": {
            "problem": "Computation Failed",
            "message_template": "An error occurred during computation: {error_detail}.",
            "suggestion": "Check your input, or try again later.",
        },
    },
}


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class TranslatedError:
    """结构化的错误信息。"""
    
    status: Literal["error"] = "error"
    error_type: str = ""  # "dimension_mismatch", "singular_matrix", etc.
    message: str = ""  # User-friendly message in selected language
    context: dict = None  # Additional data: operation, dimensions, suggestions
    visual_context: dict = None  # For UI display: matrix_labels, problem_description
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.visual_context is None:
            self.visual_context = {}


# ============================================================================
# 主翻译函数
# ============================================================================

def translate_error(
    exception: Exception | None,
    operation_code: str | None = None,
    matrices_context: dict | None = None,
    language: str = "zh",
) -> dict:
    """将异常对象翻译为用户友好的结构化错误消息。
    
    Args:
        exception: 捕获的异常对象（可能为 MatrixOperationError 等）
        operation_code: 操作代码（"add", "multiply", "determinant" 等）
        matrices_context: 包含矩阵维度信息的字典，例如 {"a_shape": (2, 3), "b_shape": (3, 4)}
        language: 语言代码（"zh" 或 "en"）
        
    Returns:
        字典形式的结构化错误：
        {
            "status": "error",
            "error_type": str,
            "message": str,
            "context": {...},
            "visual_context": {...}
        }
    """
    
    if matrices_context is None:
        matrices_context = {}
    
    # 处理 None 异常
    if exception is None:
        return {
            "status": "error",
            "error_type": "unknown_error",
            "message": "Unknown error occurred." if language == "en" else "发生未知错误。",
            "context": {},
            "visual_context": {},
        }
    
    # 确定语言
    if language not in ["zh", "en"]:
        language = "zh"
    
    # 处理 MatrixOperationError
    if isinstance(exception, MatrixOperationError):
        return _translate_matrix_operation_error(
            exception, operation_code, matrices_context, language
        )
    
    # 处理其他异常
    error_type = "computation_failed"
    error_detail = str(exception)
    
    templates = ERROR_TEMPLATES[language][error_type]
    message = templates["message_template"].format(error_detail=error_detail)
    
    return {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "context": {
            "operation": operation_code or "unknown",
            "suggestion": templates["suggestion"],
        },
        "visual_context": {
            "problem_description": templates["problem"],
            "matrix_labels": [],
        },
    }


def _translate_matrix_operation_error(
    exception: MatrixOperationError,
    operation_code: str | None,
    matrices_context: dict,
    language: str,
) -> dict:
    """将 MatrixOperationError 翻译为结构化错误。"""
    
    error_msg = str(exception).lower()
    operation_code = operation_code or "unknown"
    
    # 根据异常消息的关键字段判断错误类型
    a_shape = matrices_context.get("a_shape")
    b_shape = matrices_context.get("b_shape")
    
    error_type = _detect_error_type(error_msg, a_shape, b_shape)
    templates = ERROR_TEMPLATES[language][error_type]
    
    # 生成消息
    message = _format_message(
        templates["message_template"],
        error_type,
        operation_code,
        a_shape,
        b_shape,
        error_msg,
    )
    
    # 构建上下文
    context = {
        "operation": operation_code,
        "suggestion": templates["suggestion"],
    }
    
    if a_shape:
        context["a_shape"] = a_shape
        context["a_rows"] = a_shape[0]
        context["a_cols"] = a_shape[1]
    
    if b_shape:
        context["b_shape"] = b_shape
        context["b_rows"] = b_shape[0]
        context["b_cols"] = b_shape[1]
    
    # 构建视觉上下文
    matrix_labels = []
    if a_shape:
        matrix_labels.append(f"A: {a_shape[0]}×{b_shape[1]}" if b_shape else f"A: {a_shape[0]}×{a_shape[1]}")
    if b_shape:
        matrix_labels.append(f"B: {b_shape[0]}×{b_shape[1]}")
    
    visual_context = {
        "problem_description": templates["problem"],
        "matrix_labels": matrix_labels,
    }
    
    return {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "context": context,
        "visual_context": visual_context,
    }


def _detect_error_type(
    error_msg: str,
    a_shape: tuple | None,
    b_shape: tuple | None,
) -> str:
    """根据异常消息内容检测错误类型。"""
    
    # 检查 shape 不一致（两个矩阵元素运算）
    if "shape" in error_msg and "不一致" in error_msg:
        return "dimension_mismatch"
    
    # 检查矩阵乘法的列行不匹配
    if ("列数" in error_msg or "rows" in error_msg or "columns" in error_msg) and \
       ("不满足要求" in error_msg or "must equal" in error_msg):
        return "cols_rows_mismatch"
    
    # 检查矩阵不是方阵
    if ("行列数目应该相等" in error_msg or "not square" in error_msg or 
        "require" in error_msg and "square" in error_msg):
        return "not_square"
    
    # 检查奇异矩阵
    if ("不可逆" in error_msg or "singular" in error_msg or 
        "行列式" in error_msg or "determinant" in error_msg):
        return "singular_matrix"
    
    # 检查除以零
    if ("0" in error_msg or "zero" in error_msg) and \
       ("除" in error_msg or "division" in error_msg):
        return "division_by_zero"
    
    # 默认为通用计算失败
    return "computation_failed"


def _format_message(
    template: str,
    error_type: str,
    operation_code: str,
    a_shape: tuple | None,
    b_shape: tuple | None,
    error_msg: str,
) -> str:
    """根据模板和上下文格式化错误消息。"""
    
    replacements = {
        "operation": operation_code,
        "a_shape": str(a_shape) if a_shape else "unknown",
        "b_shape": str(b_shape) if b_shape else "unknown",
        "a_rows": a_shape[0] if a_shape else "?",
        "a_cols": a_shape[1] if a_shape else "?",
        "b_rows": b_shape[0] if b_shape else "?",
        "b_cols": b_shape[1] if b_shape else "?",
        "error_detail": error_msg,
    }
    
    # 安全的字符串格式化：只替换存在的占位符
    try:
        return template.format(**replacements)
    except KeyError:
        # 如果有未知的占位符，直接返回原始模板
        return template


# ============================================================================
# 公开接口（保持向后兼容性）
# ============================================================================

__all__ = [
    "translate_error",
    "TranslatedError",
    "ERROR_TEMPLATES",
]
