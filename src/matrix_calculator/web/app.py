"""Streamlit 主导航与外观切换。"""

from __future__ import annotations

import streamlit as st

from .pages import page_binary, page_quadratic, page_solve, page_unary
from .results import reset_result_serial
from .theme import inject_theme


THEME_LABELS = ["跟随系统", "浅色", "深色"]
_THEME_MODE_MAP = {"跟随系统": "system", "浅色": "light", "深色": "dark"}


def main() -> None:
    """渲染一次网页应用。"""
    reset_result_serial()
    page = st.sidebar.radio(
        "功能",
        ["双矩阵运算", "单矩阵运算", "解线性方程组", "二次型分析"],
    )
    st.sidebar.divider()
    theme_label = st.sidebar.segmented_control(
        "外观",
        THEME_LABELS,
        default="跟随系统",
        key="mx_theme_label",
    ) or "跟随系统"
    # 先拿到主题选择，再在主内容渲染前注入对应色板，避免切换闪烁。
    inject_theme(_THEME_MODE_MAP[theme_label])

    st.title("矩阵计算器")
    st.caption("NumPy 计算核心 · 线性代数学习工具")

    if page == "双矩阵运算":
        page_binary()
    elif page == "单矩阵运算":
        page_unary()
    elif page == "解线性方程组":
        page_solve()
    else:
        page_quadratic()
