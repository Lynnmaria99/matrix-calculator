# -*- coding: utf-8 -*-
"""矩阵计算器 Streamlit 网页界面。

运行方式：

    streamlit run streamlit_app.py

网页层只负责收集输入、调用计算核心、展示结果和错误信息。
矩阵输入使用自定义 CCv2 组件，呈现为带括号、可键盘导航的数学矩阵格子。
"""

from __future__ import annotations

import re
import sys
from itertools import count
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from matrix_calculator import (  # noqa: E402
    MatrixOperationError,
    add_matrices,
    analyze_quadratic,
    determinant,
    eigendecompose,
    format_matrix,
    hadamard_divide,
    hadamard_multiply,
    invert_matrix,
    matrix_multiply,
    singular_value_decompose,
    solve_linear,
    subtract_matrices,
    transpose,
)
from matrix_calculator.formatting import (  # noqa: E402
    matrix_to_csv_text,
    matrix_to_numpy_text,
    signed_polynomial,
)


st.set_page_config(page_title="矩阵计算器", page_icon="🧮", layout="wide")

MAX_DIMENSION = 12


# ==================== 设计 token 与主题系统 ====================
#
# 颜色、阴影、圆角全部收敛为语义变量。浅色值与历史硬编码一致（保证视觉零回归），
# 深色为独立色板；矩阵组件位于 Shadow DOM，CSS 自定义属性可穿透边界自动换肤。

_THEME_RADIUS = """
  :root {
    --mx-font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
      "Segoe UI Variable Display", "Segoe UI", "PingFang SC", "Microsoft YaHei",
      sans-serif;
    --mx-radius-pill: 999px;
    --mx-radius-card: 24px;
    --mx-radius-lg: 18px;
    --mx-radius-md: 16px;
    --mx-radius-cell: 0.9rem;
  }
"""

# 浅色 token（取值与重构前完全一致）
_THEME_LIGHT = """
  :root {
    /* 背景与表面 */
    --mx-bg-image:
      radial-gradient(1100px 520px at 8% -8%, rgba(0, 113, 227, 0.20), transparent 58%),
      radial-gradient(900px 520px at 98% -4%, rgba(175, 82, 222, 0.15), transparent 55%),
      radial-gradient(820px 620px at 78% 100%, rgba(48, 209, 188, 0.15), transparent 60%),
      linear-gradient(180deg, #f8faff 0%, #eef3fb 100%);
    --mx-sidebar-bg: rgba(255, 255, 255, 0.56);
    --mx-sidebar-border: rgba(255, 255, 255, 0.72);
    --mx-glass: rgba(255, 255, 255, 0.62);
    --mx-glass-strong: rgba(255, 255, 255, 0.78);
    --mx-glass-border: rgba(255, 255, 255, 0.76);
    --mx-inset: rgba(255, 255, 255, 0.78);
    --mx-group-bg: rgba(255, 255, 255, 0.54);
    --mx-group-border: rgba(255, 255, 255, 0.78);
    --mx-group-hover: rgba(255, 255, 255, 0.66);
    --mx-group-active: rgba(255, 255, 255, 0.94);
    --mx-ghost-bg: rgba(255, 255, 255, 0.62);
    --mx-ghost-border: rgba(255, 255, 255, 0.82);
    --mx-code-bg: rgba(255, 255, 255, 0.85);
    --mx-code-border: rgba(221, 229, 240, 0.9);
    --mx-code-inset: rgba(255, 255, 255, 0.68);
    --mx-metric-bg: rgba(255, 255, 255, 0.66);
    --mx-metric-border: rgba(255, 255, 255, 0.82);
    --mx-cell-bg: rgba(255, 255, 255, 0.70);
    --mx-cell-border: rgba(255, 255, 255, 0.88);
    --mx-cell-inset: rgba(255, 255, 255, 0.86);
    /* 文字 */
    --mx-text: #1d1d1f;
    --mx-muted: #6e6e73;
    /* 品牌色 */
    --mx-blue: #0071e3;
    --mx-blue-strong: #2b9aff;
    --mx-blue-text: #0062c4;
    --mx-blue-focus: rgba(0, 113, 227, 0.55);
    --mx-blue-ring: rgba(0, 113, 227, 0.16);
    --mx-blue-glow: rgba(0, 113, 227, 0.16);
    --mx-primary-gradient: linear-gradient(180deg, #2b9aff 0%, #0071e3 100%);
    --mx-primary-shadow: 0 12px 26px rgba(0, 113, 227, 0.22);
    --mx-primary-shadow-hover: 0 15px 32px rgba(0, 113, 227, 0.28);
    /* 状态色 */
    --mx-danger: #d64545;
    --mx-danger-bg: rgba(214, 69, 69, 0.12);
    --mx-bracket: rgba(80, 93, 124, 0.42);
    /* 阴影 */
    --mx-shadow: 0 22px 60px rgba(15, 23, 42, 0.10);
    --mx-shadow-soft: 0 10px 28px rgba(15, 23, 42, 0.08);
    --mx-shadow-btn: 0 7px 18px rgba(15, 23, 42, 0.06);
    /* 提示语义色 */
    --mx-success-text: #1c7c43;
    --mx-warning-text: #9a6700;
    --mx-error-text: #c0392b;
    --mx-info-text: #0a5fb8;
  }
"""

# 深色 token（深蓝灰底，避免纯黑；主色提亮以保证对比）
_THEME_DARK = """
  :root {
    color-scheme: dark;
    /* 背景与表面 */
    --mx-bg-image:
      radial-gradient(1100px 520px at 8% -8%, rgba(40, 120, 230, 0.18), transparent 58%),
      radial-gradient(900px 520px at 98% -4%, rgba(150, 90, 220, 0.14), transparent 55%),
      radial-gradient(820px 620px at 78% 100%, rgba(40, 190, 175, 0.10), transparent 60%),
      linear-gradient(180deg, #0b0f16 0%, #0d1422 100%);
    --mx-sidebar-bg: rgba(22, 29, 43, 0.62);
    --mx-sidebar-border: rgba(255, 255, 255, 0.08);
    --mx-glass: rgba(30, 39, 56, 0.55);
    --mx-glass-strong: rgba(33, 43, 62, 0.82);
    --mx-glass-border: rgba(255, 255, 255, 0.09);
    --mx-inset: rgba(255, 255, 255, 0.05);
    --mx-group-bg: rgba(255, 255, 255, 0.05);
    --mx-group-border: rgba(255, 255, 255, 0.10);
    --mx-group-hover: rgba(255, 255, 255, 0.10);
    --mx-group-active: rgba(60, 96, 150, 0.32);
    --mx-ghost-bg: rgba(255, 255, 255, 0.06);
    --mx-ghost-border: rgba(255, 255, 255, 0.12);
    --mx-code-bg: rgba(12, 18, 29, 0.78);
    --mx-code-border: rgba(255, 255, 255, 0.10);
    --mx-code-inset: rgba(255, 255, 255, 0.04);
    --mx-metric-bg: rgba(255, 255, 255, 0.05);
    --mx-metric-border: rgba(255, 255, 255, 0.10);
    --mx-cell-bg: rgba(255, 255, 255, 0.05);
    --mx-cell-border: rgba(255, 255, 255, 0.12);
    --mx-cell-inset: rgba(255, 255, 255, 0.05);
    /* 文字 */
    --mx-text: #e7ebf3;
    --mx-muted: #9aa6ba;
    /* 品牌色（暗底提亮） */
    --mx-blue: #4d9fff;
    --mx-blue-strong: #6cb0ff;
    --mx-blue-text: #82bdff;
    --mx-blue-focus: rgba(120, 180, 255, 0.65);
    --mx-blue-ring: rgba(120, 180, 255, 0.22);
    --mx-blue-glow: rgba(77, 159, 255, 0.28);
    --mx-primary-gradient: linear-gradient(180deg, #3d97ff 0%, #2b7fe0 100%);
    --mx-primary-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    --mx-primary-shadow-hover: 0 15px 34px rgba(40, 120, 230, 0.40);
    /* 状态色 */
    --mx-danger: #ff7a7a;
    --mx-danger-bg: rgba(255, 107, 107, 0.16);
    --mx-bracket: rgba(165, 180, 205, 0.55);
    /* 阴影（深色下更柔、更深） */
    --mx-shadow: 0 22px 60px rgba(0, 0, 0, 0.45);
    --mx-shadow-soft: 0 10px 28px rgba(0, 0, 0, 0.35);
    --mx-shadow-btn: 0 7px 18px rgba(0, 0, 0, 0.30);
    /* 提示语义色 */
    --mx-success-text: #7ee2a8;
    --mx-warning-text: #ffcf7a;
    --mx-error-text: #ff9b9b;
    --mx-info-text: #9cc8ff;
  }
"""

# 深色下对 Streamlit 原生控件的额外覆盖（浅色不需要，基线已适配）
_THEME_DARK_OVERRIDES = """
  .stApp { color-scheme: dark; }
  /* 根节点深色兜底：.stApp 渐变之外（过度滚动/边缘）不露浅色底 */
  html,
  body,
  [data-testid="stMain"],
  [data-testid="stAppViewContainer"] {
    background-color: #0d1422;
  }
  .stApp,
  .stApp .stMarkdownContainer,
  .stApp p,
  .stApp li,
  .stApp td,
  .stApp th,
  .stApp span {
    color: var(--mx-text);
  }
  .stApp a,
  .stApp a:visited {
    color: var(--mx-blue-text);
  }
  .stApp hr {
    border-color: var(--mx-glass-border);
  }
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] p {
    color: var(--mx-muted);
  }
  /* 侧边栏单选导航 */
  [data-testid="stRadio"] label p,
  [data-testid="stRadioOption"] p {
    color: var(--mx-text);
  }
  [data-testid="stRadio"] [aria-checked="true"] p {
    color: var(--mx-blue-text);
  }
  /* 原生输入类控件兜底（矩阵输入在 Shadow DOM 内自管，不受影响） */
  .stApp input,
  .stApp textarea,
  .stApp select {
    background-color: var(--mx-cell-bg);
    color: var(--mx-text);
    border-color: var(--mx-cell-border);
  }
  .stApp input::placeholder,
  .stApp textarea::placeholder {
    color: var(--mx-muted);
  }
  /* 开关、复选、选择类容器 */
  [data-testid="stToggle"] {
    color: var(--mx-text);
  }
  /* 弹出层（popover / 下载菜单等挂在 portal） */
  [data-testid="stPopover"],
  [data-testid="portal"] {
    background: var(--mx-glass-strong);
    color: var(--mx-text);
    border: 1px solid var(--mx-glass-border);
  }
  [data-testid="stPopover"] p,
  [data-testid="portal"] p {
    color: var(--mx-text);
  }
  /* popover 内容体挂在 body 下的 portal，自带浅色底，需单独覆盖 */
  [data-testid="stPopoverBody"] {
    background: var(--mx-glass-strong);
    color: var(--mx-text);
    border: 1px solid var(--mx-glass-border);
  }
  [data-testid="stPopoverBody"] p,
  [data-testid="stPopoverBody"] span {
    color: var(--mx-text);
  }
  /* 代码块内部 pre 被基础主题设为不透明白底，深色下强制换肤 */
  .stApp [data-testid="stCode"] pre,
  .stApp pre {
    background: var(--mx-code-bg) !important;
    border: 1px solid var(--mx-code-border);
    color: var(--mx-text);
  }
  .stApp [data-testid="stCode"] code,
  .stApp pre code {
    color: var(--mx-text);
  }
  /* 代码块复制按钮的白色背板（按钮外层包装 div，用 :has 命中） */
  .stApp [data-testid="stCode"] div:has(> [data-testid="stElementToolbarButton"]),
  .stApp [data-testid="stCode"] [data-testid="stElementToolbarButton"] {
    background: var(--mx-ghost-bg) !important;
    border: 1px solid var(--mx-ghost-border);
    border-radius: var(--mx-radius-md);
  }
  /* 顶栏菜单 / 部署按钮 */
  [data-testid="stMainMenu"] button,
  [data-testid="stAppDeployButton"],
  [data-testid="stBaseButton-header"],
  [data-testid="stBaseButton-headerNoPadding"] {
    background: var(--mx-ghost-bg);
    color: var(--mx-text);
    border: 1px solid var(--mx-ghost-border);
  }
  /* 提示条四种语义色 */
  [data-testid="stAlert"] {
    background: var(--mx-glass);
    border: 1px solid var(--mx-glass-border);
  }
  [data-testid="stAlertContentSuccess"],
  [data-testid="stAlertContentSuccess"] * {
    color: var(--mx-success-text);
  }
  [data-testid="stAlertContentInfo"],
  [data-testid="stAlertContentInfo"] * {
    color: var(--mx-info-text);
  }
  [data-testid="stAlertContentWarning"],
  [data-testid="stAlertContentWarning"] * {
    color: var(--mx-warning-text);
  }
  [data-testid="stAlertContentError"],
  [data-testid="stAlertContentError"] * {
    color: var(--mx-error-text);
  }
  /* 工具提示 */
  [role="tooltip"] {
    background: var(--mx-glass-strong);
    color: var(--mx-text);
    border: 1px solid var(--mx-glass-border);
  }
  /* 深色滚动条 */
  .stApp *::-webkit-scrollbar,
  .matrix-widget *::-webkit-scrollbar {
    width: 10px;
    height: 10px;
  }
  .stApp *::-webkit-scrollbar-thumb,
  .matrix-widget *::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.18);
    border-radius: var(--mx-radius-pill);
  }
  .stApp *::-webkit-scrollbar-track,
  .matrix-widget *::-webkit-scrollbar-track {
    background: transparent;
  }
"""

# 主题无关规则：只引用变量，切换 token 即整体换肤
_APP_RULES = """
  .stApp,
  .stAppViewContainer {
    background: var(--mx-bg-image) !important;
    color: var(--mx-text);
    font-family: var(--mx-font);
  }

  .stAppViewContainer {
    background-attachment: fixed;
  }

  [data-testid="stHeader"] {
    background: transparent;
  }

  [data-testid="stMainBlockContainer"] {
    max-width: 1040px;
    padding: 2.35rem 2rem 4.5rem;
  }

  .stApp h1,
  .stApp h2,
  .stApp h3 {
    letter-spacing: -0.025em;
  }

  .stApp h1 {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.2;
  }

  .stApp h2 {
    font-size: 1.45rem;
    font-weight: 650;
    line-height: 1.3;
  }

  [data-testid="stSidebar"] {
    background: var(--mx-sidebar-bg);
    border-right: 1px solid var(--mx-sidebar-border);
    box-shadow: inset -1px 0 0 var(--mx-inset);
    backdrop-filter: blur(32px) saturate(180%);
    -webkit-backdrop-filter: blur(32px) saturate(180%);
  }

  [data-testid="stSidebarContent"] {
    background: transparent;
  }

  [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
    background: var(--mx-glass) !important;
    border: 1px solid var(--mx-glass-border) !important;
    border-radius: var(--mx-radius-card) !important;
    box-shadow:
      var(--mx-shadow),
      inset 0 1px 0 var(--mx-inset) !important;
    padding: 1.15rem 1.25rem;
    backdrop-filter: blur(30px) saturate(180%);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
  }

  [class*="st-key-matrix-grid-"] {
    align-self: center;
    margin-inline: auto;
  }

  [data-testid="stButtonGroup"] {
    background: var(--mx-group-bg);
    border: 1px solid var(--mx-group-border);
    border-radius: var(--mx-radius-pill);
    box-shadow:
      inset 0 1px 0 var(--mx-inset),
      var(--mx-shadow-btn);
    gap: 2px;
    padding: 4px;
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
  }

  [data-testid="stButtonGroup"] button {
    background: transparent !important;
    border: 0 !important;
    border-radius: var(--mx-radius-pill) !important;
    color: var(--mx-text) !important;
    font-weight: 550;
    min-height: 34px;
  }

  [data-testid="stButtonGroup"] button:hover {
    background: var(--mx-group-hover) !important;
  }

  [data-testid="stButtonGroup"] button[aria-checked="true"],
  [data-testid="stButtonGroup"] button[aria-pressed="true"] {
    background: var(--mx-group-active) !important;
    box-shadow:
      var(--mx-shadow-btn),
      0 1px 2px rgba(15, 23, 42, 0.08);
    color: var(--mx-blue-text) !important;
  }

  [data-testid="stButtonGroup"] button[aria-checked="true"] *,
  [data-testid="stButtonGroup"] button[aria-pressed="true"] * {
    color: var(--mx-blue-text) !important;
  }

  .stButton button:not([kind="primary"]),
  .stDownloadButton button:not([kind="primary"]),
  .stPopover button {
    background: var(--mx-ghost-bg) !important;
    border: 1px solid var(--mx-ghost-border) !important;
    border-radius: var(--mx-radius-pill) !important;
    box-shadow:
      inset 0 1px 0 var(--mx-inset),
      var(--mx-shadow-btn);
    color: var(--mx-text) !important;
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
  }

  .stButton button:not([kind="primary"]):hover,
  .stDownloadButton button:not([kind="primary"]):hover,
  .stPopover button:hover {
    border-color: var(--mx-blue-focus) !important;
    color: var(--mx-blue-text) !important;
    transform: translateY(-1px);
  }

  .stButton button[kind="primary"],
  .stDownloadButton button[kind="primary"] {
    background: var(--mx-primary-gradient) !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    border-radius: var(--mx-radius-pill) !important;
    box-shadow: var(--mx-primary-shadow);
    font-weight: 650;
    padding: 0.58rem 1.35rem !important;
    transition: transform 160ms ease, box-shadow 160ms ease;
  }

  .stButton button[kind="primary"]:hover {
    box-shadow: var(--mx-primary-shadow-hover);
    transform: translateY(-1px);
  }

  .stButton button[kind="primary"]:active {
    transform: translateY(0);
  }

  [data-testid="stAlert"] {
    border-radius: var(--mx-radius-lg);
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
  }

  [data-testid="stText"],
  [data-testid="stCode"],
  pre {
    background: var(--mx-code-bg);
    border: 1px solid var(--mx-code-border);
    border-radius: var(--mx-radius-md);
    box-shadow: inset 0 1px 0 var(--mx-code-inset);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    line-height: 1.65;
    overflow-x: auto;
    padding: 0.85rem 1rem;
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
  }

  [data-testid="stCaptionContainer"],
  [data-testid="stCaptionContainer"] * {
    color: var(--mx-muted);
  }

  /* 结果指标统一为圆角卡片，数字降级、标签弱化 */
  [data-testid="stMetric"] {
    background: var(--mx-metric-bg);
    border: 1px solid var(--mx-metric-border);
    border-radius: var(--mx-radius-lg);
    box-shadow: var(--mx-shadow-soft);
    padding: 0.85rem 1.05rem;
  }

  [data-testid="stMetricLabel"] p,
  [data-testid="stMetricLabel"] {
    color: var(--mx-muted);
    font-size: 0.85rem;
  }

  .stApp [data-testid="stMetricValue"] {
    font-size: 1.55rem;
  }

  /* 主操作按钮更易点 */
  .stButton button[kind="primary"],
  .stDownloadButton button[kind="primary"] {
    min-width: 7.5rem;
  }

  /* 键盘焦点可见性 */
  .stApp :focus-visible {
    outline: 3px solid var(--mx-blue-focus);
    outline-offset: 2px;
  }

  @media (prefers-reduced-motion: reduce) {
    .stApp *,
    .stApp *::before,
    .stApp *::after {
      animation: none !important;
      scroll-behavior: auto !important;
      transition: none !important;
    }
  }

  @media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
      padding-left: 1rem;
      padding-right: 1rem;
    }

    .stButton button[kind="primary"],
    .stDownloadButton button[kind="primary"] {
      min-width: 0;
      width: 100%;
    }
  }
"""


def build_app_css(mode: str) -> str:
    """按主题模式组装完整样式表。mode: system / light / dark。"""
    if mode == "dark":
        dark_block = _THEME_DARK + _THEME_DARK_OVERRIDES
    elif mode == "system":
        dark_block = (
            "@media (prefers-color-scheme: dark) {\n"
            + _THEME_DARK
            + _THEME_DARK_OVERRIDES
            + "\n}"
        )
    else:
        dark_block = ""
    return (
        "<style>\n"
        + _THEME_RADIUS
        + _THEME_LIGHT
        + dark_block
        + _APP_RULES
        + "</style>"
    )


def inject_theme(mode: str = "system") -> None:
    """注入应用级 glass 风格样式，并按主题模式切换浅色 / 深色色板。"""
    st.markdown(build_app_css(mode), unsafe_allow_html=True)


MATRIX_GRID_HTML = """
<div class="matrix-widget" data-testid="matrix-widget">
  <div class="matrix-head">
    <span class="matrix-symbol"></span>
    <span class="matrix-size"></span>
    <div class="matrix-controls" aria-label="调整矩阵维度">
      <button type="button" data-action="remove-row">−行</button>
      <button type="button" data-action="add-row">+行</button>
      <button type="button" data-action="remove-col">−列</button>
      <button type="button" data-action="add-col">+列</button>
      <button type="button" data-action="remove-order">−阶</button>
      <button type="button" data-action="add-order">+阶</button>
    </div>
  </div>
  <div class="matrix-expression">
    <span class="bracket bracket-left" aria-hidden="true"></span>
    <div class="matrix-grid" role="grid"></div>
    <span class="bracket bracket-right" aria-hidden="true"></span>
  </div>
  <div class="matrix-error" role="alert"></div>
</div>
"""

MATRIX_GRID_CSS = """
.matrix-widget,
.matrix-widget *,
.matrix-widget *::before,
.matrix-widget *::after {
  box-sizing: border-box;
}

.matrix-widget {
  color: var(--mx-text, #1d1d1f);
  font-family: var(--mx-font, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
}
.matrix-head {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  margin-bottom: 0.45rem;
}
.matrix-symbol {
  font-size: 1.08rem;
  font-weight: 650;
  white-space: nowrap;
}
.matrix-size {
  color: var(--mx-muted, #6e6e73);
  font-family: var(--mx-font-family-monospace, ui-monospace, monospace);
  font-size: 0.86rem;
  opacity: 0.78;
}
.matrix-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.28rem;
}
.matrix-controls button {
  appearance: none;
  background: var(--mx-ghost-bg, rgba(255, 255, 255, 0.62));
  border: 1px solid var(--mx-ghost-border, rgba(255, 255, 255, 0.82));
  border-radius: var(--mx-radius-pill, 999px);
  box-shadow: var(--mx-shadow-btn, 0 7px 18px rgba(15, 23, 42, 0.06)),
    inset 0 1px 0 var(--mx-inset, rgba(255, 255, 255, 0.78));
  color: var(--mx-text, #1d1d1f);
  cursor: pointer;
  font: inherit;
  font-size: 0.82rem;
  line-height: 1.2;
  padding: 0.28rem 0.62rem;
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
}
.matrix-controls button:hover:not(:disabled) {
  border-color: var(--mx-blue-text, #0062c4);
  color: var(--mx-blue-text, #0062c4);
}
.matrix-controls button:disabled {
  cursor: not-allowed;
  opacity: 0.32;
}
.matrix-expression {
  display: inline-flex;
  margin: 0.15rem 0 0 0.15rem;
  max-width: 100%;
  overflow-x: auto;
  padding: 0.35rem 0.95rem;
  position: relative;
  -webkit-overflow-scrolling: touch;
}
.bracket {
  border: 2.5px solid var(--mx-bracket, rgba(80, 93, 124, 0.42));
  bottom: 0.2rem;
  pointer-events: none;
  position: absolute;
  top: 0.2rem;
  width: 10px;
}
.bracket-left {
  border-right: 0;
  border-radius: 9px 0 0 9px;
  left: 0;
}
.bracket-right {
  border-left: 0;
  border-radius: 0 9px 9px 0;
  right: 0;
}
.matrix-grid {
  display: grid;
  gap: 0.38rem;
}
.matrix-cell {
  background: var(--mx-cell-bg, rgba(255, 255, 255, 0.70));
  border: 1px solid var(--mx-cell-border, rgba(255, 255, 255, 0.88));
  border-radius: var(--mx-radius-cell, 0.9rem);
  box-shadow:
    var(--mx-shadow-btn, 0 8px 20px rgba(15, 23, 42, 0.06)),
    inset 0 1px 0 var(--mx-cell-inset, rgba(255, 255, 255, 0.86));
  color: var(--mx-text, #1d1d1f);
  font: inherit;
  font-variant-numeric: tabular-nums;
  height: 2.28rem;
  min-width: 2.5rem;
  padding: 0.1rem 0.25rem;
  text-align: center;
  width: clamp(2.5rem, 13vw, 3.65rem);
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
}
.matrix-cell:focus {
  border-color: var(--mx-blue-focus, rgba(0, 113, 227, 0.55));
  box-shadow:
    0 0 0 4px var(--mx-blue-ring, rgba(0, 113, 227, 0.16)),
    0 10px 22px var(--mx-blue-glow, rgba(0, 113, 227, 0.10));
  outline: none;
}
.matrix-cell.invalid {
  background: var(--mx-danger-bg, rgba(214, 69, 69, 0.12));
  border-color: var(--mx-danger, #d64545);
}
.matrix-error {
  color: var(--mx-danger, #d64545);
  font-size: 0.84rem;
  min-height: 1.15rem;
  padding-top: 0.15rem;
}

@media (max-width: 640px) {
  .matrix-grid {
    gap: 0.3rem;
  }

  .matrix-cell {
    border-radius: 0.7rem;
    height: 2rem;
  }

  .matrix-expression {
    padding: 0.3rem 0.7rem;
  }

  .matrix-controls {
    gap: 0.22rem;
  }

  .matrix-controls button {
    font-size: 0.78rem;
    padding: 0.26rem 0.5rem;
  }
}
"""

MATRIX_GRID_JS = r"""
const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

function clampInt(value, fallback, minValue, maxValue) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maxValue, Math.max(minValue, parsed));
}

function parseCell(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return { ok: false, value: null, message: "不能为空" };

  const parts = text.split("/");
  if (parts.length > 2) return { ok: false, value: null, message: "只能有一个除号" };
  if (parts.some((part) => !NUMBER.test(part.trim()))) {
    return { ok: false, value: null, message: "请输入数字，例如 3、-2.5、1/2" };
  }

  const numerator = Number(parts[0]);
  if (parts.length === 1) {
    return Number.isFinite(numerator)
      ? { ok: true, value: numerator, message: "" }
      : { ok: false, value: null, message: "数字不合法" };
  }

  const denominator = Number(parts[1]);
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return { ok: false, value: null, message: "分母不能为 0" };
  }
  return { ok: true, value: numerator / denominator, message: "" };
}

function collectMatrix(root, rows, cols) {
  const values = [];
  const numbers = [];
  const errors = [];

  for (let r = 0; r < rows; r += 1) {
    const rowValues = [];
    const rowNumbers = [];
    for (let c = 0; c < cols; c += 1) {
      const input = root.querySelector(`.matrix-cell[data-row="${r}"][data-col="${c}"]`);
      const raw = input ? input.value : "";
      const parsed = parseCell(raw);
      rowValues.push(raw);
      rowNumbers.push(parsed.ok ? parsed.value : null);
      if (!parsed.ok) {
        errors.push({ row: r, col: c, message: parsed.message });
      }
    }
    values.push(rowValues);
    numbers.push(rowNumbers);
  }

  return { rows, cols, values, numbers, valid: errors.length === 0, errors };
}

function commitMatrix(root) {
  const rows = clampInt(root.dataset.rows, 1, 1, 12);
  const cols = clampInt(root.dataset.cols, 1, 1, 12);
  const payload = collectMatrix(root, rows, cols);
  root._setStateValue("matrix", payload);
  return payload;
}

function resizeMatrix(root, action) {
  const oldRows = clampInt(root.dataset.rows, 1, 1, 12);
  const oldCols = clampInt(root.dataset.cols, 1, 1, 12);
  let rows = oldRows;
  let cols = oldCols;

  if (action === "remove-row") rows -= 1;
  if (action === "add-row") rows += 1;
  if (action === "remove-col") cols -= 1;
  if (action === "add-col") cols += 1;
  if (action === "remove-order") {
    rows -= 1;
    cols -= 1;
  }
  if (action === "add-order") {
    rows += 1;
    cols += 1;
  }

  const minRows = clampInt(root._data.minRows, 1, 1, 12);
  const minCols = clampInt(root._data.minCols, 1, 1, 12);
  const maxRows = clampInt(root._data.maxRows, 12, 1, 12);
  const maxCols = clampInt(root._data.maxCols, 12, 1, 12);
  rows = clampInt(rows, oldRows, minRows, maxRows);
  cols = clampInt(cols, oldCols, minCols, maxCols);
  if (rows === oldRows && cols === oldCols) return;

  const previous = collectMatrix(root, oldRows, oldCols);
  const values = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => previous.values[r]?.[c] ?? "0")
  );
  const nextData = { ...root._data, rows, cols, values };
  root._data = nextData;
  renderWidget(root, nextData);
  commitMatrix(root);

  const focusRow = action.startsWith("add") ? rows - 1 : 0;
  const focusCol = action === "add-col" ? cols - 1 : 0;
  // preventScroll：移动端宽矩阵横向滚动时，聚焦新格不会把首列顶出视口
  root.querySelector(`.matrix-cell[data-row="${focusRow}"][data-col="${focusCol}"]`)
    ?.focus({ preventScroll: true });
}

function validateCell(input) {
  const parsed = parseCell(input.value);
  input.classList.toggle("invalid", !parsed.ok);
  input.setAttribute("aria-invalid", parsed.ok ? "false" : "true");
  input.title = parsed.ok ? "" : parsed.message;
}

function focusCell(root, row, col) {
  root.querySelector(`.matrix-cell[data-row="${row}"][data-col="${col}"]`)?.focus();
}

function bindWidget(root, grid) {
  root.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (button && root.contains(button)) {
      resizeMatrix(root, button.dataset.action);
    }
  });

  grid.addEventListener("input", (event) => {
    if (event.target.matches(".matrix-cell")) validateCell(event.target);
  });

  grid.addEventListener("change", (event) => {
    if (event.target.matches(".matrix-cell")) commitMatrix(root);
  });

  grid.addEventListener("keydown", (event) => {
    if (!event.target.matches(".matrix-cell")) return;
    const rows = clampInt(root.dataset.rows, 1, 1, 12);
    const cols = clampInt(root.dataset.cols, 1, 1, 12);
    let row = Number.parseInt(event.target.dataset.row, 10);
    let col = Number.parseInt(event.target.dataset.col, 10);
    let nextRow = row;
    let nextCol = col;

    if (event.key === "ArrowRight") {
      if (col < cols - 1) nextCol += 1;
      else if (row < rows - 1) {
        nextRow += 1;
        nextCol = 0;
      }
    } else if (event.key === "ArrowLeft") {
      if (col > 0) nextCol -= 1;
      else if (row > 0) {
        nextRow -= 1;
        nextCol = cols - 1;
      }
    } else if (event.key === "ArrowDown") {
      if (row < rows - 1) nextRow += 1;
    } else if (event.key === "ArrowUp") {
      if (row > 0) nextRow -= 1;
    } else if (event.key === "Enter") {
      event.preventDefault();
      commitMatrix(root);
      if (row < rows - 1) nextRow += 1;
      else if (col < cols - 1) {
        nextRow = 0;
        nextCol += 1;
      }
    } else {
      return;
    }

    event.preventDefault();
    if (nextRow !== row || nextCol !== col) focusCell(root, nextRow, nextCol);
  });
}

function renderWidget(root, data) {
  const rows = clampInt(data.rows, 1, 1, 12);
  const cols = clampInt(data.cols, 1, 1, 12);
  const square = Boolean(data.square);
  const minRows = clampInt(data.minRows, 1, 1, 12);
  const minCols = clampInt(data.minCols, 1, 1, 12);
  const maxRows = clampInt(data.maxRows, 12, 1, 12);
  const maxCols = clampInt(data.maxCols, 12, 1, 12);

  root.dataset.rows = String(rows);
  root.dataset.cols = String(cols);
  root.setAttribute("aria-label", data.label || "矩阵输入器");

  root.querySelector(".matrix-symbol").textContent = `${data.symbol || "A"} =`;
  root.querySelector(".matrix-size").textContent = square ? `${rows} 阶` : `${rows} × ${cols}`;

  const rowButtons = ["remove-row", "add-row", "remove-col", "add-col"];
  const orderButtons = ["remove-order", "add-order"];
  rowButtons.forEach((action) => {
    root.querySelector(`[data-action="${action}"]`).hidden = square;
  });
  orderButtons.forEach((action) => {
    root.querySelector(`[data-action="${action}"]`).hidden = !square;
  });

  root.querySelector('[data-action="remove-row"]').disabled = rows <= minRows;
  root.querySelector('[data-action="add-row"]').disabled = rows >= maxRows;
  root.querySelector('[data-action="remove-col"]').disabled = cols <= minCols;
  root.querySelector('[data-action="add-col"]').disabled = cols >= maxCols;
  root.querySelector('[data-action="remove-order"]').disabled = rows <= minRows;
  root.querySelector('[data-action="add-order"]').disabled = rows >= maxRows;

  const grid = root.querySelector(".matrix-grid");
  grid.style.gridTemplateColumns = `repeat(${cols}, 3.55rem)`;
  grid.setAttribute("aria-label", data.label || "矩阵输入器");

  const wanted = rows * cols;
  while (grid.children.length < wanted) {
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.className = "matrix-cell";
    grid.appendChild(input);
  }
  while (grid.children.length > wanted) {
    grid.removeChild(grid.lastElementChild);
  }

  Array.from(grid.children).forEach((input, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    input.dataset.row = String(row);
    input.dataset.col = String(col);
    input.setAttribute("role", "gridcell");
    input.setAttribute("aria-label", `${data.label || "矩阵"} 第 ${row + 1} 行第 ${col + 1} 列`);

    const nextValue = String(data.values?.[row]?.[col] ?? "");
    if (document.activeElement !== input && input.value !== nextValue) {
      input.value = nextValue;
    }
    validateCell(input);
  });

  const firstError = Array.isArray(data.errors) ? data.errors[0] : null;
  root.querySelector(".matrix-error").textContent = firstError
    ? `第 ${firstError.row + 1} 行第 ${firstError.col + 1} 列：${firstError.message}`
    : "";
}

export default function matrixGrid(component) {
  const { parentElement, data, setStateValue } = component;
  const root = parentElement.querySelector('[data-testid="matrix-widget"]');
  if (!root || !data) return;

  root._setStateValue = setStateValue;
  root._data = data;

  if (!root.dataset.bound) {
    bindWidget(root, root.querySelector(".matrix-grid"));
    root.dataset.bound = "true";
  }

  renderWidget(root, data);
}
"""

_matrix_grid_component = st.components.v2.component(
    "matrix_calculator.matrix_grid",
    html=MATRIX_GRID_HTML,
    css=MATRIX_GRID_CSS,
    js=MATRIX_GRID_JS,
)


# ==================== 矩阵输入组件 ====================


_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _state_keys(key: str) -> tuple[str, str]:
    return f"matrix:{key}:dims", f"matrix:{key}:raw"


def _component_key(key: str) -> str:
    return f"matrix-grid:{key}"


def _clamp_dimension(value: object, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return min(maximum, max(minimum, parsed))


def _format_number(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-10:
        return str(int(round(number)))
    return f"{number:.6g}"


def _parse_cell(raw: object) -> float:
    """解析小数或简单分数；不使用 eval，避免输入表达式被执行。"""
    text = str(raw).strip()
    parts = [part.strip() for part in text.split("/")]
    if not text or len(parts) > 2 or any(not _NUMBER_PATTERN.match(part) for part in parts):
        raise ValueError("请输入数字，例如 3、-2.5、1/2")

    numerator = float(parts[0])
    if len(parts) == 1:
        return numerator

    denominator = float(parts[1])
    if denominator == 0:
        raise ValueError("分母不能为 0")
    return numerator / denominator


def _raw_from_matrix(matrix: np.ndarray) -> list[list[str]]:
    return [[_format_number(value) for value in row] for row in matrix.tolist()]


def _normalized_raw(raw: object, rows: int, cols: int) -> list[list[str]]:
    result = []
    for i in range(rows):
        source_row = raw[i] if isinstance(raw, list) and i < len(raw) and isinstance(raw[i], list) else []
        row = []
        for j in range(cols):
            try:
                row.append(str(source_row[j]).strip())
            except (IndexError, KeyError, TypeError):
                row.append("0")
        result.append(row)
    return result


def _ensure_matrix(
    key: str,
    default: np.ndarray | None = None,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
) -> None:
    dims_key, raw_key = _state_keys(key)
    if dims_key not in st.session_state or raw_key not in st.session_state:
        matrix = np.eye(2, dtype=float) if default is None else np.asarray(default, dtype=float)
        if matrix.ndim != 2:
            matrix = np.atleast_2d(matrix)
        st.session_state[dims_key] = [
            max(min_rows, int(matrix.shape[0])),
            max(min_cols, int(matrix.shape[1])),
        ]
        st.session_state[raw_key] = _raw_from_matrix(matrix)


def _sync_matrix_from_component(
    key: str,
    *,
    min_rows: int,
    min_cols: int,
    square: bool,
) -> None:
    """组件状态变化时，在本次脚本重跑前同步维度和原始文本。"""
    component_state = st.session_state.get(_component_key(key)) or {}
    payload = component_state.get("matrix") if isinstance(component_state, dict) else None
    if not isinstance(payload, dict):
        return

    dims_key, raw_key = _state_keys(key)
    fallback_rows, fallback_cols = st.session_state.get(dims_key, [min_rows, min_cols])
    effective_min_cols = min_rows if square else min_cols
    rows = _clamp_dimension(payload.get("rows"), int(fallback_rows), min_rows, MAX_DIMENSION)
    cols = rows if square else _clamp_dimension(
        payload.get("cols"), int(fallback_cols), effective_min_cols, MAX_DIMENSION
    )

    st.session_state[dims_key] = [rows, cols]
    st.session_state[raw_key] = _normalized_raw(payload.get("values"), rows, cols)


def _build_matrix_data(
    key: str,
    label: str,
    symbol: str,
    *,
    min_rows: int,
    min_cols: int,
    square: bool,
) -> tuple[dict, np.ndarray, list[str]]:
    dims_key, raw_key = _state_keys(key)
    rows, cols = (int(value) for value in st.session_state[dims_key])
    if square:
        cols = rows
    raw = _normalized_raw(st.session_state[raw_key], rows, cols)

    numbers = []
    errors = []
    error_payload = []
    for i, row in enumerate(raw):
        number_row = []
        for j, cell in enumerate(row):
            try:
                number_row.append(_parse_cell(cell))
            except ValueError as exc:
                number_row.append(0.0)
                message = str(exc)
                errors.append(f"第 {i + 1} 行第 {j + 1} 列：{message}")
                error_payload.append({"row": i, "col": j, "message": message})
        numbers.append(number_row)

    payload = {
        "rows": rows,
        "cols": cols,
        "values": raw,
        "numbers": numbers,
        "valid": not errors,
        "errors": error_payload,
    }
    data = {
        **payload,
        "label": label,
        "symbol": symbol,
        "square": square,
        "minRows": min_rows,
        "minCols": min_rows if square else min_cols,
        "maxRows": MAX_DIMENSION,
        "maxCols": MAX_DIMENSION,
    }
    return data, np.asarray(numbers, dtype=float), errors


def matrix_input(
    label: str,
    key: str,
    default: np.ndarray | None = None,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
    square: bool = False,
    symbol: str = "A",
) -> np.ndarray:
    """渲染一个接近数学书写形式的自定义矩阵输入器。"""
    _ensure_matrix(key, default, min_rows=min_rows, min_cols=min_cols)
    data, matrix, errors = _build_matrix_data(
        key,
        label,
        symbol,
        min_rows=min_rows,
        min_cols=min_cols,
        square=square,
    )

    _matrix_grid_component(
        key=_component_key(key),
        data=data,
        default={"matrix": data},
        width="content",
        on_matrix_change=lambda: _sync_matrix_from_component(
            key,
            min_rows=min_rows,
            min_cols=min_cols,
            square=square,
        ),
    )

    if errors:
        st.error(
            "矩阵中有不合法的输入："
            + "；".join(errors[:3])
            + "。支持整数、小数、科学计数法（如 1e-3）和简单分数（如 1/2）。"
        )
        st.stop()
    return matrix


RESULT_SERIAL = count(1)




def show_text_matrix(label: str, value) -> None:
    """展示教材风格矩阵；st.code 自带复制按钮，另提供 TXT/CSV/NumPy 导出。"""
    result_id = next(RESULT_SERIAL)
    matrix_text = format_matrix(value)
    csv_text = matrix_to_csv_text(value)
    numpy_text = matrix_to_numpy_text(value)

    with st.container(border=True):
        title_col, export_col = st.columns([3, 1], vertical_alignment="center")
        title_col.markdown(f"**{label}**")
        with export_col:
            with st.popover("导出", width="stretch"):
                st.download_button(
                    "下载 TXT",
                    data=matrix_text,
                    file_name=f"matrix-{result_id}.txt",
                    mime="text/plain",
                    key=f"download-txt-{result_id}",
                    width="stretch",
                )
                st.download_button(
                    "下载 CSV",
                    data=csv_text,
                    file_name=f"matrix-{result_id}.csv",
                    mime="text/csv",
                    key=f"download-csv-{result_id}",
                    width="stretch",
                )
                st.download_button(
                    "NumPy 文本",
                    data=numpy_text,
                    file_name=f"matrix-{result_id}.py",
                    mime="text/x-python",
                    key=f"download-numpy-{result_id}",
                    width="stretch",
                )
        st.code(matrix_text, language="text", wrap_lines=True)


def show_error(exc: Exception) -> None:
    st.error(f"输入或计算有误：{exc}")


def shape_text(matrix: np.ndarray) -> str:
    return f"{matrix.shape[0]} × {matrix.shape[1]}"


def explain_binary_error(
    exc: Exception,
    operation_code: str,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
) -> str:
    """把核心库的通用异常翻译成网页上可操作的维度提示。"""
    if operation_code == "multiply" and mat_a.shape[1] != mat_b.shape[0]:
        return (
            f"A × B 要求 A 的列数等于 B 的行数。当前 A 是 {shape_text(mat_a)}，"
            f"有 {mat_a.shape[1]} 列；B 是 {shape_text(mat_b)}，有 {mat_b.shape[0]} 行。"
            "请调整矩阵维度，或改选 A + B / A ⊙ B。"
        )

    elementwise_operations = {
        "add": ("A + B", "相加"),
        "subtract": ("A − B", "相减"),
        "hadamard_multiply": ("A ⊙ B", "对应元素相乘"),
        "hadamard_divide": ("A ⊘ B", "对应元素相除"),
    }
    if operation_code in elementwise_operations and mat_a.shape != mat_b.shape:
        symbol, action = elementwise_operations[operation_code]
        return (
            f"{symbol} 要求两个矩阵形状完全相同，才能逐元素{action}。"
            f"当前 A 是 {shape_text(mat_a)}，B 是 {shape_text(mat_b)}。"
            "请用 +行 / +列 调整成相同形状。"
        )

    if operation_code == "hadamard_divide" and mat_b.size and np.any(mat_b == 0):
        return "A ⊘ B 是对应元素相除；B 中包含 0，会产生无穷大，因此已取消计算。请把 0 改成非零数。"

    return str(exc)


def explain_unary_error(exc: Exception, operation_code: str, mat: np.ndarray) -> str:
    square_only = {"determinant", "inverse", "eigendecomposition"}
    labels = {
        "determinant": "det(A)",
        "inverse": "A⁻¹",
        "eigendecomposition": "特征分解",
    }
    if operation_code in square_only and mat.shape[0] != mat.shape[1]:
        return (
            f"{labels[operation_code]} 只能用于方阵。当前矩阵是 {shape_text(mat)}，"
            "请调整成相同行数和列数；非方阵可以选择 SVD 或 Aᵀ。"
        )
    return str(exc)


def _matrix_signature(matrix: np.ndarray) -> tuple[tuple[int, int], tuple[float, ...]]:
    """生成可比较的矩阵输入签名，用于判断上次结果是否已经过期。"""
    array = np.asarray(matrix, dtype=float)
    return array.shape, tuple(array.ravel().tolist())


def render_matrix_analysis(
    matrix: np.ndarray,
    operation_code: str,
    symbol: str = "A",
) -> None:
    """渲染单矩阵分析结果，供单矩阵页和双矩阵结果复用。"""
    if operation_code == "determinant":
        with st.container(border=True):
            st.metric(f"行列式 det({symbol})", f"{determinant(matrix):.6g}")
    elif operation_code == "inverse":
        show_text_matrix(f"逆矩阵 {symbol}⁻¹", invert_matrix(matrix))
    elif operation_code == "transpose":
        show_text_matrix(f"转置 {symbol}ᵀ", transpose(matrix))
    elif operation_code == "eigendecomposition":
        eigenvalues, eigenvectors = eigendecompose(matrix)
        show_text_matrix("特征值", eigenvalues)
        show_text_matrix("特征向量（每列对应一个特征值）", eigenvectors)
        if np.iscomplexobj(eigenvalues) and np.any(np.abs(eigenvalues.imag) > 1e-9):
            st.info("该矩阵存在复数特征值或特征向量。")
    elif operation_code == "svd":
        u, singular_values, vt = singular_value_decompose(matrix)
        show_text_matrix("U", u)
        show_text_matrix("奇异值 S", singular_values)
        show_text_matrix("Vᵀ", vt)


# ==================== 页面：双矩阵运算 ====================


BINARY_OPERATIONS = [
    ("add", "A + B", add_matrices),
    ("subtract", "A − B", subtract_matrices),
    ("multiply", "A × B", matrix_multiply),
    ("hadamard_multiply", "A ⊙ B", hadamard_multiply),
    ("hadamard_divide", "A ⊘ B", hadamard_divide),
]
BINARY_OPTION_LABELS = {code: label for code, label, _ in BINARY_OPERATIONS}
BINARY_OPERATION_FUNCTIONS = {code: func for code, _, func in BINARY_OPERATIONS}


BINARY_RESULT_KEY = "binary:last_result"
BINARY_RESULT_ANALYSIS_KEY = "binary:result_analysis"
BINARY_RESULT_ANALYSIS_LABELS = {
    "eigendecomposition": "特征分解",
    "svd": "SVD",
    "transpose": "Cᵀ",
}


BINARY_EXAMPLES = [
    {
        "label": "I＋交换矩阵",
        "operation": "add",
        "a": np.eye(2),
        "b": np.array([[0, 1], [1, 0]], dtype=float),
    },
    {
        "label": "2×3 × 3×2",
        "operation": "multiply",
        "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
        "b": np.array([[7, 8], [9, 10], [11, 12]], dtype=float),
    },
    {
        "label": "非方阵相加",
        "operation": "add",
        "a": np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
        "b": np.array([[0, 1, 0], [1, 0, 0]], dtype=float),
    },
]

UNARY_EXAMPLES = [
    {
        "label": "旋转矩阵",
        "operation": "determinant",
        "matrix": np.array([[0, -1], [1, 0]], dtype=float),
    },
    {
        "label": "奇异矩阵",
        "operation": "determinant",
        "matrix": np.array([[1, 1], [1, 1]], dtype=float),
    },
    {
        "label": "可逆三角阵",
        "operation": "inverse",
        "matrix": np.array([[2, 1], [0, 3]], dtype=float),
    },
    {
        "label": "3×3 可逆",
        "operation": "determinant",
        "matrix": np.array([[2, 1, 0], [1, 3, 2], [0, 2, 4]], dtype=float),
    },
]

SOLVE_EXAMPLES = [
    {
        "label": "唯一解",
        "matrix": np.array([[1, 1, 3], [2, -1, 0]], dtype=float),
    },
    {
        "label": "无穷多解",
        "matrix": np.array([[1, 1, 1], [2, 2, 2]], dtype=float),
    },
    {
        "label": "无解",
        "matrix": np.array([[1, 1, 1], [1, 1, 2]], dtype=float),
    },
]

QUADRATIC_EXAMPLES = [
    {"label": "正定", "matrix": np.eye(2)},
    {"label": "负定", "matrix": -np.eye(2)},
    {"label": "不定", "matrix": np.diag([1.0, -1.0])},
    {"label": "半正定", "matrix": np.array([[1, 1], [1, 1]], dtype=float)},
]


def _set_matrix_example(key: str, matrix: np.ndarray) -> None:
    array = np.asarray(matrix, dtype=float)
    dims_key, raw_key = _state_keys(key)
    st.session_state[dims_key] = [int(array.shape[0]), int(array.shape[1])]
    st.session_state[raw_key] = _raw_from_matrix(array)
    st.session_state.pop(_component_key(key), None)


def render_example_buttons(examples: list[dict], apply_example, prefix: str) -> None:
    with st.container(border=True):
        st.caption("示例：点击即可载入，然后再点页面下方的计算按钮。")
        columns = st.columns(min(4, len(examples)), wrap=True)
        for index, (column, example) in enumerate(zip(columns, examples)):
            with column:
                if st.button(
                    example["label"],
                    key=f"{prefix}-example-{index}",
                    width="stretch",
                    wrap=True,
                ):
                    apply_example(example)
                    st.rerun()


def _apply_binary_example(example: dict) -> None:
    st.session_state["binary_operation"] = example["operation"]
    _set_matrix_example("binary-a", example["a"])
    _set_matrix_example("binary-b", example["b"])
    st.session_state.pop(BINARY_RESULT_KEY, None)
    st.session_state.pop(BINARY_RESULT_ANALYSIS_KEY, None)


def _apply_unary_example(example: dict) -> None:
    st.session_state["unary_operation"] = example["operation"]
    _set_matrix_example("unary-a", example["matrix"])


def _apply_solve_example(example: dict) -> None:
    _set_matrix_example("solve-ab", example["matrix"])


def _apply_quadratic_example(example: dict) -> None:
    _set_matrix_example("quadratic-a", example["matrix"])


def _binary_input_signature(
    operation_code: str,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
) -> tuple[str, tuple, tuple]:
    return (
        operation_code,
        _matrix_signature(mat_a),
        _matrix_signature(mat_b),
    )


def _render_binary_result(
    result: np.ndarray,
    operation_code: str,
    *,
    stale: bool,
) -> None:
    """展示 C=A*B，并把常用单矩阵分析直接接到结果下面。"""
    operation_label = BINARY_OPTION_LABELS[operation_code]
    if stale:
        st.warning("A、B 或运算方式已改变，下面是上一次结果；点击“计算”后可继续分析。")

    show_text_matrix(f"运算结果 C = {operation_label}", result)

    rows, cols = result.shape
    is_square = rows == cols
    determinant_value = None
    inverse_matrix = None

    if is_square:
        determinant_value = determinant(result)
        try:
            inverse_matrix = invert_matrix(result)
        except (MatrixOperationError, np.linalg.LinAlgError):
            inverse_matrix = None

        col_shape, col_determinant, col_invertible = st.columns(3)
        with col_shape:
            with st.container(border=True):
                st.metric("形状 C", f"{rows} × {cols}")
        with col_determinant:
            with st.container(border=True):
                st.metric("行列式 det(C)", f"{determinant_value:.6g}")
        with col_invertible:
            with st.container(border=True):
                st.metric("C 是否可逆", "可逆" if inverse_matrix is not None else "不可逆")

        if inverse_matrix is not None:
            show_text_matrix("逆矩阵 C⁻¹", inverse_matrix)
        else:
            st.info("det(C)=0（或数值上接近奇异），所以 C 不存在可用的逆矩阵。")
    else:
        col_shape, _, _ = st.columns(3)
        with col_shape:
            with st.container(border=True):
                st.metric("形状 C", f"{rows} × {cols}")
        st.caption("非方阵没有行列式和逆矩阵；仍可继续查看 SVD 或转置。")

    analysis_options = (
        ["eigendecomposition", "svd", "transpose"]
        if is_square
        else ["svd", "transpose"]
    )
    if st.session_state.get(BINARY_RESULT_ANALYSIS_KEY) not in analysis_options:
        st.session_state[BINARY_RESULT_ANALYSIS_KEY] = None

    with st.container(border=True):
        st.caption("继续分析结果 C，选择一项即可；det(C) 和可逆时的 C⁻¹ 已自动显示。")
        result_operation = st.segmented_control(
            "继续分析结果 C",
            options=analysis_options,
            format_func=BINARY_RESULT_ANALYSIS_LABELS.get,
            default=None,
            required=False,
            label_visibility="collapsed",
            width="stretch",
            key=BINARY_RESULT_ANALYSIS_KEY,
            disabled=stale,
        )

    if result_operation is not None and not stale:
        render_matrix_analysis(result, result_operation, symbol="C")


def page_binary() -> None:
    st.header("双矩阵运算")

    render_example_buttons(BINARY_EXAMPLES, _apply_binary_example, "binary")

    with st.container(border=True):
        operation_code = st.segmented_control(
            "选择双矩阵运算",
            options=[code for code, _, _ in BINARY_OPERATIONS],
            format_func=BINARY_OPTION_LABELS.get,
            default="add",
            required=True,
            label_visibility="collapsed",
            width="stretch",
            key="binary_operation",
        )
        st.caption("A ⊙ B 和 A ⊘ B 表示对应元素相乘 / 相除；A × B 是矩阵乘法。")

    with st.container(border=True):
        mat_a = matrix_input("矩阵 A", "binary-a", symbol="A")
        st.caption("对下面的矩阵 B 应用上面选择的运算。")
        mat_b = matrix_input("矩阵 B", "binary-b", symbol="B")

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_binary = st.button("计算", type="primary", key="run_binary")

    current_signature = _binary_input_signature(operation_code, mat_a, mat_b)
    if run_binary:
        try:
            result = BINARY_OPERATION_FUNCTIONS[operation_code](mat_a, mat_b)
            st.session_state[BINARY_RESULT_KEY] = {
                "operation": operation_code,
                "matrix": result,
                "signature": current_signature,
            }
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            st.session_state.pop(BINARY_RESULT_KEY, None)
            message = explain_binary_error(exc, operation_code, mat_a, mat_b)
            show_error(MatrixOperationError(message))

    cached_result = st.session_state.get(BINARY_RESULT_KEY)
    if cached_result is not None:
        _render_binary_result(
            cached_result["matrix"],
            cached_result["operation"],
            stale=cached_result["signature"] != current_signature,
        )


# ==================== 页面：单矩阵运算 ====================


UNARY_OPERATIONS = {
    "determinant": "det(A)",
    "inverse": "A⁻¹",
    "eigendecomposition": "特征分解",
    "svd": "SVD",
    "transpose": "Aᵀ",
}


def page_unary() -> None:
    st.header("单矩阵运算")

    render_example_buttons(UNARY_EXAMPLES, _apply_unary_example, "unary")

    with st.container(border=True):
        operation_code = st.segmented_control(
            "选择单矩阵运算",
            options=list(UNARY_OPERATIONS),
            format_func=UNARY_OPERATIONS.get,
            default="determinant",
            required=True,
            label_visibility="collapsed",
            width="stretch",
            key="unary_operation",
        )

    with st.container(border=True):
        mat = matrix_input("矩阵 A", "unary-a", symbol="A")

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_unary = st.button("计算", type="primary", key="run_unary")

    if run_unary:
        try:
            render_matrix_analysis(mat, operation_code, symbol="A")
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            message = explain_unary_error(exc, operation_code, mat)
            show_error(MatrixOperationError(message))


# ==================== 页面：线性方程组 ====================


def page_solve() -> None:
    st.header("解线性方程组 Ax = b")

    render_example_buttons(SOLVE_EXAMPLES, _apply_solve_example, "solve")

    with st.container(border=True):
        st.caption("输入增广矩阵 [A | b]：最后一列自动作为常数项 b，可通过 +列 增加未知数。")
        augmented = matrix_input(
            "增广矩阵 [A | b]",
            "solve-ab",
            default=np.zeros((2, 3)),
            min_cols=2,
            symbol="[A | b]",
        )

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_solve = st.button("求解", type="primary", key="run_solve")

    if run_solve:
        m, n_plus_one = augmented.shape
        n = n_plus_one - 1
        a = augmented[:, :n]
        b = augmented[:, -1]
        try:
            status, info = solve_linear(a, b)
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.metric("r(A)", info["rA"])
            with col2:
                with st.container(border=True):
                    st.metric("r(A|b)", info["rAb"])
            with col3:
                with st.container(border=True):
                    st.metric("未知数 n", info["n"])

            if status == "unique":
                st.success("方程组有唯一解。")
                show_text_matrix("唯一解 x", info["x"].reshape(-1, 1))
            elif status == "infinite":
                st.warning("方程组有无穷多解。")
                st.write(f"自由未知数个数：{n - info['rA']}")
                show_text_matrix("一个特解 xp", info["xp"].reshape(-1, 1))
                show_text_matrix("基础解系（每列一个向量）", info["null"])
            else:
                st.error("方程组无解（矛盾方程组）。下面是最小二乘近似解。")
                show_text_matrix("最小二乘近似解", info["x_ls"].reshape(-1, 1))
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 页面：二次型 ====================


def page_quadratic() -> None:
    st.header("二次型 f = xᵀAx")

    render_example_buttons(QUADRATIC_EXAMPLES, _apply_quadratic_example, "quadratic")

    with st.container(border=True):
        st.caption("输入方阵阶数可通过 −阶 / +阶 调整；默认是 2 阶单位矩阵。")
        mat = matrix_input(
            "矩阵 A",
            "quadratic-a",
            default=np.eye(2),
            square=True,
            min_rows=1,
            symbol="A",
        )

    with st.container(horizontal=True, horizontal_alignment="right"):
        run_quad = st.button("分析", type="primary", key="run_quad")

    if run_quad:
        try:
            result = analyze_quadratic(mat)
            if not result["symmetric"]:
                st.info("输入矩阵不对称；二次型只取决于对称部分，已自动使用 S=(A+Aᵀ)/2。")

            show_text_matrix("用于分析的对称矩阵 S", result["matrix"])
            show_text_matrix("特征值（升序）", result["eigenvalues"])
            show_text_matrix("正交变换矩阵 Q", result["Q"])
            standard_form = signed_polynomial(result["standard_terms"])
            st.markdown(f"经正交变换 x = Qy，标准形为：**f = {standard_form}**")

            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.metric("正惯性指数 p", result["pos"])
            with c2:
                with st.container(border=True):
                    st.metric("负惯性指数 q", result["neg"])
            with c3:
                with st.container(border=True):
                    st.metric("零特征值个数", result["zero"])
            st.success(f"判定结果：{result['kind']}")

            minors = result["minors"]
            minors_text = "，".join(
                f"Δ{i} = {float(value):.4g}" for i, value in enumerate(minors, start=1)
            )
            st.caption(f"顺序主子式（Sylvester 判据）：{minors_text}")
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 主导航 ====================


THEME_LABELS = ["跟随系统", "浅色", "深色"]
_THEME_MODE_MAP = {"跟随系统": "system", "浅色": "light", "深色": "dark"}


def main() -> None:
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
    # 先拿到主题选择，再在主内容渲染前注入对应色板，避免切换闪烁
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


if __name__ == "__main__":
    main()

