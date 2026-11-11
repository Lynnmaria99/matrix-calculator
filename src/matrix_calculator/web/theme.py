"""Streamlit 应用的设计 token、明暗色板与全局样式。"""

from __future__ import annotations

import streamlit as st


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
