# -*- coding: utf-8 -*-
"""Matrix Calculator 的 Streamlit 启动入口。

运行：``streamlit run streamlit_app.py``。具体界面位于
``matrix_calculator.web``，本文件只负责本地源码路径、页面配置与启动。
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="矩阵计算器", page_icon="🧮", layout="wide")

from matrix_calculator.web.app import main  # noqa: E402

# Streamlit 每次交互都会重跑入口，不使用 __main__ 保护。
main()
