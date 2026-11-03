"""pytest 收集顺序：规避 PySide6 与 Streamlit AppTest 的已知冲突。

shiboken6 会在导入 PySide6 时安装签名加载器的 import 钩子；Streamlit
AppTest 在工作线程中执行脚本，触发该钩子时存在非线程安全的竞争，可能
导致原生崩溃（access violation）。因此保证 Streamlit 测试先于任何
PySide6 导入运行，Qt 桌面测试放在最后。
"""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    def sort_key(item):
        name = item.path.name
        if name == "test_streamlit_app.py":
            return 0
        if name == "test_desktop.py":
            return 2
        return 1

    items.sort(key=sort_key)
