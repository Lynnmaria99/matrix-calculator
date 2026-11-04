"""桌面 UI 测试子进程入口：pytest 结束后立即退出，绕开 Qt 清理崩溃。"""

from __future__ import annotations

import os
import sys

import pytest


def main() -> None:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.update(env)
    tests_path = os.path.join(os.path.dirname(__file__), "desktop_ui_tests.py")
    code = pytest.main([tests_path, "-q"])
    os._exit(int(code))


if __name__ == "__main__":
    main()
