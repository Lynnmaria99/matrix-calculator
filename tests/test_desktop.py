"""以子进程运行桌面 UI 测试。

Qt/PySide6 在解释器关闭阶段的对象清理存在偶发的原生崩溃（Windows 上为
access violation），且与 Streamlit AppTest 不宜同进程。因此真正的桌面
测试放在 ``desktop_ui_tests.py``，由这里的引导脚本在子进程中执行，并在
pytest 结束后用 ``os._exit`` 直接退出，绕开解释器关闭阶段。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)


def test_desktop_ui_suite():
    """子进程偶尔在退出阶段触发 Qt 原生崩溃（与测试结果无关），重试一次。"""
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    entry = os.path.join(_HERE, "_desktop_entry.py")
    last = None
    for _ in range(2):
        last = subprocess.run(
            [sys.executable, entry],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if last.returncode == 0:
            return
    assert last is not None and last.returncode == 0, (
        "桌面 UI 测试子进程失败：\n" + (last.stdout if last else "") + (last.stderr if last else "")
    )
