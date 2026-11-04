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

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)


def test_desktop_ui_suite():
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        [sys.executable, os.path.join(_HERE, "_desktop_entry.py")],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "桌面 UI 测试子进程失败：\n" + result.stdout + result.stderr
    )
