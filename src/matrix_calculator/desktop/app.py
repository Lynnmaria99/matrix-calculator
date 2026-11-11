"""PySide6 桌面应用启动入口。"""

from __future__ import annotations


def main() -> int:
    """创建 QApplication 与主窗口并进入事件循环。"""
    import sys

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .style import STYLESHEET

    # PySide6 6.12 在 Linux/Python 3.11 下，若在控件回调中懒加载
    # matplotlib 的 QtAgg 后端，其导入期触发的 GC 会在 shiboken
    # 托管对象上发生段错误。必须在 QApplication 创建前预加载图示
    # 后端；matplotlib 仍为可选依赖，缺失时静默降级为无图示模式。
    try:
        from . import plots  # noqa: F401
    except ImportError:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("矩阵计算器")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
