"""PySide6 桌面应用启动入口。"""

from __future__ import annotations


def main() -> int:
    """创建 QApplication 与主窗口并进入事件循环。"""
    import sys

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .style import STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("矩阵计算器")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
