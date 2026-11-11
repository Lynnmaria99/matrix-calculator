"""PyInstaller 的 Windows 桌面入口与包内功能自检。"""

from __future__ import annotations

import multiprocessing


def _run_package_smoke_test(report_path: str) -> int:
    """在打包环境内验证矩阵计算和微积分绘图，并写出机器可读报告。"""
    import json
    import os
    import traceback
    from pathlib import Path

    report = Path(report_path).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        import numpy as np
        import PySide6
        from PySide6 import QtCore, QtGui
        from PySide6.QtWidgets import QApplication

        from matrix_calculator.desktop.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            window.nav_buttons["单矩阵运算"].click()
            window._select_op("det(A)")
            window.input_a.set_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
            window._on_compute()
            assert "-2" in window.result_label.text(), window.result_label.text()

            window.nav_buttons["微积分"].click()
            window.calc_expr.setText("x^2")
            window._on_compute()
            assert "黎曼和" in window.calc_integral_label.text()
            assert window.calc_func_canvas.figure.axes
            assert window.calc_int_canvas.figure.axes
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()
            app.quit()

        report.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "matrix_computation": "passed",
                    "calculus_plotting": "passed",
                    "numpy": np.__version__,
                    "pyside6": PySide6.__version__,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:  # 打包自检需要把失败详情写入报告，而非只显示窗口错误。
        report.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    import sys

    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
        raise SystemExit(_run_package_smoke_test(sys.argv[2]))

    from matrix_calculator.desktop.app import main

    raise SystemExit(main())
