from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
PAGES = {
    "双矩阵运算": "计算",
    "单矩阵运算": "计算",
    "解线性方程组": "求解",
    "二次型分析": "分析",
}


def _start_page(page_name):
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.run()
    app.sidebar.radio[0].set_value(page_name).run()
    return app


def _click(app, label):
    matches = [button for button in app.button if button.label == label]
    assert matches, f"没有找到按钮：{label}"
    return matches[-1].click().run()


def _load_page(page_name):
    app = _start_page(page_name)
    _click(app, PAGES[page_name])
    return app


def test_streamlit_app_initial_render():
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.run()

    assert not app.exception
    assert app.title
    assert "矩阵计算器" in app.title[0].value
    assert app.sidebar.radio


@pytest.mark.parametrize("page_name", PAGES.keys())
def test_streamlit_pages_default_action(page_name):
    app = _load_page(page_name)

    assert not app.exception
    assert not app.error


def test_streamlit_quadratic_identity_is_positive_definite():
    app = _load_page("二次型分析")

    assert app.success
    assert "正定" in app.success[0].value


def test_streamlit_quadratic_supports_three_order_matrix_state():
    app = _start_page("二次型分析")

    # 自定义前端组件点击 +阶 后，回调会写入这两个 Session State。
    app.session_state["matrix:quadratic-a:dims"] = [3, 3]
    app.session_state["matrix:quadratic-a:raw"] = [
        ["1", "0", "0"],
        ["0", "1", "0"],
        ["0", "0", "1"],
    ]
    app.run()

    assert not app.exception
    _click(app, "分析")
    assert app.success
    assert "正定" in app.success[0].value


def test_streamlit_matrix_reports_invalid_cell():
    app = _start_page("二次型分析")
    app.session_state["matrix:quadratic-a:raw"] = [
        ["1", "0"],
        ["abc", "1"],
    ]
    app.run()

    assert not app.exception
    assert app.error
    assert "不合法" in app.error[0].value
