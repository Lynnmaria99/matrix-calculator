from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
PAGES = ["双矩阵运算", "单矩阵运算", "解线性方程组", "二次型分析"]


def _load_page(page_name):
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.run()
    app.sidebar.radio[0].set_value(page_name).run()
    app.button[0].click().run()
    return app


def test_streamlit_app_initial_render():
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.run()

    assert not app.exception
    assert app.title
    assert "矩阵计算器" in app.title[0].value
    assert app.sidebar.radio


@pytest.mark.parametrize("page_name", PAGES)
def test_streamlit_pages_default_action(page_name):
    app = _load_page(page_name)

    assert not app.exception
    assert not app.error


def test_streamlit_quadratic_identity_is_positive_definite():
    app = _load_page("二次型分析")

    assert app.success
    assert "正定" in app.success[0].value