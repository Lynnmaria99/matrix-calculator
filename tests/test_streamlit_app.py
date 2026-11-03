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



def test_streamlit_binary_operations_are_visible_segments():
    app = _start_page("双矩阵运算")

    assert not app.selectbox
    assert len(app.segmented_control) == 1
    assert app.segmented_control[0].value == "add"

    app.segmented_control[0].set_value("multiply").run()
    _click(app, "计算")

    assert not app.exception
    assert not app.error


def test_streamlit_unary_operations_are_visible_segments():
    app = _start_page("单矩阵运算")

    assert not app.selectbox
    assert len(app.segmented_control) == 1
    assert app.segmented_control[0].value == "determinant"

    app.segmented_control[0].set_value("transpose").run()
    _click(app, "计算")

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

def test_streamlit_binary_result_shows_determinant_and_inverse():
    app = _load_page("双矩阵运算")

    assert not app.exception
    assert any(metric.label == "形状 C" and metric.value == "2 × 2" for metric in app.metric)
    assert any(metric.label == "行列式 det(C)" and metric.value == "4" for metric in app.metric)
    assert any(metric.label == "C 是否可逆" and metric.value == "可逆" for metric in app.metric)
    assert any("逆矩阵 C⁻¹" in markdown.value for markdown in app.markdown)
    assert any("0.5" in text.value for text in app.text)


def test_streamlit_binary_singular_result_reports_missing_inverse():
    app = _start_page("双矩阵运算")
    app.session_state["matrix:binary-b:raw"] = [
        ["0", "1"],
        ["1", "0"],
    ]
    app.run()
    _click(app, "计算")

    assert not app.exception
    assert any(metric.label == "行列式 det(C)" and metric.value == "0" for metric in app.metric)
    assert any(metric.label == "C 是否可逆" and metric.value == "不可逆" for metric in app.metric)
    assert app.info
    assert "不存在可用的逆矩阵" in app.info[0].value
    assert not any("逆矩阵 C⁻¹" in markdown.value for markdown in app.markdown)


def test_streamlit_binary_result_followup_svd_persists_after_rerun():
    app = _load_page("双矩阵运算")

    assert len(app.segmented_control) == 2
    app.segmented_control[1].set_value("svd").run()

    assert not app.exception
    assert any(markdown.value.strip() == "**U**" for markdown in app.markdown)
    assert any("奇异值 S" in markdown.value for markdown in app.markdown)
    assert any("Vᵀ" in markdown.value for markdown in app.markdown)


def test_streamlit_non_square_binary_result_offers_svd_and_transpose():
    app = _start_page("双矩阵运算")
    app.session_state["matrix:binary-a:dims"] = [2, 3]
    app.session_state["matrix:binary-a:raw"] = [
        ["1", "0", "0"],
        ["0", "1", "0"],
    ]
    app.session_state["matrix:binary-b:dims"] = [2, 3]
    app.session_state["matrix:binary-b:raw"] = [
        ["0", "1", "0"],
        ["1", "0", "0"],
    ]
    app.run()
    _click(app, "计算")

    assert not app.exception
    assert any(metric.label == "形状 C" and metric.value == "2 × 3" for metric in app.metric)
    assert not any(metric.label.startswith("行列式") for metric in app.metric)
    assert app.segmented_control[1].options == ["SVD", "Cᵀ"]

    app.segmented_control[1].set_value("transpose").run()
    assert not app.exception
    assert any("转置 Cᵀ" in markdown.value for markdown in app.markdown)
