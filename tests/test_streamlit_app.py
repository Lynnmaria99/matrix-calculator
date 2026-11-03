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
    assert any("0.5" in code.value for code in app.code)


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


def test_streamlit_result_matrix_has_copy_code_and_export_actions():
    app = _load_page("双矩阵运算")

    assert app.code
    assert any("2" in code.value for code in app.code)
    labels = [button.label for button in app.download_button]
    assert "下载 TXT" in labels
    assert "下载 CSV" in labels
    assert "NumPy 文本" in labels


@pytest.mark.parametrize(
    ("page_name", "example_label", "action_label", "expected_text"),
    [
        ("双矩阵运算", "I＋交换矩阵", "计算", "不可逆"),
        ("单矩阵运算", "奇异矩阵", "计算", "行列式 det(A)"),
        ("解线性方程组", "唯一解", "求解", "唯一解"),
        ("二次型分析", "半正定", "分析", "半正定"),
    ],
)
def test_streamlit_examples_load_from_buttons(
    page_name,
    example_label,
    action_label,
    expected_text,
):
    app = _start_page(page_name)
    _click(app, example_label)
    _click(app, action_label)

    assert not app.exception
    rendered_text = "\n".join(
        [
            *[f"{m.label} {m.value}" for m in app.metric],
            *(m.value for m in app.markdown),
            *(m.value for m in app.success),
        ]
    )
    assert expected_text in rendered_text


def test_streamlit_shape_mismatch_error_is_actionable():
    app = _start_page("双矩阵运算")
    app.session_state["binary_operation"] = "multiply"
    app.session_state["matrix:binary-a:dims"] = [2, 3]
    app.session_state["matrix:binary-a:raw"] = [
        ["1", "0", "0"],
        ["0", "1", "0"],
    ]
    app.session_state["matrix:binary-b:dims"] = [2, 2]
    app.session_state["matrix:binary-b:raw"] = [
        ["1", "0"],
        ["0", "1"],
    ]
    app.run()
    _click(app, "计算")

    assert app.error
    message = app.error[0].value
    assert "A × B 要求 A 的列数等于 B 的行数" in message
    assert "2 × 3" in message
    assert "2 × 2" in message

def test_streamlit_unary_non_square_error_is_actionable():
    page_name = "\u5355\u77e9\u9635\u8fd0\u7b97"
    app = _start_page(page_name)
    app.session_state["unary_operation"] = "determinant"
    app.session_state["matrix:unary-a:dims"] = [2, 3]
    app.session_state["matrix:unary-a:raw"] = [
        ["1", "0", "0"],
        ["0", "1", "0"],
    ]
    app.run()
    _click(app, "\u8ba1\u7b97")

    assert app.error
    message = app.error[0].value
    assert "det(A) \u53ea\u80fd\u7528\u4e8e\u65b9\u9635" in message
    assert "2 \u00d7 3" in message


def test_streamlit_hadamard_division_zero_error_is_actionable():
    page_name = "\u53cc\u77e9\u9635\u8fd0\u7b97"
    app = _start_page(page_name)
    app.session_state["binary_operation"] = "hadamard_divide"
    app.session_state["matrix:binary-b:raw"] = [
        ["1", "0"],
        ["1", "1"],
    ]
    app.run()
    _click(app, "\u8ba1\u7b97")

    assert app.error
    assert "B \u4e2d\u5305\u542b 0" in app.error[0].value
