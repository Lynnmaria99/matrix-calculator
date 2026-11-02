# -*- coding: utf-8 -*-
"""矩阵计算器 Streamlit 网页界面。

运行方式：

    streamlit run streamlit_app.py

网页层只负责收集输入、调用计算核心、展示结果和错误信息。
矩阵输入使用自定义 CCv2 组件，呈现为带括号、可键盘导航的数学矩阵格子。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from matrix_calculator import (  # noqa: E402
    MatrixOperationError,
    add_matrices,
    analyze_quadratic,
    determinant,
    eigendecompose,
    format_matrix,
    hadamard_divide,
    hadamard_multiply,
    invert_matrix,
    matrix_multiply,
    singular_value_decompose,
    solve_linear,
    subtract_matrices,
    transpose,
)
from matrix_calculator.formatting import signed_polynomial  # noqa: E402


st.set_page_config(page_title="矩阵计算器", page_icon="🧮", layout="wide")

MAX_DIMENSION = 12

MATRIX_GRID_HTML = """
<div class="matrix-widget" data-testid="matrix-widget">
  <div class="matrix-head">
    <span class="matrix-symbol"></span>
    <span class="matrix-size"></span>
    <div class="matrix-controls" aria-label="调整矩阵维度">
      <button type="button" data-action="remove-row">−行</button>
      <button type="button" data-action="add-row">+行</button>
      <button type="button" data-action="remove-col">−列</button>
      <button type="button" data-action="add-col">+列</button>
      <button type="button" data-action="remove-order">−阶</button>
      <button type="button" data-action="add-order">+阶</button>
    </div>
  </div>
  <div class="matrix-expression">
    <span class="bracket bracket-left" aria-hidden="true"></span>
    <div class="matrix-grid" role="grid"></div>
    <span class="bracket bracket-right" aria-hidden="true"></span>
  </div>
  <div class="matrix-error" role="alert"></div>
</div>
"""

MATRIX_GRID_CSS = """
.matrix-widget {
  color: var(--st-text-color);
  font-family: var(--st-font-family, system-ui, sans-serif);
}
.matrix-head {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  margin-bottom: 0.45rem;
}
.matrix-symbol {
  font-size: 1.08rem;
  font-weight: 650;
  white-space: nowrap;
}
.matrix-size {
  color: var(--st-secondary-text-color, var(--st-text-color));
  font-family: var(--st-font-family-monospace, ui-monospace, monospace);
  font-size: 0.86rem;
  opacity: 0.78;
}
.matrix-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.28rem;
}
.matrix-controls button {
  appearance: none;
  background: var(--st-secondary-background-color, rgba(128, 128, 128, 0.12));
  border: 1px solid var(--st-border-color, rgba(128, 128, 128, 0.45));
  border-radius: 0.45rem;
  color: var(--st-text-color);
  cursor: pointer;
  font: inherit;
  font-size: 0.82rem;
  line-height: 1.2;
  padding: 0.22rem 0.48rem;
}
.matrix-controls button:hover:not(:disabled) {
  border-color: var(--st-primary-color);
  color: var(--st-primary-color);
}
.matrix-controls button:disabled {
  cursor: not-allowed;
  opacity: 0.32;
}
.matrix-expression {
  display: inline-flex;
  margin: 0.15rem 0 0 0.15rem;
  padding: 0.35rem 0.95rem;
  position: relative;
}
.bracket {
  border: 2.5px solid var(--st-border-color, rgba(128, 128, 128, 0.72));
  bottom: 0.2rem;
  pointer-events: none;
  position: absolute;
  top: 0.2rem;
  width: 10px;
}
.bracket-left {
  border-right: 0;
  border-radius: 9px 0 0 9px;
  left: 0;
}
.bracket-right {
  border-left: 0;
  border-radius: 0 9px 9px 0;
  right: 0;
}
.matrix-grid {
  display: grid;
  gap: 0.28rem;
}
.matrix-cell {
  background: var(--st-secondary-background-color, rgba(128, 128, 128, 0.08));
  border: 1px solid var(--st-border-color, rgba(128, 128, 128, 0.42));
  border-radius: 0.35rem;
  color: var(--st-text-color);
  font: inherit;
  font-variant-numeric: tabular-nums;
  height: 2.18rem;
  min-width: 3.55rem;
  padding: 0.1rem 0.25rem;
  text-align: center;
  width: 3.55rem;
}
.matrix-cell:focus {
  border-color: var(--st-primary-color);
  box-shadow: 0 0 0 1px var(--st-primary-color);
  outline: none;
}
.matrix-cell.invalid {
  background: rgba(214, 69, 69, 0.12);
  border-color: #d64545;
}
.matrix-error {
  color: #d64545;
  font-size: 0.84rem;
  min-height: 1.15rem;
  padding-top: 0.15rem;
}
"""

MATRIX_GRID_JS = r"""
const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

function clampInt(value, fallback, minValue, maxValue) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maxValue, Math.max(minValue, parsed));
}

function parseCell(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return { ok: false, value: null, message: "不能为空" };

  const parts = text.split("/");
  if (parts.length > 2) return { ok: false, value: null, message: "只能有一个除号" };
  if (parts.some((part) => !NUMBER.test(part.trim()))) {
    return { ok: false, value: null, message: "请输入数字，例如 3、-2.5、1/2" };
  }

  const numerator = Number(parts[0]);
  if (parts.length === 1) {
    return Number.isFinite(numerator)
      ? { ok: true, value: numerator, message: "" }
      : { ok: false, value: null, message: "数字不合法" };
  }

  const denominator = Number(parts[1]);
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return { ok: false, value: null, message: "分母不能为 0" };
  }
  return { ok: true, value: numerator / denominator, message: "" };
}

function collectMatrix(root, rows, cols) {
  const values = [];
  const numbers = [];
  const errors = [];

  for (let r = 0; r < rows; r += 1) {
    const rowValues = [];
    const rowNumbers = [];
    for (let c = 0; c < cols; c += 1) {
      const input = root.querySelector(`.matrix-cell[data-row="${r}"][data-col="${c}"]`);
      const raw = input ? input.value : "";
      const parsed = parseCell(raw);
      rowValues.push(raw);
      rowNumbers.push(parsed.ok ? parsed.value : null);
      if (!parsed.ok) {
        errors.push({ row: r, col: c, message: parsed.message });
      }
    }
    values.push(rowValues);
    numbers.push(rowNumbers);
  }

  return { rows, cols, values, numbers, valid: errors.length === 0, errors };
}

function commitMatrix(root) {
  const rows = clampInt(root.dataset.rows, 1, 1, 12);
  const cols = clampInt(root.dataset.cols, 1, 1, 12);
  const payload = collectMatrix(root, rows, cols);
  root._setStateValue("matrix", payload);
  return payload;
}

function resizeMatrix(root, action) {
  const oldRows = clampInt(root.dataset.rows, 1, 1, 12);
  const oldCols = clampInt(root.dataset.cols, 1, 1, 12);
  let rows = oldRows;
  let cols = oldCols;

  if (action === "remove-row") rows -= 1;
  if (action === "add-row") rows += 1;
  if (action === "remove-col") cols -= 1;
  if (action === "add-col") cols += 1;
  if (action === "remove-order") {
    rows -= 1;
    cols -= 1;
  }
  if (action === "add-order") {
    rows += 1;
    cols += 1;
  }

  const minRows = clampInt(root._data.minRows, 1, 1, 12);
  const minCols = clampInt(root._data.minCols, 1, 1, 12);
  const maxRows = clampInt(root._data.maxRows, 12, 1, 12);
  const maxCols = clampInt(root._data.maxCols, 12, 1, 12);
  rows = clampInt(rows, oldRows, minRows, maxRows);
  cols = clampInt(cols, oldCols, minCols, maxCols);
  if (rows === oldRows && cols === oldCols) return;

  const previous = collectMatrix(root, oldRows, oldCols);
  const values = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => previous.values[r]?.[c] ?? "0")
  );
  const nextData = { ...root._data, rows, cols, values };
  root._data = nextData;
  renderWidget(root, nextData);
  commitMatrix(root);

  const focusRow = action.startsWith("add") ? rows - 1 : 0;
  const focusCol = action === "add-col" ? cols - 1 : 0;
  root.querySelector(`.matrix-cell[data-row="${focusRow}"][data-col="${focusCol}"]`)?.focus();
}

function validateCell(input) {
  const parsed = parseCell(input.value);
  input.classList.toggle("invalid", !parsed.ok);
  input.setAttribute("aria-invalid", parsed.ok ? "false" : "true");
  input.title = parsed.ok ? "" : parsed.message;
}

function focusCell(root, row, col) {
  root.querySelector(`.matrix-cell[data-row="${row}"][data-col="${col}"]`)?.focus();
}

function bindWidget(root, grid) {
  root.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (button && root.contains(button)) {
      resizeMatrix(root, button.dataset.action);
    }
  });

  grid.addEventListener("input", (event) => {
    if (event.target.matches(".matrix-cell")) validateCell(event.target);
  });

  grid.addEventListener("change", (event) => {
    if (event.target.matches(".matrix-cell")) commitMatrix(root);
  });

  grid.addEventListener("keydown", (event) => {
    if (!event.target.matches(".matrix-cell")) return;
    const rows = clampInt(root.dataset.rows, 1, 1, 12);
    const cols = clampInt(root.dataset.cols, 1, 1, 12);
    let row = Number.parseInt(event.target.dataset.row, 10);
    let col = Number.parseInt(event.target.dataset.col, 10);
    let nextRow = row;
    let nextCol = col;

    if (event.key === "ArrowRight") {
      if (col < cols - 1) nextCol += 1;
      else if (row < rows - 1) {
        nextRow += 1;
        nextCol = 0;
      }
    } else if (event.key === "ArrowLeft") {
      if (col > 0) nextCol -= 1;
      else if (row > 0) {
        nextRow -= 1;
        nextCol = cols - 1;
      }
    } else if (event.key === "ArrowDown") {
      if (row < rows - 1) nextRow += 1;
    } else if (event.key === "ArrowUp") {
      if (row > 0) nextRow -= 1;
    } else if (event.key === "Enter") {
      event.preventDefault();
      commitMatrix(root);
      if (row < rows - 1) nextRow += 1;
      else if (col < cols - 1) {
        nextRow = 0;
        nextCol += 1;
      }
    } else {
      return;
    }

    event.preventDefault();
    if (nextRow !== row || nextCol !== col) focusCell(root, nextRow, nextCol);
  });
}

function renderWidget(root, data) {
  const rows = clampInt(data.rows, 1, 1, 12);
  const cols = clampInt(data.cols, 1, 1, 12);
  const square = Boolean(data.square);
  const minRows = clampInt(data.minRows, 1, 1, 12);
  const minCols = clampInt(data.minCols, 1, 1, 12);
  const maxRows = clampInt(data.maxRows, 12, 1, 12);
  const maxCols = clampInt(data.maxCols, 12, 1, 12);

  root.dataset.rows = String(rows);
  root.dataset.cols = String(cols);
  root.setAttribute("aria-label", data.label || "矩阵输入器");

  root.querySelector(".matrix-symbol").textContent = `${data.symbol || "A"} =`;
  root.querySelector(".matrix-size").textContent = square ? `${rows} 阶` : `${rows} × ${cols}`;

  const rowButtons = ["remove-row", "add-row", "remove-col", "add-col"];
  const orderButtons = ["remove-order", "add-order"];
  rowButtons.forEach((action) => {
    root.querySelector(`[data-action="${action}"]`).hidden = square;
  });
  orderButtons.forEach((action) => {
    root.querySelector(`[data-action="${action}"]`).hidden = !square;
  });

  root.querySelector('[data-action="remove-row"]').disabled = rows <= minRows;
  root.querySelector('[data-action="add-row"]').disabled = rows >= maxRows;
  root.querySelector('[data-action="remove-col"]').disabled = cols <= minCols;
  root.querySelector('[data-action="add-col"]').disabled = cols >= maxCols;
  root.querySelector('[data-action="remove-order"]').disabled = rows <= minRows;
  root.querySelector('[data-action="add-order"]').disabled = rows >= maxRows;

  const grid = root.querySelector(".matrix-grid");
  grid.style.gridTemplateColumns = `repeat(${cols}, 3.55rem)`;
  grid.setAttribute("aria-label", data.label || "矩阵输入器");

  const wanted = rows * cols;
  while (grid.children.length < wanted) {
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.className = "matrix-cell";
    grid.appendChild(input);
  }
  while (grid.children.length > wanted) {
    grid.removeChild(grid.lastElementChild);
  }

  Array.from(grid.children).forEach((input, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    input.dataset.row = String(row);
    input.dataset.col = String(col);
    input.setAttribute("role", "gridcell");
    input.setAttribute("aria-label", `${data.label || "矩阵"} 第 ${row + 1} 行第 ${col + 1} 列`);

    const nextValue = String(data.values?.[row]?.[col] ?? "");
    if (document.activeElement !== input && input.value !== nextValue) {
      input.value = nextValue;
    }
    validateCell(input);
  });

  const firstError = Array.isArray(data.errors) ? data.errors[0] : null;
  root.querySelector(".matrix-error").textContent = firstError
    ? `第 ${firstError.row + 1} 行第 ${firstError.col + 1} 列：${firstError.message}`
    : "";
}

export default function matrixGrid(component) {
  const { parentElement, data, setStateValue } = component;
  const root = parentElement.querySelector('[data-testid="matrix-widget"]');
  if (!root || !data) return;

  root._setStateValue = setStateValue;
  root._data = data;

  if (!root.dataset.bound) {
    bindWidget(root, root.querySelector(".matrix-grid"));
    root.dataset.bound = "true";
  }

  renderWidget(root, data);
}
"""

_matrix_grid_component = st.components.v2.component(
    "matrix_calculator.matrix_grid",
    html=MATRIX_GRID_HTML,
    css=MATRIX_GRID_CSS,
    js=MATRIX_GRID_JS,
)


# ==================== 矩阵输入组件 ====================


_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _state_keys(key: str) -> tuple[str, str]:
    return f"matrix:{key}:dims", f"matrix:{key}:raw"


def _component_key(key: str) -> str:
    return f"matrix-grid:{key}"


def _clamp_dimension(value: object, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return min(maximum, max(minimum, parsed))


def _format_number(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-10:
        return str(int(round(number)))
    return f"{number:.6g}"


def _parse_cell(raw: object) -> float:
    """解析小数或简单分数；不使用 eval，避免输入表达式被执行。"""
    text = str(raw).strip()
    parts = [part.strip() for part in text.split("/")]
    if not text or len(parts) > 2 or any(not _NUMBER_PATTERN.match(part) for part in parts):
        raise ValueError("请输入数字，例如 3、-2.5、1/2")

    numerator = float(parts[0])
    if len(parts) == 1:
        return numerator

    denominator = float(parts[1])
    if denominator == 0:
        raise ValueError("分母不能为 0")
    return numerator / denominator


def _raw_from_matrix(matrix: np.ndarray) -> list[list[str]]:
    return [[_format_number(value) for value in row] for row in matrix.tolist()]


def _normalized_raw(raw: object, rows: int, cols: int) -> list[list[str]]:
    result = []
    for i in range(rows):
        source_row = raw[i] if isinstance(raw, list) and i < len(raw) and isinstance(raw[i], list) else []
        row = []
        for j in range(cols):
            try:
                row.append(str(source_row[j]).strip())
            except (IndexError, KeyError, TypeError):
                row.append("0")
        result.append(row)
    return result


def _ensure_matrix(
    key: str,
    default: np.ndarray | None = None,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
) -> None:
    dims_key, raw_key = _state_keys(key)
    if dims_key not in st.session_state or raw_key not in st.session_state:
        matrix = np.eye(2, dtype=float) if default is None else np.asarray(default, dtype=float)
        if matrix.ndim != 2:
            matrix = np.atleast_2d(matrix)
        st.session_state[dims_key] = [
            max(min_rows, int(matrix.shape[0])),
            max(min_cols, int(matrix.shape[1])),
        ]
        st.session_state[raw_key] = _raw_from_matrix(matrix)


def _sync_matrix_from_component(
    key: str,
    *,
    min_rows: int,
    min_cols: int,
    square: bool,
) -> None:
    """组件状态变化时，在本次脚本重跑前同步维度和原始文本。"""
    component_state = st.session_state.get(_component_key(key)) or {}
    payload = component_state.get("matrix") if isinstance(component_state, dict) else None
    if not isinstance(payload, dict):
        return

    dims_key, raw_key = _state_keys(key)
    fallback_rows, fallback_cols = st.session_state.get(dims_key, [min_rows, min_cols])
    effective_min_cols = min_rows if square else min_cols
    rows = _clamp_dimension(payload.get("rows"), int(fallback_rows), min_rows, MAX_DIMENSION)
    cols = rows if square else _clamp_dimension(
        payload.get("cols"), int(fallback_cols), effective_min_cols, MAX_DIMENSION
    )

    st.session_state[dims_key] = [rows, cols]
    st.session_state[raw_key] = _normalized_raw(payload.get("values"), rows, cols)


def _build_matrix_data(
    key: str,
    label: str,
    symbol: str,
    *,
    min_rows: int,
    min_cols: int,
    square: bool,
) -> tuple[dict, np.ndarray, list[str]]:
    dims_key, raw_key = _state_keys(key)
    rows, cols = (int(value) for value in st.session_state[dims_key])
    if square:
        cols = rows
    raw = _normalized_raw(st.session_state[raw_key], rows, cols)

    numbers = []
    errors = []
    error_payload = []
    for i, row in enumerate(raw):
        number_row = []
        for j, cell in enumerate(row):
            try:
                number_row.append(_parse_cell(cell))
            except ValueError as exc:
                number_row.append(0.0)
                message = str(exc)
                errors.append(f"第 {i + 1} 行第 {j + 1} 列：{message}")
                error_payload.append({"row": i, "col": j, "message": message})
        numbers.append(number_row)

    payload = {
        "rows": rows,
        "cols": cols,
        "values": raw,
        "numbers": numbers,
        "valid": not errors,
        "errors": error_payload,
    }
    data = {
        **payload,
        "label": label,
        "symbol": symbol,
        "square": square,
        "minRows": min_rows,
        "minCols": min_rows if square else min_cols,
        "maxRows": MAX_DIMENSION,
        "maxCols": MAX_DIMENSION,
    }
    return data, np.asarray(numbers, dtype=float), errors


def matrix_input(
    label: str,
    key: str,
    default: np.ndarray | None = None,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
    square: bool = False,
    symbol: str = "A",
) -> np.ndarray:
    """渲染一个接近数学书写形式的自定义矩阵输入器。"""
    _ensure_matrix(key, default, min_rows=min_rows, min_cols=min_cols)
    data, matrix, errors = _build_matrix_data(
        key,
        label,
        symbol,
        min_rows=min_rows,
        min_cols=min_cols,
        square=square,
    )

    _matrix_grid_component(
        key=_component_key(key),
        data=data,
        default={"matrix": data},
        width="content",
        on_matrix_change=lambda: _sync_matrix_from_component(
            key,
            min_rows=min_rows,
            min_cols=min_cols,
            square=square,
        ),
    )

    if errors:
        st.error("矩阵中有不合法的输入：" + "；".join(errors[:3]))
        st.stop()
    return matrix
def show_text_matrix(label: str, value) -> None:
    st.markdown(f"**{label}**")
    st.text(format_matrix(value))


def show_error(exc: Exception) -> None:
    st.error(f"输入或计算有误：{exc}")



# ==================== 页面：双矩阵运算 ====================


def page_binary() -> None:
    st.header("双矩阵运算")
    operation_name = st.selectbox(
        "选择运算",
        [
            "矩阵加法（A + B）",
            "矩阵减法（A - B）",
            "矩阵乘法（A × B）",
            "对应元素相乘（Hadamard）",
            "对应元素相除（A / B）",
        ],
    )

    mat_a = matrix_input("矩阵 A", "binary-a", symbol="A")
    st.caption("对下面的矩阵 B 应用上面选择的运算。")
    mat_b = matrix_input("矩阵 B", "binary-b", symbol="B")

    if st.button("计算", type="primary", key="run_binary"):
        operations = {
            "矩阵加法（A + B）": add_matrices,
            "矩阵减法（A - B）": subtract_matrices,
            "矩阵乘法（A × B）": matrix_multiply,
            "对应元素相乘（Hadamard）": hadamard_multiply,
            "对应元素相除（A / B）": hadamard_divide,
        }
        try:
            result = operations[operation_name](mat_a, mat_b)
            show_text_matrix("运算结果", result)
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 页面：单矩阵运算 ====================


def page_unary() -> None:
    st.header("单矩阵运算")
    operation_name = st.selectbox(
        "选择运算",
        ["行列式", "逆矩阵", "特征分解", "奇异值分解 SVD", "转置"],
    )
    mat = matrix_input("矩阵 A", "unary-a", symbol="A")

    if st.button("计算", type="primary", key="run_unary"):
        try:
            if operation_name == "行列式":
                st.metric("行列式 det(A)", f"{determinant(mat):.6g}")
            elif operation_name == "逆矩阵":
                show_text_matrix("逆矩阵 A⁻¹", invert_matrix(mat))
            elif operation_name == "转置":
                show_text_matrix("转置 Aᵀ", transpose(mat))
            elif operation_name == "特征分解":
                eigenvalues, eigenvectors = eigendecompose(mat)
                show_text_matrix("特征值", eigenvalues)
                show_text_matrix("特征向量（每列对应一个特征值）", eigenvectors)
                if np.iscomplexobj(eigenvalues) and np.any(np.abs(eigenvalues.imag) > 1e-9):
                    st.info("该矩阵存在复数特征值或特征向量。")
            elif operation_name == "奇异值分解 SVD":
                u, singular_values, vt = singular_value_decompose(mat)
                show_text_matrix("U", u)
                show_text_matrix("奇异值 S", singular_values)
                show_text_matrix("Vᵀ", vt)
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 页面：线性方程组 ====================


def page_solve() -> None:
    st.header("解线性方程组 Ax = b")
    st.caption("输入增广矩阵 [A | b]：最后一列自动作为常数项 b，可通过 +列 增加未知数。")
    augmented = matrix_input(
        "增广矩阵 [A | b]",
        "solve-ab",
        default=np.zeros((2, 3)),
        min_cols=2,
        symbol="[A | b]",
    )

    if st.button("求解", type="primary", key="run_solve"):
        m, n_plus_one = augmented.shape
        n = n_plus_one - 1
        a = augmented[:, :n]
        b = augmented[:, -1]
        try:
            status, info = solve_linear(a, b)
            col1, col2, col3 = st.columns(3)
            col1.metric("r(A)", info["rA"])
            col2.metric("r(A|b)", info["rAb"])
            col3.metric("未知数 n", info["n"])

            if status == "unique":
                st.success("方程组有唯一解。")
                show_text_matrix("唯一解 x", info["x"].reshape(-1, 1))
            elif status == "infinite":
                st.warning("方程组有无穷多解。")
                st.write(f"自由未知数个数：{n - info['rA']}")
                show_text_matrix("一个特解 xp", info["xp"].reshape(-1, 1))
                show_text_matrix("基础解系（每列一个向量）", info["null"])
            else:
                st.error("方程组无解（矛盾方程组）。下面是最小二乘近似解。")
                show_text_matrix("最小二乘近似解", info["x_ls"].reshape(-1, 1))
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 页面：二次型 ====================


def page_quadratic() -> None:
    st.header("二次型 f = xᵀAx")
    st.caption("输入方阵阶数可通过 −阶 / +阶 调整；默认是 2 阶单位矩阵。")
    mat = matrix_input(
        "矩阵 A",
        "quadratic-a",
        default=np.eye(2),
        square=True,
        min_rows=1,
        symbol="A",
    )

    if st.button("分析", type="primary", key="run_quad"):
        try:
            result = analyze_quadratic(mat)
            if not result["symmetric"]:
                st.info("输入矩阵不对称；二次型只取决于对称部分，已自动使用 S=(A+Aᵀ)/2。")

            show_text_matrix("用于分析的对称矩阵 S", result["matrix"])
            show_text_matrix("特征值（升序）", result["eigenvalues"])
            show_text_matrix("正交变换矩阵 Q", result["Q"])
            standard_form = signed_polynomial(result["standard_terms"])
            st.markdown(f"经正交变换 x = Qy，标准形为：**f = {standard_form}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("正惯性指数 p", result["pos"])
            c2.metric("负惯性指数 q", result["neg"])
            c3.metric("零特征值个数", result["zero"])
            st.success(f"判定结果：{result['kind']}")

            minors = result["minors"]
            minors_text = "，".join(
                f"Δ{i} = {float(value):.4g}" for i, value in enumerate(minors, start=1)
            )
            st.caption(f"顺序主子式（Sylvester 判据）：{minors_text}")
        except (MatrixOperationError, np.linalg.LinAlgError) as exc:
            show_error(exc)


# ==================== 主导航 ====================


def main() -> None:
    st.title("🧮 矩阵计算器")
    st.caption("NumPy 计算核心 + Streamlit 网页界面")

    page = st.sidebar.radio(
        "功能",
        ["双矩阵运算", "单矩阵运算", "解线性方程组", "二次型分析"],
    )
    if page == "双矩阵运算":
        page_binary()
    elif page == "单矩阵运算":
        page_unary()
    elif page == "解线性方程组":
        page_solve()
    else:
        page_quadratic()


if __name__ == "__main__":
    main()