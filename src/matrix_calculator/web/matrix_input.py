"""基于 Streamlit CCv2 的矩阵逐格输入组件。"""

from __future__ import annotations

import re

import numpy as np
import streamlit as st


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
.matrix-widget,
.matrix-widget *,
.matrix-widget *::before,
.matrix-widget *::after {
  box-sizing: border-box;
}

.matrix-widget {
  color: var(--mx-text, #1d1d1f);
  font-family: var(--mx-font, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
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
  color: var(--mx-muted, #6e6e73);
  font-family: var(--mx-font-family-monospace, ui-monospace, monospace);
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
  background: var(--mx-ghost-bg, rgba(255, 255, 255, 0.62));
  border: 1px solid var(--mx-ghost-border, rgba(255, 255, 255, 0.82));
  border-radius: var(--mx-radius-pill, 999px);
  box-shadow: var(--mx-shadow-btn, 0 7px 18px rgba(15, 23, 42, 0.06)),
    inset 0 1px 0 var(--mx-inset, rgba(255, 255, 255, 0.78));
  color: var(--mx-text, #1d1d1f);
  cursor: pointer;
  font: inherit;
  font-size: 0.82rem;
  line-height: 1.2;
  padding: 0.28rem 0.62rem;
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
}
.matrix-controls button:hover:not(:disabled) {
  border-color: var(--mx-blue-text, #0062c4);
  color: var(--mx-blue-text, #0062c4);
}
.matrix-controls button:disabled {
  cursor: not-allowed;
  opacity: 0.32;
}
.matrix-expression {
  display: inline-flex;
  margin: 0.15rem 0 0 0.15rem;
  max-width: 100%;
  overflow-x: auto;
  padding: 0.35rem 0.95rem;
  position: relative;
  -webkit-overflow-scrolling: touch;
}
.bracket {
  border: 2.5px solid var(--mx-bracket, rgba(80, 93, 124, 0.42));
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
  gap: 0.38rem;
}
.matrix-cell {
  background: var(--mx-cell-bg, rgba(255, 255, 255, 0.70));
  border: 1px solid var(--mx-cell-border, rgba(255, 255, 255, 0.88));
  border-radius: var(--mx-radius-cell, 0.9rem);
  box-shadow:
    var(--mx-shadow-btn, 0 8px 20px rgba(15, 23, 42, 0.06)),
    inset 0 1px 0 var(--mx-cell-inset, rgba(255, 255, 255, 0.86));
  color: var(--mx-text, #1d1d1f);
  font: inherit;
  font-variant-numeric: tabular-nums;
  height: 2.28rem;
  min-width: 2.5rem;
  padding: 0.1rem 0.25rem;
  text-align: center;
  width: clamp(2.5rem, 13vw, 3.65rem);
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
}
.matrix-cell:focus {
  border-color: var(--mx-blue-focus, rgba(0, 113, 227, 0.55));
  box-shadow:
    0 0 0 4px var(--mx-blue-ring, rgba(0, 113, 227, 0.16)),
    0 10px 22px var(--mx-blue-glow, rgba(0, 113, 227, 0.10));
  outline: none;
}
.matrix-cell.invalid {
  background: var(--mx-danger-bg, rgba(214, 69, 69, 0.12));
  border-color: var(--mx-danger, #d64545);
}
.matrix-error {
  color: var(--mx-danger, #d64545);
  font-size: 0.84rem;
  min-height: 1.15rem;
  padding-top: 0.15rem;
}

@media (max-width: 640px) {
  .matrix-grid {
    gap: 0.3rem;
  }

  .matrix-cell {
    border-radius: 0.7rem;
    height: 2rem;
  }

  .matrix-expression {
    padding: 0.3rem 0.7rem;
  }

  .matrix-controls {
    gap: 0.22rem;
  }

  .matrix-controls button {
    font-size: 0.78rem;
    padding: 0.26rem 0.5rem;
  }
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
  // preventScroll：移动端宽矩阵横向滚动时，聚焦新格不会把首列顶出视口
  root.querySelector(`.matrix-cell[data-row="${focusRow}"][data-col="${focusCol}"]`)
    ?.focus({ preventScroll: true });
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

def _create_matrix_grid_component():
    """在当前 Streamlit 运行上下文中创建组件，避免跨会话共享状态。"""
    return st.components.v2.component(
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

    _create_matrix_grid_component()(
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
        st.error(
            "矩阵中有不合法的输入："
            + "；".join(errors[:3])
            + "。支持整数、小数、科学计数法（如 1e-3）和简单分数（如 1/2）。"
        )
        st.stop()
    return matrix
