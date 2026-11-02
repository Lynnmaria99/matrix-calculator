"""线性代数专题算法：方程组、零空间和二次型判定。"""

from __future__ import annotations

import numpy as np

from .constants import EPS


def rank_and_nullspace(a: np.ndarray):
    """基于 SVD 计算秩与零空间基。

    返回 ``(秩 r, 零空间基矩阵 N, 奇异值阈值)``；
    ``N`` 的每一列是一个零空间基向量。
    """
    m, n = a.shape
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    tol = max(m, n) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > tol))
    null = vt[r:].T if r < n else np.zeros((n, 0))
    return r, null, tol


def solve_linear(a: np.ndarray, b: np.ndarray):
    """解 Ax=b。

    返回 ``(状态码, 信息字典)``：
    - ``unique``：唯一解；
    - ``infinite``：无穷多解，包含一个特解和零空间基；
    - ``none``：无解，包含最小二乘近似解。
    """
    m, n = a.shape
    r_a, null, _ = rank_and_nullspace(a)
    aug = np.hstack([a, b.reshape(-1, 1)])
    r_ab = int(np.linalg.matrix_rank(aug))
    info = {"rA": r_a, "rAb": r_ab, "n": n, "aug": aug}

    if r_a < r_ab:
        x_ls, *_ = np.linalg.lstsq(a, b, rcond=None)
        info["x_ls"] = x_ls
        return "none", info

    if r_a == n:
        if m == n:
            x = np.linalg.solve(a, b)
        else:
            x, *_ = np.linalg.lstsq(a, b, rcond=None)
        info["x"] = x
        return "unique", info

    xp, *_ = np.linalg.lstsq(a, b, rcond=None)
    info["xp"] = xp
    info["null"] = null
    return "infinite", info


def classify_quadratic(eigenvalues: np.ndarray, tol: float = EPS):
    """根据对称矩阵特征值判定二次型正定性。

    返回 ``(kind, pos, neg, zero)``。
    """
    w = np.asarray(eigenvalues)
    n = w.size
    pos = int(np.sum(w > tol))
    neg = int(np.sum(w < -tol))
    zero = n - pos - neg

    if pos == n:
        kind = "【正定】(特征值全为正)"
    elif neg == n:
        kind = "【负定】(特征值全为负)"
    elif neg == 0 and zero > 0:
        kind = "【半正定】(特征值非负且含 0)"
    elif pos == 0 and zero > 0:
        kind = "【半负定】(特征值非正且含 0)"
    elif pos > 0 and neg > 0:
        kind = "【不定】(特征值有正有负)"
    else:
        kind = "判定需人工复核"
    return kind, pos, neg, zero
