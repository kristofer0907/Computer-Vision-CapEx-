"""Optimal rectangular assignment (Hungarian / Jonker-Volgenant potentials).

Written out rather than imported from scipy on purpose. scipy is a ~40 MB
install on a Pi 5 and this is the only thing the project would want from it,
for a matrix that is at most 18x18. O(n^3) on 18 rows is microseconds.

`hungarian` is the exact solver. `assign_with_gate` is what the tracker
actually calls: it solves globally, then drops pairs that exceed the gate.

Order matters and is deliberate - gate after solving, not before. Solving on a
pre-gated matrix lets one impossible pair distort the whole solution, because
the solver has to route around an infinity rather than simply being told
afterwards that a cheap-looking pair was not physically plausible.
"""

from __future__ import annotations

import numpy as np

INF = float("inf")


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment of rows to columns.

    Returns (row, col) pairs, one per row when rows <= cols, otherwise one
    per column. Costs must be finite; use a large sentinel, not inf, for
    pairs you want discouraged.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix must be finite")

    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T

    n, m = cost.shape
    # 1-indexed potentials; index 0 is the algorithm's virtual free column.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)     # p[j] = row matched to column j
    way = np.zeros(m + 1, dtype=np.int64)   # column predecessor on the path

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def assign_with_gate(cost: np.ndarray, gate: float
                     ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve, then reject pairs costing more than `gate`.

    Returns (matched pairs, unmatched row indices, unmatched column indices).
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {cost.shape}")
    n_rows, n_cols = cost.shape

    if cost.size == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    matched = [(r, c) for r, c in hungarian(cost) if cost[r, c] <= gate]
    used_rows = {r for r, _ in matched}
    used_cols = {c for _, c in matched}
    return (matched,
            [r for r in range(n_rows) if r not in used_rows],
            [c for c in range(n_cols) if c not in used_cols])
