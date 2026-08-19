"""The Hungarian solver, on cases where the greedy answer is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.assignment import assign_with_gate, hungarian


def test_identity():
    cost = np.array([[1.0, 9.0], [9.0, 1.0]])
    assert hungarian(cost) == [(0, 0), (1, 1)]


def test_beats_greedy():
    """Greedy takes the 1 and is then forced into the 100. Total 101.

    Optimal takes 2 + 3 = 5. This is the whole reason for not writing three
    lines of nearest-neighbour matching instead - and it is not a contrived
    case: it is two vials whose paths cross on the conveyor.
    """
    cost = np.array([[1.0, 2.0], [3.0, 100.0]])
    assert hungarian(cost) == [(0, 1), (1, 0)]


def test_rectangular_more_columns():
    cost = np.array([[4.0, 1.0, 9.0], [2.0, 8.0, 3.0]])
    pairs = hungarian(cost)
    assert len(pairs) == 2
    assert {r for r, _ in pairs} == {0, 1}
    assert sum(cost[r, c] for r, c in pairs) == 3.0


def test_rectangular_more_rows():
    """Transposed internally; every column must still be used exactly once."""
    cost = np.array([[4.0, 1.0], [2.0, 8.0], [9.0, 3.0]])
    pairs = hungarian(cost)
    assert len(pairs) == 2
    assert len({c for _, c in pairs}) == 2


def test_empty():
    assert hungarian(np.zeros((0, 0))) == []
    assert hungarian(np.zeros((0, 4))) == []


def test_rejects_infinities():
    with pytest.raises(ValueError):
        hungarian(np.array([[1.0, np.inf]]))


def test_gate_rejects_after_solving():
    """The gate drops implausible pairs; the rest of the solution survives."""
    cost = np.array([[1.0, 500.0], [500.0, 2.0]])
    matched, rows, cols = assign_with_gate(cost, gate=10.0)
    assert matched == [(0, 0), (1, 1)]
    assert not rows and not cols

    matched, rows, cols = assign_with_gate(np.array([[400.0]]), gate=10.0)
    assert matched == []
    assert rows == [0] and cols == [0]


def test_gate_reports_unmatched():
    cost = np.array([[1.0, 900.0, 900.0]])
    matched, rows, cols = assign_with_gate(cost, gate=50.0)
    assert matched == [(0, 0)]
    assert rows == []
    assert cols == [1, 2]


def test_matches_bruteforce_on_random_matrices():
    """Exhaustive check against every permutation, on sizes small enough to."""
    from itertools import permutations

    rng = np.random.default_rng(7)
    for n in (3, 4, 5):
        for _ in range(20):
            cost = rng.uniform(0, 50, (n, n))
            best = min(sum(cost[i, p[i]] for i in range(n))
                       for p in permutations(range(n)))
            got = sum(cost[r, c] for r, c in hungarian(cost))
            assert got == pytest.approx(best, abs=1e-9)
