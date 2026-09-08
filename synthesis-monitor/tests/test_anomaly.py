"""MissingLid: a crucible heated without a lid.

lid_score is stubbed here so these test the detector's own decisions - when
it looks, when it stays quiet, what it raises - rather than re-testing the
lid feature, which tests/test_lid.py covers against the labelled set.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DETECTION
from pipeline.anomaly import MissingLid

FRAME = np.zeros((200, 200, 3), np.uint8)
LID = DETECTION.lid_score_threshold + 10
OPEN = DETECTION.lid_score_threshold - 10


@pytest.fixture
def scored(monkeypatch):
    """Drive lid_score from a dict of slot -> score."""
    table: dict[tuple[float, float], float] = {}

    def fake(img, cx, cy, *a, **k):
        return table[(cx, cy)]

    monkeypatch.setattr("pipeline.anomaly.lid_score", fake)
    return table


def test_an_open_crucible_on_a_heater_raises_an_alert(scored):
    scored[(10.0, 10.0)] = OPEN
    events = MissingLid().check(FRAME, {1: (10.0, 10.0)}, 100.0, frame_id=7)

    assert len(events) == 1
    e = events[0]
    assert e.kind == "missing_lid"
    assert e.severity == "alert"          # a safety matter, not an observation
    assert e.data["heater_slot"] == 1
    assert e.frame_id == 7


def test_a_lidded_crucible_raises_nothing(scored):
    scored[(10.0, 10.0)] = LID
    assert MissingLid().check(FRAME, {1: (10.0, 10.0)}, 100.0) == []


def test_it_fires_once_on_arrival_not_every_frame(scored):
    scored[(10.0, 10.0)] = OPEN
    m = MissingLid()
    first = m.check(FRAME, {1: (10.0, 10.0)}, 100.0)
    later = [m.check(FRAME, {1: (10.0, 10.0)}, 100.0 + 30 * i) for i in range(1, 4)]

    assert len(first) == 1
    assert all(e == [] for e in later)


def test_a_slot_refilled_after_emptying_is_checked_again(scored):
    scored[(10.0, 10.0)] = OPEN
    m = MissingLid()
    assert len(m.check(FRAME, {1: (10.0, 10.0)}, 100.0)) == 1
    assert m.check(FRAME, {}, 130.0) == []                 # slot empties
    assert len(m.check(FRAME, {1: (10.0, 10.0)}, 160.0)) == 1


def test_both_heaters_are_checked_independently(scored):
    scored[(10.0, 10.0)] = OPEN
    scored[(50.0, 50.0)] = LID
    events = MissingLid().check(
        FRAME, {0: (10.0, 10.0), 1: (50.0, 50.0)}, 100.0)

    assert [e.data["heater_slot"] for e in events] == [0]


def test_an_empty_heater_is_not_an_alert(scored):
    assert MissingLid().check(FRAME, {}, 100.0) == []


def test_reset_forgets_what_it_has_seen(scored):
    scored[(10.0, 10.0)] = OPEN
    m = MissingLid()
    m.check(FRAME, {1: (10.0, 10.0)}, 100.0)
    m.reset()
    assert len(m.check(FRAME, {1: (10.0, 10.0)}, 130.0)) == 1
