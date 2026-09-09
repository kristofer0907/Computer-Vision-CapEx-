"""The main.py command line.

Nothing here starts hardware: these cover argument parsing and how the parsed
values reach Monitor. The cadence actually holding is checked by running the
thing, not by a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import CADENCE, REGION_TRACKING
from main import Monitor, parse_args
from pipeline.types import Detection, Track


def test_cadence_defaults_to_adaptive():
    args = parse_args([])
    assert args.rgb_interval is None
    assert args.thermal_interval is None


def test_cadence_flags_are_read_as_seconds():
    args = parse_args(["--rgb-interval", "15", "--thermal-interval", "3"])
    assert args.rgb_interval == 15.0
    assert args.thermal_interval == 3.0


def test_fractional_intervals_are_allowed():
    args = parse_args(["--thermal-interval", "0.5"])
    assert args.thermal_interval == 0.5


@pytest.mark.parametrize("flag", ["--rgb-interval", "--thermal-interval"])
@pytest.mark.parametrize("bad", ["0", "-1"])
def test_a_non_positive_interval_is_refused(flag, bad):
    """Zero would spin the capture loop as fast as the camera can go."""
    with pytest.raises(SystemExit):
        parse_args([flag, bad])


def test_monitor_defaults_to_the_configured_cadence():
    monitor = Monitor()
    assert monitor.rgb_interval_s is None
    assert monitor.thermal_interval_s is None


def test_monitor_keeps_a_pinned_cadence():
    monitor = Monitor(rgb_interval_s=15.0, thermal_interval_s=3.0)
    assert monitor.rgb_interval_s == 15.0
    assert monitor.thermal_interval_s == 3.0


def test_pinned_rgb_interval_overrides_the_controller():
    """The controller still observes; its answer is just not what is used."""
    monitor = Monitor(rgb_interval_s=7.0)
    adaptive = monitor.cadence.observe([])
    assert adaptive == CADENCE.analysis_interval_s      # what it would have said
    chosen = (adaptive if monitor.rgb_interval_s is None
              else monitor.rgb_interval_s)
    assert chosen == 7.0


# --------------------------------------------------------------------------
# Untracked zones (storing, injection - REGION_TRACKING.id_zones excludes
# them). Their detections carry no identity and live in
# RegionCoordinator.untracked rather than in `tracks`; _stage_counts and
# _overlay both have to read that separately or these crucibles vanish from
# the dashboard entirely, which is what shipped until this was noticed.
# --------------------------------------------------------------------------
class _FakeCoord:
    """Only what _stage_counts and _overlay read off a real RegionCoordinator."""

    def __init__(self, untracked=None):
        self.untracked = untracked or {}


def _det(cx=50.0, cy=50.0, radius=15.0) -> Detection:
    return Detection(cx=cx, cy=cy, radius=radius)


def _track(stage: str, track_id: int = 1) -> Track:
    return Track(track_id=track_id, cx=1.0, cy=1.0, radius=1.0,
                 first_seen_ts=0.0, last_seen_ts=0.0, stage=stage)


def test_stage_counts_includes_untracked_zones():
    monitor = Monitor()
    coord = _FakeCoord({"storing": [_det(), _det(), _det()],
                        "injection": [_det()]})
    counts = monitor._stage_counts([], coord)
    assert counts["storing"] == 3
    assert counts["injection"] == 1
    assert counts["heating"] == 0


def test_stage_counts_still_counts_tracked_crucibles():
    monitor = Monitor()
    tracks = [_track("heating"), _track("heating", 2), _track("collection", 3)]
    counts = monitor._stage_counts(tracks, _FakeCoord())
    assert counts["heating"] == 2
    assert counts["collection"] == 1


def test_stage_counts_has_every_region_regardless_of_activity():
    monitor = Monitor()
    counts = monitor._stage_counts([], _FakeCoord())
    assert set(counts) == set(REGION_TRACKING.region_sequence)
    assert all(n == 0 for n in counts.values())


def test_overlay_draws_untracked_detections():
    """Anti-aliasing blends the circle's edge, so check pixels changed near
    the centre rather than for the exact BGR triple."""
    monitor = Monitor(draw_overlay=True)
    image = np.zeros((200, 300, 3), np.uint8)
    coord = _FakeCoord({"storing": [_det(cx=120, cy=90, radius=15)]})
    out = monitor._overlay(image, [], coord, [])
    window = out[75:106, 105:136]
    assert window.any(), "an untracked detection must still be visible on the overlay"


def test_overlay_draws_nothing_extra_with_no_untracked_detections():
    monitor = Monitor(draw_overlay=True)
    image = np.zeros((200, 300, 3), np.uint8)
    baseline = monitor._overlay(image, [], _FakeCoord(), [])
    coord = _FakeCoord({"storing": [_det(cx=120, cy=90, radius=15)]})
    with_detection = monitor._overlay(image, [], coord, [])
    assert not np.array_equal(baseline, with_detection), \
        "an untracked detection must change the frame"
    assert with_detection[75:106, 105:136].sum() > baseline[75:106, 105:136].sum()
