"""Stage commitment, oven inference and the analysis cadence.

These used to be driven through the whole-platform assignment tracker. That
tracker is gone; the logic it exercised lives in pipeline/zones.StageTracker
and pipeline/tracking.CadenceController, and is tested directly here.
"""

from __future__ import annotations

import pytest

from config import GEOMETRY, TRACKING
from pipeline.tracking import CadenceController
from pipeline.types import Track
from pipeline.zones import StageTracker, ZoneMap, mm_to_px

W, H = GEOMETRY.frame_width_px, GEOMETRY.frame_height_px


def rect(x0, x1, y0=0.3, y1=0.7):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.fixture
def zones():
    return ZoneMap(W, H, {
        "storing": rect(0.00, 0.30),
        "injection": rect(0.30, 0.60),
        "heating": rect(0.60, 0.90),
    })


def track_at(nx, ny=0.5, r=17.0, ts=0.0):
    """A track positioned at normalised frame coordinates."""
    return Track(track_id=1, cx=nx * W, cy=ny * H, radius=r,
                 first_seen_ts=ts, last_seen_ts=ts, confirmed=True)


def move(track, nx, ny=0.5):
    track.cx, track.cy = nx * W, ny * H
    return track


# --------------------------------------------------------------------------
# Stage commitment
# --------------------------------------------------------------------------
def test_first_stage_commits_immediately(zones):
    stages = StageTracker(zones, hysteresis_n=1)
    t = track_at(0.10)
    assert stages.update(t, 0.0) == "storing"
    assert t.stage == "storing"


def test_stage_transition_needs_hysteresis(zones):
    """N agreeing frames before a stage change is committed."""
    stages = StageTracker(zones, hysteresis_n=2)
    t = track_at(0.25)
    # N applies to the first commitment too, so it takes two frames to settle.
    stages.update(t, 0.0)
    stages.update(t, 45.0)
    assert t.stage == "storing"

    stages.update(move(t, 0.35), 90.0)
    assert t.stage == "storing", "one frame is not enough"

    stages.update(move(t, 0.36), 135.0)
    assert t.stage == "injection"
    assert [s for s, _ in t.stage_log] == ["storing", "injection"]


def test_boundary_flicker_does_not_commit(zones):
    """A centroid oscillating across a zone edge must not log transitions."""
    stages = StageTracker(zones, hysteresis_n=2)
    t = track_at(0.29)
    stages.update(t, 0.0)
    stages.update(t, 45.0)          # settle on "storing" first
    for i, x in enumerate([0.31, 0.29, 0.31, 0.29]):
        stages.update(move(t, x), 90.0 + 45.0 * i)
    assert t.stage == "storing"
    assert len(t.stage_log) == 1


def test_point_outside_every_polygon_is_unstaged(zones):
    stages = StageTracker(zones, hysteresis_n=1)
    t = track_at(0.5, ny=0.05)
    assert stages.update(t, 0.0) is None
    assert t.stage is None


# --------------------------------------------------------------------------
# Oven inference - the documented blind spot, pinned so nobody "fixes" it
# into a silent failure.
# --------------------------------------------------------------------------
def test_disappearing_after_the_end_rack_is_inferred_oven_entry(zones):
    stages = StageTracker(zones, hysteresis_n=1)
    t = track_at(0.10)
    t.stage = TRACKING.oven_entry_from
    assert stages.close_reason(t) == "oven"


def test_disappearing_elsewhere_is_lost(zones):
    stages = StageTracker(zones, hysteresis_n=1)
    t = track_at(0.70)
    t.stage = "heating"
    assert stages.close_reason(t) == "lost"


# --------------------------------------------------------------------------
# Cadence
# --------------------------------------------------------------------------
def test_cadence_speeds_up_on_motion_and_releases_slowly():
    from config import CADENCE

    cadence = CadenceController()
    t = track_at(0.10)
    assert cadence.observe([t]) == CADENCE.analysis_interval_s

    moved_px = mm_to_px(CADENCE.busy_motion_mm * 2)
    t.cx += moved_px
    assert cadence.observe([t]) == CADENCE.analysis_interval_busy_s

    # Staying still does not release immediately - it takes several quiet
    # frames, so one noisy centroid cannot flap the cadence.
    for _ in range(CADENCE.busy_release_frames - 1):
        assert cadence.observe([t]) == CADENCE.analysis_interval_busy_s
    assert cadence.observe([t]) == CADENCE.analysis_interval_s
