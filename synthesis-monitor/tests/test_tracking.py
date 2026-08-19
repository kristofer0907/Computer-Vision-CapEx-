"""Tracking and staging against synthetic centroids.

Deliberately independent of the camera: these are the behaviours that decide
whether a vial keeps its identity for the length of a run, and they can be
pinned down before any hardware or any localiser exists.
"""

from __future__ import annotations

import pytest

from config import GEOMETRY
from pipeline.tracking import CadenceController, HungarianTracker
from pipeline.types import Detection
from pipeline.zones import StageTracker, ZoneMap, mm_to_px

W, H = GEOMETRY.frame_width_px, GEOMETRY.frame_height_px


def rect(x0, x1, y0=0.3, y1=0.7):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.fixture
def zones():
    return ZoneMap(W, H, {
        "filling": rect(0.00, 0.30),
        "conveyor": rect(0.30, 0.60),
        "heating": rect(0.60, 0.90),
    })


def det(nx, ny=0.5, r=17.0):
    """Detection at normalised frame coordinates."""
    return Detection(cx=nx * W, cy=ny * H, radius=r)


def test_new_detections_become_tracks(zones):
    tracker = HungarianTracker(zones, min_hits_to_confirm=1)
    tracks, closed = tracker.update([det(0.1), det(0.15)], 0.0)
    assert len(tracks) == 2
    assert not closed
    assert {t.track_id for t in tracks} == {1, 2}


def test_identity_survives_movement(zones):
    tracker = HungarianTracker(zones, min_hits_to_confirm=1)
    tracker.update([det(0.10), det(0.20)], 0.0)
    tracks, _ = tracker.update([det(0.12), det(0.22)], 45.0)
    by_id = {t.track_id: t for t in tracks}
    assert by_id[1].cx == pytest.approx(0.12 * W)
    assert by_id[2].cx == pytest.approx(0.22 * W)


def test_crossing_paths_do_not_swap_ids(zones):
    """Two vials whose nearest neighbour is the other one.

    Greedy matching gets this wrong. This is the case the global solve exists
    for, and it is what a handover on the conveyor looks like.
    """
    tracker = HungarianTracker(zones, min_hits_to_confirm=1)
    tracker.update([det(0.30), det(0.40)], 0.0)
    tracks, _ = tracker.update([det(0.33), det(0.43)], 10.0)
    by_id = {t.track_id: t.cx for t in tracks}
    assert by_id[1] < by_id[2]


def test_gate_rejects_teleportation(zones):
    """A detection beyond the plausible gate starts a new track, not a jump."""
    tracker = HungarianTracker(zones, min_hits_to_confirm=1,
                               max_assignment_mm=50.0)
    tracker.update([det(0.10)], 0.0)
    tracks, _ = tracker.update([det(0.85)], 45.0)
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}


def test_track_survives_a_missed_frame_then_closes(zones):
    tracker = HungarianTracker(zones, min_hits_to_confirm=1, max_missed_frames=2)
    tracker.update([det(0.10)], 0.0)

    for i in range(2):
        tracks, closed = tracker.update([], 10.0 * (i + 1))
        assert len(tracks) == 1, "should tolerate a brief occlusion"
        assert not closed

    tracks, closed = tracker.update([], 40.0)
    assert not tracks
    assert len(closed) == 1


def test_min_hits_suppresses_one_frame_ghosts(zones):
    tracker = HungarianTracker(zones, min_hits_to_confirm=2)
    tracks, _ = tracker.update([det(0.10)], 0.0)
    assert not tracks[0].confirmed
    tracks, _ = tracker.update([det(0.11)], 45.0)
    assert tracks[0].confirmed


def test_first_stage_commits_immediately(zones):
    tracker = HungarianTracker(zones, min_hits_to_confirm=1)
    tracks, _ = tracker.update([det(0.10)], 0.0)
    assert tracks[0].stage == "filling"


def test_stage_transition_needs_hysteresis(zones):
    """N agreeing frames before a stage change is committed."""
    tracker = HungarianTracker(zones, min_hits_to_confirm=1,
                               stage_tracker=StageTracker(zones, hysteresis_n=2))
    tracker.update([det(0.25)], 0.0)

    # 0.25 -> 0.35 is ~102 mm, well inside the assignment gate, so this is the
    # same vial moving rather than a new one appearing.
    tracks, _ = tracker.update([det(0.35)], 45.0)
    assert tracks[0].stage == "filling", "one frame is not enough"

    tracks, _ = tracker.update([det(0.36)], 90.0)
    assert tracks[0].stage == "conveyor"
    assert [s for s, _ in tracks[0].stage_log] == ["filling", "conveyor"]


def test_boundary_flicker_does_not_commit(zones):
    """A centroid oscillating across a zone edge must not log transitions."""
    tracker = HungarianTracker(zones, min_hits_to_confirm=1,
                               stage_tracker=StageTracker(zones, hysteresis_n=2))
    tracker.update([det(0.29)], 0.0)
    for i, x in enumerate([0.31, 0.29, 0.31, 0.29]):
        tracks, _ = tracker.update([det(x)], 45.0 * (i + 1))
    assert tracks[0].stage == "filling"
    assert len(tracks[0].stage_log) == 1


def test_disappearing_after_cooling_is_inferred_oven_entry():
    """The blind spot, pinned so nobody 'fixes' it into a silent failure."""
    zones = ZoneMap(W, H, {"cooling": rect(0.0, 1.0)})
    tracker = HungarianTracker(zones, min_hits_to_confirm=1, max_missed_frames=1)
    tracker.update([det(0.5)], 0.0)
    tracker.update([], 45.0)
    _, closed = tracker.update([], 90.0)
    assert closed[0].closed_reason == "oven"


def test_disappearing_elsewhere_is_lost(zones):
    tracker = HungarianTracker(zones, min_hits_to_confirm=1, max_missed_frames=1)
    tracker.update([det(0.7)], 0.0)     # heating
    tracker.update([], 45.0)
    _, closed = tracker.update([], 90.0)
    assert closed[0].closed_reason == "lost"


def test_point_outside_every_polygon_is_unstaged(zones):
    tracker = HungarianTracker(zones, min_hits_to_confirm=1)
    tracks, _ = tracker.update([det(0.5, ny=0.05)], 0.0)
    assert tracks[0].stage is None


def test_cadence_speeds_up_on_motion_and_releases_slowly():
    from config import CADENCE

    zones = ZoneMap(W, H, {"z": rect(0.0, 1.0)})
    tracker = HungarianTracker(zones, min_hits_to_confirm=1)
    cadence = CadenceController()

    tracks, _ = tracker.update([det(0.10)], 0.0)
    assert cadence.observe(tracks) == CADENCE.analysis_interval_s

    moved_px = mm_to_px(CADENCE.busy_motion_mm * 2)
    tracks, _ = tracker.update([Detection(0.10 * W + moved_px, 0.5 * H, 17.0)], 45.0)
    assert cadence.observe(tracks) == CADENCE.analysis_interval_busy_s
    assert cadence.busy

    for _ in range(CADENCE.busy_release_frames - 1):
        interval = cadence.observe(tracks)
        assert interval == CADENCE.analysis_interval_busy_s, "must not flap"
    assert cadence.observe(tracks) == CADENCE.analysis_interval_s
