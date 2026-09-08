"""SlotTracker, FifoTracker and RegionCoordinator against synthetic data.

Deliberately independent of the camera and of pipeline/tracking.py's
vial-flow HungarianTracker - these are the real-hardware crucible zone
mechanics (nearest fixed slot, order along a lane, handoff across a zone
boundary), pinned down before any real capture pipeline is wired to them.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from config import GEOMETRY
from pipeline.region_trackers import (FifoTracker, RegionCoordinator, RegionTracker,
                                      SlotTracker)
from pipeline.types import Detection
from pipeline.zones import ZoneMap

W, H = GEOMETRY.frame_width_px, GEOMETRY.frame_height_px


def rect(x0, x1, y0=0.3, y1=0.7):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def det(cx, cy, r=17.0):
    return Detection(cx=cx, cy=cy, radius=r)


def write_slots(path, slots, radius=20.0, image_size=(W, H)):
    path.write_text(json.dumps({
        "stage": "test",
        "image": "ref.jpg",
        "image_size": list(image_size),
        "slot_radius_px": radius,
        "slots": [{"id": i, "x": x, "y": y} for i, (x, y) in enumerate(slots)],
    }))


def write_lane(path, entry, exit_, image_size=(W, H)):
    path.write_text(json.dumps({
        "stage": "test",
        "image": "ref.jpg",
        "image_size": list(image_size),
        "entry": list(entry),
        "exit": list(exit_),
    }))


# --------------------------------------------------------------------------
# SlotTracker
# --------------------------------------------------------------------------
def test_slot_tracker_spawns_and_holds_identity(tmp_path):
    path = tmp_path / "slots.json"
    write_slots(path, [(80, 300), (180, 400)])
    tracker = SlotTracker("storing", (W, H), slots_path=path)
    tracker.start()

    tracks, closed = tracker.update([det(82, 302)], 0.0)
    assert len(tracks) == 1 and not closed
    assert tracks[0].track_id == 1
    assert tracks[0].slot_id == 0
    assert tracks[0].stage == "storing"

    tracks, closed = tracker.update([det(79, 301)], 45.0)
    assert tracks[0].track_id == 1  # same slot, same identity
    assert tracks[0].hits == 2


def test_slot_tracker_two_detections_bind_to_distinct_slots(tmp_path):
    path = tmp_path / "slots.json"
    write_slots(path, [(80, 300), (180, 400)])
    tracker = SlotTracker("storing", (W, H), slots_path=path)
    tracker.start()

    tracks, _ = tracker.update([det(82, 302), det(178, 398)], 0.0)
    by_slot = {t.slot_id: t.track_id for t in tracks}
    assert by_slot == {0: 1, 1: 2}


def test_slot_tracker_closes_after_vacated(tmp_path):
    path = tmp_path / "slots.json"
    write_slots(path, [(80, 300)])
    tracker = SlotTracker("storing", (W, H), slots_path=path, max_missed_frames=1)
    tracker.start()

    tracker.update([det(80, 300)], 0.0)
    tracks, closed = tracker.update([], 45.0)
    assert tracks and not closed  # missed=1, not yet over the gate

    tracks, closed = tracker.update([], 90.0)
    assert not tracks
    assert closed and closed[0].closed_reason == "vacated"


def test_slot_tracker_rejects_offslot_detection(tmp_path):
    path = tmp_path / "slots.json"
    write_slots(path, [(80, 300)], radius=10.0)
    tracker = SlotTracker("storing", (W, H), slots_path=path)
    tracker.start()

    tracks, closed = tracker.update([det(500, 500)], 0.0)
    assert not tracks and not closed


def test_slot_tracker_rescales_to_frame_size(tmp_path):
    path = tmp_path / "slots.json"
    write_slots(path, [(160, 600)], radius=20.0, image_size=(2 * W, 2 * H))
    tracker = SlotTracker("storing", (W, H), slots_path=path)
    tracker.start()

    tracks, _ = tracker.update([det(80, 300)], 0.0)
    assert len(tracks) == 1  # 160,600 scaled by 0.5 -> 80,300


# --------------------------------------------------------------------------
# FifoTracker
# --------------------------------------------------------------------------
def test_fifo_tracker_orders_new_detections_by_lane_position(tmp_path):
    path = tmp_path / "lane.json"
    write_lane(path, entry=(280, 360), exit_=(480, 360))
    tracker = FifoTracker("injection", (W, H), lane_path=path)
    tracker.start()

    tracks, _ = tracker.update([det(450, 360), det(300, 360)], 0.0)
    assert tracker.queue_order() == sorted(t.track_id for t in tracks)
    by_id = {t.track_id: t.cx for t in tracks}
    # id 1 was processed first because it projects closer to entry
    assert by_id[min(by_id)] < by_id[max(by_id)]


def test_fifo_tracker_identity_survives_movement(tmp_path):
    path = tmp_path / "lane.json"
    write_lane(path, entry=(280, 360), exit_=(480, 360))
    tracker = FifoTracker("injection", (W, H), lane_path=path, step_gate_mm=200.0)
    tracker.start()

    tracker.update([det(300, 360)], 0.0)
    tracks, _ = tracker.update([det(340, 360)], 10.0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 2


def test_fifo_tracker_cannot_swap_order(tmp_path):
    """Two crucibles on the lane; the one nearer the entry never overtakes."""
    path = tmp_path / "lane.json"
    write_lane(path, entry=(280, 360), exit_=(480, 360))
    tracker = FifoTracker("injection", (W, H), lane_path=path, step_gate_mm=200.0)
    tracker.start()

    tracker.update([det(300, 360), det(400, 360)], 0.0)
    tracks, _ = tracker.update([det(330, 360), det(430, 360)], 10.0)
    by_id = {t.track_id: t.cx for t in tracks}
    assert by_id[1] < by_id[2]


def test_fifo_tracker_exit_closes_as_advanced(tmp_path):
    path = tmp_path / "lane.json"
    write_lane(path, entry=(280, 360), exit_=(480, 360))
    tracker = FifoTracker("injection", (W, H), lane_path=path,
                          max_missed_frames=0, step_gate_mm=200.0)
    tracker.start()

    tracker.update([det(300, 360)], 0.0)
    tracker.update([det(460, 360)], 10.0)  # advanced to near the exit
    tracks, closed = tracker.update([], 20.0)
    assert not tracks
    assert closed and closed[0].closed_reason == "advanced"


def test_fifo_tracker_midlane_disappearance_is_lost(tmp_path):
    path = tmp_path / "lane.json"
    write_lane(path, entry=(280, 360), exit_=(480, 360))
    tracker = FifoTracker("injection", (W, H), lane_path=path, max_missed_frames=0)
    tracker.start()

    tracker.update([det(300, 360)], 0.0)  # stays near entry, never advances
    tracks, closed = tracker.update([], 10.0)
    assert not tracks
    assert closed and closed[0].closed_reason == "lost"


# --------------------------------------------------------------------------
# RegionCoordinator
# --------------------------------------------------------------------------
def test_coordinator_handoff_across_zone_boundary(tmp_path):
    slots_path = tmp_path / "slots_storing.json"
    write_slots(slots_path, [(80, 300)])
    lane_path = tmp_path / "lane_injection.json"
    write_lane(lane_path, entry=(400, 360), exit_=(600, 360))

    zone_map = ZoneMap(W, H, {
        "storing": rect(0.00, 0.20),
        "injection": rect(0.20, 0.60),
    })
    id_source = itertools.count(1)
    storing = SlotTracker("storing", (W, H), slots_path=slots_path,
                          id_source=id_source)
    injection = FifoTracker("injection", (W, H), lane_path=lane_path,
                            id_source=id_source, max_missed_frames=0,
                            step_gate_mm=250.0)
    coord = RegionCoordinator(zone_map, {"storing": storing, "injection": injection},
                              region_sequence=("storing", "injection"))
    coord.start()

    coord.update([det(420, 360)], 0.0)              # injection: spawn id 1
    coord.update([det(580, 360)], 10.0)              # injection: id 1 advances near exit
    # injection loses its detection (about to close as "advanced") while a
    # crucible appears in the adjacent storing slot in the very same frame.
    results = coord.update([det(80, 300)], 20.0)

    storing_tracks, _ = results["storing"]
    injection_tracks, injection_closed = results["injection"]
    assert not injection_tracks
    assert injection_closed and injection_closed[0].closed_reason == "advanced"
    assert len(storing_tracks) == 1
    assert storing_tracks[0].track_id == 1  # inherited, not a fresh id
    assert coord.handoffs_this_frame() == [(1, "injection", "storing")]


def test_coordinator_does_not_hand_off_between_non_adjacent_zones(tmp_path):
    slots_a = tmp_path / "slots_a.json"
    write_slots(slots_a, [(80, 300)])
    slots_b = tmp_path / "slots_b.json"
    write_slots(slots_b, [(900, 300)])

    zone_map = ZoneMap(W, H, {
        "storing": rect(0.00, 0.20),
        "injection": rect(0.20, 0.40),
        "collection": rect(0.60, 0.80),
    })
    id_source = itertools.count(1)
    storing = SlotTracker("storing", (W, H), slots_path=slots_a, id_source=id_source,
                          max_missed_frames=0)
    collection = SlotTracker("collection", (W, H), slots_path=slots_b,
                             id_source=id_source)
    coord = RegionCoordinator(
        zone_map, {"storing": storing, "collection": collection},
        region_sequence=("storing", "injection", "collection"))
    coord.start()

    coord.update([det(80, 300)], 0.0)                # storing: spawn id 1
    results = coord.update([det(900, 300)], 10.0)     # storing vacates, collection fills
    collection_tracks, _ = results["collection"]
    assert collection_tracks[0].track_id == 2         # fresh id, not 1
    assert coord.handoffs_this_frame() == []


def test_coordinator_does_not_bridge_a_skipped_zone(tmp_path):
    """A crucible whose whole transit through the middle zone falls between
    two frames gets a fresh id, not the donor's.

    This is a known, accepted limitation of keying handoffs on immediate
    neighbours in the process order: at a slow cadence a crucible can enter
    and leave `injection` entirely unobserved, and nothing in the frames
    then connects the storing track to the collection one. Locked in as a
    test so the behaviour is deliberate rather than discovered later.
    """
    slots_a = tmp_path / "slots_storing.json"
    write_slots(slots_a, [(80, 300)])
    slots_b = tmp_path / "slots_collection.json"
    write_slots(slots_b, [(900, 300)])

    zone_map = ZoneMap(W, H, {
        "storing": rect(0.00, 0.20),
        "injection": rect(0.20, 0.40),
        "collection": rect(0.60, 0.80),
    })
    id_source = itertools.count(1)
    storing = SlotTracker("storing", (W, H), slots_path=slots_a,
                          id_source=id_source, max_missed_frames=0)
    collection = SlotTracker("collection", (W, H), slots_path=slots_b,
                             id_source=id_source)
    coord = RegionCoordinator(
        zone_map, {"storing": storing, "collection": collection},
        region_sequence=("storing", "injection", "collection"))
    coord.start()

    coord.update([det(80, 300)], 0.0)                  # storing: id 1
    # next frame: gone from storing, already sitting in collection - the
    # injection leg was never observed.
    results = coord.update([det(900, 300)], 10.0)
    collection_tracks, _ = results["collection"]
    assert collection_tracks[0].track_id != 1
    assert coord.handoffs_this_frame() == []


def test_coordinator_reslot_within_same_zone_keeps_identity(tmp_path):
    slots_path = tmp_path / "slots.json"
    write_slots(slots_path, [(80, 300), (180, 300)])

    zone_map = ZoneMap(W, H, {"storing": rect(0.00, 0.30)})
    storing = SlotTracker("storing", (W, H), slots_path=slots_path,
                          max_missed_frames=0)
    coord = RegionCoordinator(zone_map, {"storing": storing},
                              region_sequence=("storing",))
    coord.start()

    coord.update([det(80, 300)], 0.0)                 # slot 0
    results = coord.update([det(180, 300)], 10.0)      # slot 0 vacated, slot 1 filled
    tracks, _ = results["storing"]
    assert tracks[0].track_id == 1
    assert tracks[0].slot_id == 1
    assert coord.handoffs_this_frame() == [(1, "storing", "storing")]


def test_coordinator_reset_clears_every_zone(tmp_path):
    slots_path = tmp_path / "slots.json"
    write_slots(slots_path, [(80, 300)])
    zone_map = ZoneMap(W, H, {"storing": rect(0.00, 0.30)})
    storing = SlotTracker("storing", (W, H), slots_path=slots_path)
    coord = RegionCoordinator(zone_map, {"storing": storing},
                              region_sequence=("storing",))
    coord.start()

    coord.update([det(80, 300)], 0.0)
    assert coord.tracks
    coord.reset()
    assert not coord.tracks
    assert coord.handoffs_this_frame() == []


# --------------------------------------------------------------------------
# RegionTracker - the Tracker-ABC adapter the pipeline drives
# --------------------------------------------------------------------------
def _two_zone_tracker(tmp_path):
    slots_path = tmp_path / "slots_storing.json"
    write_slots(slots_path, [(80, 300), (180, 300)])
    lane_path = tmp_path / "lane_injection.json"
    write_lane(lane_path, entry=(400, 360), exit_=(600, 360))

    zone_map = ZoneMap(W, H, {
        "storing": rect(0.00, 0.20),
        "injection": rect(0.20, 0.60),
    })
    id_source = itertools.count(1)
    trackers = {
        "storing": SlotTracker("storing", (W, H), slots_path=slots_path,
                               id_source=id_source),
        "injection": FifoTracker("injection", (W, H), lane_path=lane_path,
                                 id_source=id_source),
    }
    coord = RegionCoordinator(zone_map, trackers,
                              region_sequence=("storing", "injection"))
    return RegionTracker(coordinator=coord)


def test_region_tracker_flattens_zones_to_the_tracker_interface(tmp_path):
    tracker = _two_zone_tracker(tmp_path)
    tracker.start()

    active, closed = tracker.update([det(80, 300), det(420, 360)], 0.0)
    assert isinstance(active, list) and isinstance(closed, list)
    assert len(active) == 2                      # one per zone, flattened
    assert {t.stage for t in active} == {"storing", "injection"}
    assert not closed
    assert len(tracker.tracks) == 2


def test_region_tracker_reset_clears_state(tmp_path):
    tracker = _two_zone_tracker(tmp_path)
    tracker.start()
    tracker.update([det(80, 300)], 0.0)
    assert tracker.tracks
    tracker.reset()
    assert not tracker.tracks


def test_region_tracker_is_usable_as_a_tracker(tmp_path):
    """The pipeline only ever touches the ABC surface - pin that down."""
    from pipeline.tracking import Tracker

    tracker = _two_zone_tracker(tmp_path)
    assert isinstance(tracker, Tracker)
    tracker.start()
    result = tracker.update([], 0.0)
    assert isinstance(result, tuple) and len(result) == 2


def test_region_tracker_does_not_report_a_handoff_as_closed(tmp_path):
    """A crucible moving zones must not look like a disappearance.

    pipeline/runner.py fires a closure event and calls history.forget() on
    everything in the closed list, so leaking handoff donors into it would
    drop the crucible's feature history and raise a false alarm on every
    normal zone transition.
    """
    slots_path = tmp_path / "slots_storing.json"
    write_slots(slots_path, [(80, 300)])
    lane_path = tmp_path / "lane_injection.json"
    write_lane(lane_path, entry=(400, 360), exit_=(600, 360))

    zone_map = ZoneMap(W, H, {
        "storing": rect(0.00, 0.20),
        "injection": rect(0.20, 0.60),
    })
    id_source = itertools.count(1)
    trackers = {
        "storing": SlotTracker("storing", (W, H), slots_path=slots_path,
                               id_source=id_source, max_missed_frames=0),
        "injection": FifoTracker("injection", (W, H), lane_path=lane_path,
                                 id_source=id_source),
    }
    coord = RegionCoordinator(zone_map, trackers,
                              region_sequence=("storing", "injection"))
    tracker = RegionTracker(coordinator=coord)
    tracker.start()

    tracker.update([det(80, 300)], 0.0)                    # storing: id 1
    active, closed = tracker.update([det(420, 360)], 10.0)  # moved to injection

    assert coord.handoffs_this_frame() == [(1, "storing", "injection")]
    assert not closed                       # the donor is not a disappearance
    assert [t.track_id for t in active] == [1]
    assert active[0].stage == "injection"


def test_region_tracker_still_reports_a_real_disappearance(tmp_path):
    """The inverse of the above - a genuine loss must still surface."""
    slots_path = tmp_path / "slots.json"
    write_slots(slots_path, [(80, 300)])
    zone_map = ZoneMap(W, H, {"storing": rect(0.00, 0.30)})
    coord = RegionCoordinator(
        zone_map,
        {"storing": SlotTracker("storing", (W, H), slots_path=slots_path,
                                max_missed_frames=0)},
        region_sequence=("storing",))
    tracker = RegionTracker(coordinator=coord)
    tracker.start()

    tracker.update([det(80, 300)], 0.0)
    active, closed = tracker.update([], 10.0)      # nothing anywhere
    assert not active
    assert [t.closed_reason for t in closed] == ["vacated"]


def test_create_tracker_auto_is_still_hungarian():
    """Default behaviour must not change for anything that did not opt in."""
    from pipeline.tracking import HungarianTracker, create_tracker

    assert isinstance(create_tracker("auto"), HungarianTracker)
    assert isinstance(create_tracker(), HungarianTracker)


def test_create_tracker_rejects_an_unknown_name():
    from pipeline.tracking import create_tracker

    with pytest.raises(ValueError, match="unknown tracker"):
        create_tracker("nonsense")


# --------------------------------------------------------------------------
# CrucibleLocalizer
# --------------------------------------------------------------------------
def test_crucible_localizer_wraps_detect_crucibles(monkeypatch):
    """It is a thin adapter: circles in, Detections out, no identity."""
    import pipeline.features
    from drivers.base import Frame
    from pipeline.localize import CrucibleLocalizer

    monkeypatch.setattr(pipeline.features, "detect_crucibles",
                        lambda img, **kw: [(100.0, 200.0, 60.0), (300.0, 400.0, 55.0)])

    frame = Frame(image=np.zeros((H, W, 3), np.uint8), timestamp=0.0,
                  frame_id=1, source="test")
    loc = CrucibleLocalizer()
    loc.start()
    dets = loc.locate(frame)

    assert [(d.cx, d.cy, d.radius) for d in dets] == [
        (100.0, 200.0, 60.0), (300.0, 400.0, 55.0)]
    assert all(d.meta["detector"] == "hough_crucible" for d in dets)
    assert loc.real_capable


def test_detect_crucibles_survives_a_frame_smaller_than_its_roi():
    """Its second-pass ROI is in full-capture pixels.

    On anything smaller - a 1280x720 preview, which is what the live camera
    is configured for - the slice came back empty and OpenCV threw, so live
    capture died on the first frame.
    """
    from pipeline.features import detect_crucibles

    small = np.zeros((720, 1280, 3), np.uint8)
    assert detect_crucibles(small) == []          # no crash, no detections

    tall = np.zeros((3040, 1280, 3), np.uint8)    # wide enough, short enough
    assert isinstance(detect_crucibles(tall), list)


def test_lid_state_reads_the_calibrated_threshold(monkeypatch):
    from config import DETECTION
    from pipeline.types import Track
    from tools.track_regions import lid_state

    import tools.track_regions as tr
    monkeypatch.setattr("pipeline.features.lid_score",
                        lambda *a, **k: DETECTION.lid_score_threshold + 1)
    t = Track(track_id=1, cx=10.0, cy=10.0, radius=5.0,
              first_seen_ts=0.0, last_seen_ts=0.0)
    state, score = lid_state(np.zeros((40, 40, 3), np.uint8), t)
    assert state == "lid"

    monkeypatch.setattr("pipeline.features.lid_score",
                        lambda *a, **k: DETECTION.lid_score_threshold - 1)
    assert lid_state(np.zeros((40, 40, 3), np.uint8), t)[0] == "open"


def test_track_label_is_three_digits():
    from tools.track_regions import track_label

    assert track_label(1) == "001"
    assert track_label(42) == "042"
    assert track_label(999) == "999"
    # past 999 it keeps counting rather than wrapping onto a live id
    assert track_label(1000) == "1000"


def test_create_localizer_knows_crucible_but_auto_is_unchanged():
    from pipeline.localize import CrucibleLocalizer, create_localizer

    assert isinstance(create_localizer("crucible"), CrucibleLocalizer)
    assert not isinstance(create_localizer("auto"), CrucibleLocalizer)
