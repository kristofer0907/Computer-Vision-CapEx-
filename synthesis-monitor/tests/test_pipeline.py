"""Detector host, localisers and the file source.

These used to drive PipelineRunner against the synthetic platform. The runner
is gone - main.py:Monitor is the loop now - and the simulator draws the old
whole-platform bench, not the marked crucible rack, so end-to-end assertions
against it would be testing a drawing. What survives here is everything that
does not need that runner: the detector registry and its failure isolation,
the localisers, and the file source.

End-to-end coverage of the real pipeline now comes from running
`python main.py --rgb file` over a capture folder.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drivers.base import Frame
from pipeline.detectors import DetectorHost, load_detectors
from pipeline.detectors.base import Detector, DetectionContext
from pipeline.localize import GroundTruthLocalizer
from pipeline.zones import ZoneMap


def _ctx(frame_id: int = 1) -> DetectionContext:
    """The smallest context a detector can legally be handed."""
    image = np.zeros((64, 64, 3), np.uint8)
    return DetectionContext(
        frame=Frame(image, 0.0, frame_id, "test", True),
        timestamp=0.0, frame_id=frame_id,
        tracks=[], reports=[], crops={}, masks={}, boxes={},
        zones={}, zone_map=ZoneMap(), history=None,
        closed_tracks=[], frame_ref="", interval_s=45.0)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def test_detectors_all_load():
    detectors = load_detectors()
    assert {d.name for d in detectors} == {
        "turbidity", "sol_gel_transition", "color_change", "spill",
        "missing_crucible", "misplaced_labware", "missing_lid"}
    # missing_lid is the one that actually looks at pixels.
    assert {d.name for d in detectors if d.implemented} == {"missing_lid"}


def test_unknown_detector_is_a_hard_error():
    with pytest.raises(ValueError):
        load_detectors(["turbidity", "not_a_detector"])


def test_stub_detectors_produce_nothing():
    host = DetectorHost([d for d in load_detectors() if not d.implemented])
    host.start()
    try:
        results, warnings = host.run(_ctx())
    finally:
        host.stop()
    assert results == []
    assert warnings == []


# --------------------------------------------------------------------------
# Detector isolation: the property that lets half-written detection logic be
# developed against a live system without taking it down.
# --------------------------------------------------------------------------
class _Exploding(Detector):
    name = "exploding"
    max_consecutive_errors = 2

    def check(self, ctx):
        raise RuntimeError("boom")


class _Quiet(Detector):
    name = "quiet"

    def __init__(self):
        self.calls = 0

    def check(self, ctx):
        self.calls += 1
        return []


def test_a_raising_detector_does_not_stop_the_others():
    quiet = _Quiet()
    host = DetectorHost([_Exploding(), quiet])
    host.start()
    try:
        _results, warnings = host.run(_ctx())
    finally:
        host.stop()
    assert quiet.calls == 1
    assert any("boom" in w for w in warnings)


def test_a_repeatedly_raising_detector_gets_disabled():
    host = DetectorHost([_Exploding()])
    host.start()
    try:
        for i in range(3):
            host.run(_ctx(i))
        assert not host.health()[0]["enabled"]
        # And once disabled it stops adding a warning every frame.
        _results, warnings = host.run(_ctx(9))
    finally:
        host.stop()
    assert warnings == []


# --------------------------------------------------------------------------
# Localisers
# --------------------------------------------------------------------------
def test_ground_truth_localizer_is_blind_on_a_real_frame():
    """It must not silently pretend to work once a real camera is attached."""
    loc = GroundTruthLocalizer()
    loc.start()
    real = Frame(np.zeros((10, 10, 3), np.uint8), 0.0, 1, "picamera2", False)
    assert loc.locate(real) == []
    assert loc.real_capable is False


def test_auto_localizer_is_the_crucible_one():
    """"auto" is classical Hough - the settled decision."""
    from pipeline.localize import CrucibleLocalizer, create_localizer

    assert isinstance(create_localizer("auto"), CrucibleLocalizer)
    assert create_localizer("auto").real_capable is True


# --------------------------------------------------------------------------
# Zone configuration robustness. The simulator reads the same polygons the
# staging does, so tracing real zones must not break the ability to keep
# regression-testing against synthetic frames.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("polygons", [
    pytest.param({}, id="no-zones"),
    pytest.param({"rack": [(0.02, 0.3), (0.3, 0.3), (0.3, 0.7), (0.02, 0.7)]},
                 id="renamed-zones"),
    pytest.param({"storing": [(0.05, 0.35), (0.28, 0.30), (0.30, 0.72),
                              (0.03, 0.68)]},
                 id="non-rectangular"),
])
def test_scene_renders_against_arbitrary_zone_sets(monkeypatch, polygons):
    from config import ZoneConfig
    import drivers.scene as scene

    monkeypatch.setattr(scene, "ZONES", ZoneConfig(polygons=polygons))
    platform = scene.SyntheticPlatform()
    image = platform.render(120.0)
    assert image.shape[2] == 3
    assert platform.truth_at(120.0)["crucibles"], "crucibles must still be placed"


# --------------------------------------------------------------------------
# ManualLocalizer: how your own captured images get through the pipeline
# before the Hough parameters are tuned for a new bench layout.
# --------------------------------------------------------------------------
def _marks_file(tmp_path, **entries):
    import json
    path = tmp_path / "marks.json"
    path.write_text(json.dumps(entries))
    return path


def _file_frame(image, name):
    return Frame(image, 0.0, 1, "file", True, {"name": name, "path": name})


def test_manual_localizer_reads_marks(tmp_path):
    from pipeline.localize import ManualLocalizer

    path = _marks_file(tmp_path, default=[
        {"cx": 100, "cy": 200, "radius": 17},
        {"cx": 140, "cy": 200, "radius": 17},
    ])
    loc = ManualLocalizer(path)
    loc.start()
    dets = loc.locate(_file_frame(np.zeros((720, 1280, 3), np.uint8), "any.jpg"))
    assert [(d.cx, d.cy) for d in dets] == [(100.0, 200.0), (140.0, 200.0)]
    assert loc.real_capable is True


def test_manual_localizer_prefers_a_per_image_entry(tmp_path):
    from pipeline.localize import ManualLocalizer

    path = _marks_file(
        tmp_path,
        default=[{"cx": 10, "cy": 10, "radius": 5}],
        **{"shot_02.jpg": [{"cx": 900, "cy": 400, "radius": 17}]})
    loc = ManualLocalizer(path)
    loc.start()
    image = np.zeros((720, 1280, 3), np.uint8)

    specific = loc.locate(_file_frame(image, "shot_02.jpg"))
    assert (specific[0].cx, specific[0].cy) == (900.0, 400.0)

    fallback = loc.locate(_file_frame(image, "shot_99.jpg"))
    assert (fallback[0].cx, fallback[0].cy) == (10.0, 10.0)


def test_manual_localizer_scales_to_a_different_frame_size(tmp_path):
    """Marks made on a 1280-wide capture, replayed at half that."""
    import json

    from pipeline.localize import ManualLocalizer

    path = tmp_path / "marks.json"
    path.write_text(json.dumps({
        "default": [{"cx": 640, "cy": 360, "radius": 20}],
        "_image_size": [1280, 720],
    }))
    loc = ManualLocalizer(path)
    loc.start()
    det = loc.locate(_file_frame(np.zeros((360, 640, 3), np.uint8), "x.jpg"))[0]
    assert (det.cx, det.cy) == (320.0, 180.0)
    assert det.radius == pytest.approx(10.0)


def test_manual_localizer_says_what_to_do_when_marks_are_missing(tmp_path):
    from pipeline.localize import ManualLocalizer

    loc = ManualLocalizer(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="mark_vials"):
        loc.start()


def test_file_source_reports_which_file_a_frame_came_from(tmp_path):
    """ManualLocalizer's per-image lookup depends on this."""
    from drivers.rgb_cam import FileCameraSource

    for i in range(2):
        cv2.imwrite(str(tmp_path / f"shot_{i}.png"),
                    np.full((40, 60, 3), 10 * (i + 1), np.uint8))

    source = FileCameraSource(str(tmp_path), loop=False, latency_s=0.0)
    source.start()
    try:
        names = [source.capture().truth["name"] for _ in range(2)]
    finally:
        source.stop()
    assert names == ["shot_0.png", "shot_1.png"]
