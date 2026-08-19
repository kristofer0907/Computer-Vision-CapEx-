"""End-to-end: synthetic frames in, PipelineResult out.

These are the tests that would have caught every integration bug worth
catching before hardware existed. They run the real runner, the real tracker,
the real detector host and the real ROI code against the synthetic platform.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drivers.rgb_cam import MockCameraSource
from pipeline.detectors import DetectorHost, load_detectors
from pipeline.detectors.base import Detector, DetectionContext
from pipeline.features import FeatureExtractor, NullFeatureExtractor
from pipeline.localize import GroundTruthLocalizer, NullLocalizer
from pipeline.runner import PipelineRunner


@pytest.fixture
def camera():
    cam = MockCameraSource(time_scale=60.0, latency_s=0.0)
    cam.start()
    yield cam
    cam.stop()


@pytest.fixture
def runner():
    r = PipelineRunner(localizer=GroundTruthLocalizer(),
                       extractor=NullFeatureExtractor(),
                       draw_overlay=False)
    r.start()
    yield r
    r.stop()


def test_processes_a_frame(camera, runner):
    result = runner.process(camera.capture())
    assert result.frame_id == 1
    assert result.simulated is True
    assert "localize" in result.timings_ms


def test_finds_and_confirms_vials(camera, runner):
    for _ in range(3):
        result = runner.process(camera.capture())
    assert result.n_vials > 0
    assert all(v.radius > 0 for v in result.vials)


def test_ids_are_stable_across_frames(camera, runner):
    seen = []
    for _ in range(5):
        result = runner.process(camera.capture())
        seen.append({v.track_id for v in result.vials})
    # The first vials tracked must still be tracked at the end - the batch is
    # stationary in filling for most of a run, so churn here means the gate
    # or the assignment is wrong.
    assert seen[1] & seen[-1]


def test_vials_get_staged(camera, runner):
    for _ in range(4):
        result = runner.process(camera.capture())
    stages = {v.stage for v in result.vials}
    assert stages - {None}, "no vial landed in any zone polygon"
    assert sum(result.stage_counts.values()) > 0


def test_null_localizer_yields_an_empty_but_valid_result(camera):
    runner = PipelineRunner(localizer=NullLocalizer(),
                            extractor=NullFeatureExtractor(),
                            draw_overlay=False)
    runner.start()
    try:
        result = runner.process(camera.capture())
        assert result.n_vials == 0
        assert result.events == []
        assert result.stage_counts  # keys present, all zero
    finally:
        runner.stop()


def test_overlay_is_encoded_jpeg(camera):
    runner = PipelineRunner(localizer=GroundTruthLocalizer(),
                            extractor=NullFeatureExtractor(),
                            draw_overlay=True)
    runner.start()
    try:
        result = runner.process(camera.capture())
        assert result.overlay_jpeg[:2] == b"\xff\xd8"    # JPEG SOI
    finally:
        runner.stop()


def test_stub_detectors_all_load_and_produce_nothing():
    detectors = load_detectors()
    assert {d.name for d in detectors} == {
        "turbidity", "solgel", "color_change", "spill", "vial_presence"}
    assert all(not d.implemented for d in detectors)


def test_unknown_detector_is_a_hard_error():
    with pytest.raises(ValueError):
        load_detectors(["turbidity", "not_a_detector"])


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


def test_a_raising_detector_does_not_stop_the_others(camera):
    quiet = _Quiet()
    host = DetectorHost([_Exploding(), quiet])
    runner = PipelineRunner(localizer=GroundTruthLocalizer(),
                            extractor=NullFeatureExtractor(),
                            detectors=host, draw_overlay=False)
    runner.start()
    try:
        result = runner.process(camera.capture())
        assert quiet.calls == 1
        assert any("boom" in w for w in result.warnings)
    finally:
        runner.stop()


def test_a_repeatedly_raising_detector_gets_disabled(camera):
    host = DetectorHost([_Exploding()])
    runner = PipelineRunner(localizer=GroundTruthLocalizer(),
                            extractor=NullFeatureExtractor(),
                            detectors=host, draw_overlay=False)
    runner.start()
    try:
        for _ in range(3):
            result = runner.process(camera.capture())
        assert not host.health()[0]["enabled"]
        # And once disabled it stops adding a warning every frame.
        result = runner.process(camera.capture())
        assert not result.warnings
    finally:
        runner.stop()


def test_a_raising_extractor_does_not_stop_the_frame(camera):
    class Broken(FeatureExtractor):
        name = "broken"

        def extract(self, crop, mask, track, prev=None):
            raise ValueError("nope")

    runner = PipelineRunner(localizer=GroundTruthLocalizer(),
                            extractor=Broken(), draw_overlay=False)
    runner.start()
    try:
        for _ in range(2):
            result = runner.process(camera.capture())
        assert result.n_vials > 0
        assert any("nope" in w for w in result.warnings)
    finally:
        runner.stop()


# --------------------------------------------------------------------------
# The context a detector is handed
# --------------------------------------------------------------------------
class _Capturing(Detector):
    name = "capturing"

    def __init__(self):
        self.ctx: DetectionContext | None = None

    def check(self, ctx):
        self.ctx = ctx
        return [self.event(ctx, "test", "hello", severity="info",
                           track_id=ctx.tracks[0].track_id if ctx.tracks else None)]


def test_context_carries_crops_masks_and_zones(camera):
    spy = _Capturing()
    runner = PipelineRunner(localizer=GroundTruthLocalizer(),
                            extractor=NullFeatureExtractor(),
                            detectors=DetectorHost([spy]), draw_overlay=False)
    runner.start()
    try:
        for _ in range(3):
            result = runner.process(camera.capture())
    finally:
        runner.stop()

    ctx = spy.ctx
    assert ctx.tracks
    tid = ctx.tracks[0].track_id
    assert ctx.crops[tid].ndim == 3
    assert ctx.masks[tid].shape == ctx.crops[tid].shape[:2]
    assert ctx.masks[tid].max() == 255
    assert ctx.zones, "zone views must be built"

    view = next(iter(ctx.zones.values()))
    assert view.bench_mask.shape == view.zone_mask.shape
    assert view.bench_mask.sum() <= view.zone_mask.sum(), "vials must be punched out"
    assert any(e.kind == "test" for e in result.events)


def test_previous_crop_is_available_and_size_matched(camera):
    spy = _Capturing()
    runner = PipelineRunner(localizer=GroundTruthLocalizer(),
                            extractor=NullFeatureExtractor(),
                            detectors=DetectorHost([spy]), draw_overlay=False)
    runner.start()
    try:
        first = runner.process(camera.capture())
        # Nothing is confirmed on frame 1 (min_hits_to_confirm), so there is
        # nothing to have a previous crop of yet.
        assert first.n_vials == 0
        for _ in range(3):
            runner.process(camera.capture())
    finally:
        runner.stop()

    ctx = spy.ctx
    tid = ctx.tracks[0].track_id
    prev = ctx.previous_crop(tid)
    assert prev is not None, "the colour-change comparison needs this"
    assert prev.shape == ctx.crops[tid].shape, "must be diffable without resizing"


def test_feature_column_reads_across_the_batch(camera):
    class Fake(FeatureExtractor):
        name = "fake"

        def extract(self, crop, mask, track, prev=None):
            return {"brightness": float(track.track_id)}

    spy = _Capturing()
    runner = PipelineRunner(localizer=GroundTruthLocalizer(), extractor=Fake(),
                            detectors=DetectorHost([spy]), draw_overlay=False)
    runner.start()
    try:
        for _ in range(3):
            runner.process(camera.capture())
    finally:
        runner.stop()

    ids, values = spy.ctx.feature_column("brightness")
    assert len(ids) == len(values) > 1
    assert values == [float(i) for i in ids]
    assert spy.ctx.feature_column("does_not_exist") == ([], [])


def test_history_records_features_over_time(camera):
    class Counting(FeatureExtractor):
        name = "counting"

        def __init__(self):
            self.n = 0

        def extract(self, crop, mask, track, prev=None):
            self.n += 1
            return {"n": float(self.n)}

    runner = PipelineRunner(localizer=GroundTruthLocalizer(),
                            extractor=Counting(), draw_overlay=False)
    runner.start()
    try:
        for _ in range(4):
            result = runner.process(camera.capture())
        tid = result.vials[0].track_id
        assert len(runner.history.series(tid, "n")) >= 2
    finally:
        runner.stop()


def test_ground_truth_localizer_is_blind_on_a_real_frame():
    """It must not silently pretend to work once a real camera is attached."""
    from drivers.base import Frame

    loc = GroundTruthLocalizer()
    loc.start()
    real = Frame(np.zeros((10, 10, 3), np.uint8), 0.0, 1, "picamera2", False)
    assert loc.locate(real) == []
    assert loc.real_capable is False


# --------------------------------------------------------------------------
# Zone configuration robustness. The simulator reads the same polygons the
# staging does, so tracing real zones must not break the ability to keep
# regression-testing against synthetic frames.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("polygons", [
    pytest.param({}, id="no-zones"),
    pytest.param({"rack": [(0.02, 0.3), (0.3, 0.3), (0.3, 0.7), (0.02, 0.7)]},
                 id="renamed-zones"),
    pytest.param({"filling": [(0.05, 0.35), (0.28, 0.30), (0.30, 0.72),
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
    assert platform.truth_at(120.0)["vials"], "vials must still be placed"


# --------------------------------------------------------------------------
# ManualLocalizer: how your own captured images get through the pipeline
# before a real localiser exists.
# --------------------------------------------------------------------------
def _marks_file(tmp_path, **entries):
    import json
    path = tmp_path / "vials.json"
    path.write_text(json.dumps(entries))
    return path


def _file_frame(image, name):
    from drivers.base import Frame
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

    path = tmp_path / "vials.json"
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
