"""The analysis pass: one frame in, one PipelineResult out.

Order is fixed and each stage only depends on the ones before it:

    localise -> track -> stage -> crop -> features -> detect -> report

The runner owns no threads, no queues and no sockets. It is a plain object
with a process() method, which is what makes the whole pipeline testable by
handing it a synthetic frame and reading the result - no camera, no
multiprocessing, no Flask.

It contains no detection logic. Every decision about what is normal lives in
pipeline/features.py and pipeline/detectors/. This file decides only what runs
in what order and what gets published.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from config import CADENCE, DASHBOARD
from drivers.base import Frame
from drivers.rgb_cam import encode_jpeg
from pipeline import roi
from pipeline.detectors import DetectionContext, DetectorHost, ZoneView
from pipeline.features import FeatureExtractor, create_extractor
from pipeline.history import VialHistory
from pipeline.localize import Localizer, create_localizer
from pipeline.tracking import CadenceController, HungarianTracker, Tracker
from pipeline.types import Event, PipelineResult, Track, VialReport
from pipeline.zones import ZoneMap

log = logging.getLogger(__name__)

SEVERITY_BGR = {
    "info": (200, 170, 90),
    "warning": (60, 168, 224),
    "alert": (82, 82, 224),
}
TRACK_BGR = (110, 199, 98)
UNCONFIRMED_BGR = (120, 120, 120)


class _Timer:
    """Accumulates per-stage wall time so a slow stage is visible, not guessed."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round((now - self._t) * 1000.0, 2)
        self._t = now


class PipelineRunner:
    """Owns the analysis stages and the per-vial memory between frames."""

    def __init__(self,
                 localizer: Localizer | None = None,
                 extractor: FeatureExtractor | None = None,
                 tracker: Tracker | None = None,
                 detectors: DetectorHost | None = None,
                 zone_map: ZoneMap | None = None,
                 history: VialHistory | None = None,
                 draw_overlay: bool = True) -> None:
        self.zones = zone_map or ZoneMap()
        self.localizer = localizer or create_localizer()
        self.extractor = extractor or create_extractor()
        self.tracker = tracker or HungarianTracker(self.zones)
        self.detectors = detectors or DetectorHost()
        self.history = history or VialHistory()
        self.cadence = CadenceController()
        self.draw_overlay = draw_overlay

        self._started = False
        self._frames = 0
        self._last_ts: float | None = None
        self._warned_localizer = False
        self._next_interval_s = CADENCE.analysis_interval_s

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._started:
            return
        self.localizer.start()
        self.extractor.start()
        self.detectors.start()
        self._started = True
        log.info("pipeline started: localizer=%s extractor=%s detectors=%s",
                 self.localizer.name, self.extractor.name,
                 ",".join(d.name for d in self.detectors.detectors) or "none")
        if not self.zones.polygons_px:
            log.warning("no zone polygons - every vial will be unstaged")

    def stop(self) -> None:
        if not self._started:
            return
        for closer in (self.detectors.stop, self.extractor.stop, self.localizer.stop):
            try:
                closer()
            except Exception:
                log.warning("shutdown step failed", exc_info=True)
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    @property
    def next_interval_s(self) -> float:
        """Seconds the caller should wait before the next analysis frame.

        A plain read of state computed during the last process() call. The
        cadence controller is advanced exactly once per frame, inside
        process(); advancing it from a property read would let the number of
        times something asked change what the answer is.
        """
        return self._next_interval_s

    @property
    def busy(self) -> bool:
        """True while the batch is moving and the fast cadence is in effect."""
        return self.cadence.busy

    # -------------------------------------------------------------- process
    def process(self, frame: Frame) -> PipelineResult:
        if not self._started:
            raise RuntimeError("process() before start()")

        timer = _Timer()
        warnings: list[str] = []
        image = frame.image
        now = frame.timestamp
        interval_s = 0.0 if self._last_ts is None else max(0.0, now - self._last_ts)

        # 1. localise --------------------------------------------------------
        detections = self.localizer.locate(frame)
        timer.mark("localize")
        if not detections and not self._warned_localizer:
            self._warned_localizer = True
            msg = (f"localizer {self.localizer.name!r} returned no detections - "
                   "vial localisation is not implemented yet")
            log.warning(msg)
        if not self.localizer.real_capable and not frame.simulated:
            warnings.append(
                f"localizer {self.localizer.name!r} cannot work on real frames")

        # 2. track and stage -------------------------------------------------
        # The gate comes from the cadence decided on the *previous* frame,
        # which is the interval this frame actually waited out.
        if hasattr(self.tracker, "set_gate_mm"):
            self.tracker.set_gate_mm(self.cadence.gate_mm())
        tracks, closed = self.tracker.update(detections, now)
        confirmed = [t for t in tracks if t.confirmed]
        self._next_interval_s = self.cadence.observe(confirmed)
        timer.mark("track")

        # 3. crops -----------------------------------------------------------
        crops: dict[int, np.ndarray] = {}
        masks: dict[int, np.ndarray] = {}
        boxes: dict[int, tuple[int, int, int, int]] = {}
        for t in confirmed:
            crop, mask, box = roi.crop_with_mask(image, t.cx, t.cy, t.radius)
            if crop.size == 0:
                # Fully off-frame. Should not happen, but a zero-size array
                # would make every downstream cv2 call throw.
                continue
            crops[t.track_id], masks[t.track_id], boxes[t.track_id] = crop, mask, box
        timer.mark("crop")

        # 4. features --------------------------------------------------------
        try:
            self.extractor.frame_context(image)
        except Exception as exc:
            log.exception("feature extractor frame_context failed")
            warnings.append(f"features.frame_context: {exc}")

        features: dict[int, dict[str, float]] = {}
        for t in confirmed:
            tid = t.track_id
            if tid not in crops:
                continue
            prev = self.history.previous_crop_matched(tid, crops[tid])
            try:
                features[tid] = self.extractor.extract(crops[tid], masks[tid], t, prev)
            except Exception as exc:
                log.exception("feature extraction failed for track %d", tid)
                warnings.append(f"features[{tid}]: {exc}")
                features[tid] = {}
        timer.mark("features")

        # 5. history ---------------------------------------------------------
        # Written after extraction so that `prev` above is genuinely the
        # previous frame's crop and not the one just taken.
        for t in confirmed:
            tid = t.track_id
            if tid in crops:
                self.history.record(tid, now, features.get(tid), crops[tid])
        for t in closed:
            self.history.forget(t.track_id)
        self.history.prune({t.track_id for t in tracks})

        reports = [self._report(t, features.get(t.track_id, {}), now)
                   for t in confirmed]

        # 6. detect ----------------------------------------------------------
        ctx = DetectionContext(
            frame=frame, timestamp=now, frame_id=frame.frame_id,
            tracks=confirmed, reports=reports,
            crops=crops, masks=masks, boxes=boxes,
            zones=self._zone_views(image, confirmed),
            zone_map=self.zones, history=self.history,
            closed_tracks=closed, interval_s=interval_s,
        )
        timer.mark("context")

        events, detector_warnings = self.detectors.run(ctx)
        warnings.extend(detector_warnings)
        events.extend(self._closure_events(closed, frame.frame_id, now))
        timer.mark("detect")

        # 7. publish ---------------------------------------------------------
        overlay = None
        if self.draw_overlay:
            overlay = encode_jpeg(self._overlay(image, tracks, events),
                                  DASHBOARD.jpeg_quality)
            timer.mark("overlay")

        self._frames += 1
        self._last_ts = now

        return PipelineResult(
            frame_id=frame.frame_id, timestamp=now, source=frame.source,
            simulated=frame.simulated, vials=reports, events=events,
            stage_counts=self._stage_counts(confirmed),
            timings_ms=timer.marks, overlay_jpeg=overlay, warnings=warnings,
        )

    # --------------------------------------------------------------- helpers
    def _report(self, t: Track, features: dict[str, float],
                now: float) -> VialReport:
        return VialReport(
            track_id=t.track_id, cx=t.cx, cy=t.cy, radius=t.radius,
            stage=t.stage, hits=t.hits, missed=t.missed, age_s=t.age_s,
            time_in_stage_s=t.time_in_stage_s(now),
            features={k: float(v) for k, v in features.items()},
        )

    def _stage_counts(self, tracks: list[Track]) -> dict[str, int]:
        counts = {name: 0 for name in self.zones.names}
        for t in tracks:
            if t.stage in counts:
                counts[t.stage] += 1
        return counts

    def _zone_views(self, image: np.ndarray,
                    tracks: list[Track]) -> dict[str, ZoneView]:
        """One ZoneView per zone, with the vials punched out of the bench mask."""
        views: dict[str, ZoneView] = {}
        for name in self.zones.names:
            bounds = self.zones.bounds(name)
            crop, box = roi.zone_crop(image, bounds)
            if crop.size == 0:
                continue
            x0, y0, x1, y1 = box
            zone_mask = self.zones.mask(name)[y0:y1, x0:x1]
            inside = [t for t in tracks
                      if x0 <= t.cx < x1 and y0 <= t.cy < y1]
            bench = roi.exclude_discs(
                zone_mask, [(t.cx, t.cy, t.radius) for t in inside], (x0, y0))
            views[name] = ZoneView(
                name=name, image=crop.copy(), bounds=box, zone_mask=zone_mask,
                bench_mask=bench, track_ids=[t.track_id for t in inside],
            )
        return views

    def _closure_events(self, closed: list[Track], frame_id: int,
                        now: float) -> list[Event]:
        """Events for tracks that ended. Not a detector - a bookkeeping fact.

        "oven" is recorded as info because it is the normal end of a vial's
        visible life. It is an inference from disappearance, not an
        observation: the oven is outside the camera's view. A failure during
        cooling would produce this same event, which is the acknowledged
        blind spot.
        """
        out: list[Event] = []
        for t in closed:
            if t.closed_reason == "oven":
                out.append(Event(
                    kind="oven_entry_inferred", severity="info",
                    message=(f"vial {t.track_id} disappeared after cooling - "
                             "inferred oven entry (not observed)"),
                    timestamp=now, frame_id=frame_id, track_id=t.track_id,
                    detector="tracker", zone=t.stage,
                    data={"last_stage": t.stage, "age_s": round(t.age_s, 1),
                          "inferred": True},
                ))
            else:
                out.append(Event(
                    kind="track_lost", severity="warning",
                    message=(f"vial {t.track_id} disappeared from "
                             f"{t.stage or 'an unstaged position'}"),
                    timestamp=now, frame_id=frame_id, track_id=t.track_id,
                    detector="tracker", zone=t.stage,
                    data={"last_stage": t.stage, "age_s": round(t.age_s, 1),
                          "hits": t.hits},
                ))
        return out

    # --------------------------------------------------------------- overlay
    def _overlay(self, image: np.ndarray, tracks: list[Track],
                 events: list[Event]) -> np.ndarray:
        """Annotated preview: zones, tracked vials, and anything that fired."""
        out = self.zones.draw(image)
        flagged = {e.track_id: e.severity for e in events if e.track_id is not None}

        for t in tracks:
            color = TRACK_BGR if t.confirmed else UNCONFIRMED_BGR
            sev = flagged.get(t.track_id)
            if sev:
                color = SEVERITY_BGR.get(sev, color)
            c = (int(round(t.cx)), int(round(t.cy)))
            r = int(round(t.radius))
            cv2.circle(out, c, r, color, 2 if sev else 1, cv2.LINE_AA)
            label = f"{t.track_id}"
            if t.stage:
                label += f" {t.stage[:4]}"
            cv2.putText(out, label, (c[0] - r, c[1] - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        w = out.shape[1]
        if w > DASHBOARD.preview_width_px:
            scale = DASHBOARD.preview_width_px / w
            out = cv2.resize(out, (DASHBOARD.preview_width_px,
                                   int(out.shape[0] * scale)))
        return out
