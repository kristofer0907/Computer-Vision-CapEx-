"""Entry point: capture, locate, track, check, serve. One process.

    python main.py                          # auto-detect hardware, simulate the rest
    python main.py --rgb mock               # synthetic platform, no hardware
    python main.py --rgb file --file run/   # replay a folder of stills or a video
    python main.py --no-persist             # do not write to SQLite
    python main.py --no-thermal             # skip the MLX90640

Open http://<host>:5000/. From the laptop over the ICS link that is the Pi's
address on 192.168.137.x.

One thread does all the vision work, on purpose. At a 15-60 s cadence a frame
costs far less than the interval it is followed by, so there is nothing to
parallelise; a queue between stages would only add ways to be wrong. The
thermal logger is the one exception - it keeps its own thread and writes its
own SQLite rows, because an MLX90640 read blocks for ~500 ms and throws
checksum errors routinely, and none of that belongs in the vision loop.

Runs anywhere. On a machine with no Pi camera stack the drivers report the
hardware unavailable and fall back to the simulator rather than failing to
import, which is why picamera2 is imported inside start() and never at module
level. `ModuleNotFoundError: No module named 'libcamera'` means something
imported picamera2 at module scope; that is the bug, not the missing package.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime

import numpy as np

from config import (DASHBOARD, GEOMETRY, REGION_TRACKING, SOURCES, STORAGE,
                    ZONES, ensure_dirs)
from drivers.base import Frame, HardwareUnavailable
from drivers.rgb_cam import create_camera, encode_jpeg
from pipeline import roi
from pipeline.detectors import DetectorHost
from pipeline.detectors.base import DetectionContext, ZoneView
from pipeline.history import CrucibleHistory
from pipeline.lineage import VialLineage
from pipeline.localize import create_localizer
from pipeline.region_trackers import (RegionTracker, SlotTracker,
                                      create_region_coordinator)
from pipeline.tracking import CadenceController
from pipeline.types import AnomalyResult, Event, PipelineResult, CrucibleReport
from pipeline.zones import ZoneMap
from runtime.bus import LatestSlot, RingBuffer
from runtime.messages import PreviewMessage, WorkerStatus
from storage.db import Database
from storage.images import SnapshotStore

log = logging.getLogger("main")


def slot_occupancy(coord, zone: str) -> dict[int, bool]:
    """Which slots of `zone` read full. All pipeline/lineage.py is given."""
    tracker = coord.trackers.get(zone)
    if not isinstance(tracker, SlotTracker):
        return {}
    return {sid: occupant is not None
            for sid, occupant in tracker.slot_status().items()}


def slot_positions(coord, zone: str) -> dict[int, tuple[float, float]]:
    """Slot -> where its crucible was detected, so lineage can tell a
    replaced jar from the same one sitting still."""
    tracker = coord.trackers.get(zone)
    if not isinstance(tracker, SlotTracker):
        return {}
    return {t.slot_id: (t.cx, t.cy) for t in tracker.tracks
            if t.slot_id is not None}


class Monitor:
    """The whole vision pipeline, in one object the dashboard can read from.

    Exposes the slots the dashboard consumes (preview/result/thermal, events,
    status) so a request handler never touches the camera or the pipeline. It
    reads the last thing that was published and returns.
    """

    def __init__(self, rgb_backend: str | None = None,
                 thermal_backend: str | None = None,
                 localizer: str = "auto",
                 enable_thermal: bool = True,
                 persist: bool = True,
                 store_thermal_grid: bool = False,
                 note: str | None = None,
                 draw_overlay: bool = True) -> None:
        self.rgb_backend = rgb_backend
        self.thermal_backend = thermal_backend
        self.localizer_name = localizer
        self.enable_thermal = enable_thermal
        self.persist = persist
        self.store_thermal_grid = store_thermal_grid
        self.note = note
        self.draw_overlay = draw_overlay

        # Read by the dashboard, written here. LatestSlot is last-write-wins:
        # a stale frame is worse than no frame at this cadence.
        self.preview: LatestSlot = LatestSlot()
        self.result: LatestSlot = LatestSlot()
        self.thermal: LatestSlot = LatestSlot()
        self.events: RingBuffer = RingBuffer()
        self.status: dict[str, WorkerStatus] = {}
        self.run_id: int | None = None

        self.camera = None
        self.db: Database | None = None
        self.snapshots = SnapshotStore()
        self.localizer = None
        self.tracker: RegionTracker | None = None
        self.lineage = VialLineage()
        self.detectors = DetectorHost()
        self.cadence = CadenceController()
        self.history = CrucibleHistory()
        self.zones = ZoneMap(GEOMETRY.frame_width_px, GEOMETRY.frame_height_px)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._thermal_q: queue.Queue = queue.Queue(maxsize=4)
        self._started_at = 0.0
        self._frames = 0
        self._interval_s = 0.0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._started_at = time.time()
        self.camera = create_camera(self.rgb_backend)
        self.camera.start()

        # The mock renders a drawing, and the Hough parameters were tuned
        # against photographs, so the mock gets ground truth instead. Keyed on
        # the mock by name, not on `simulated`: the file source also sets that
        # flag and its frames are real captures that Hough must handle.
        name = self.localizer_name
        if name == "auto" and self.camera.name == "mock":
            name = "ground_truth"
            log.info("mock camera - using the ground-truth localiser")
        self.localizer = create_localizer(name)
        self.localizer.start()

        frame_size = (GEOMETRY.frame_width_px, GEOMETRY.frame_height_px)
        self.tracker = RegionTracker(
            create_region_coordinator(frame_size, self.zones))
        self.tracker.start()
        self.detectors.start()

        if self.persist:
            self.db = Database()
            self.run_id = self.db.start_run(
                rgb_source=self.camera.name, note=self.note,
                simulated=self.camera.simulated)

        if self.enable_thermal:
            self._spawn(self._thermal_loop, "thermal")
            self._spawn(self._thermal_drain, "thermal-drain")
        self._spawn(self._vision_loop, "vision")

        log.info("monitor started: camera=%s localiser=%s zones=%s",
                 self.camera.name, self.localizer.name,
                 ", ".join(self.tracker.coordinator.trackers) or "none")

    def _spawn(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        for closer in (self.detectors.stop,
                       getattr(self.localizer, "stop", None),
                       getattr(self.camera, "stop", None)):
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                log.warning("shutdown step failed", exc_info=True)
        if self.db is not None:
            self.db.end_run()
            self.db.close()
        log.info("monitor stopped after %d frames", self._frames)

    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def health(self) -> dict:
        return {
            "run_id": self.run_id,
            "uptime_s": time.time() - self._started_at,
            "persisting": self.persist,
            "analysis_interval_s": self._interval_s,
            "workers": {n: s.__dict__ for n, s in self.status.items()},
        }

    # ------------------------------------------------------------ the loop
    def _vision_loop(self) -> None:
        status = WorkerStatus(worker="vision", source=self.camera.name,
                              simulated=self.camera.simulated)
        self.status["processing"] = status
        while not self._stop.is_set():
            started = time.time()
            try:
                frame = self.camera.capture()
                self._publish_preview(frame)
                result = self.process(frame)
                self.result.set(result)
                status.frames += 1
                status.last_frame_ts = result.timestamp
            except HardwareUnavailable as exc:
                status.errors += 1
                status.error = str(exc)
                log.error("camera gone: %s", exc)
                break
            except Exception as exc:
                status.errors += 1
                status.error = f"{type(exc).__name__}: {exc}"
                log.exception("frame failed - continuing")

            # Sleep the remainder of the interval, not the whole of it, so a
            # slow frame does not push the cadence out.
            self._stop.wait(max(0.0, self._interval_s - (time.time() - started)))
        status.alive = False

    def process(self, frame: Frame) -> PipelineResult:
        """One frame, end to end. The only place the pipeline order lives."""
        now = frame.timestamp
        image = frame.image
        warnings: list[str] = []

        detections = self.localizer.locate(frame)
        tracks, closed = self.tracker.update(detections, now)
        coord = self.tracker.coordinator

        # Lineage runs on slot occupancy alone, at the heater->cooling
        # boundary. It is not folded into the tracker or the detectors: it
        # answers "which crucible is this", they answer "is this wrong".
        self.lineage.update(
            slot_occupancy(coord, REGION_TRACKING.heater_zone),
            slot_occupancy(coord, REGION_TRACKING.cooling_zone),
            now, heater_pos=slot_positions(coord, REGION_TRACKING.heater_zone))

        frame_ref = ""
        if self.persist and self._frames % max(1, STORAGE.snapshot_every_n_frames) == 0:
            frame_ref = self.snapshots.save_frame(image, frame.frame_id, now) or ""

        crops, masks, boxes = {}, {}, {}
        for t in tracks:
            crop, mask, box = roi.crop_with_mask(image, t.cx, t.cy, t.radius)
            if crop.size:
                crops[t.track_id], masks[t.track_id], boxes[t.track_id] = \
                    crop, mask, box

        reports = [CrucibleReport(track_id=t.track_id, cx=t.cx, cy=t.cy,
                              radius=t.radius, stage=t.stage, hits=t.hits,
                              missed=t.missed, age_s=t.age_s,
                              time_in_stage_s=t.time_in_stage_s(now))
                   for t in tracks]

        ctx = DetectionContext(
            frame=frame, timestamp=now, frame_id=frame.frame_id,
            tracks=tracks, reports=reports,
            crops=crops, masks=masks, boxes=boxes,
            zones=self._zone_views(image, tracks, coord),
            zone_map=self.zones, history=self.history,
            closed_tracks=closed, frame_ref=frame_ref,
            interval_s=self._interval_s,
        )

        results, detector_warnings = self.detectors.run(ctx)
        warnings.extend(detector_warnings)
        self._record(results, frame.frame_id)

        # A latched hazard halts the capture loop. The dashboard keeps serving,
        # because the frame that stopped the run is the thing to look at. This
        # stops our analysis only - nothing here can stop the platform.
        if any(getattr(d, "stop_requested", False) for d in self.detectors.detectors):
            if not self._stop.is_set():
                log.error("stop latched - halting capture. Dashboard stays up.")
                warnings.append("run stopped by a latched hazard")
            self._stop.set()

        self._interval_s = self.cadence.observe(tracks)
        self._frames += 1

        overlay = None
        if self.draw_overlay:
            overlay = encode_jpeg(self._overlay(image, tracks, coord, results),
                                  DASHBOARD.jpeg_quality)

        return PipelineResult(
            frame_id=frame.frame_id, timestamp=now, source=frame.source,
            simulated=frame.simulated, crucibles=reports,
            events=[_as_event(r, frame.frame_id) for r in results],
            stage_counts=self._stage_counts(tracks),
            overlay_jpeg=overlay, warnings=warnings,
        )

    def _record(self, results: list[AnomalyResult], frame_id: int) -> None:
        """Tripped checks go to SQLite and to the dashboard's event ring."""
        for r in results:
            if not getattr(r, "tripped", False):
                continue
            self.events.extend([_as_event(r, frame_id)])
            if self.db is not None:
                self.db.write_event(r)
            log.error("ANOMALY %s in %s (frame %s)",
                      r.failure_type, r.zone, r.frame_ref or frame_id)

    # --------------------------------------------------------------- helpers
    def _zone_views(self, image: np.ndarray, tracks, coord
                    ) -> dict[str, ZoneView]:
        """One ZoneView per tracked zone, crucibles punched out of the bench."""
        views: dict[str, ZoneView] = {}
        for name in coord.trackers:
            if name not in self.zones.names:
                continue
            crop, box = roi.zone_crop(image, self.zones.bounds(name))
            if crop.size == 0:
                continue
            x0, y0, x1, y1 = box
            zone_mask = self.zones.mask(name)[y0:y1, x0:x1]
            inside = [t for t in coord.trackers[name].tracks]
            views[name] = ZoneView(
                name=name, image=crop.copy(), bounds=box, zone_mask=zone_mask,
                bench_mask=roi.exclude_discs(
                    zone_mask, [(t.cx, t.cy, t.radius) for t in inside],
                    (x0, y0)),
                track_ids=[t.track_id for t in inside])
        return views

    def _stage_counts(self, tracks) -> dict[str, int]:
        counts = {name: 0 for name in REGION_TRACKING.region_sequence}
        for t in tracks:
            if t.stage in counts:
                counts[t.stage] += 1
        return counts

    def _overlay(self, image: np.ndarray, tracks, coord,
                 results: list[AnomalyResult]) -> np.ndarray:
        import cv2

        out = self.zones.draw(image)
        tripped = {r.zone for r in results if r.tripped}
        for t in tracks:
            zone = getattr(t, "stage", None)
            color = (0, 0, 255) if zone in tripped else (0, 200, 0)
            c = (int(round(t.cx)), int(round(t.cy)))
            cv2.circle(out, c, int(round(t.radius)), color, 2, cv2.LINE_AA)
            label = str(t.track_id)
            if t.slot_id is not None:
                label += f"@{t.slot_id}"
            cv2.putText(out, label, (c[0] - int(t.radius), c[1] - int(t.radius) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        for zone in sorted(tripped):
            cv2.putText(out, f"ANOMALY: {zone}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        cv2.LINE_AA)

        w = out.shape[1]
        if w > DASHBOARD.preview_width_px:
            scale = DASHBOARD.preview_width_px / w
            out = cv2.resize(out, (DASHBOARD.preview_width_px,
                                   int(out.shape[0] * scale)))
        return out

    def _publish_preview(self, frame: Frame) -> None:
        self.preview.set(PreviewMessage(
            jpeg=encode_jpeg(frame.image, DASHBOARD.jpeg_quality),
            timestamp=frame.timestamp, frame_id=frame.frame_id,
            source=frame.source, simulated=frame.simulated))

    # --------------------------------------------------------------- thermal
    def _thermal_loop(self) -> None:
        """The thermal logger, unchanged, driven on a thread instead of a
        process. It writes its own SQLite rows and nothing here reads them."""
        from runtime.thermal import thermal_worker

        status_q: queue.Queue = queue.Queue(maxsize=8)
        try:
            thermal_worker(self._thermal_q, status_q, self._stop,
                           run_id=self.run_id, backend=self.thermal_backend,
                           store_grid=self.store_thermal_grid)
        except Exception:
            log.warning("thermal logger stopped", exc_info=True)

    def _thermal_drain(self) -> None:
        """Move thermal frames into the slot the dashboard reads."""
        while not self._stop.is_set():
            try:
                self.thermal.set(self._thermal_q.get(timeout=0.5))
            except queue.Empty:
                continue


def _as_event(result: AnomalyResult, frame_id: int) -> Event:
    """AnomalyResult -> Event, for the dashboard's event list.

    SQLite gets the AnomalyResult; the dashboard's list, its severity colours
    and its API all predate it and speak Event.
    """
    return Event(
        kind=result.failure_type,
        severity="alert" if result.tripped else "info",
        message=f"{result.failure_type} in {result.zone}",
        timestamp=result.timestamp.timestamp(),
        frame_id=frame_id,
        detector=result.failure_type,
        zone=result.zone,
        data={"frame_ref": result.frame_ref, "tripped": result.tripped},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CapEx synthesis monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_argument_group("sources")
    src.add_argument("--rgb", choices=["auto", "picamera2", "mock", "file"],
                     default=None, help="RGB backend (default: config / auto)")
    src.add_argument("--thermal", choices=["auto", "mlx90640", "mock"],
                     default=None, help="thermal backend")
    src.add_argument("--file", default=None,
                     help="video or stills directory for --rgb file")
    src.add_argument("--no-thermal", action="store_true",
                     help="do not start the thermal logger")

    pipe = p.add_argument_group("pipeline")
    pipe.add_argument("--localizer", default="auto",
                      help="localiser to use; see pipeline/localize.py")
    pipe.add_argument("--no-overlay", action="store_true",
                      help="skip drawing the annotated preview")

    store = p.add_argument_group("storage")
    store.add_argument("--no-persist", action="store_true",
                       help="do not write to SQLite")
    store.add_argument("--store-thermal-grid", action="store_true",
                       help="also store the raw 24x32 field (~130 MB/day)")
    store.add_argument("--note", default=None,
                       help="free-text note recorded against this run")

    p.add_argument("--host", default=DASHBOARD.host)
    p.add_argument("--port", type=int, default=DASHBOARD.port)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    ensure_dirs()

    if args.file:
        os.environ["CAPEX_RGB_FILE"] = args.file
        if args.rgb is None:
            args.rgb = "file"
    if args.rgb == "file" and not (args.file or SOURCES.rgb_file_path):
        log.error("--rgb file needs --file <video or directory>")
        return 2

    if not ZONES.calibrated:
        log.warning("zone polygons are placeholders - trace the real ones with "
                    "`python -m tools.edit_zones`; every stage assignment "
                    "until then is against made-up geometry")

    from dashboard.app import create_app

    monitor = Monitor(
        rgb_backend=args.rgb,
        thermal_backend=args.thermal,
        localizer=args.localizer,
        enable_thermal=not args.no_thermal,
        persist=not args.no_persist,
        store_thermal_grid=args.store_thermal_grid,
        note=args.note,
        draw_overlay=not args.no_overlay,
    )

    def handle_signal(signum, _frame):
        # Flask's dev server installs no signal handler, so without this a
        # Ctrl-C leaves the capture thread holding the camera and the next
        # start fails with a device-busy error.
        log.info("signal %s received, shutting down", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    monitor.start()
    app = create_app(monitor)

    log.info("dashboard on http://%s:%d", args.host, args.port)
    try:
        # debug=True spawns a reloader child, which would start a second
        # capture thread and open the camera twice. Never enable it here.
        app.run(host=args.host, port=args.port, debug=False,
                use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
