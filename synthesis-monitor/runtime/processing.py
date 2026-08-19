"""Processing process: runs the pipeline, persists the result, publishes it.

Separate from capture for one reason that matters and one that will: the
pipeline's cost is unbounded (a YOLO forward pass, if that route wins, is
seconds on a Pi 5 CPU) and the preview must stay smooth regardless. Running
them in one process would mean every analysis frame visibly stalls the live
view, and on a shared GIL even threads would not fix it.

It also owns all pipeline writes to SQLite. One writer for pipeline data, one
for thermal, and WAL to let them coexist.

The cadence loop closes here: after each frame the runner reports what
interval it wants next, and that number is written into the shared value the
capture process reads. Fast while the conveyor moves, slow while it does not.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time

from config import STORAGE
from pipeline.runner import PipelineRunner
from pipeline.types import severity_rank
from runtime.bus import put_drop_oldest
from runtime.messages import WorkerStatus
from storage.images import SnapshotStore

log = logging.getLogger(__name__)


def processing_worker(analysis_q: mp.Queue, result_q: mp.Queue,
                      status_q: mp.Queue, stop: mp.Event,
                      analysis_interval: mp.Value, run_id: int | None = None,
                      localizer: str = "auto", extractor: str = "auto",
                      draw_overlay: bool = True) -> None:
    """Entry point for the processing process. Returns when `stop` is set."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from pipeline.features import create_extractor
    from pipeline.localize import create_localizer
    from storage.db import Database   # per-process connection

    status = WorkerStatus(worker="processing", source="pipeline")

    try:
        runner = PipelineRunner(
            localizer=create_localizer(localizer),
            extractor=create_extractor(extractor),
            draw_overlay=draw_overlay,
        )
        runner.start()
    except Exception as exc:
        status.alive = False
        status.error = f"{type(exc).__name__}: {exc}"
        put_drop_oldest(status_q, status)
        log.exception("processing: pipeline failed to start")
        return

    status.extra["localizer"] = runner.localizer.name
    status.extra["extractor"] = runner.extractor.name
    status.extra["detectors"] = runner.detectors.health()

    db = None
    snapshots = None
    if run_id is not None:
        try:
            db = Database()
            db.attach_run(run_id)
            snapshots = SnapshotStore()
        except Exception:
            log.exception("processing: storage unavailable - "
                          "continuing without persistence")
            db = None

    processed = 0
    last_status = 0.0

    try:
        while not stop.is_set():
            try:
                # Blocking with a timeout, not a spin: the next frame is up to
                # a minute away and polling for it would burn a core.
                frame = analysis_q.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - last_status > 2.0:
                    put_drop_oldest(status_q, status.copy())
                    last_status = time.monotonic()
                continue

            try:
                result = runner.process(frame)
            except Exception as exc:
                status.errors += 1
                status.error = f"{type(exc).__name__}: {exc}"
                # The pipeline raising is a bug, not a condition. Keep the
                # process alive so the dashboard keeps working and the error
                # is visible rather than the whole panel going dark.
                log.exception("processing: pipeline raised on frame %d",
                              frame.frame_id)
                continue

            processed += 1
            status.frames += 1
            status.last_frame_ts = result.timestamp
            status.error = None
            status.extra["busy"] = runner.busy
            status.extra["interval_s"] = runner.next_interval_s
            status.extra["detectors"] = runner.detectors.health()
            status.extra["timings_ms"] = result.timings_ms

            # Close the cadence loop before storage: a slow disk should not
            # also delay the next capture.
            analysis_interval.value = float(runner.next_interval_s)

            if db is not None:
                _persist(db, snapshots, runner, result, frame, processed, status)

            if not put_drop_oldest(result_q, result):
                status.dropped += 1

            now = time.monotonic()
            if now - last_status > 2.0:
                put_drop_oldest(status_q, status.copy())
                last_status = now
    finally:
        runner.stop()
        if db is not None:
            db.close()
        status.alive = False
        put_drop_oldest(status_q, status.copy())
        log.info("processing: stopped after %d frames (%d errors)",
                 status.frames, status.errors)


def _persist(db, snapshots, runner, result, frame, processed: int,
             status: WorkerStatus) -> None:
    """Write one frame's results, plus any images worth keeping.

    Storage failures are logged and counted but never propagated. Losing a row
    is bad; losing the live monitor because the SD card filled up is worse.
    """
    snapshot_path = None
    try:
        every = STORAGE.snapshot_every_n_frames
        if snapshots is not None and every and processed % every == 0:
            if result.overlay_jpeg:
                # Already encoded by the runner's overlay pass - no reason to
                # decode and re-encode the same pixels.
                snapshot_path = snapshots.save_frame_jpeg(
                    result.overlay_jpeg, frame.frame_id, result.timestamp)
            else:
                snapshot_path = snapshots.save_frame(
                    frame.image, frame.frame_id, result.timestamp)
    except Exception:
        log.exception("processing: snapshot failed")

    crop_paths: dict[int, str] = {}
    try:
        if (snapshots is not None and STORAGE.save_crop_on_event
                and result.events):
            crop_paths = _save_event_crops(snapshots, runner, result, frame)
    except Exception:
        log.exception("processing: event crop save failed")

    try:
        db.log_result(result, snapshot_path)
        if crop_paths:
            # log_result already stored the events; the crop paths are
            # attached afterwards rather than threaded through, so a failing
            # image write can never roll back the event row itself.
            _attach_crops(db, result, crop_paths)
    except Exception:
        status.errors += 1
        log.exception("processing: database write failed")


def _save_event_crops(snapshots, runner, result, frame) -> dict[int, str]:
    """One close-up per flagged vial: the picture the event was raised on.

    Only for warning and above - info events include every normal oven entry,
    and writing an image for each of those would be one file per vial per run
    for no benefit.
    """
    from pipeline import roi

    paths: dict[int, str] = {}
    for event in result.events:
        if event.track_id is None or event.track_id in paths:
            continue
        if severity_rank(event.severity) < severity_rank("warning"):
            continue
        vial = next((v for v in result.vials if v.track_id == event.track_id), None)
        if vial is None:
            continue
        crop, _ = roi.crop(frame.image, vial.cx, vial.cy, vial.radius)
        if crop.size == 0:
            continue
        path = snapshots.save_crop(crop, frame.frame_id, event.track_id,
                                   event.kind, result.timestamp)
        if path:
            paths[event.track_id] = path
    return paths


def _attach_crops(db, result, crop_paths: dict[int, str]) -> None:
    """Point the just-written event rows at their crops."""
    rows = db.conn.execute(
        "SELECT id, track_id FROM events WHERE run_id=? AND frame_id=? "
        "AND track_id IS NOT NULL",
        (db.run_id, result.frame_id)).fetchall()
    for row in rows:
        path = crop_paths.get(row["track_id"])
        if path:
            db.set_event_crop(int(row["id"]), path)
