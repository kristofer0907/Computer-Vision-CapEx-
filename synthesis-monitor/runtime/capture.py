"""Capture process: owns the RGB camera, feeds the preview and the pipeline.

One camera, two consumers running at very different rates:

    preview   ~5 fps, JPEG encoded here, for the dashboard
    analysis  one frame per 10-60 s, raw pixels, for the pipeline

Both come from the same capture loop. The loop runs at the preview rate and
simply forwards whichever frame it happens to be holding when the analysis
interval elapses. Capturing separately for analysis would mean two opens of
the sensor, or a second capture that blocks the preview for 100 ms every time.

The analysis interval is not fixed: the processing process publishes what it
wants next into a shared double, so the cadence can drop to ~10 s while the
conveyor is moving and rise again when the platform is idle. A shared value
rather than a control queue because it is one number, always the newest one
matters, and nothing needs to be delivered exactly once.

JPEG encoding happens here rather than in the dashboard for a boring but real
reason: this process is idle between captures, and the dashboard is not.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time

import cv2

from config import CADENCE, DASHBOARD
from drivers.base import HardwareUnavailable
from drivers.rgb_cam import create_camera, encode_jpeg
from runtime.bus import put_drop_oldest
from runtime.messages import PreviewMessage, WorkerStatus

log = logging.getLogger(__name__)


def _shrink(image, width: int = DASHBOARD.preview_width_px):
    h, w = image.shape[:2]
    if w <= width:
        return image
    return cv2.resize(image, (width, int(h * width / w)),
                      interpolation=cv2.INTER_AREA)


def capture_worker(preview_q: mp.Queue, analysis_q: mp.Queue,
                   status_q: mp.Queue, stop: mp.Event,
                   analysis_interval: mp.Value, backend: str | None = None,
                   preview_interval_s: float | None = None) -> None:
    """Entry point for the capture process. Returns when `stop` is set."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    interval = (CADENCE.preview_interval_s if preview_interval_s is None
                else preview_interval_s)
    status = WorkerStatus(worker="capture")

    try:
        camera = create_camera(backend)
    except HardwareUnavailable as exc:
        # create_camera falls back to the simulator, so reaching here means
        # even that failed - which is a bug, not a missing Pi.
        status.alive = False
        status.error = str(exc)
        put_drop_oldest(status_q, status)
        log.error("capture: no camera at all: %s", exc)
        return

    status.source = camera.name
    status.simulated = camera.simulated
    log.info("capture: running on %s", camera.describe())

    next_analysis = time.monotonic()     # forward the very first frame
    last_status = 0.0

    try:
        while not stop.is_set():
            began = time.monotonic()
            try:
                frame = camera.capture()
            except StopIteration:
                log.info("capture: source exhausted")
                break
            except Exception as exc:
                status.errors += 1
                status.error = f"{type(exc).__name__}: {exc}"
                log.exception("capture: capture() failed")
                stop.wait(1.0)
                continue

            status.frames += 1
            status.last_frame_ts = frame.timestamp
            status.error = None

            if not put_drop_oldest(preview_q, PreviewMessage(
                    jpeg=encode_jpeg(_shrink(frame.image), DASHBOARD.jpeg_quality),
                    timestamp=frame.timestamp, frame_id=frame.frame_id,
                    source=camera.name, simulated=camera.simulated)):
                status.dropped += 1

            now = time.monotonic()
            if now >= next_analysis:
                # Full-resolution frame, unencoded. The pipeline needs pixels.
                if not put_drop_oldest(analysis_q, frame):
                    status.dropped += 1
                    # A drop here means the pipeline is slower than the
                    # cadence it asked for. Worth a log line, not a crash.
                    log.warning("capture: analysis queue full, frame %d dropped",
                                frame.frame_id)
                next_analysis = now + max(1.0, float(analysis_interval.value))

            if now - last_status > 2.0:
                put_drop_oldest(status_q, status.copy())
                last_status = now

            slack = interval - (time.monotonic() - began)
            if slack > 0:
                stop.wait(slack)
    finally:
        camera.stop()
        status.alive = False
        put_drop_oldest(status_q, status.copy())
        log.info("capture: stopped after %d frames (%d errors, %d dropped)",
                 status.frames, status.errors, status.dropped)
