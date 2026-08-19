"""Thermal process: polls the MLX90640, logs it, publishes it for display.

Scope, and it is narrow on purpose: this is a passive data logger for the
researchers to look at. No model, no algorithm and no part of the anomaly
logic runs on thermal data, and nothing in pipeline/ receives it. That is a
settled decision, not an omission.

The resolution is the reason. 32x24 pixels over a 110 degree field of view at
800 mm is about 2.4 cm per pixel, so a vial covers one or two pixels. There is
nothing there to run anything on. The planned fix is to remount the sensor low
over the heater pad alone - about 1.3 cm/px at 150 mm - before spending
anything on a Lepton, and even then it stays a viewing aid.

It runs as its own process rather than a thread in the capture process because
a frame read takes ~500 ms at 2 Hz and the part throws checksum errors
routinely. Isolated, a bad I2C day costs thermal frames and nothing else.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time

from config import CADENCE
from drivers.base import HardwareUnavailable
from drivers.rgb_cam import encode_jpeg
from drivers.thermal_cam import colorize, create_thermal
from runtime.bus import put_drop_oldest
from runtime.messages import ThermalMessage, WorkerStatus

log = logging.getLogger(__name__)


def thermal_worker(thermal_q: mp.Queue, status_q: mp.Queue, stop: mp.Event,
                   run_id: int | None = None, backend: str | None = None,
                   interval_s: float | None = None,
                   store_grid: bool = False) -> None:
    """Entry point for the thermal process. Returns when `stop` is set."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from storage.db import Database   # opened here: connections are per-process

    interval = CADENCE.thermal_interval_s if interval_s is None else interval_s
    status = WorkerStatus(worker="thermal")

    try:
        source = create_thermal(backend)
    except HardwareUnavailable as exc:
        status.alive = False
        status.error = str(exc)
        put_drop_oldest(status_q, status)
        log.error("thermal: no source at all: %s", exc)
        return

    status.source = source.name
    status.simulated = source.simulated
    log.info("thermal: running on %s", source.describe())

    db = None
    if run_id is not None:
        try:
            db = Database()
            db.attach_run(run_id)
        except Exception:
            log.exception("thermal: could not open the database - "
                          "continuing without logging")

    last_status = 0.0
    try:
        while not stop.is_set():
            began = time.monotonic()
            try:
                frame = source.capture()
            except Exception as exc:
                status.errors += 1
                status.error = f"{type(exc).__name__}: {exc}"
                # Checksum and I2C hiccups are routine on this part; the
                # driver already retries internally, so reaching here means
                # several consecutive failures.
                log.warning("thermal: read failed: %s", exc)
                stop.wait(1.0)
                continue

            lo, hi, mean = frame.stats
            status.frames += 1
            status.last_frame_ts = frame.timestamp
            status.error = None

            if not put_drop_oldest(thermal_q, ThermalMessage(
                    jpeg=encode_jpeg(colorize(frame.celsius)),
                    timestamp=frame.timestamp, frame_id=frame.frame_id,
                    source=source.name, simulated=source.simulated,
                    min_c=round(lo, 2), mean_c=round(mean, 2),
                    max_c=round(hi, 2))):
                status.dropped += 1

            if db is not None:
                try:
                    db.log_thermal(
                        frame.timestamp, frame.frame_id, lo, mean, hi,
                        simulated=source.simulated,
                        # The raw field is 3 kB a sample - ~130 MB a day at
                        # the 2 s poll. Off unless a researcher wants the
                        # field back for offline review.
                        grid=(frame.celsius.astype("float32").tobytes()
                              if store_grid else None))
                except Exception:
                    status.errors += 1
                    log.exception("thermal: database write failed")

            now = time.monotonic()
            if now - last_status > 2.0:
                put_drop_oldest(status_q, status.copy())
                last_status = now

            slack = interval - (time.monotonic() - began)
            if slack > 0:
                stop.wait(slack)
    finally:
        source.stop()
        if db is not None:
            db.close()
        status.alive = False
        put_drop_oldest(status_q, status.copy())
        log.info("thermal: stopped after %d frames (%d errors)",
                 status.frames, status.errors)
