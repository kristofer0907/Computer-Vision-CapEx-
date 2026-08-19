"""Starts the worker processes, drains their queues, shuts them down cleanly.

Process layout:

    capture      owns the camera        -> preview_q (JPEG), analysis_q (raw)
    processing   owns the pipeline + DB -> result_q
    thermal      owns the MLX90640 + DB -> thermal_q
    main         owns Flask, drains every queue into LatestSlots

The Flask app runs in the main process and never touches a device. Everything
it renders arrives over a queue, which is what makes `debug=True`'s reloader a
correctness problem rather than a style one: a second interpreter would fork a
second set of workers and open the camera twice.

Start method is forced to "spawn". The default on Linux is fork, and forking a
process that has already imported picamera2 or opened an I2C bus gives the
child a copy of a file descriptor it does not own - which manifests as the
camera working for a while and then failing in ways that look like a hardware
fault. spawn costs about a second per process at start-up and buys not having
to debug that.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time

from config import CADENCE, DASHBOARD, QUEUES, ensure_dirs
from runtime.bus import LatestSlot, RingBuffer, drain_all, drain_latest
from runtime.capture import capture_worker
from runtime.processing import processing_worker
from runtime.thermal import thermal_worker

log = logging.getLogger(__name__)


class Supervisor:
    """Owns the workers and the shared state the dashboard reads."""

    def __init__(self, rgb_backend: str | None = None,
                 thermal_backend: str | None = None,
                 localizer: str = "auto", extractor: str = "auto",
                 enable_thermal: bool = True, enable_pipeline: bool = True,
                 persist: bool = True, store_thermal_grid: bool = False,
                 note: str | None = None) -> None:
        self.rgb_backend = rgb_backend
        self.thermal_backend = thermal_backend
        self.localizer = localizer
        self.extractor = extractor
        self.enable_thermal = enable_thermal
        self.enable_pipeline = enable_pipeline
        self.persist = persist
        self.store_thermal_grid = store_thermal_grid
        self.note = note

        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.analysis_interval = self.ctx.Value("d", CADENCE.analysis_interval_s)

        self.preview_q = self.ctx.Queue(maxsize=QUEUES.preview_depth)
        self.analysis_q = self.ctx.Queue(maxsize=QUEUES.analysis_depth)
        self.result_q = self.ctx.Queue(maxsize=QUEUES.result_depth)
        self.thermal_q = self.ctx.Queue(maxsize=QUEUES.thermal_depth)
        self.status_q = self.ctx.Queue(maxsize=64)

        # What the dashboard reads. All written by the drain thread.
        self.preview = LatestSlot()
        self.thermal = LatestSlot()
        self.result = LatestSlot()
        self.events = RingBuffer(capacity=DASHBOARD.events_page_size * 4)
        self.status: dict[str, object] = {}

        self.run_id: int | None = None
        self._db = None
        self._procs: dict[str, mp.Process] = {}
        self._drain: threading.Thread | None = None
        self._started_at = 0.0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        ensure_dirs()
        self._started_at = time.time()

        if self.persist:
            self._open_run()

        self._procs["capture"] = self.ctx.Process(
            target=capture_worker, name="capture",
            args=(self.preview_q, self.analysis_q, self.status_q,
                  self.stop_event, self.analysis_interval, self.rgb_backend),
            daemon=True)

        if self.enable_pipeline:
            self._procs["processing"] = self.ctx.Process(
                target=processing_worker, name="processing",
                args=(self.analysis_q, self.result_q, self.status_q,
                      self.stop_event, self.analysis_interval, self.run_id,
                      self.localizer, self.extractor),
                daemon=True)

        if self.enable_thermal:
            self._procs["thermal"] = self.ctx.Process(
                target=thermal_worker, name="thermal",
                args=(self.thermal_q, self.status_q, self.stop_event,
                      self.run_id, self.thermal_backend, None,
                      self.store_thermal_grid),
                daemon=True)

        for name, proc in self._procs.items():
            proc.start()
            log.info("started %s (pid %d)", name, proc.pid)

        self._drain = threading.Thread(target=self._drain_loop, name="drain",
                                       daemon=True)
        self._drain.start()

    def _open_run(self) -> None:
        from storage.db import Database
        try:
            self._db = Database()
            self.run_id = self._db.start_run(
                rgb_source=self.rgb_backend or "auto",
                thermal_source=(self.thermal_backend or "auto"
                                if self.enable_thermal else None),
                note=self.note)
            self._db.prune()
        except Exception:
            log.exception("could not open the database - running without "
                          "persistence")
            self._db = None
            self.run_id = None

    def stop(self, timeout_s: float | None = None) -> None:
        """Signal every worker, drain the queues, then join.

        Order matters. A multiprocessing.Queue's feeder thread blocks a child
        from exiting while items it wrote are still unread, so joining before
        draining deadlocks until the timeout. Drain first, then join, then
        terminate whatever is still alive.
        """
        timeout = QUEUES.join_timeout_s if timeout_s is None else timeout_s
        self.stop_event.set()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(
                p.is_alive() for p in self._procs.values()):
            self._drain_once()
            time.sleep(0.05)

        for name, proc in self._procs.items():
            proc.join(timeout=max(0.1, deadline - time.monotonic()))
            if proc.is_alive():
                log.warning("%s did not exit - terminating", name)
                proc.terminate()
                proc.join(timeout=2.0)

        for q in (self.preview_q, self.analysis_q, self.result_q,
                  self.thermal_q, self.status_q):
            q.close()
            q.cancel_join_thread()

        if self._db is not None:
            self._db.end_run()
            self._db.close()
            self._db = None
        log.info("supervisor stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ----------------------------------------------------------------- drain
    def _drain_loop(self) -> None:
        while not self.stop_event.is_set():
            self._drain_once()
            time.sleep(0.05)

    def _drain_once(self) -> None:
        """Move everything waiting on the queues into the shared slots.

        Preview and thermal take only the newest item - an old frame has no
        value. Results and statuses take everything, because every result
        carries events that must not be skipped.
        """
        preview = drain_latest(self.preview_q)
        if preview is not None:
            self.preview.set(preview)

        thermal = drain_latest(self.thermal_q)
        if thermal is not None:
            self.thermal.set(thermal)

        for result in drain_all(self.result_q):
            self.result.set(result)
            if result.events:
                self.events.extend(result.events)

        for status in drain_all(self.status_q):
            self.status[status.worker] = status

    # ---------------------------------------------------------------- health
    def health(self) -> dict:
        """Everything the dashboard needs to say what is and is not working."""
        workers = {}
        for name, proc in self._procs.items():
            status = self.status.get(name)
            workers[name] = {
                "pid": proc.pid,
                "alive": proc.is_alive(),
                "exitcode": proc.exitcode,
                "status": status.__dict__ if status else None,
            }
        return {
            "run_id": self.run_id,
            "uptime_s": round(time.time() - self._started_at, 1)
            if self._started_at else 0.0,
            "persisting": self._db is not None,
            "analysis_interval_s": round(float(self.analysis_interval.value), 1),
            "workers": workers,
        }

    def alive(self) -> bool:
        """True while every worker that was started is still running.

        Used by /healthz. A worker that died leaves the dashboard serving a
        frozen frame with no other outward sign, which is exactly the failure
        a health check exists to catch.
        """
        return all(p.is_alive() for p in self._procs.values())
