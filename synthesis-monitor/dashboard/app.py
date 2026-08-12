"""Flask dashboard: live RGB and thermal views.

Self-contained. It opens the sources through the driver factories, polls each
one on its own background thread, and serves the newest frame as MJPEG. No
analysis, no persistence — those are yours to build.

    python -m dashboard.app          # from the project root

The polling threads exist because the two sources have very different natural
rates (an IMX477 still is ~100 ms, an MLX90640 frame is ~500 ms at 2 Hz) and a
blocking capture() must never sit inside a request handler. Threads are enough
here: capture() releases the GIL while it waits on the device.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from config import CADENCE, DASHBOARD, GEOMETRY
from drivers.base import HardwareUnavailable
from drivers.rgb_cam import create_camera, encode_jpeg
from drivers.thermal_cam import colorize, create_thermal

log = logging.getLogger(__name__)

BOUNDARY = "frame"
PREVIEW_WIDTH = 960          # downscaled for the browser


class LatestFrame:
    """One slot, last value wins.

    Deliberately not a queue: a viewer that falls behind wants the newest
    frame, not a backlog of stale ones.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._ts: float = 0.0
        self._meta: dict = {}

    def publish(self, jpeg: bytes, **meta) -> None:
        with self._lock:
            self._jpeg, self._ts, self._meta = jpeg, time.time(), meta

    def get(self) -> tuple[bytes | None, float, dict]:
        with self._lock:
            return self._jpeg, self._ts, dict(self._meta)


class SourceRunner:
    """Polls one source on a thread and publishes the newest frame."""

    def __init__(self, name: str, interval_s: float) -> None:
        self.name = name
        self.interval_s = interval_s
        self.latest = LatestFrame()
        self.info: dict = {"name": None, "simulated": None, "error": None}
        self.counters = {"frames": 0, "errors": 0}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, open_source, render) -> None:
        """open_source() -> started source; render(frame) -> BGR image."""
        def loop() -> None:
            try:
                source = open_source()
            except HardwareUnavailable as exc:
                self.info["error"] = str(exc)
                log.error("%s: no source could be started: %s", self.name, exc)
                return

            self.info.update(name=source.name, simulated=source.simulated)
            log.info("%s running on %s", self.name, source.describe())
            try:
                while not self._stop.is_set():
                    began = time.monotonic()
                    try:
                        frame = source.capture()
                    except StopIteration:
                        log.info("%s: source exhausted", self.name)
                        break
                    except Exception:
                        self.counters["errors"] += 1
                        log.exception("%s: capture failed", self.name)
                        self._stop.wait(1.0)
                        continue

                    self.counters["frames"] += 1
                    self.latest.publish(encode_jpeg(render(frame)),
                                        frame_id=frame.frame_id,
                                        timestamp=frame.timestamp)
                    slack = self.interval_s - (time.monotonic() - began)
                    if slack > 0:
                        self._stop.wait(slack)
            finally:
                source.stop()
                log.info("%s stopped", self.name)

        self._thread = threading.Thread(target=loop, name=self.name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def _placeholder(text: str, size=(480, 270)) -> bytes:
    img = np.full((size[1], size[0], 3), 24, np.uint8)
    cv2.putText(img, text, (14, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _mjpeg(runner: SourceRunner, label: str):
    """Yield the newest JPEG, re-sending it when nothing is new.

    Re-sending matters for the thermal feed: at a 2 s cadence a browser that
    receives nothing in between can drop the connection.
    """
    interval = max(runner.interval_s, 0.1)
    last_ts = -1.0
    payload = _placeholder(f"waiting for {label}...")
    while True:
        jpeg, ts, _ = runner.latest.get()
        if jpeg and ts != last_ts:
            payload, last_ts = jpeg, ts
        yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
               + payload + b"\r\n")
        time.sleep(interval)


def _shrink(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= PREVIEW_WIDTH:
        return image
    scale = PREVIEW_WIDTH / w
    return cv2.resize(image, (PREVIEW_WIDTH, int(h * scale)))


def create_app(start_sources: bool = True) -> Flask:
    app = Flask(__name__)

    rgb = SourceRunner("rgb", CADENCE.preview_interval_s)
    thermal = SourceRunner("thermal", CADENCE.thermal_interval_s)
    app.config["RUNNERS"] = {"rgb": rgb, "thermal": thermal}

    if start_sources:
        rgb.start(create_camera, lambda f: _shrink(f.image))
        thermal.start(create_thermal, lambda f: colorize(f.celsius))

    @app.route("/")
    def index():
        return render_template("index.html", n_vials=GEOMETRY.n_vials,
                               refresh_ms=2000)

    @app.route("/rgb_feed")
    def rgb_feed():
        return Response(_mjpeg(rgb, "camera"),
                        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/thermal_feed")
    def thermal_feed():
        return Response(_mjpeg(thermal, "thermal"),
                        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/api/state")
    def api_state():
        out = {"server_time": time.time(), "sources": {}}
        for key, runner in app.config["RUNNERS"].items():
            _, ts, meta = runner.latest.get()
            out["sources"][key] = {
                **runner.info,
                "counters": dict(runner.counters),
                "last_frame_ts": ts or None,
                "last_frame_age_s": round(time.time() - ts, 1) if ts else None,
                "frame_id": meta.get("frame_id"),
            }
        return jsonify(out)

    @app.route("/healthz")
    def healthz():
        _, ts, _ = rgb.latest.get()
        healthy = bool(ts) and (time.time() - ts) < 30.0
        return jsonify({"ok": healthy}), (200 if healthy else 503)

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    app = create_app()
    log.info("dashboard on http://localhost:%d", DASHBOARD.port)
    # debug=True spawns a reloader child that would open the camera twice.
    app.run(host=DASHBOARD.host, port=DASHBOARD.port, debug=False,
            use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
