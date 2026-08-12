"""Run the thermal camera and stream it over Flask.

    python main.py                  # auto-detect, fall back to the simulator
    python main.py --thermal mock   # force the simulator
    python main.py --port 8080

Open http://<host>:5000/ in a browser. On the Pi, reach it from the laptop at
the Pi's address on the ICS link, e.g. http://192.168.137.x:5000/

The capture loop runs on its own thread rather than inside the request
handler. An MLX90640 frame takes ~500 ms at 2 Hz, so capturing per request
would mean every viewer blocks the sensor and two open tabs would halve the
rate. One capture loop, N viewers reading the newest frame.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

from flask import Flask, Response, jsonify, render_template_string

from config import CADENCE, DASHBOARD
from drivers.base import HardwareUnavailable
from drivers.rgb_cam import encode_jpeg
from drivers.thermal_cam import colorize, create_thermal

log = logging.getLogger("main")

BOUNDARY = "frame"

PAGE = """<!doctype html>
<title>CapEx thermal</title>
<style>
  body { background:#14161a; color:#dfe3ea; margin:0;
         font:14px ui-sans-serif, system-ui, sans-serif; }
  header { padding:12px 18px; border-bottom:1px solid #2c313a; }
  h1 { font-size:16px; margin:0 0 4px; font-weight:600; }
  .sub { color:#868f9e; font-size:12px; }
  main { padding:14px 18px; max-width:680px; }
  img { width:100%; display:block; background:#000; border-radius:8px; }
  .stats { display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; font-size:12px; }
  .stats span { color:#868f9e; }
  .stats b { font-variant-numeric:tabular-nums; }
</style>
<header>
  <h1>CapEx thermal</h1>
  <div class="sub" id="src">starting…</div>
</header>
<main>
  <img src="/thermal_feed" alt="thermal">
  <div class="stats" id="stats"></div>
</main>
<script>
setInterval(async () => {
  const s = await (await fetch('/api/state')).json();
  document.getElementById('src').textContent =
    s.source + (s.simulated ? ' (simulated)' : '') + (s.error ? ' — ' + s.error : '');
  document.getElementById('stats').innerHTML =
    `<div><span>min</span> <b>${s.min_c ?? '—'} °C</b></div>
     <div><span>mean</span> <b>${s.mean_c ?? '—'} °C</b></div>
     <div><span>max</span> <b>${s.max_c ?? '—'} °C</b></div>
     <div><span>frames</span> <b>${s.frames}</b></div>
     <div><span>errors</span> <b>${s.errors}</b></div>`;
}, 1000);
</script>
"""


class ThermalStreamer:
    """Owns the sensor. Captures on a thread, publishes the newest JPEG."""

    def __init__(self, interval_s: float = CADENCE.thermal_interval_s,
                 backend: str | None = None) -> None:
        self.interval_s = interval_s
        self.backend = backend          # None -> whatever config.SOURCES says
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._ts = 0.0
        self.state = {"source": None, "simulated": None, "error": None,
                      "frames": 0, "errors": 0,
                      "min_c": None, "mean_c": None, "max_c": None}
        self._stop = threading.Event()

    # ------------------------------------------------------------- capture
    def start(self) -> None:
        threading.Thread(target=self._loop, name="thermal", daemon=True).start()

    def _loop(self) -> None:
        try:
            source = create_thermal(self.backend)
        except HardwareUnavailable as exc:
            self.state["error"] = str(exc)
            log.error("no thermal source could be started: %s", exc)
            return

        self.state.update(source=source.name, simulated=source.simulated)
        log.info("streaming from %s", source.describe())
        try:
            while not self._stop.is_set():
                began = time.monotonic()
                try:
                    frame = source.capture()
                except Exception:
                    self.state["errors"] += 1
                    log.exception("thermal read failed")
                    self._stop.wait(1.0)
                    continue

                lo, hi, mean = frame.stats
                jpeg = encode_jpeg(colorize(frame.celsius))
                with self._lock:
                    self._jpeg, self._ts = jpeg, time.time()
                self.state.update(frames=self.state["frames"] + 1,
                                  min_c=round(lo, 1), mean_c=round(mean, 1),
                                  max_c=round(hi, 1))

                slack = self.interval_s - (time.monotonic() - began)
                if slack > 0:
                    self._stop.wait(slack)
        finally:
            source.stop()
            log.info("thermal loop stopped")

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- stream
    def mjpeg(self):
        """Yield the newest frame, re-sending it when nothing is new.

        Re-sending keeps the connection alive: at a 2 s cadence a browser
        receiving nothing in between may drop it.
        """
        last_ts = -1.0
        payload: bytes | None = None
        while True:
            with self._lock:
                jpeg, ts = self._jpeg, self._ts
            if jpeg and ts != last_ts:
                payload, last_ts = jpeg, ts
            if payload:
                yield (b"--" + BOUNDARY.encode()
                       + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n")
            time.sleep(min(self.interval_s, 0.5))


def create_app(streamer: ThermalStreamer) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/thermal_feed")
    def thermal_feed():
        return Response(streamer.mjpeg(),
                        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/api/state")
    def api_state():
        return jsonify(streamer.state)

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stream the thermal camera")
    p.add_argument("--thermal", choices=["auto", "mlx90640", "mock"],
                   help="override the thermal backend (default: auto)")
    p.add_argument("--port", type=int, default=DASHBOARD.port)
    p.add_argument("--interval", type=float, default=CADENCE.thermal_interval_s,
                   help="seconds between captures (default: %(default)s)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    # Passed straight to the factory, not via os.environ: config.SOURCES is
    # built when config is first imported, so setting the env var here would
    # be read too late and silently ignored.
    streamer = ThermalStreamer(args.interval, backend=args.thermal)
    streamer.start()

    app = create_app(streamer)
    log.info("thermal stream on http://localhost:%d", args.port)
    try:
        # debug=True spawns a reloader child that would open the sensor twice.
        app.run(host=DASHBOARD.host, port=args.port, debug=False,
                use_reloader=False, threaded=True)
    finally:
        streamer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
