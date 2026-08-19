"""Flask dashboard.

Reads only. It owns no camera, no sensor and no pipeline - everything it
renders arrives from the worker processes through the Supervisor's slots, and
history comes from SQLite. That separation is what lets a request handler be
slow, or a browser tab be left open for a week, without any of it reaching the
hardware.

    python -m dashboard.app          # standalone, starts its own supervisor
    python main.py                   # normal entry point

Two things are load-bearing and easy to break:

  * debug=True must stay off. Flask's reloader forks a second interpreter,
    which would start a second set of worker processes and open the camera
    twice. Not a style preference.
  * MJPEG generators must never block on a device. They read a slot that some
    other thread fills, and re-send the last frame when nothing is new -
    otherwise a browser on the 2 s thermal feed drops the connection.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from config import DASHBOARD, DETECTION, GEOMETRY, TRACKING, ZONES
from runtime.supervisor import Supervisor
from storage.db import Database
from storage.images import SnapshotStore

log = logging.getLogger(__name__)

BOUNDARY = "frame"


def _placeholder(text: str, size=(480, 270)) -> bytes:
    img = np.full((size[1], size[0], 3), 24, np.uint8)
    cv2.putText(img, text, (14, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _mjpeg(read, label: str, interval_s: float):
    """Stream whatever `read()` returns, re-sending when nothing is new.

    `read` returns (jpeg bytes or None, timestamp). Re-sending matters at the
    slow cadences: a browser that receives nothing for two seconds on the
    thermal feed, or forty-five on the overlay, will give up on the
    connection.
    """
    last_ts = -1.0
    payload = _placeholder(f"waiting for {label}...")
    while True:
        jpeg, ts = read()
        if jpeg and ts != last_ts:
            payload, last_ts = jpeg, ts
        yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
               + payload + b"\r\n")
        time.sleep(max(0.05, min(interval_s, 1.0)))


def create_app(supervisor: Supervisor) -> Flask:
    app = Flask(__name__)
    app.config["SUPERVISOR"] = supervisor
    snapshots = SnapshotStore()

    def _db() -> Database | None:
        """A connection per request thread. sqlite3 handles are not shareable.

        Cached on the Flask application context so a single request that
        queries twice does not open twice, and so the handle is closed when
        the request ends.
        """
        from flask import g
        if not supervisor.persist:
            return None
        if "db" not in g:
            try:
                db = Database()
                if supervisor.run_id is not None:
                    db.attach_run(supervisor.run_id)
                g.db = db
            except Exception:
                log.exception("dashboard: could not open the database")
                g.db = None
        return g.db

    @app.teardown_appcontext
    def _close_db(_exc):
        from flask import g
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # ------------------------------------------------------------------ page
    @app.route("/")
    def index():
        return render_template(
            "index.html",
            n_vials=GEOMETRY.n_vials,
            stages=ZONES.order(),
            zones_calibrated=ZONES.calibrated,
            hysteresis_n=TRACKING.stage_hysteresis_n,
            refresh_ms=2000,
        )

    # ---------------------------------------------------------------- feeds
    @app.route("/feed/live")
    def feed_live():
        def read():
            msg, ts = supervisor.preview.get()
            return (msg.jpeg if msg else None), ts
        return Response(_mjpeg(read, "camera", 0.2),
                        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/feed/overlay")
    def feed_overlay():
        """The analysed frame with zones, track IDs and any flagged vials.

        Updates at the analysis cadence, so tens of seconds between frames.
        That is not a stall - the live feed next to it shows the real rate.
        """
        def read():
            result, ts = supervisor.result.get()
            return (result.overlay_jpeg if result else None), ts
        return Response(_mjpeg(read, "pipeline overlay", 1.0),
                        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/feed/thermal")
    def feed_thermal():
        def read():
            msg, ts = supervisor.thermal.get()
            return (msg.jpeg if msg else None), ts
        return Response(_mjpeg(read, "thermal", 1.0),
                        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    # ------------------------------------------------------------------ API
    @app.route("/api/state")
    def api_state():
        """One poll, everything the page needs. Deliberately one endpoint.

        Five endpoints polled independently would show five slightly
        different moments of the same system, and reconciling them in the
        browser is not worth the bytes saved.
        """
        now = time.time()
        preview, preview_ts = supervisor.preview.get()
        thermal, thermal_ts = supervisor.thermal.get()
        result, result_ts = supervisor.result.get()

        return jsonify({
            "server_time": now,
            "health": supervisor.health(),
            "sources": {
                "rgb": _source_state(preview, preview_ts, now),
                "thermal": {**_source_state(thermal, thermal_ts, now),
                            "min_c": getattr(thermal, "min_c", None),
                            "mean_c": getattr(thermal, "mean_c", None),
                            "max_c": getattr(thermal, "max_c", None)},
            },
            "pipeline": _pipeline_state(result, result_ts, now),
            "events": [_event_json(e) for e in
                       supervisor.events.latest(DASHBOARD.events_page_size)],
            "zones_calibrated": ZONES.calibrated,
        })

    @app.route("/api/vials")
    def api_vials():
        result, _ = supervisor.result.get()
        if result is None:
            return jsonify({"vials": [], "frame_id": None})
        return jsonify({
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "vials": [v.__dict__ for v in result.vials],
        })

    @app.route("/api/events")
    def api_events():
        """Recent events. From SQLite when persisting, else the in-memory ring.

        The ring is bounded and only covers this process's lifetime, so the
        database is preferred whenever there is one - it is the only place a
        run from yesterday still exists.
        """
        limit = min(int(request.args.get("limit", DASHBOARD.events_page_size)), 500)
        severity = request.args.get("severity")
        db = _db()
        if db is not None:
            try:
                return jsonify({"events": db.recent_events(limit, min_severity=severity),
                                "source": "database"})
            except Exception:
                log.exception("dashboard: event query failed")
        return jsonify({
            "events": [_event_json(e) for e in supervisor.events.latest(limit)],
            "source": "memory",
        })

    @app.route("/api/vials/<int:track_id>/series")
    def api_vial_series(track_id: int):
        """One feature's history for one vial. Empty until features exist."""
        feature = request.args.get("feature", "")
        if not feature:
            return jsonify({"error": "feature query parameter is required"}), 400
        db = _db()
        if db is None:
            return jsonify({"error": "persistence is disabled"}), 503
        try:
            return jsonify({"track_id": track_id, "feature": feature,
                            "points": db.vial_series(track_id, feature)})
        except Exception as exc:
            log.exception("dashboard: series query failed")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/detectors")
    def api_detectors():
        status = supervisor.status.get("processing")
        detectors = (status.extra.get("detectors") if status else None) or []
        return jsonify({"detectors": detectors,
                        "enabled_in_config": list(DETECTION.enabled)})

    @app.route("/snapshot/<path:relative>")
    def snapshot(relative: str):
        """Serve a stored image by the relative path held in the database."""
        target = snapshots.resolve(relative).resolve()
        root = snapshots.root.resolve()
        # Path traversal guard: the relative path comes from the database, but
        # the URL is user-controlled and nothing else validates it.
        if root not in target.parents or not target.is_file():
            return jsonify({"error": "not found"}), 404
        return send_file(target, mimetype="image/jpeg")

    @app.route("/healthz")
    def healthz():
        _, preview_ts = supervisor.preview.get()
        fresh = bool(preview_ts) and (time.time() - preview_ts) < 30.0
        ok = fresh and supervisor.alive()
        return jsonify({
            "ok": ok,
            "workers_alive": supervisor.alive(),
            "preview_fresh": fresh,
        }), (200 if ok else 503)

    return app


# ------------------------------------------------------------------ helpers
def _source_state(msg, ts: float, now: float) -> dict:
    return {
        "name": getattr(msg, "source", None),
        "simulated": getattr(msg, "simulated", None),
        "frame_id": getattr(msg, "frame_id", None),
        "last_frame_ts": ts or None,
        "age_s": round(now - ts, 1) if ts else None,
    }


def _pipeline_state(result, ts: float, now: float) -> dict:
    if result is None:
        return {"frame_id": None, "n_vials": 0, "stage_counts": {},
                "age_s": None, "warnings": [], "timings_ms": {}}
    return {
        "frame_id": result.frame_id,
        "timestamp": result.timestamp,
        "age_s": round(now - ts, 1) if ts else None,
        "n_vials": result.n_vials,
        "stage_counts": result.stage_counts,
        "worst_severity": result.worst_severity(),
        "warnings": result.warnings,
        "timings_ms": result.timings_ms,
        "vials": [v.__dict__ for v in result.vials],
    }


def _event_json(event) -> dict:
    return {
        "timestamp": event.timestamp,
        "kind": event.kind,
        "severity": event.severity,
        "detector": event.detector,
        "track_id": event.track_id,
        "zone": event.zone,
        "message": event.message,
        "data": event.data,
    }


def main() -> None:
    """Standalone dashboard: starts its own supervisor and serves it."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    supervisor = Supervisor()
    supervisor.start()
    app = create_app(supervisor)
    log.info("dashboard on http://localhost:%d", DASHBOARD.port)
    try:
        app.run(host=DASHBOARD.host, port=DASHBOARD.port, debug=False,
                use_reloader=False, threaded=True)
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
