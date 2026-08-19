"""Entry point: start the workers, serve the dashboard, shut down cleanly.

    python main.py                        # auto-detect hardware, simulate the rest
    python main.py --rgb mock             # force the synthetic platform
    python main.py --no-thermal           # skip the MLX90640
    python main.py --no-persist           # do not write to SQLite
    python main.py --rgb file --file run.mp4   # replay a recording
    python main.py --port 8080

Open http://<host>:5000/. From the laptop over the ICS link that is the Pi's
address on 192.168.137.x.

Runs anywhere. On a machine with no Pi camera stack the drivers report the
hardware as unavailable and fall back to the simulator rather than failing to
import - which is the whole reason picamera2 and the MLX90640 libraries are
imported inside start() and never at module level. If `ModuleNotFoundError:
No module named 'libcamera'` ever surfaces again it means something imported
picamera2 at module scope; that is the bug, not the missing package.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from config import DASHBOARD, SOURCES, ZONES, ensure_dirs

log = logging.getLogger("main")


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
                     help="do not start the thermal process")

    pipe = p.add_argument_group("pipeline")
    pipe.add_argument("--localizer", default="auto",
                      help="localiser to use; see pipeline/localize.py")
    pipe.add_argument("--extractor", default="auto",
                      help="feature extractor; see pipeline/features.py")
    pipe.add_argument("--no-pipeline", action="store_true",
                      help="preview and thermal only, no analysis")

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
        # SOURCES is built when config is first imported, so this has to be in
        # the environment before the child processes import it. Setting it
        # after that import - as an earlier version of this file did for the
        # backend - is read too late and silently ignored.
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

    # Imported after logging is configured, and after the env var above.
    from dashboard.app import create_app
    from runtime.supervisor import Supervisor

    supervisor = Supervisor(
        rgb_backend=args.rgb,
        thermal_backend=args.thermal,
        localizer=args.localizer,
        extractor=args.extractor,
        enable_thermal=not args.no_thermal,
        enable_pipeline=not args.no_pipeline,
        persist=not args.no_persist,
        store_thermal_grid=args.store_thermal_grid,
        note=args.note,
    )

    shutdown = threading.Event()

    def handle_signal(signum, _frame):
        # Flask's dev server does not run a signal handler of its own, so
        # without this a Ctrl-C leaves the worker processes holding the
        # camera and the next start fails with a device-busy error.
        log.info("signal %s received, shutting down", signum)
        shutdown.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    supervisor.start()
    app = create_app(supervisor)

    log.info("dashboard on http://%s:%d", args.host, args.port)
    try:
        # debug=True spawns a reloader child, which would start a second set
        # of workers and open the camera twice. Never enable it here.
        app.run(host=args.host, port=args.port, debug=False,
                use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
