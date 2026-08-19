"""Run the pipeline over a recording or the simulator, without a dashboard.

    python -m tools.replay --frames 40
    python -m tools.replay --rgb file --file data/snapshots/2026-08-14 --frames 200

The fastest way to exercise the analysis path while writing detection logic:
no Flask, no multiprocessing, no browser, and a printed summary per frame. It
uses the same PipelineRunner the processing process uses, so anything that
works here works there.

--persist writes to the real database, so a replay can be inspected with the
same queries as a live run. Off by default: replaying the same recording ten
times while tuning a threshold should not pollute the run history.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

log = logging.getLogger("replay")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb", default="mock", help="camera backend")
    p.add_argument("--file", default=None, help="video or stills directory")
    p.add_argument("--frames", type=int, default=20, help="frames to analyse")
    p.add_argument("--localizer", default="auto")
    p.add_argument("--extractor", default="auto")
    p.add_argument("--interval", type=float, default=0.0,
                   help="seconds to wait between frames (0 = as fast as possible)")
    p.add_argument("--persist", action="store_true",
                   help="write results to the database")
    p.add_argument("--events-only", action="store_true",
                   help="print only frames that produced events")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")
    if args.file:
        os.environ["CAPEX_RGB_FILE"] = args.file

    from drivers.rgb_cam import create_camera
    from pipeline.features import create_extractor
    from pipeline.localize import create_localizer
    from pipeline.runner import PipelineRunner

    camera = create_camera(args.rgb)
    runner = PipelineRunner(localizer=create_localizer(args.localizer),
                            extractor=create_extractor(args.extractor),
                            draw_overlay=False)
    runner.start()

    db = None
    if args.persist:
        from config import ensure_dirs
        from storage.db import Database
        ensure_dirs()
        db = Database()
        db.start_run(rgb_source=camera.name, simulated=camera.simulated,
                     note="replay")

    total_events = 0
    began = time.monotonic()
    try:
        for i in range(args.frames):
            try:
                frame = camera.capture()
            except StopIteration:
                log.info("source exhausted after %d frames", i)
                break

            result = runner.process(frame)
            total_events += len(result.events)

            if not args.events_only or result.events:
                stages = " ".join(f"{k}={v}" for k, v in result.stage_counts.items() if v)
                log.info("frame %-4d  vials=%-3d %s  next=%.0fs  %s",
                         result.frame_id, result.n_vials, stages or "-",
                         runner.next_interval_s,
                         " ".join(f"{k}:{v}ms" for k, v in result.timings_ms.items()))
            for e in result.events:
                log.info("    [%s] %s: %s", e.severity, e.detector, e.message)
            for w in result.warnings:
                log.warning("    %s", w)

            if db is not None:
                db.log_result(result)
            if args.interval:
                time.sleep(args.interval)
    finally:
        runner.stop()
        camera.stop()
        if db is not None:
            db.end_run()
            db.close()

    elapsed = time.monotonic() - began
    log.info("done: %d events over %.1fs", total_events, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
