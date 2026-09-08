"""Run SlotTracker/FifoTracker/RegionCoordinator over real captures, headless.

    python -m tools.track_regions --images capture/second_iteration/crucibles/undistored
    python -m tools.track_regions --images capture/second_iteration/clean --undistort
    python -m tools.track_regions --images ... --frames 40 --out data/region_tracks.json

Standalone validation for pipeline/region_trackers.py, same spirit as
tools/replay.py but without a PipelineRunner: no dashboard, no storage, no
main.py. Nothing here is wired into the live pipeline - that is a deliberate
follow-up once this is proven against real captures.

Before running this once: mark the fixed slot positions (already done, see
data/slots_filling_*.json) and the injection lane's entry/exit points:

    python -m tools.mark_slots --stage injection --lane

--undistort matters. The slot files and data/reference_images/filling_reference.jpg
are registered to *undistorted* pixels. capture/second_iteration/crucibles/undistored/
is already undistorted at the matching size - use it as-is. Anything else
(capture/second_iteration/clean/, or the raw capture/second_iteration/crucibles/
parent folder) needs --undistort, or slot/lane positions will be off by however
much the lens distorts near the frame edges.

Writes one annotated JPEG per frame to --overlay-dir and one JSON dump to
--out. Console line per frame, in the spirit of tools/replay.py's summary:

    frame 007  storing=4/12  heating=1/2  collection=2/15  injection(queue)=[11,14]  handoffs=1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DATA_DIR, ensure_dirs
from pipeline.region_trackers import (FifoTracker, RegionCoordinator, SlotTracker,
                                      create_region_coordinator)

log = logging.getLogger("track_regions")

OUT_JSON = DATA_DIR / "region_tracks.json"
OUT_DIR = DATA_DIR / "inspect_regions"

BASE_BGR = (90, 110, 130)
SLOT_BGR = (98, 170, 235)
LANE_BGR = (201, 134, 31)
CONTINUING_BGR = (110, 199, 98)
FRESH_BGR = (60, 168, 224)
HANDOFF_BGR = (230, 210, 70)
CLOSED_BGR = (82, 82, 224)


def grab(images_dir: str, undistort: bool):
    """Yield Frames from a folder, optionally undistorted (see module docstring)."""
    from drivers.rgb_cam import FileCameraSource

    source = FileCameraSource(images_dir, loop=False)
    source.start()
    try:
        while True:
            try:
                frame = source.capture()
            except StopIteration:
                return
            if undistort:
                from tools.mark_slots import undistort_image
                frame = dataclasses.replace(frame, image=undistort_image(frame.image))
            yield frame
    finally:
        source.stop()


def build_coordinator(frame_size: tuple[int, int]) -> RegionCoordinator:
    return create_region_coordinator(frame_size)


def summarize(coord: RegionCoordinator, results: dict, frame_idx: int) -> str:
    parts = [f"frame {frame_idx:03d}"]
    for zone, tracker in coord.trackers.items():
        if isinstance(tracker, SlotTracker):
            status = tracker.slot_status()
            occupied = sum(1 for v in status.values() if v is not None)
            parts.append(f"{zone}={occupied}/{len(status)}")
        elif isinstance(tracker, FifoTracker):
            parts.append(f"{zone}(queue)={tracker.queue_order()}")
    parts.append(f"handoffs={len(coord.handoffs_this_frame())}")
    return "  ".join(parts)


def _text(out: np.ndarray, s: str, org: tuple[int, int], color, scale: float,
          thick: int, center: bool = False) -> None:
    """Label with a dark outline, so it stays readable over bright aluminium
    and dark pegboard alike. `center` treats `org` as the text's midpoint."""
    if center:
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (org[0] - tw // 2, org[1] + th // 2)
    cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 22, 26),
                thick + 2, cv2.LINE_AA)
    cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def draw_overlay(image: np.ndarray, coord: RegionCoordinator,
                 results: dict) -> np.ndarray:
    out = coord.zone_map.draw(image, color=BASE_BGR)
    handed_off = {tid for tid, _from, _to in coord.handoffs_this_frame()}

    # These frames are 4056 px wide; text sized for a 1280 px preview is
    # unreadable on them. Scale the labels to the image instead.
    scale = max(0.45, out.shape[1] / 2600.0)
    thick = max(1, int(round(out.shape[1] / 1400.0)))

    for zone, tracker in coord.trackers.items():
        if isinstance(tracker, SlotTracker):
            for sid, x, y in tracker.slots():
                c = (int(round(x)), int(round(y)))
                cv2.circle(out, c, int(round(tracker.slot_radius_px)),
                          SLOT_BGR, 1, cv2.LINE_AA)
                _text(out, str(sid), (c[0] + 8, c[1] - 8), SLOT_BGR,
                      scale * 0.5, 1)
        elif isinstance(tracker, FifoTracker):
            entry, exit_ = tracker.lane_endpoints()
            a = tuple(int(round(v)) for v in entry)
            b = tuple(int(round(v)) for v in exit_)
            cv2.line(out, a, b, LANE_BGR, thick, cv2.LINE_AA)
            cv2.drawMarker(out, a, LANE_BGR, cv2.MARKER_TRIANGLE_UP, 24, thick)
            cv2.drawMarker(out, b, LANE_BGR, cv2.MARKER_SQUARE, 24, thick)
            _text(out, f"{zone} entry", (a[0] + 14, a[1]), LANE_BGR,
                  scale * 0.6, 1)
            _text(out, f"{zone} exit", (b[0] + 14, b[1]), LANE_BGR,
                  scale * 0.6, 1)

    for zone, (active, closed) in results.items():
        queue = (coord.trackers[zone].queue_order()
                 if isinstance(coord.trackers[zone], FifoTracker) else [])
        for t in active:
            color = (HANDOFF_BGR if t.track_id in handed_off
                     else FRESH_BGR if t.hits == 1 else CONTINUING_BGR)
            c = (int(round(t.cx)), int(round(t.cy)))
            r = int(round(t.radius))
            cv2.circle(out, c, r, color, thick, cv2.LINE_AA)

            # The id goes inside the disc and the zone just under it: slots
            # sit ~140 px apart, so a wide label above each one collides with
            # its neighbour's.
            if t.slot_id is not None:
                where = f"{zone[:4]} {t.slot_id}"
            elif t.track_id in queue:
                where = f"{zone[:4]} q{queue.index(t.track_id) + 1}"
            else:
                where = zone[:4]
            _text(out, str(t.track_id), c, color, scale * 0.62, thick,
                  center=True)
            _text(out, where, (c[0], c[1] + r + int(round(26 * scale))),
                  color, scale * 0.46, max(1, thick - 1), center=True)

        for t in closed:
            c = (int(round(t.cx)), int(round(t.cy)))
            cv2.drawMarker(out, c, CLOSED_BGR, cv2.MARKER_TILTED_CROSS,
                           28, thick)
            _text(out, f"ID {t.track_id} {t.closed_reason}",
                  (c[0] + 16, c[1] + 8), CLOSED_BGR, scale * 0.72,
                  max(1, thick - 1))

    return out


def as_json(name: str, results: dict, coord: RegionCoordinator) -> dict:
    zones = {}
    for zone, (active, _closed) in results.items():
        tracker = coord.trackers[zone]
        queue_pos = ({tid: i for i, tid in enumerate(tracker.queue_order())}
                    if isinstance(tracker, FifoTracker) else {})
        zones[zone] = [
            {"track_id": t.track_id, "slot_id": t.slot_id, "cx": t.cx, "cy": t.cy,
             "stage": t.stage, "queue_pos": queue_pos.get(t.track_id)}
            for t in active
        ]
    return {
        "frame": name,
        "zones": zones,
        "handoffs": [{"track_id": tid, "from": frm, "to": to}
                    for tid, frm, to in coord.handoffs_this_frame()],
        "closed": [{"zone": zone, "track_id": t.track_id, "reason": t.closed_reason}
                  for zone, (_active, closed) in results.items() for t in closed],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of captures")
    p.add_argument("--undistort", action="store_true",
                   help="undistort each frame first (see module docstring)")
    p.add_argument("--localizer", default="crucible",
                   help="where crucibles come from (default: detect_crucibles)")
    p.add_argument("--frames", type=int, default=None, help="cap frames processed")
    p.add_argument("--out", default=str(OUT_JSON))
    p.add_argument("--overlay-dir", default=str(OUT_DIR))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    ensure_dirs()
    overlay_dir = Path(args.overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.localize import create_localizer

    localizer = create_localizer(args.localizer)
    localizer.start()

    coord: RegionCoordinator | None = None
    dumped: list[dict] = []
    n = 0
    try:
        for frame in grab(args.images, args.undistort):
            if args.frames is not None and n >= args.frames:
                break
            if coord is None:
                h, w = frame.image.shape[:2]
                coord = build_coordinator((w, h))
                try:
                    coord.start()
                except FileNotFoundError as exc:
                    log.error("%s", exc)
                    return 2
                log.info("tracking zones: %s  (frame %dx%d)",
                         ", ".join(coord.trackers), w, h)

            detections = localizer.locate(frame)
            results = coord.update(detections, frame.timestamp)

            name = str((frame.truth or {}).get("name", "") or f"frame_{frame.frame_id}")
            log.info(summarize(coord, results, n))
            dumped.append(as_json(name, results, coord))

            overlay = draw_overlay(frame.image, coord, results)
            cv2.imwrite(str(overlay_dir / f"{n:03d}_{Path(name).stem}.jpg"), overlay)
            n += 1
    finally:
        localizer.stop()

    if n == 0:
        log.error("no images processed")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dumped, indent=2))
    log.info("wrote %d frame(s) of tracks to %s", n, out_path)
    log.info("wrote overlays to %s", overlay_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
