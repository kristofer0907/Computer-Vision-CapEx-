"""Run SlotTracker/FifoTracker/RegionCoordinator over captures, headless.

Replay a folder:

    python -m tools.track_regions --images capture/second_iteration/crucibles/undistored
    python -m tools.track_regions --images capture/second_iteration/clean --undistort

Or capture live. Like capture/capture_images.py it asks where to put the
images and how far apart to take them, then runs until ctrl-c:

    python -m tools.track_regions --live
        Folder name to save images to: runs/monday
        Interval between photos (seconds): 30

Pass either as a flag to skip that question, for scripting:

    python -m tools.track_regions --live picamera2 --interval 10 --overlay-dir runs/x

Each crucible is drawn with its id (001-999), the zone and slot it is in,
and a colour for whether it looks lidded - see DETECTION.lid_score_threshold
for how far to trust that (96.4% against the hand-labelled set, and the
classes do overlap).

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
import time
from pathlib import Path

import cv2
import numpy as np

from config import DATA_DIR, DETECTION, REGION_TRACKING, ensure_dirs
from pipeline.lineage import VialLineage
from pipeline.region_trackers import (FifoTracker, RegionCoordinator, SlotTracker,
                                      create_region_coordinator)

log = logging.getLogger("track_regions")

OUT_JSON = DATA_DIR / "region_tracks.json"
OUT_DIR = DATA_DIR / "inspect_regions"

BASE_BGR = (90, 110, 130)
SLOT_BGR = (98, 170, 235)
LANE_BGR = (201, 134, 31)
HANDOFF_BGR = (230, 210, 70)
CLOSED_BGR = (82, 82, 224)

# Lid state is what colours a crucible, since that is the thing being read
# off these frames. Track state (just appeared / just handed off) is a ring
# outside the disc instead, so the two never compete for the same channel.
LID_BGR = (110, 199, 98)        # green - lid detected
OPEN_BGR = (60, 168, 224)       # amber - open jar
UNKNOWN_BGR = (150, 150, 150)   # grey - could not be scored


def grab(images_dir: str | None, undistort: bool, rgb: str | None = None,
         interval_s: float = 0.0):
    """Yield Frames, either replaying a folder or capturing live.

    A folder ends when it runs out. A live source runs until interrupted -
    that is the point of it - so the caller decides when to stop.
    """
    from drivers.rgb_cam import FileCameraSource, create_camera

    if images_dir:
        source = FileCameraSource(images_dir, loop=False)
    else:
        source = create_camera(rgb)
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
            if interval_s:
                time.sleep(interval_s)
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
    for zone, dets in coord.untracked.items():
        parts.append(f"{zone}={len(dets)}(unnumbered)")
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


def slot_occupancy(coord: RegionCoordinator, zone: str) -> dict[int, bool]:
    """Which slots of `zone` read full this frame. The only thing
    pipeline/lineage.py needs, and all it is allowed to see."""
    tracker = coord.trackers.get(zone)
    if not isinstance(tracker, SlotTracker):
        return {}
    return {sid: occupant is not None
            for sid, occupant in tracker.slot_status().items()}


def short_vial_id(vial_id: str | None) -> str:
    """`heater-0-1757339234` -> `h0.9234`, so it fits under a crucible."""
    if not vial_id:
        return ""
    try:
        _, slot, ts = vial_id.split("-")
        return f"h{slot}.{ts[-4:]}"
    except ValueError:
        return vial_id


def track_label(track_id: int) -> str:
    """Ids read as 001-999. Past 999 it keeps counting rather than wrapping -
    a reused id would silently merge two crucibles' histories."""
    return f"{track_id:03d}"


def lid_state(image: np.ndarray, t) -> tuple[str, float | None]:
    """("lid" | "open" | "unknown", score) for one tracked crucible.

    Scored per frame off the current image rather than carried on the Track:
    a lid can be put on or taken off between frames, so it is an observation,
    not an identity. See DETECTION.lid_score_threshold for how reliable this
    is - the classes overlap, so single-frame verdicts do flip.
    """
    from pipeline.features import lid_score
    try:
        score = lid_score(image, t.cx, t.cy, t.radius)
    except Exception:
        return "unknown", None
    return ("lid" if score >= DETECTION.lid_score_threshold else "open"), score


def draw_overlay(image: np.ndarray, coord: RegionCoordinator,
                 results: dict, lineage: VialLineage | None = None) -> np.ndarray:
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

    # Crucibles in zones that do not carry identity (storing, the injection
    # lane): shown so you can see them, deliberately without a number.
    for zone, dets in coord.untracked.items():
        for d in dets:
            state, _score = lid_state(image, d)
            color = {"lid": LID_BGR, "open": OPEN_BGR}.get(state, UNKNOWN_BGR)
            c = (int(round(d.cx)), int(round(d.cy)))
            r = int(round(d.radius))
            cv2.circle(out, c, r, color, max(1, thick - 1), cv2.LINE_AA)
            _text(out, "-", c, color, scale * 0.62, thick, center=True)
            _text(out, zone[:4], (c[0], c[1] + r + int(round(26 * scale))),
                  color, scale * 0.46, max(1, thick - 1), center=True)

    for zone, (active, closed) in results.items():
        queue = (coord.trackers[zone].queue_order()
                 if isinstance(coord.trackers[zone], FifoTracker) else [])
        for t in active:
            state, _score = lid_state(image, t)
            color = {"lid": LID_BGR, "open": OPEN_BGR}.get(state, UNKNOWN_BGR)
            c = (int(round(t.cx)), int(round(t.cy)))
            r = int(round(t.radius))
            cv2.circle(out, c, r, color, thick, cv2.LINE_AA)
            if t.track_id in handed_off:
                # just arrived from another zone - a ring outside the disc,
                # so it does not fight the lid colour
                cv2.circle(out, c, r + int(round(9 * scale)), HANDOFF_BGR,
                           max(1, thick - 1), cv2.LINE_AA)

            # The id goes inside the disc and the zone just under it: slots
            # sit ~140 px apart, so a wide label above each one collides with
            # its neighbour's.
            vial_id = None
            if lineage is not None and t.slot_id is not None:
                if zone == REGION_TRACKING.heater_zone:
                    vial_id = lineage.vial_on_heater(t.slot_id)
                elif zone == REGION_TRACKING.cooling_zone:
                    vial_id = lineage.vial_on_cooling(t.slot_id)

            if vial_id:
                where = short_vial_id(vial_id)
            elif t.slot_id is not None:
                where = f"{zone[:4]} {t.slot_id}"
            elif t.track_id in queue:
                where = f"{zone[:4]} q{queue.index(t.track_id) + 1}"
            else:
                where = zone[:4]
            _text(out, track_label(t.track_id), c, color, scale * 0.62, thick,
                  center=True)
            _text(out, where, (c[0], c[1] + r + int(round(26 * scale))),
                  color, scale * 0.46, max(1, thick - 1), center=True)

        for t in closed:
            c = (int(round(t.cx)), int(round(t.cy)))
            cv2.drawMarker(out, c, CLOSED_BGR, cv2.MARKER_TILTED_CROSS,
                           28, thick)
            _text(out, f"{track_label(t.track_id)} {t.closed_reason}",
                  (c[0] + 16, c[1] + 8), CLOSED_BGR, scale * 0.72,
                  max(1, thick - 1))

    _legend(out, scale, thick)
    return out


def _legend(out: np.ndarray, scale: float, thick: int) -> None:
    """What the colours mean, on the frame itself - these get looked at
    days later, away from the terminal that produced them."""
    x, y = int(round(24 * scale)), int(round(40 * scale))
    step = int(round(34 * scale))
    for color, text in ((LID_BGR, "lid detected"),
                        (OPEN_BGR, "open"),
                        (UNKNOWN_BGR, "not scored"),
                        (HANDOFF_BGR, "outer ring: changed zone"),
                        (CLOSED_BGR, "x: track ended")):
        cv2.circle(out, (x, y - int(round(6 * scale))),
                   int(round(9 * scale)), color, -1, cv2.LINE_AA)
        _text(out, text, (x + int(round(22 * scale)), y), color,
              scale * 0.46, max(1, thick - 1))
        y += step


def as_json(name: str, results: dict, coord: RegionCoordinator,
            image: np.ndarray, lineage: VialLineage | None = None) -> dict:
    zones = {}
    for zone, (active, _closed) in results.items():
        tracker = coord.trackers[zone]
        queue_pos = ({tid: i for i, tid in enumerate(tracker.queue_order())}
                    if isinstance(tracker, FifoTracker) else {})
        entries = []
        for t in active:
            state, score = lid_state(image, t)
            entries.append(
                {"track_id": t.track_id, "label": track_label(t.track_id),
                 "slot_id": t.slot_id, "cx": t.cx, "cy": t.cy,
                 "stage": t.stage, "queue_pos": queue_pos.get(t.track_id),
                 "lid": state,
                 "lid_score": None if score is None else round(score, 1)})
        zones[zone] = entries
    return {
        "frame": name,
        "zones": zones,
        "lineage": lineage.as_dict() if lineage else None,
        "handoffs": [{"track_id": tid, "from": frm, "to": to}
                    for tid, frm, to in coord.handoffs_this_frame()],
        "closed": [{"zone": zone, "track_id": t.track_id, "reason": t.closed_reason}
                  for zone, (_active, closed) in results.items() for t in closed],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="replay a folder of captures")
    src.add_argument("--live", metavar="BACKEND", nargs="?", const="auto",
                     help="capture from the camera instead: auto | picamera2 | mock")
    p.add_argument("--interval", type=float, default=None,
                   help="seconds between live captures; asked for if omitted")
    und = p.add_mutually_exclusive_group()
    und.add_argument("--undistort", action="store_true", default=None,
                     help="lens-correct each frame before tracking. On by "
                          "default when --live, since the camera's raw output "
                          "is distorted and the slot layouts are not")
    und.add_argument("--no-undistort", dest="undistort", action="store_false",
                     help="skip lens correction (the right choice for a "
                          "folder that is already undistorted)")
    p.add_argument("--localizer", default="crucible",
                   help="where crucibles come from (default: detect_crucibles)")
    p.add_argument("--frames", type=int, default=None,
                   help="stop after this many frames (default: unlimited when "
                        "live, whole folder when replaying)")
    p.add_argument("--out", default=None,
                   help="tracks JSON (default: <overlay-dir>/region_tracks.json)")
    p.add_argument("--overlay-dir", default=None,
                   help="folder to save the annotated frames into; asked for "
                        "if omitted on a live run")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    # The hand-marked slot and lane layouts live in undistorted pixels, so a
    # raw camera frame has to be corrected before anything is matched against
    # them. At the storing rack that is worth ~100 px against a 45 px gate -
    # skipping it does not degrade matching there, it stops it working.
    if args.undistort is None:
        args.undistort = bool(args.live)

    ensure_dirs()

    # A live run asks for the folder and the interval the way
    # capture/capture_images.py does, rather than making you remember flag
    # names at the bench. Passing either flag skips its question, so this
    # still scripts.
    if args.live:
        if args.overlay_dir is None:
            args.overlay_dir = input("Folder name to save images to: ").strip()
            while not args.overlay_dir:
                args.overlay_dir = input("Folder name to save images to: ").strip()
        if args.interval is None:
            while True:
                raw = input("Interval between photos (seconds): ").strip()
                try:
                    args.interval = float(raw)
                except ValueError:
                    print("  a number, please")
                    continue
                if args.interval < 0:
                    print("  must not be negative")
                    continue
                break
    if args.overlay_dir is None:
        args.overlay_dir = str(OUT_DIR)
    if args.interval is None:
        args.interval = 10.0

    overlay_dir = Path(args.overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else overlay_dir / "region_tracks.json"

    from pipeline.localize import create_localizer

    localizer = create_localizer(args.localizer)
    localizer.start()

    log.info("lens correction: %s",
             "on" if args.undistort else "off (frames assumed already undistorted)")
    if args.live:
        # flush: logging goes to stderr, so an unflushed stdout banner turns
        # up after the frames it was meant to introduce
        print(f"\nCapturing to '{overlay_dir}' every {args.interval:g}s, "
              f"with tracking overlays.", flush=True)
        print("Press ctrl-c to stop.\n", flush=True)

    lineage = VialLineage()
    coord: RegionCoordinator | None = None
    dumped: list[dict] = []
    n = 0
    try:
        for frame in grab(args.images, args.undistort, rgb=args.live,
                          interval_s=args.interval if args.live else 0.0):
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
            lineage.update(slot_occupancy(coord, REGION_TRACKING.heater_zone),
                           slot_occupancy(coord, REGION_TRACKING.cooling_zone),
                           frame.timestamp)

            name = str((frame.truth or {}).get("name", "")
                       or time.strftime("%Y%m%d_%H%M%S"))
            log.info(summarize(coord, results, n))
            dumped.append(as_json(name, results, coord, frame.image, lineage))

            overlay = draw_overlay(frame.image, coord, results, lineage)
            cv2.imwrite(str(overlay_dir / f"{n:04d}_{Path(name).stem}.jpg"), overlay)
            # Written every frame, not just at the end: a live run is stopped
            # with ctrl-c, and losing the whole record to that would be daft.
            out_path.write_text(json.dumps(dumped, indent=2))
            n += 1
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        localizer.stop()

    if n == 0:
        log.error("no frames processed")
        return 1

    log.info("wrote %d frame(s): overlays in %s, tracks in %s",
             n, overlay_dir, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
