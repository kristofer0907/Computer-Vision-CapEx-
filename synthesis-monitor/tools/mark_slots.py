"""Click the fixed slot positions for a stage (e.g. the crucible rack seen in
a calibration shot), save them as a reference JSON.

    python -m tools.mark_slots --stage filling
    python -m tools.mark_slots --stage filling --source-image path/to/img.jpg
    python -m tools.mark_slots --stage filling --radius 22
    python -m tools.mark_slots --stage injection --lane

By default this undistorts the first image in
capture/second_iteration/clean/ (using the saved calibration at
data/camera_calibration.npz) and uses that as the reference frame. Pass
--source-image to use a different photo instead. Either way the undistorted
frame is saved once to data/reference_images/<stage>_reference.jpg, and
re-used on future runs of the same --stage unless --source-image forces a
fresh one.

Controls:
    left click      place a slot at the cursor
    right click     remove the nearest slot
    + / -           grow / shrink the radius (applies to all slots)
    c               clear all slots on this image
    s               save and quit
    q / ESC         quit without saving

--radius is optional. If you never set one (neither --radius nor +/- during
the session), the output simply omits "slot_radius_px" instead of writing a
guessed default.

Output (data/slots_<stage>.json by default):

    {
      "stage": "filling",
      "image": "filling_reference.jpg",
      "image_size": [4056, 3040],
      "slot_radius_px": 22,
      "slots": [
        {"id": 0, "x": 1134, "y": 788},
        {"id": 1, "x": 1290, "y": 788}
      ]
    }

Why this exists: mirrors tools/mark_vials.py, but for the *fixed* slot
layout (crucible rack holes, lidding stations, etc.) rather than per-frame
vial detections - a one-time reference to check live detections against,
not something re-marked every capture.

--lane switches to a second mode for a region with no fixed slots at all: a
narrow lane a crucible moves through single-file (see
pipeline.region_trackers.FifoTracker). Instead of N slot positions, click
exactly 2 points - where a crucible enters the lane, then where it exits.
Controls are the same minus the radius keys, which don't apply.

Output (data/lane_<stage>.json by default):

    {
      "stage": "injection",
      "image": "injection_reference.jpg",
      "image_size": [4056, 3040],
      "entry": [1820, 1210],
      "exit": [2340, 1690]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DATA_DIR, ensure_dirs
from tools.calibrate_camera import CALIBRATION_FILE

log = logging.getLogger("mark_slots")

WINDOW = "mark slots - click to place, s save, q quit"
SLOT_BGR = (98, 170, 235)
REFERENCE_DIR = DATA_DIR / "reference_images"
DEFAULT_IMAGES_DIR = Path("capture/second_iteration/clean")
FALLBACK_DISPLAY_RADIUS_PX = 22.0  # only used to render circles when no radius was set


def undistort_image(img: np.ndarray) -> np.ndarray:
    if not CALIBRATION_FILE.exists():
        raise SystemExit(
            f"No calibration file at {CALIBRATION_FILE}. "
            f"Run `python -m tools.calibrate_camera` first.")
    data = np.load(CALIBRATION_FILE)
    return cv2.undistort(img, data["camera_matrix"], data["dist_coeffs"])


def pick_source_image(source_image: str | None) -> Path:
    if source_image:
        return Path(source_image)
    candidates = sorted(DEFAULT_IMAGES_DIR.glob("*.jpg"))
    if not candidates:
        raise SystemExit(f"no .jpg images found in {DEFAULT_IMAGES_DIR}")
    return candidates[0]


def build_reference(stage: str, source_image: str | None, force: bool) -> Path:
    """Return path to the undistorted reference image for this stage,
    undistorting + saving it if it doesn't exist yet (or --source-image/--force)."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REFERENCE_DIR / f"{stage}_reference.jpg"

    if ref_path.exists() and not source_image and not force:
        log.info("using existing reference image %s", ref_path)
        return ref_path

    src_path = pick_source_image(source_image)
    img = cv2.imread(str(src_path))
    if img is None:
        raise SystemExit(f"could not read {src_path}")

    fixed = undistort_image(img)
    cv2.imwrite(str(ref_path), fixed)
    log.info("undistorted %s -> %s", src_path, ref_path)
    return ref_path


class SlotMarker:
    def __init__(self, image: np.ndarray, radius: float, radius_set: bool,
                 lane: bool = False) -> None:
        self.image = image
        self.radius = radius
        self.radius_set = radius_set
        self.lane = lane
        self.points: list[tuple[float, float]] = []

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.lane and len(self.points) >= 2:
                return  # entry/exit only - right click one off first
            self.points.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            nearest = min(self.points,
                          key=lambda p: math.hypot(p[0] - x, p[1] - y))
            self.points.remove(nearest)

    def render(self) -> np.ndarray:
        out = self.image.copy()
        labels = ("entry", "exit") if self.lane else None

        if self.lane and len(self.points) == 2:
            a, b = (int(round(v)) for v in self.points[0]), (int(round(v)) for v in self.points[1])
            cv2.line(out, tuple(a), tuple(b), SLOT_BGR, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(self.points):
            c = (int(round(x)), int(round(y)))
            label = labels[i] if labels else str(i)
            if not self.lane:
                cv2.circle(out, c, int(round(self.radius)), SLOT_BGR, 2, cv2.LINE_AA)
            cv2.drawMarker(out, c, SLOT_BGR, cv2.MARKER_CROSS, 10, 2)
            cv2.putText(out, label, (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, SLOT_BGR, 1, cv2.LINE_AA)

        if self.lane:
            banner = (f"{len(self.points)}/2 points   "
                      f"[left click entry then exit, right click to remove, "
                      f"c clear, s save, q quit]")
        else:
            radius_label = f"r={self.radius:.0f}px" if self.radius_set else "r=unset"
            banner = (f"{len(self.points)} slots   {radius_label}   "
                      f"[+/- radius, c clear, s save, q quit]")
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 22, 26), -1)
        cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (223, 227, 234), 1, cv2.LINE_AA)
        return out

    def slots(self) -> list[dict]:
        return [{"id": i, "x": int(round(x)), "y": int(round(y))}
                for i, (x, y) in enumerate(self.points)]


def _edit(marker: SlotMarker) -> str:
    """Interactive loop. Returns 'save' or 'quit'."""
    while True:
        cv2.imshow(WINDOW, marker.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "save"
        if not marker.lane and key in (ord("+"), ord("=")):
            marker.radius += 1
            marker.radius_set = True
        elif not marker.lane and key in (ord("-"), ord("_")):
            marker.radius = max(2.0, marker.radius - 1)
            marker.radius_set = True
        elif key == ord("c"):
            marker.points.clear()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   help="stage name this slot layout belongs to, e.g. filling")
    p.add_argument("--source-image",
                   help="image to undistort and use as the reference frame "
                        "(default: first *.jpg in "
                        f"{DEFAULT_IMAGES_DIR})")
    p.add_argument("--force-reundistort", action="store_true",
                   help="re-undistort even if a reference image for this "
                        "stage already exists")
    p.add_argument("--radius", type=float, default=None,
                   help="slot radius in pixels; optional - if omitted (and "
                        "never adjusted with +/-), the output has no "
                        "slot_radius_px field. Ignored with --lane.")
    p.add_argument("--lane", action="store_true",
                   help="mark a 2-point entry/exit lane instead of N slots "
                        "(see pipeline.region_trackers.FifoTracker)")
    p.add_argument("--out",
                   help="output JSON path (default: data/slots_<stage>.json, "
                        "or data/lane_<stage>.json with --lane)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ensure_dirs()

    ref_path = build_reference(args.stage, args.source_image, args.force_reundistort)
    image = cv2.imread(str(ref_path))
    if image is None:
        raise SystemExit(f"could not read reference image {ref_path}")

    radius_set = args.radius is not None
    radius = args.radius if radius_set else FALLBACK_DISPLAY_RADIUS_PX
    marker = SlotMarker(image, radius, radius_set, lane=args.lane)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1500, image.shape[1]), min(950, image.shape[0]))
    cv2.setMouseCallback(WINDOW, marker.on_mouse)

    try:
        action = _edit(marker)
    finally:
        cv2.destroyAllWindows()

    if action == "quit":
        log.info("quit without saving")
        return 1

    if not marker.points:
        log.warning("no slots marked, not writing")
        return 1

    if args.lane:
        if len(marker.points) != 2:
            log.warning("--lane needs exactly 2 points (entry, exit), got %d - "
                       "not writing", len(marker.points))
            return 1
        result = {
            "stage": args.stage,
            "image": ref_path.name,
            "image_size": [image.shape[1], image.shape[0]],
            "entry": [int(round(marker.points[0][0])), int(round(marker.points[0][1]))],
            "exit": [int(round(marker.points[1][0])), int(round(marker.points[1][1]))],
        }
        default_out = DATA_DIR / f"lane_{args.stage}.json"
    else:
        result = {
            "stage": args.stage,
            "image": ref_path.name,
            "image_size": [image.shape[1], image.shape[0]],
        }
        if marker.radius_set:
            result["slot_radius_px"] = round(marker.radius, 1)
        result["slots"] = marker.slots()
        default_out = DATA_DIR / f"slots_{args.stage}.json"

    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("wrote %d point(s) to %s", len(marker.points), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
