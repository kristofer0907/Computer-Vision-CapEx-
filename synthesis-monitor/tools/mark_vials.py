"""Click the vials in your own captures, save them to data/vials.json.

    python -m tools.mark_vials --image capture.jpg
    python -m tools.mark_vials --images captures/          # a whole folder
    python -m tools.mark_vials --image capture.jpg --key default

Controls:
    left click      place a vial at the cursor
    right click     remove the nearest vial
    + / -           grow / shrink the radius (applies to all)
    g               auto-place a grid inside a zone (then nudge by hand)
    c               clear this image
    n / ENTER       next image (multi-image mode)
    s               save and quit
    q / ESC         quit without saving

Why this exists: until a real localiser is written, nothing can find vials in
a real photograph, so a real capture goes through the pipeline showing zero
vials and every downstream check is vacuously fine. Marking them once by hand
turns a folder of your own images into a working end-to-end fixture - real
pixels, real colours, real illumination gradients - for the tracker, the ROI
crops, the masks, the features and the detectors.

It doubles as the reference set later: run a real localiser over the same
images and compare against these marks to get an actual localisation error,
rather than an impression.

Marks are stored in pixels of the image they were made on, keyed by filename,
with "default" applying to any image without its own entry.
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

from config import GEOMETRY, VIALS_FILE, ZONES, ensure_dirs
from pipeline.zones import ZoneMap, px_to_mm

log = logging.getLogger("mark")

WINDOW = "mark vials - click to place, s save, q quit"
VIAL_BGR = (110, 199, 98)
ZONE_BGR = (90, 110, 130)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Marker:
    """One image's worth of marks, plus the mouse handling."""

    def __init__(self, image: np.ndarray, name: str, radius: float) -> None:
        self.image = image
        self.name = name
        self.radius = radius
        self.points: list[tuple[float, float]] = []
        self.zones = ZoneMap(image.shape[1], image.shape[0])

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            nearest = min(self.points,
                          key=lambda p: math.hypot(p[0] - x, p[1] - y))
            self.points.remove(nearest)

    def auto_grid(self, zone: str | None = None, cols: int = 9,
                  rows: int = 2) -> None:
        """Drop a rows x cols grid inside a zone, as a starting point.

        The filling rack is a regular 2x9 array, so placing 18 points by hand
        is wasted effort - place the grid, then drag the few that are off.
        """
        name = zone or (self.zones.names[0] if self.zones.names else None)
        if name is None:
            log.warning("no zones configured, cannot auto-grid")
            return
        x0, y0, x1, y1 = self.zones.bounds(name)
        for r in range(rows):
            for c in range(cols):
                self.points.append((
                    x0 + (x1 - x0) * (c + 0.5) / cols,
                    y0 + (y1 - y0) * (r + 0.5) / rows,
                ))
        log.info("placed %d points in %s - nudge them by hand", rows * cols, name)

    def render(self) -> np.ndarray:
        out = self.zones.draw(self.image, ZONE_BGR)
        for i, (x, y) in enumerate(self.points):
            c = (int(round(x)), int(round(y)))
            cv2.circle(out, c, int(round(self.radius)), VIAL_BGR, 1, cv2.LINE_AA)
            cv2.drawMarker(out, c, VIAL_BGR, cv2.MARKER_CROSS, 6, 1)
            cv2.putText(out, str(i), (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, VIAL_BGR, 1, cv2.LINE_AA)

        banner = (f"{self.name}   {len(self.points)} vials   "
                  f"r={self.radius:.0f}px ({px_to_mm(self.radius) * 2:.0f}mm dia)"
                  f"   [+/- radius, g grid, c clear, n next, s save]")
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (20, 22, 26), -1)
        cv2.putText(out, banner, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (223, 227, 234), 1, cv2.LINE_AA)
        return out

    def marks(self) -> list[dict]:
        return [{"index": i, "cx": round(x, 1), "cy": round(y, 1),
                 "radius": round(self.radius, 1)}
                for i, (x, y) in enumerate(self.points)]


def load_images(image: str | None, images: str | None) -> list[tuple[str, np.ndarray]]:
    if image:
        img = cv2.imread(image)
        if img is None:
            raise SystemExit(f"could not read {image}")
        return [(Path(image).name, img)]

    folder = Path(images)
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"no images in {folder}")
    out = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            log.warning("skipping unreadable %s", f)
            continue
        out.append((f.name, img))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="one capture to mark")
    src.add_argument("--images", help="a folder of captures to mark in sequence")
    p.add_argument("--key", help="store under this key instead of the filename "
                                 "(use 'default' to apply to every image)")
    p.add_argument("--radius", type=float, default=GEOMETRY.vial_radius_px,
                   help="starting vial radius in pixels")
    p.add_argument("--grid", nargs="?", const="", metavar="ZONE",
                   help="pre-place a 2x9 grid in this zone")
    p.add_argument("--out", default=str(VIALS_FILE))
    p.add_argument("--merge", action="store_true",
                   help="keep entries already in the output file")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ensure_dirs()

    frames = load_images(args.image, args.images)
    log.info("marking %d image(s); zones are %s",
             len(frames), "traced" if ZONES.calibrated else "PLACEHOLDERS")

    out_path = Path(args.out)
    result: dict = {}
    if args.merge and out_path.exists():
        result = json.loads(out_path.read_text())
        log.info("merging into %d existing entries", len(result) - 1)

    first_shape = frames[0][1].shape
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1500, first_shape[1]), min(950, first_shape[0]))

    radius = args.radius
    try:
        index = 0
        while index < len(frames):
            name, image = frames[index]
            marker = Marker(image, name, radius)
            if args.grid is not None:
                marker.auto_grid(args.grid or None)
            cv2.setMouseCallback(WINDOW, marker.on_mouse)

            action = _edit_one(marker)
            radius = marker.radius

            key = args.key or name
            if marker.points:
                result[key] = marker.marks()

            if action == "quit":
                log.info("quit without saving")
                return 1
            if action == "save":
                break
            index += 1
    finally:
        cv2.destroyAllWindows()

    if not any(isinstance(v, list) for v in result.values()):
        log.warning("nothing marked, not writing")
        return 1

    result["_image_size"] = [first_shape[1], first_shape[0]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    entries = {k: v for k, v in result.items() if isinstance(v, list)}
    log.info("wrote %d entries to %s", len(entries), out_path)
    for k, v in entries.items():
        log.info("  %-24s %d vials", k, len(v))
    log.info("now: python -m tools.replay --rgb file --file <your images> "
             "--localizer manual")
    return 0


def _edit_one(marker: Marker) -> str:
    """Interactive loop for one image. Returns 'next', 'save' or 'quit'."""
    while True:
        cv2.imshow(WINDOW, marker.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "save"
        if key in (13, 10, ord("n")):
            return "next"
        if key in (ord("+"), ord("=")):
            marker.radius += 1
        elif key in (ord("-"), ord("_")):
            marker.radius = max(2.0, marker.radius - 1)
        elif key == ord("g"):
            marker.auto_grid()
        elif key == ord("c"):
            marker.points.clear()


if __name__ == "__main__":
    sys.exit(main())
