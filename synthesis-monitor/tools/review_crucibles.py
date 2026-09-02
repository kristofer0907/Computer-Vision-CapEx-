"""Review pipeline.features.detect_crucibles() output by hand.

    python -m tools.review_crucibles --images capture/second_iteration/crucibles/undistored/

Controls:
    left click on a detected circle    toggle it confirmed / rejected (red X)
    left click+drag on empty space     draw a circle around a missed crucible
    left click (no drag) on empty space   drop a missed-crucible circle at the
                                           default radius (+/- to change it)
    right click near a drawn circle    remove that manual mark
    + / -           grow / shrink the default radius for new circles
    c               clear this image's manual (missed) marks
    n / ENTER       next image, keeping this image's review
    s               save and quit
    q / ESC         quit without saving

Why this exists: pipeline/features.py:detect_crucibles() is tuned against a
handful of frames inspected by hand, not against ground truth. This walks
every frame, shows what it found, and lets a human say what's actually
right - correct detections need zero clicks (everything is "confirmed"
unless you mark it otherwise), only mistakes and misses need attention.

Output, keyed by filename like tools/mark_vials.py's data/vials.json:

    {
      "capture_01.jpg": {
        "detections": [{"cx": .., "cy": .., "r": .., "status": "confirmed"}],
        "missed":     [{"cx": .., "cy": .., "r": ..}]
      },
      "_image_size": [4056, 3040]
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
from pipeline.features import detect_crucibles

log = logging.getLogger("review")

WINDOW = "review crucibles - click detections, drag misses, s save, q quit"
OUT_FILE = DATA_DIR / "crucible_review.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CONFIRMED_BGR = (255, 0, 255)   # matches detect_crucibles' own debug color
REJECTED_BGR = (40, 40, 220)
MISSED_BGR = (90, 220, 90)
DRAG_BGR = (60, 220, 230)
MIN_CLICK_DRAG_PX = 10           # below this, a drag is treated as a plain click


class ReviewFrame:
    """One image's detections plus the human's corrections."""

    def __init__(self, image: np.ndarray, name: str, default_radius: float) -> None:
        self.image = image
        self.name = name
        self.default_radius = default_radius
        self.detections = [
            {"cx": cx, "cy": cy, "r": r, "status": "confirmed"}
            for cx, cy, r in detect_crucibles(image)
        ]
        self.missed: list[dict] = []
        self._dragging = False
        self._drag_start = (0.0, 0.0)
        self._drag_now = (0.0, 0.0)

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._detection_at(x, y)
            if hit is not None:
                hit["status"] = ("rejected" if hit["status"] == "confirmed"
                                  else "confirmed")
                return
            self._dragging = True
            self._drag_start = (float(x), float(y))
            self._drag_now = (float(x), float(y))
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._drag_now = (float(x), float(y))
        elif event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            cx, cy = self._drag_start
            r = math.hypot(x - cx, y - cy)
            if r < MIN_CLICK_DRAG_PX:
                r = self.default_radius
            self.missed.append({"cx": cx, "cy": cy, "r": r})
        elif event == cv2.EVENT_RBUTTONDOWN and self.missed:
            nearest = min(self.missed,
                          key=lambda m: math.hypot(m["cx"] - x, m["cy"] - y))
            if math.hypot(nearest["cx"] - x, nearest["cy"] - y) <= nearest["r"] + 20:
                self.missed.remove(nearest)

    def _detection_at(self, x: float, y: float) -> dict | None:
        """The detection whose circle contains (x, y), nearest center first."""
        hits = [d for d in self.detections
                if math.hypot(d["cx"] - x, d["cy"] - y) <= max(d["r"], 30)]
        if not hits:
            return None
        return min(hits, key=lambda d: math.hypot(d["cx"] - x, d["cy"] - y))

    def render(self) -> np.ndarray:
        out = self.image.copy()
        for i, d in enumerate(self.detections):
            c = (int(round(d["cx"])), int(round(d["cy"])))
            r = int(round(d["r"]))
            color = CONFIRMED_BGR if d["status"] == "confirmed" else REJECTED_BGR
            cv2.circle(out, c, r, color, 3, cv2.LINE_AA)
            if d["status"] == "rejected":
                cv2.line(out, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), REJECTED_BGR, 3)
                cv2.line(out, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), REJECTED_BGR, 3)
            cv2.putText(out, str(i), (c[0] + r + 4, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        for i, m in enumerate(self.missed):
            c = (int(round(m["cx"])), int(round(m["cy"])))
            cv2.circle(out, c, int(round(m["r"])), MISSED_BGR, 3, cv2.LINE_AA)
            cv2.putText(out, f"M{i}", (c[0] + int(m["r"]) + 4, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, MISSED_BGR, 2, cv2.LINE_AA)

        if self._dragging:
            cx, cy = self._drag_start
            r = math.hypot(self._drag_now[0] - cx, self._drag_now[1] - cy)
            cv2.circle(out, (int(cx), int(cy)), int(max(r, self.default_radius)),
                       DRAG_BGR, 3, cv2.LINE_AA)

        n_confirmed = sum(1 for d in self.detections if d["status"] == "confirmed")
        n_rejected = len(self.detections) - n_confirmed
        banner = (f"{self.name}   {n_confirmed} confirmed  {n_rejected} rejected  "
                  f"{len(self.missed)} missed   r={self.default_radius:.0f}px   "
                  f"[click=toggle, drag=add miss, +/- radius, c clear, n next, s save]")
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (20, 22, 26), -1)
        cv2.putText(out, banner, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (223, 227, 234), 2, cv2.LINE_AA)
        return out

    def result(self) -> dict:
        return {
            "detections": [
                {"cx": round(d["cx"], 1), "cy": round(d["cy"], 1),
                 "r": round(d["r"], 1), "status": d["status"]}
                for d in self.detections
            ],
            "missed": [
                {"cx": round(m["cx"], 1), "cy": round(m["cy"], 1), "r": round(m["r"], 1)}
                for m in self.missed
            ],
        }


def load_images(images: str) -> list[tuple[str, np.ndarray]]:
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of captures to review")
    p.add_argument("--radius", type=float, default=70.0,
                   help="starting radius for manually-added (missed) circles")
    p.add_argument("--out", default=str(OUT_FILE))
    p.add_argument("--merge", action="store_true",
                   help="keep entries already in the output file")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ensure_dirs()

    frames = load_images(args.images)
    log.info("reviewing %d image(s)", len(frames))

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
            frame = ReviewFrame(image, name, radius)
            cv2.setMouseCallback(WINDOW, frame.on_mouse)

            action = _review_one(frame)
            radius = frame.default_radius

            result[name] = frame.result()

            if action == "quit":
                log.info("quit without saving")
                return 1
            if action == "save":
                break
            index += 1
    finally:
        cv2.destroyAllWindows()

    result["_image_size"] = [first_shape[1], first_shape[0]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    entries = {k: v for k, v in result.items() if isinstance(v, dict)}
    log.info("wrote %d entries to %s", len(entries), out_path)
    total_rejected = sum(
        1 for v in entries.values() for d in v["detections"] if d["status"] == "rejected")
    total_missed = sum(len(v["missed"]) for v in entries.values())
    log.info("  %d detections rejected, %d misses drawn in total",
             total_rejected, total_missed)
    return 0


def _review_one(frame: ReviewFrame) -> str:
    """Interactive loop for one image. Returns 'next', 'save' or 'quit'."""
    while True:
        cv2.imshow(WINDOW, frame.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "save"
        if key in (13, 10, ord("n")):
            return "next"
        if key in (ord("+"), ord("=")):
            frame.default_radius += 2
        elif key in (ord("-"), ord("_")):
            frame.default_radius = max(5.0, frame.default_radius - 2)
        elif key == ord("c"):
            frame.missed.clear()


if __name__ == "__main__":
    sys.exit(main())
