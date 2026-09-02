"""Label lid/no-lid ground truth for detected crucibles, by hand.

    python -m tools.review_lids --images capture/second_iteration/crucibles/undistored/

Why this exists: pipeline/features.py:lid_score() measures local contrast in
the middle of a detected crucible, and across the reviewed crucible set it
separates cleanly outside a score band of roughly 300-650 - but nothing
confirms which side of "has a lid" that band actually falls on, because
nobody has said which jars are really lidded. This walks every detection,
pre-labels the obvious ones from the score so only real judgment calls need
a click, and reports whether a clean score threshold exists once you're done.

Controls:
    left click on a circle    toggle its label: lid <-> open
    n / ENTER       next image, keeping this image's labels
    s               save and quit
    q / ESC         quit without saving

Circles with a dashed dark ring are in the ambiguous score band (300-650)
and were only guessed at - look at those first.

Output, keyed by filename:

    {
      "<filename>": [{"cx": .., "cy": .., "r": .., "score": .., "label": "lid"|"open"}],
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
from pipeline.features import detect_crucibles, lid_score

log = logging.getLogger("review_lids")

WINDOW = "review lids - click to toggle lid/open, s save, q quit"
OUT_FILE = DATA_DIR / "lid_review.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

LID_BGR = (90, 220, 90)
OPEN_BGR = (220, 150, 60)
UNCERTAIN_BAND = (300.0, 650.0)  # lid_score()'s own fuzzy band - see its docstring


def guess_label(score: float) -> str:
    lo, hi = UNCERTAIN_BAND
    return "lid" if score >= (lo + hi) / 2 else "open"


class LidReviewFrame:
    """One image's detections, each with a score-based guess and a label."""

    def __init__(self, image: np.ndarray, name: str) -> None:
        self.image = image
        self.name = name
        self.marks = []
        for cx, cy, r in detect_crucibles(image):
            score = lid_score(image, cx, cy, r)
            self.marks.append({"cx": cx, "cy": cy, "r": r, "score": score,
                                "label": guess_label(score)})

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not self.marks:
            return
        hits = [m for m in self.marks
                if math.hypot(m["cx"] - x, m["cy"] - y) <= max(m["r"], 30)]
        if not hits:
            return
        nearest = min(hits, key=lambda m: math.hypot(m["cx"] - x, m["cy"] - y))
        nearest["label"] = "open" if nearest["label"] == "lid" else "lid"

    def render(self) -> np.ndarray:
        out = self.image.copy()
        lo, hi = UNCERTAIN_BAND
        for m in self.marks:
            c = (int(round(m["cx"])), int(round(m["cy"])))
            r = int(round(m["r"]))
            color = LID_BGR if m["label"] == "lid" else OPEN_BGR
            cv2.circle(out, c, r, color, 3, cv2.LINE_AA)
            if lo <= m["score"] <= hi:
                cv2.circle(out, c, r + 10, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.putText(out, f"{m['label']} {m['score']:.0f}",
                        (c[0] + r + 4, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2, cv2.LINE_AA)

        n_lid = sum(1 for m in self.marks if m["label"] == "lid")
        n_uncertain = sum(1 for m in self.marks if lo <= m["score"] <= hi)
        banner = (f"{self.name}   {n_lid} lid  {len(self.marks) - n_lid} open  "
                  f"{n_uncertain} in the uncertain band (dark ring)   "
                  f"[click=toggle, n next, s save]")
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (20, 22, 26), -1)
        cv2.putText(out, banner, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (223, 227, 234), 2, cv2.LINE_AA)
        return out

    def result(self) -> list[dict]:
        return [
            {"cx": round(m["cx"], 1), "cy": round(m["cy"], 1), "r": round(m["r"], 1),
             "score": round(m["score"], 1), "label": m["label"]}
            for m in self.marks
        ]


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

    try:
        index = 0
        while index < len(frames):
            name, image = frames[index]
            frame = LidReviewFrame(image, name)
            cv2.setMouseCallback(WINDOW, frame.on_mouse)

            action = _review_one(frame)
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

    entries = {k: v for k, v in result.items() if isinstance(v, list)}
    lid_scores = [m["score"] for v in entries.values() for m in v if m["label"] == "lid"]
    open_scores = [m["score"] for v in entries.values() for m in v if m["label"] == "open"]
    log.info("wrote %d entries to %s (%d lid, %d open)",
             len(entries), out_path, len(lid_scores), len(open_scores))
    if lid_scores and open_scores:
        min_lid, max_open = min(lid_scores), max(open_scores)
        if min_lid > max_open:
            log.info("clean separation: every 'open' scores <= %.0f, every "
                      "'lid' scores >= %.0f - a threshold anywhere in between "
                      "works, e.g. %.0f", max_open, min_lid,
                      (min_lid + max_open) / 2)
        else:
            overlap = [s for s in open_scores if s > min_lid] + \
                      [s for s in lid_scores if s < max_open]
            log.warning("no clean threshold: 'open' scores go up to %.0f and "
                        "'lid' scores go down to %.0f - %d label(s) sit in "
                        "that overlap, lid_score() alone can't separate them",
                        max_open, min_lid, len(overlap))
    return 0


def _review_one(frame: LidReviewFrame) -> str:
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


if __name__ == "__main__":
    sys.exit(main())
