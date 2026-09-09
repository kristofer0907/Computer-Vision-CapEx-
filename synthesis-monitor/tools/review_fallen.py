"""Score capture/fail_safe for tip-over and report the class separation.

    python -m tools.review_fallen
    python -m tools.review_fallen --overlay out/          # write marked frames

Why this exists: pipeline/features.py:rim_circularity() is the tip-over
measure and DETECTION.fallen_circularity_threshold is the number that turns
it into a verdict. Neither came from theory - both came from the seven frames
in capture/fail_safe, which are the only pictures of a tipped crucible this
project has. This is the script that produced them, so re-running them on a
larger set is a command rather than an afternoon.

Per frame it scores all fifteen plate positions in FAIL_SAFE_SLOTS, looks the
label up in FAIL_SAFE_TRUTH, and at the end prints each class' range and
whether the threshold still sits in the gap. --overlay draws the verdict on
the frame, which is the faster way to see a disagreement.

Two things to know about what the numbers mean.

The score is anchored on where a crucible is *expected* to be, not on a
detection. That is deliberate: detect_crucibles() finds none of the three
tipped crucibles in these frames - a crucible on its side stops being a
circle - so a check that only looks at detections would never see any of
them. --detections prints that gap rather than hiding it. The corollary is
that a crucible vanishing from the detections is itself the loudest tip-over
signal, and that belongs to lineage, which is what knows a crucible was there
a frame ago.

The evidence is six objects. Three tipped and three upright crucibles, each
in seven near-identical frames of one scene under one lighting setup. The gap
is wide and it is the same in every frame, but 105 rows here are not 105
samples, and a threshold from six objects is a working default.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DETECTION
from pipeline.features import detect_crucibles, rim_circularity, rim_gradient

log = logging.getLogger(__name__)

IMAGE_DIR = Path("capture/fail_safe")

# The 3x5 crucible plate in these frames, in full-capture pixels. Fitted by
# least squares to the bright slot-hole centroids over all seven frames
# (1.9 px mean residual, 5.0 px worst), not hand-marked - the measure wants
# the anchor within a few px and hand-marking is not reliably that good. See
# data/slots_storing.json's note for the same lesson on the storing zone.
_ORIGIN = (2181.9, 1356.2)
_COL = (135.1, 1.4)      # step per column
_ROW = (-0.8, 160.0)     # step per row
FAIL_SAFE_SLOTS = {
    r * 3 + c: (_ORIGIN[0] + c * _COL[0] + r * _ROW[0],
                _ORIGIN[1] + c * _COL[1] + r * _ROW[1])
    for r in range(5) for c in range(3)
}

# Hand-labelled off the frames. Slot 3 holds an upright crucible for the first
# three frames and is empty afterwards - it is lifted off the plate between
# 16:04:31 and 16:05:04 - so its label depends on the frame index.
FAIL_SAFE_TRUTH = {2: "fallen", 9: "fallen", 12: "fallen",
                   11: "upright", 14: "upright"}
SLOT_3_UPRIGHT_UNTIL = 3

#: A detection further than this from every slot centre is not on the plate.
MATCH_RADIUS_PX = 90.0


def label_for(slot: int, frame_index: int) -> str:
    if slot == 3:
        return "upright" if frame_index < SLOT_3_UPRIGHT_UNTIL else "empty"
    return FAIL_SAFE_TRUTH.get(slot, "empty")


def nearest_slot(cx: float, cy: float) -> int | None:
    best, best_d = None, MATCH_RADIUS_PX
    for slot, (sx, sy) in FAIL_SAFE_SLOTS.items():
        d = float(np.hypot(cx - sx, cy - sy))
        if d < best_d:
            best, best_d = slot, d
    return best


def frame_paths(image_dir: Path) -> list[Path]:
    # The undistorted duplicate of frame 0 is a second projection of a frame
    # already in the set; scoring both would count one crucible twice.
    paths = sorted(p for p in image_dir.glob("*.jpg")
                   if "undistorted" not in p.name)
    if not paths:
        raise SystemExit(f"no images in {image_dir}")
    return paths


def draw(image: np.ndarray, slot: int, score: float, fallen: bool) -> None:
    cx, cy = FAIL_SAFE_SLOTS[slot]
    colour = (0, 0, 220) if fallen else (0, 170, 0)
    radius = 72 if fallen else 60
    cv2.circle(image, (int(cx), int(cy)), radius, colour, 5 if fallen else 2)
    cv2.putText(image, f"{score:.2f}", (int(cx) - 38, int(cy) + radius + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2)
    if fallen:
        cv2.putText(image, "FALLEN", (int(cx) - 72, int(cy) - 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)


def score_dir(image_dir: Path, overlay_dir: Path | None = None,
              show_detections: bool = False) -> dict[str, list[float]]:
    """Score every plate position in every frame. Returns label -> scores."""
    limit = DETECTION.fallen_circularity_threshold
    scores: dict[str, list[float]] = {}
    for index, path in enumerate(frame_paths(image_dir)):
        image = cv2.imread(str(path))
        if image is None:
            log.warning("unreadable: %s", path)
            continue
        mag = rim_gradient(image)
        marked = image.copy() if overlay_dir else None
        for slot, (cx, cy) in sorted(FAIL_SAFE_SLOTS.items()):
            label = label_for(slot, index)
            score = rim_circularity(image, cx, cy, _mag=mag)
            scores.setdefault(label, []).append(score)
            verdict = "fallen" if score < limit else "upright-or-empty"
            flag = "" if (score < limit) == (label == "fallen") else "   <-- WRONG"
            print(f"{path.name}  slot {slot:2d}  truth={label:8s} "
                  f"circularity={score:.3f}  reads {verdict}{flag}")
            if marked is not None:
                draw(marked, slot, score, score < limit)
        if show_detections:
            found = {nearest_slot(cx, cy) for cx, cy, _ in detect_crucibles(image)}
            missed = sorted(s for s, l in
                            ((s, label_for(s, index)) for s in FAIL_SAFE_SLOTS)
                            if l == "fallen" and s not in found)
            print(f"{path.name}  detect_crucibles missed tipped slots: "
                  f"{missed or 'none'}")
        if marked is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_dir / path.name), marked)
    return scores


def report(scores: dict[str, list[float]]) -> int:
    limit = DETECTION.fallen_circularity_threshold
    print()
    for label in ("upright", "fallen", "empty"):
        v = scores.get(label)
        if not v:
            continue
        print(f"{label:8s} n={len(v):3d}  min={min(v):.3f} "
              f"median={float(np.median(v)):.3f} max={max(v):.3f}")

    fallen = scores.get("fallen")
    # An empty slot scores like an upright one - its own rim is a circle -
    # so both are on the same side of the threshold and both belong here.
    standing = (scores.get("upright") or []) + (scores.get("empty") or [])
    if not fallen or not standing:
        print("\nboth classes are needed to judge the threshold")
        return 1
    gap_lo, gap_hi = max(fallen), min(standing)
    print(f"\ngap: tipped tops out at {gap_lo:.3f}, upright-or-empty bottoms "
          f"out at {gap_hi:.3f}")
    if gap_lo >= gap_hi:
        print(f"the classes OVERLAP - no threshold separates them, so "
              f"{limit:.2f} cannot be right")
        return 1
    where = "inside" if gap_lo < limit < gap_hi else "OUTSIDE"
    print(f"threshold {limit:.2f} sits {where} the gap; the midpoint would "
          f"be {(gap_lo + gap_hi) / 2:.3f}")
    wrong = sum(s < limit for s in standing) + sum(s >= limit for s in fallen)
    total = len(standing) + len(fallen)
    print(f"{total - wrong}/{total} scored positions classified correctly")
    return 0 if wrong == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=Path, default=IMAGE_DIR)
    ap.add_argument("--overlay", type=Path, default=None,
                    help="directory to write marked-up frames into")
    ap.add_argument("--detections", action="store_true",
                    help="also report which tipped crucibles "
                         "detect_crucibles() fails to find")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return report(score_dir(args.images, args.overlay, args.detections))


if __name__ == "__main__":
    sys.exit(main())
