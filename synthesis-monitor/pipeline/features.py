"""Per-vial feature extraction.  ***YOURS TO IMPLEMENT.***

The runner calls extract() once per tracked vial per analysis frame and passes
the returned dict, untouched, to every detector, into VialReport.features, and
into storage as JSON. Nothing between here and the database knows or cares
what the keys are, so adding a feature needs no change anywhere else.

Contract, and it is the only thing that matters:

    extract(...) -> dict[str, float]

Plain floats. No numpy scalars (they survive pickling but serialise to JSON
badly), no nested structures, no images. Same keys every frame for a given
vial where possible - the batch statistics compare vial to vial within one
frame, so a key present on 12 vials and missing on 6 quietly shrinks the
sample the median is taken over.

What you have to work with, all handed to you:

    crop    BGR copy of the region around the vial, already cropped
    mask    uint8, 255 over the liquid disc, rim excluded
    prev    the same vial's crop from a previous frame, resized to match
            `crop`, or None on the first sighting of that vial

The obvious first set, from the design notes: mean/median HSV inside the mask,
texture variance, edge density, brightness, and a frame-difference magnitude
against `prev`. Do not guess thresholds here - this module produces numbers,
the detectors decide what they mean, and neither can be calibrated until there
are real captured vial images to calibrate against.

Flat-field correction belongs here or upstream of here, not in the detectors:
the LED panel has a real illumination gradient across the platform, so an
uncorrected brightness feature partly encodes *where on the bench a vial is*,
and batch-median scoring would then flag the far end of the platform as
anomalous every single run.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from pipeline.types import Track

log = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Turns one vial's pixels into a flat dict of numbers."""

    name: str = "unset"

    def start(self) -> None:
        """Called once before the first frame."""

    def stop(self) -> None:
        """Called once at shutdown. Must be safe to call twice."""

    @abstractmethod
    def extract(self, crop: np.ndarray, mask: np.ndarray, track: Track,
                prev: np.ndarray | None = None) -> dict[str, float]:
        """Features for one vial in one frame.

        crop  BGR uint8, HxWx3
        mask  uint8 HxW, 255 over the liquid disc
        track the vial's Track, for stage / age / dwell-dependent features
        prev  the same vial's previous crop, resized to `crop`, or None
        """

    def frame_context(self, image: np.ndarray) -> None:
        """Optional hook, called once per frame before any extract() call.

        Use it for anything that is per-frame rather than per-vial: computing
        a flat-field gain map, a background model, a global white balance
        reference off a known grey patch on the bench.
        """


class NullFeatureExtractor(FeatureExtractor):
    """Returns nothing. The pipeline runs, the detectors get empty dicts.

    This is what ships until extract() above is written. It keeps the runner,
    storage and dashboard exercisable without pretending to measure anything.
    """

    name = "null"

    def extract(self, crop, mask, track, prev=None) -> dict[str, float]:
        return {}


def create_extractor(name: str = "auto") -> FeatureExtractor:
    """Build a feature extractor by name. Register yours here."""
    key = (name or "auto").lower()
    if key in ("auto", "null", "none"):
        return NullFeatureExtractor()
    raise ValueError(
        f"unknown feature extractor {name!r}. Implement it in "
        "pipeline/features.py and register it here."
    )



import cv2


# One plate slot has a double-walled jar that never clears the main Hough
# threshold. Rechecked here with a looser threshold instead of loosening
# globally, which would pull in empty pegboard holes. Fixes 8 frames, no
# new false positives (data/crucible_review.json).
_SECOND_PASS_ROI = (2050, 1250, 2450, 1650)  # x0, y0, x1, y1


def _hough_pass(gray0: np.ndarray, min_r: int, max_r: int, param1: int,
                 param2: int, min_dist: int,
                 min_mean: float) -> list[tuple[float, float, float]]:
    """Blur + close + HoughCircles + a brightness floor, on one grayscale image.

    The brightness floor kills a dark bolt elsewhere in frame that otherwise
    reads as a circle (real crucibles score >=54 mean gray, the bolt ~9).
    """
    gray = cv2.GaussianBlur(gray0, (9, 9), 2)
    # bridges small gaps in a crucible's rim (glare, thin reflection breaks)
    # without being big enough to fill in the gripper's much larger opening
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_dist,
                                param1=param1, param2=param2,
                                minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return []

    h, w = gray0.shape
    out: list[tuple[float, float, float]] = []
    for cx, cy, r in circles[0]:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r * 0.85), 255, -1)
        if gray0[mask == 255].mean() >= min_mean:
            out.append((float(cx), float(cy), float(r)))
    return out


def detect_crucibles(img: np.ndarray, min_r: int = 50, max_r: int = 100,
                      param1: int = 90, param2: int = 55, min_dist: int = 100,
                      min_mean: float = 35.0) -> list[tuple[float, float, float]]:
    """Find standing crucibles (glass or metal jars) in one full-frame image.

    minRadius/maxRadius/param2 exclude empty pegboard holes - don't loosen
    globally for a rare miss, use tools/review_crucibles.py instead.

    Returns (cx, cy, r) in pixels, one per accepted circle.
    """
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = _hough_pass(gray0, min_r, max_r, param1, param2, min_dist, min_mean)

    x0, y0, x1, y1 = _SECOND_PASS_ROI
    roi_hits = _hough_pass(gray0[y0:y1, x0:x1], min_r=60, max_r=max_r,
                            param1=param1, param2=35, min_dist=min_dist,
                            min_mean=100.0)
    for cx, cy, r in roi_hits:
        cx, cy = cx + x0, cy + y0
        if all((cx - px) ** 2 + (cy - py) ** 2 > 40 ** 2 for px, py, _ in out):
            out.append((cx, cy, r))
    return out


def lid_score(img: np.ndarray, cx: float, cy: float, r: float,
              frac: float = 0.55) -> float:
    """Local contrast (Laplacian variance) in the central `frac` of a crucible.

    A lid breaks up the smooth open-jar interior whether it reads bright
    (metal jars) or dark (glass) - contrast catches both, brightness alone
    didn't. Threshold calibrated against data/lid_review.json.
    """
    h, w = img.shape[:2]
    rr = r * frac
    x0, y0 = max(int(cx - rr), 0), max(int(cy - rr), 0)
    x1, y1 = min(int(cx + rr), w), min(int(cy + rr), h)
    crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(crop, cv2.CV_32F, ksize=3).var())


# Tuned on data/lid_review.json (225 labels): 8 misses, all open->lid, from
# glare in one tray slot plus a flush glass disc. Fix with a better feature,
# not by nudging this number.
LID_SCORE_THRESHOLD = 585.0


def has_lid(img: np.ndarray, cx: float, cy: float, r: float,
            threshold: float = LID_SCORE_THRESHOLD) -> bool:
    """Best-effort yes/no over lid_score() - see LID_SCORE_THRESHOLD."""
    return lid_score(img, cx, cy, r) >= threshold


def segment(path: str) -> None:
    """Manual sanity check: run detect_crucibles on one file and show it."""
    img = cv2.imread(path)
    for cx, cy, r in detect_crucibles(img):
        center = (int(cx), int(cy))
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        cv2.circle(img, center, int(r), (255, 0, 255), 3)

    cv2.namedWindow("detected circles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detected circles", 1200, 800)  # adjust to whatever fits your screen

    cv2.imshow("detected circles", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    segment("capture/second_iteration/crucibles/undistored/20260828_164353_0000.jpg")