"""Per-crucible feature extraction.  ***YOURS TO IMPLEMENT.***

The runner calls extract() once per tracked crucible per analysis frame and passes
the returned dict, untouched, to every detector, into CrucibleReport.features, and
into storage as JSON. Nothing between here and the database knows or cares
what the keys are, so adding a feature needs no change anywhere else.

Contract, and it is the only thing that matters:

    extract(...) -> dict[str, float]

Plain floats. No numpy scalars (they survive pickling but serialise to JSON
badly), no nested structures, no images. Same keys every frame for a given
crucible where possible - the batch statistics compare crucible to crucible within one
frame, so a key present on 12 crucibles and missing on 6 quietly shrinks the
sample the median is taken over.

What you have to work with, all handed to you:

    crop    BGR copy of the region around the crucible, already cropped
    mask    uint8, 255 over the liquid disc, rim excluded
    prev    the same crucible's crop from a previous frame, resized to match
            `crop`, or None on the first sighting of that crucible

The obvious first set, from the design notes: mean/median HSV inside the mask,
texture variance, edge density, brightness, and a frame-difference magnitude
against `prev`. Do not guess thresholds here - this module produces numbers,
the detectors decide what they mean, and neither can be calibrated until there
are real captured crucible images to calibrate against.

Flat-field correction belongs here or upstream of here, not in the detectors:
the LED panel has a real illumination gradient across the platform, so an
uncorrected brightness feature partly encodes *where on the bench a crucible is*,
and batch-median scoring would then flag the far end of the platform as
anomalous every single run.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from config import GEOMETRY
from pipeline.types import Track

log = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Turns one crucible's pixels into a flat dict of numbers."""

    name: str = "unset"

    def start(self) -> None:
        """Called once before the first frame."""

    def stop(self) -> None:
        """Called once at shutdown. Must be safe to call twice."""

    @abstractmethod
    def extract(self, crop: np.ndarray, mask: np.ndarray, track: Track,
                prev: np.ndarray | None = None) -> dict[str, float]:
        """Features for one crucible in one frame.

        crop  BGR uint8, HxWx3
        mask  uint8 HxW, 255 over the liquid disc
        track the crucible's Track, for stage / age / dwell-dependent features
        prev  the same crucible's previous crop, resized to `crop`, or None
        """

    def frame_context(self, image: np.ndarray) -> None:
        """Optional hook, called once per frame before any extract() call.

        Use it for anything that is per-frame rather than per-crucible: computing
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

    # The ROI is in full-capture pixels. On a smaller frame - a downscaled
    # preview, the mock platform - the slice comes back empty and every cv2
    # call below it throws, so skip the pass rather than crash: it is a
    # recheck of one known-awkward jar, not something the result depends on.
    h, w = gray0.shape
    x0, y0, x1, y1 = _SECOND_PASS_ROI
    if x1 > w or y1 > h:
        return out

    roi_hits = _hough_pass(gray0[y0:y1, x0:x1], min_r=60, max_r=max_r,
                            param1=param1, param2=35, min_dist=min_dist,
                            min_mean=100.0)
    for cx, cy, r in roi_hits:
        cx, cy = cx + x0, cy + y0
        if all((cx - px) ** 2 + (cy - py) ** 2 > 40 ** 2 for px, py, _ in out):
            out.append((cx, cy, r))
    return out


def lid_score(img: np.ndarray, cx: float, cy: float, r: float | None = None,
              half_px: float | None = None) -> float:
    """Mean gradient magnitude in a fixed window on a crucible's centre.

    A lid breaks up the smooth open-jar interior whether it reads bright
    (metal jars) or dark (glass), so an edge measure catches both where
    brightness alone did not.

    The window is a fixed number of pixels rather than a fraction of `r`.
    Hough's radius wanders - 60 to 78 px across the labelled set for jars of
    one size - and scaling the window with it means a jar that happened to
    measure large gets more of its own smooth rim averaged in and scores
    lower for it. That is what put one lidded crucible at 578 against a 585
    threshold while its identical neighbours sat at 799-955.

    `r` is accepted and ignored, so existing callers keep working.

    Measured on data/lid_review.json (225 hand labels): 99.1%, 98.7% under
    5-fold cross-validation, against 96.4% for the Laplacian-variance version
    this replaces. Lids run 56.0-82.7, open jars 4.8-67.7 - still overlapping,
    but the two remaining errors are both open jars read as lidded, and every
    lid is caught.
    """
    half_px = LID_WINDOW_PX if half_px is None else half_px
    h, w = img.shape[:2]
    x0, y0 = max(int(cx - half_px), 0), max(int(cy - half_px), 0)
    x1, y1 = min(int(cx + half_px), w), min(int(cy + half_px), h)
    crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if crop.size == 0:
        return 0.0
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)))


# Half-width of the window lid_score() samples, in pixels. Swept over the
# labelled set: 18, 20 and 22 px all score 98.7% cross-validated, so this
# sits in the middle of a plateau rather than on a peak.
LID_WINDOW_PX = 20.0

# Tuned on data/lid_review.json (225 labels): 2 misses, both open->lid.
# Fix with a better feature, not by nudging this number.
LID_SCORE_THRESHOLD = 55.7


# --------------------------------------------------------------------------
# Tip-over
# --------------------------------------------------------------------------
# Angular samples around the crucible, and the radial band searched for its
# rim, as a multiple of the crucible radius. The band starts below 1.0 and
# ends well above it because Hough's radius wanders and because the rim of a
# crucible sitting in a plate slot is only a little wider than the slot.
RIM_ANGLES = 120
RIM_BAND = (0.85, 1.60)
# Sobel magnitude at or above this counts as rim. Swept over the fail-safe
# set: 60 is the middle of the plateau. At 40 every class saturates at 1.0,
# at 100 real upright rims start dropping out.
RIM_EDGE_THRESHOLD = 60.0
# The centre is re-tried over a +-32 px grid at 4 px steps and the best score
# kept. The reason is parallax, not sloppiness: the rim sits ~40 mm above the
# plate, so away from the optical axis it projects outward from the slot it
# stands in - 28 px at the far corner of the fail-safe plate. Without the
# search an upright crucible scored from its slot centre reads 0.62-0.77,
# indistinguishable from a tipped one; with it, 0.97-1.00 against 0.73-0.88.
RIM_SEARCH_PX = 32.0
RIM_SEARCH_STEP_PX = 4


def rim_circularity(img: np.ndarray, cx: float, cy: float,
                    r: float | None = None,
                    search_px: float = RIM_SEARCH_PX,
                    edge_threshold: float = RIM_EDGE_THRESHOLD,
                    _mag: np.ndarray | None = None) -> float:
    """How much of a full circle the crucible's rim traces, 0..1.

    An upright crucible is a cylinder seen down its axis: its rim closes a
    complete circle about the centre. A crucible lying on its side presents
    the cylinder wall instead - long arcs down one side, and a rim ellipse
    displaced by roughly the crucible's height - so no radius about the
    reported centre has edge support all the way round.

    So: sample the Sobel magnitude on a polar grid about (cx, cy), and for
    each candidate radius count the fraction of the 120 angles carrying an
    edge within +-2 px of it. The score is the best radius' fraction. 1.0 is
    an unbroken ring; a tipped crucible leaves a gap wherever its own body,
    not its rim, sits under the sampling circle.

    `r` defaults to the nominal crucible radius from GEOMETRY. The band is
    wide enough that the exact value does not matter - scoring the fail-safe
    set with each detection's own Hough radius and with the single nominal
    radius gives identical numbers to three decimals.

    `_mag` lets a caller scoring many crucibles in one frame compute the
    Sobel magnitude once and pass it in; it is otherwise computed here.

    Measured on capture/fail_safe (7 frames, tools/review_fallen.py),
    scored at each crucible's slot centre: upright 0.97-1.00, tipped
    0.76-0.84. An empty slot also scores 0.97-1.00 - the slot's own rim is a
    perfect circle - so this measure says "upright or nothing", never
    "occupied". Occupancy is the tracker's to know.

    That is three tipped crucibles and three upright ones seen seven times
    each, not 105 independent samples. The gap is wide and stable across
    every frame; it is still six objects in one scene.
    """
    r = GEOMETRY.crucible_radius_px if r is None else r
    mag = rim_gradient(img) if _mag is None else _mag
    lo, hi = RIM_BAND
    radii = np.arange(int(r * lo), int(r * hi) + 1, dtype=np.float32)
    if radii.size == 0:
        return 0.0
    ang = np.arange(RIM_ANGLES, dtype=np.float32) / RIM_ANGLES * 2 * np.pi
    cos_r = np.outer(np.cos(ang), radii)
    sin_r = np.outer(np.sin(ang), radii)

    step = RIM_SEARCH_STEP_PX
    offsets = range(-int(search_px), int(search_px) + 1, step) or (0,)
    best = 0.0
    for dy in offsets:
        for dx in offsets:
            xs = (cx + dx + cos_r).astype(np.float32)
            ys = (cy + dy + sin_r).astype(np.float32)
            polar = cv2.remap(mag, xs, ys, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            hit = polar > edge_threshold
            # +-2 px of radial slack, so a rim slightly out of round still
            # closes rather than being punished once per angle for one offset.
            slack = hit.copy()
            for shift in (-2, -1, 1, 2):
                slack |= np.roll(hit, shift, axis=1)
            best = max(best, float(slack.mean(axis=0).max()))
    return best


def rim_gradient(img: np.ndarray) -> np.ndarray:
    """Sobel magnitude of one frame - what rim_circularity samples.

    Exposed so a caller scoring a whole frame's worth of crucibles computes
    it once instead of once per crucible; on a 4056x3040 frame it costs far
    more than the polar sampling does.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


# Mid-gap between the two classes on capture/fail_safe: upright bottoms out
# at 0.97, tipped tops out at 0.84. Six objects, one scene, one lighting - a
# working threshold, not a calibrated one.
RIM_CIRCULARITY_THRESHOLD = 0.90


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