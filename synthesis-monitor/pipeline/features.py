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


def segment(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    # bridges small gaps in a vial's rim (glare, thin reflection breaks)
    # without being big enough to fill in the gripper's much larger opening
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=90,param2=55,
                               minRadius=50,maxRadius=90)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
    
    cv2.namedWindow("detected circles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detected circles", 1200, 800)  # adjust to whatever fits your screen
    
    cv2.imshow("detected circles", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    segment("/home/kkristjansson/DTU/CAPeX/Computer-Vision-CapEx-/synthesis-monitor/capture/second_iteration/crucibles/20260828_164353_0000_undistorted.jpg")