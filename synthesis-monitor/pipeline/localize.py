"""Crucible localisation.  ***YOURS TO IMPLEMENT.***

Method is settled: classical CV - Hough circles on a flat-field-corrected
frame, in CrucibleLocalizer below, which is what "auto" resolves to. Learned
detectors were considered and dropped: they need labelled data this project
does not have, and the crucibles are near-identical discs on a flat bench,
which is the case classical CV handles well.

What the rest of the pipeline needs from you:

    a list[Detection] in pixel coordinates, per frame, that's all.

Not identity - the tracker assigns that. Not stage - the zone map assigns
that. Not correctness - a detection that turns out to be a reflection will be
dropped by the tracker's min_hits_to_confirm after one frame. Over-detecting
is cheaper than under-detecting here.

Three implementations ship, none of which is an algorithm:

  GroundTruthLocalizer  reads Frame.truth, which only the simulator fills.
                        A test oracle: it lets the tracker, stage machine,
                        storage and dashboard be exercised end to end before
                        this decision is made. Blind on a real frame, by
                        design.

  ManualLocalizer       reads crucible positions you marked by hand, from a JSON
                        file (see tools/mark_vials.py). This is how a folder
                        of real captured images gets through the pipeline
                        before a real localiser exists - and, later, how you
                        measure one: run both over the same images and the
                        difference is localisation error, with the hand marks
                        as the reference.

  NullLocalizer         returns nothing, always. Use it to run the capture and
                        dashboard stack with the pipeline attached but silent.

Anything added later goes behind this same ABC, and any heavy dependency it
needs gets imported inside start() the way the drivers do - a module-level
import would make this file unimportable on a machine that lacks it.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from config import GEOMETRY, CRUCIBLES_FILE
from drivers.base import Frame
from pipeline.types import Detection

log = logging.getLogger(__name__)


class Localizer(ABC):
    """Finds crucibles in one frame. Stateless with respect to identity."""

    name: str = "unset"
    #: False when this implementation cannot work on real camera frames.
    real_capable: bool = True

    def start(self) -> None:
        """Load models or warm caches. Called once before the first frame."""

    def stop(self) -> None:
        """Release anything start() acquired. Must be safe to call twice."""

    @abstractmethod
    def locate(self, frame: Frame) -> list[Detection]:
        """Return every crucible candidate in `frame`, in pixel coordinates."""

    def describe(self) -> str:
        return self.name


class GroundTruthLocalizer(Localizer):
    """Reads the simulator's ground truth. Blind on real hardware, on purpose.

    This exists so that "does the pipeline work" and "does the localiser work"
    stay separate questions. When a real localiser lands, run both against the
    same recorded frames and the difference is purely localisation error.
    """

    name = "ground_truth"
    real_capable = False

    def __init__(self, jitter_px: float = 0.0) -> None:
        # Non-zero jitter perturbs the true centroids, which is how to check
        # that the assignment gate and hysteresis are not silently relying on
        # pixel-perfect input.
        self.jitter_px = jitter_px
        self._rng = None

    def start(self) -> None:
        import numpy as np
        self._rng = np.random.default_rng(20260814)

    def locate(self, frame: Frame) -> list[Detection]:
        truth = frame.truth or {}
        crucibles = truth.get("crucibles", [])
        if not crucibles and not frame.simulated:
            # A real frame has no truth to read. Say so once per process
            # rather than silently reporting an empty platform.
            _warn_once("GroundTruthLocalizer sees no truth on a real frame - "
                       "it cannot localise real captures")
            return []

        out: list[Detection] = []
        for v in crucibles:
            cx, cy = float(v["cx"]), float(v["cy"])
            if self.jitter_px and self._rng is not None:
                cx += float(self._rng.normal(0.0, self.jitter_px))
                cy += float(self._rng.normal(0.0, self.jitter_px))
            out.append(Detection(
                cx=cx, cy=cy,
                radius=float(v.get("radius", GEOMETRY.crucible_radius_px)),
                confidence=1.0,
                meta={"truth_index": v.get("index"), "truth_stage": v.get("stage")},
            ))
        return out


class ManualLocalizer(Localizer):
    """Crucible positions you marked by hand, loaded from JSON.

    File format, all coordinates in pixels of the image they were marked on:

        {
          "default":       [{"cx": 120, "cy": 300, "radius": 17}, ...],
          "capture_01.jpg":[{"cx": 118, "cy": 299, "radius": 17}, ...]
        }

    Lookup is by the source file's name, which FileCameraSource puts on
    Frame.truth, falling back to "default". So one "default" entry is enough
    for a stack of images of a stationary rack, and per-file entries override
    it wherever the crucibles actually moved.

    Coordinates are stored in pixels rather than normalised, because they are
    marked against one specific capture and silently rescaling them to a
    differently-sized frame would be worse than refusing: the numbers would
    still look plausible. If the frame size does differ from the marked size,
    this scales them and says so, once.
    """

    name = "manual"
    real_capable = True     # it works on real frames; it just is not automatic

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else CRUCIBLES_FILE
        self._marks: dict[str, list[dict]] = {}
        self._marked_size: tuple[int, int] | None = None

    def start(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"no hand-marked crucibles at {self.path}. Create it with "
                f"`python -m tools.mark_vials --image <your capture>`.")
        raw = json.loads(self.path.read_text())
        size = raw.pop("_image_size", None)
        self._marked_size = tuple(size) if size else None
        self._marks = {k: v for k, v in raw.items() if isinstance(v, list)}
        if not self._marks:
            raise ValueError(f"{self.path} contains no crucible marks")
        log.info("manual localiser: %d entries from %s (%s)",
                 len(self._marks), self.path,
                 ", ".join(sorted(self._marks)[:4]))

    def locate(self, frame: Frame) -> list[Detection]:
        key = str((frame.truth or {}).get("name", "")) or "default"
        marks = self._marks.get(key) or self._marks.get("default")
        if marks is None:
            _warn_once(f"no crucible marks for {key!r} and no 'default' entry")
            return []

        sx, sy = self._scale_for(frame)
        out: list[Detection] = []
        for i, m in enumerate(marks):
            out.append(Detection(
                cx=float(m["cx"]) * sx,
                cy=float(m["cy"]) * sy,
                radius=float(m.get("radius", GEOMETRY.crucible_radius_px))
                * (sx + sy) * 0.5,
                confidence=1.0,
                meta={"manual_index": m.get("index", i), "source": key},
            ))
        return out

    def _scale_for(self, frame: Frame) -> tuple[float, float]:
        """Rescale marks if the frame is not the size they were marked on."""
        if not self._marked_size:
            return 1.0, 1.0
        mw, mh = self._marked_size
        h, w = frame.image.shape[:2]
        if (w, h) == (mw, mh):
            return 1.0, 1.0
        _warn_once(f"crucible marks were made on {mw}x{mh} but this frame is "
                   f"{w}x{h} - scaling them, which is only valid if the "
                   "framing is otherwise identical")
        return w / mw, h / mh


class NullLocalizer(Localizer):
    """Finds nothing. The honest default until a real localiser exists."""

    name = "null"

    def locate(self, frame: Frame) -> list[Detection]:
        return []


class CrucibleLocalizer(Localizer):
    """Wraps pipeline.features.detect_crucibles(). Classical CV, real-capable.

    This is what "auto" resolves to: the MVP localiser, used by main.py and
    by tools/track_regions.py.

    Does no undistortion. A localiser answers "where are the crucibles", not
    "please fix my lens distortion first" - that is the caller's job, since
    it also has to apply consistently to the zone polygons and slot
    coordinates the caller matches these detections against.
    """

    name = "crucible"
    real_capable = True

    def __init__(self, min_r: int = 50, max_r: int = 100, param1: int = 90,
                 param2: int = 55, min_dist: int = 100,
                 min_mean: float = 35.0) -> None:
        self._kwargs = dict(min_r=min_r, max_r=max_r, param1=param1,
                            param2=param2, min_dist=min_dist, min_mean=min_mean)

    def locate(self, frame: Frame) -> list[Detection]:
        from pipeline.features import detect_crucibles  # heavier cv2 pass, lazy like the others
        out: list[Detection] = []
        for cx, cy, r in detect_crucibles(frame.image, **self._kwargs):
            out.append(Detection(cx=cx, cy=cy, radius=r, confidence=1.0,
                                 meta={"detector": "hough_crucible"}))
        return out


_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        log.warning(message)


def create_localizer(name: str = "auto") -> Localizer:
    """Build a localiser by name.

    "auto" is CrucibleLocalizer: classical Hough circles, the settled
    decision. Simulated frames still go to ground truth, because the
    synthetic platform's rendering is not what the Hough parameters were
    tuned against and matching them would be tuning against a drawing.
    """
    key = (name or "auto").lower()
    if key in ("ground_truth", "truth"):
        return GroundTruthLocalizer()
    if key in ("manual", "marks"):
        return ManualLocalizer()
    if key in ("null", "none"):
        return NullLocalizer()
    if key in ("crucible", "hough_crucible"):
        return CrucibleLocalizer()
    if key == "auto":
        return CrucibleLocalizer()
    raise ValueError(
        f"unknown localizer {name!r}. Implement it in pipeline/localize.py "
        "and register it here."
    )
