"""Zone polygons, pixel/millimetre conversion, and the stage state machine.

Zone polygons are stored in config as normalised (0..1) frame coordinates so
they survive a change of capture resolution. Everything in this module works
in pixels and converts at the boundary.

Nothing here is detection logic: it answers "which polygon is this point in"
and "has it been there long enough to believe it", not "is anything wrong".
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from config import GEOMETRY, TRACKING, ZONES
from pipeline.types import Track

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
def px_to_mm(px: float) -> float:
    return px / GEOMETRY.px_per_mm


def mm_to_px(mm: float) -> float:
    return mm * GEOMETRY.px_per_mm


def distance_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_mm(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Ground distance between two image points.

    Valid only because the camera looks straight down at a flat platform, so
    the scale is uniform across the frame. If the mount ever ends up tilted
    this needs a homography instead, and every mm figure downstream (the
    assignment gate, spacing checks) is wrong until it gets one.
    """
    return px_to_mm(distance_px(a, b))


# --------------------------------------------------------------------------
# Polygons
# --------------------------------------------------------------------------
class ZoneMap:
    """Normalised polygons resolved to pixels for one frame size."""

    def __init__(self, width: int | None = None, height: int | None = None,
                 polygons: dict[str, list[tuple[float, float]]] | None = None) -> None:
        self.width = width or GEOMETRY.frame_width_px
        self.height = height or GEOMETRY.frame_height_px
        src = polygons if polygons is not None else ZONES.polygons
        self.polygons_px: dict[str, np.ndarray] = {
            name: np.array([(x * self.width, y * self.height) for x, y in poly],
                           dtype=np.float32)
            for name, poly in src.items()
        }
        self.names = [n for n in ZONES.order() if n in self.polygons_px]

    def zone_at(self, x: float, y: float) -> str | None:
        """Name of the zone containing (x, y), or None.

        Overlapping polygons resolve to the first in process order. Trace them
        so they do not overlap; the tie-break exists so a sloppy calibration
        degrades predictably instead of flickering between two answers.
        """
        pt = (float(x), float(y))
        for name in self.names:
            if cv2.pointPolygonTest(self.polygons_px[name], pt, False) >= 0:
                return name
        return None

    def nearest_zone(self, x: float, y: float) -> tuple[str | None, float]:
        """(name, signed distance in px) of the closest polygon edge.

        Used for diagnostics: a vial that is consistently 5 px outside a zone
        means the polygon is traced wrong, not that the vial is off-platform.
        """
        best: tuple[str | None, float] = (None, -math.inf)
        pt = (float(x), float(y))
        for name in self.names:
            d = cv2.pointPolygonTest(self.polygons_px[name], pt, True)
            if d > best[1]:
                best = (name, d)
        return best

    def bounds(self, name: str) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) axis-aligned bounding box of a zone, in pixels."""
        poly = self.polygons_px[name]
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        return int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))

    def mask(self, name: str) -> np.ndarray:
        """uint8 mask of one zone at full frame size, 255 inside."""
        m = np.zeros((self.height, self.width), np.uint8)
        cv2.fillPoly(m, [self.polygons_px[name].astype(np.int32)], 255)
        return m

    def draw(self, image: np.ndarray, color=(90, 110, 130),
             label: bool = True) -> np.ndarray:
        """Outline every zone on a copy of `image`."""
        out = image.copy()
        sx = out.shape[1] / self.width
        sy = out.shape[0] / self.height
        for name in self.names:
            poly = (self.polygons_px[name] * (sx, sy)).astype(np.int32)
            cv2.polylines(out, [poly], True, color, 1, cv2.LINE_AA)
            if label:
                x, y = poly.min(axis=0)
                cv2.putText(out, name, (int(x) + 4, int(y) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return out


# --------------------------------------------------------------------------
# Stage machine
# --------------------------------------------------------------------------
class StageTracker:
    """Commits a track's stage only after N consecutive agreeing frames.

    Without hysteresis a vial sitting on a zone boundary, or one centroid
    jittering by a few pixels, would emit a stage transition every frame and
    the timing statistics built on those transitions would be noise.

    N is TRACKING.stage_hysteresis_n and is a placeholder: the right value
    falls out of the per-stage timings, which are still open with the
    researchers. At the 45 s cadence, N=2 confirms a transition ~90 s late.
    """

    def __init__(self, zone_map: ZoneMap, hysteresis_n: int | None = None) -> None:
        self.zones = zone_map
        self.n = TRACKING.stage_hysteresis_n if hysteresis_n is None else hysteresis_n

    def update(self, track: Track, timestamp: float) -> str | None:
        """Feed one observation. Returns the new stage if one was committed."""
        observed = self.zones.zone_at(track.cx, track.cy)

        if observed is None or observed == track.stage:
            # Outside every polygon, or agreeing with what is already
            # committed: either way there is nothing pending any more.
            track.pending_stage = None
            track.pending_count = 0
            return None

        if observed == track.pending_stage:
            track.pending_count += 1
        else:
            track.pending_stage = observed
            track.pending_count = 1

        if track.pending_count < self.n:
            return None

        track.stage = observed
        track.stage_since_ts = timestamp
        track.stage_log.append((observed, timestamp))
        track.pending_stage = None
        track.pending_count = 0
        return observed

    def close_reason(self, track: Track) -> str:
        """Why a track that stopped being detected ended.

        A vial last confirmed in cooling that disappears is inferred to have
        entered the oven, which is outside the camera's view.

        KNOWN BLIND SPOT, not solved here: a genuine failure during cooling -
        the vial knocked over, removed by hand, spilled - produces exactly the
        same observation as a normal oven entry. The inference is recorded as
        an inference so nothing downstream can mistake it for an observation,
        but it cannot currently be distinguished. Anything that resolves this
        needs a signal from outside this camera (an oven door sensor, a
        scheduler event), not more image processing.
        """
        if track.stage == TRACKING.oven_entry_from:
            return "oven"
        return "lost"
