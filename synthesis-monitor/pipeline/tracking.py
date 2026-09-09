"""The Tracker interface, and the cadence controller that paces the loop.

The implementations live in pipeline/region_trackers.py: SlotTracker for the
fixed-slot zones, FifoTracker for the injection lane. The whole-platform
assignment tracker that used to live here is gone - this bench has marked
slots and one lane, not free movement across a platform, and matching against
those is both simpler and correct.

The Tracker ABC stays so that decision remains reversible: if the platform
ever gets a continuous-video zone, that zone can run a different Tracker
implementation without the rest of the pipeline noticing.

No detection logic here. This module answers "which crucible is this", never
"is this crucible in trouble".
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from config import CADENCE, TRACKING
from pipeline.assignment import assign_with_gate
from pipeline.types import Detection, Track
from pipeline.zones import StageTracker, ZoneMap, distance_mm

log = logging.getLogger(__name__)


class Tracker(ABC):
    """Swappable identity assignment."""

    @abstractmethod
    def update(self, detections: list[Detection], timestamp: float
               ) -> tuple[list[Track], list[Track]]:
        """Absorb one frame of detections.

        Returns (active tracks, tracks closed by this update). Closed tracks
        carry `closed_reason`; they are returned once and then forgotten.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget everything. Used when a run ends or the source restarts."""

    @property
    @abstractmethod
    def tracks(self) -> list[Track]:
        """Currently live tracks, confirmed or not."""


class CadenceController:
    """Picks the analysis interval from how much the batch is moving.

    Slow by default; drops to the fast interval while anything is travelling.
    The release counter stops it flapping between 45 s and 10 s on one noisy
    centroid - once busy, it stays busy until several consecutive frames are
    quiet.
    """

    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, float]] = {}
        self._quiet_frames = 0
        self.busy = False

    def observe(self, tracks: list[Track]) -> float:
        """Feed the frame's tracks, get the interval to wait before the next."""
        moved = 0.0
        for t in tracks:
            prev = self._prev.get(t.track_id)
            if prev is not None:
                moved = max(moved, distance_mm(prev, t.center))
        self._prev = {t.track_id: t.center for t in tracks}

        if moved >= CADENCE.busy_motion_mm:
            self.busy = True
            self._quiet_frames = 0
        elif self.busy:
            self._quiet_frames += 1
            if self._quiet_frames >= CADENCE.busy_release_frames:
                self.busy = False

        return (CADENCE.analysis_interval_busy_s if self.busy
                else CADENCE.analysis_interval_s)

    def gate_mm(self) -> float:
        return (TRACKING.max_assignment_busy_mm if self.busy
                else TRACKING.max_assignment_mm)

    def reset(self) -> None:
        self._prev.clear()
        self._quiet_frames = 0
        self.busy = False


def create_tracker(name: str = "auto", frame_size: tuple[int, int] | None = None,
                   zone_map: ZoneMap | None = None) -> Tracker:
    """Build a tracker by name, mirroring pipeline.localize.create_localizer.

    One implementation: per-zone slot matching plus a FIFO lane, with identity
    handed across zone boundaries. It needs the hand-marked slot and lane
    files (see tools/mark_slots.py) and raises FileNotFoundError at start()
    without them.
    """
    key = (name or "auto").lower()
    if key in ("auto", "region", "crucible", "zones", "slot"):
        # Imported here, not at module level: pipeline.region_trackers imports
        # this module for the Tracker ABC, so a top-level import would be
        # circular.
        from pipeline.region_trackers import RegionTracker
        return RegionTracker(frame_size=frame_size, zone_map=zone_map)
    raise ValueError(
        f"unknown tracker {name!r}. Implement it in pipeline/ and register it here.")
