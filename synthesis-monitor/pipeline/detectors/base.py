"""Detector interface and the context every detector is handed.

***The files next to this one are yours.*** This one is the contract.

A detector is a small object with one method:

    check(ctx) -> list[Event]

It is called once per analysis frame, after localisation, tracking, staging
and feature extraction have all run. It gets everything those stages produced
and is expected to return zero or more Events. Returning nothing is the normal
case and costs nothing.

Rules the runner enforces so a half-written detector cannot take the system
down with it:

  * A detector that raises is caught, logged, counted, and its exception is
    surfaced as a warning on the result. The other detectors still run and the
    frame is still stored. It is disabled after `max_consecutive_errors`
    failures in a row, so a detector broken by a bad refactor does not fill
    the log at one screenful per frame forever.
  * A detector must not mutate ctx.frame, ctx.tracks or another detector's
    ROIs. The crops in ctx are copies, so scribbling on one is contained, but
    the tracks are live objects and editing them corrupts identity.
  * Detectors are called in registration order and must not depend on it.
    If two detectors need to share work, compute it in features.py instead.

Deliberately absent from the context: thermal data. The MLX90640 is a passive
logger for researchers to look at - no model, no algorithm and no part of the
anomaly logic runs on it. At 2.4 cm/px a vial covers one or two pixels, so
there is nothing there to run anything on. It is logged and displayed, and
that is the whole scope.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from drivers.base import Frame
from pipeline.history import VialHistory
from pipeline.types import Event, Track, VialReport
from pipeline.zones import ZoneMap

log = logging.getLogger(__name__)


@dataclass
class ZoneView:
    """One zone's pixels, with the vials optionally punched out.

    `bench_mask` is the zone polygon minus a disc around every vial in it -
    i.e. the bare aluminium (or the filter paper, if that route is taken).
    That is the surface a spill would show up on, and masking the vials out
    stops a vial's own colour from reading as a wet patch.
    """

    name: str
    image: np.ndarray                      # BGR crop of the zone's bounding box
    bounds: tuple[int, int, int, int]      # (x0, y0, x1, y1) in frame coords
    zone_mask: np.ndarray                  # 255 inside the polygon
    bench_mask: np.ndarray                 # zone_mask minus the vials
    track_ids: list[int] = field(default_factory=list)

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Crop coordinates -> frame coordinates."""
        return x + self.bounds[0], y + self.bounds[1]


@dataclass
class DetectionContext:
    """Everything one analysis frame produced, handed to every detector."""

    frame: Frame
    timestamp: float
    frame_id: int

    #: Confirmed, currently-visible vials.
    tracks: list[Track]
    #: Same vials, with their features. Indexed the same as `tracks` is not
    #: guaranteed - match on track_id.
    reports: list[VialReport]

    #: track_id -> BGR crop copy around that vial.
    crops: dict[int, np.ndarray]
    #: track_id -> uint8 mask, 255 over the liquid disc, in crop coordinates.
    masks: dict[int, np.ndarray]
    #: track_id -> (x0, y0, x1, y1) of the crop in frame coordinates.
    boxes: dict[int, tuple[int, int, int, int]]

    #: zone name -> ZoneView, for surface checks that are not about one vial.
    zones: dict[str, ZoneView]
    zone_map: ZoneMap

    #: Rolling per-vial memory: past features and past crops.
    history: VialHistory

    #: Tracks that ended on this frame. closed_reason is "oven" for an
    #: inferred oven entry or "lost" for a vial that vanished somewhere it
    #: should not have. Note the known blind spot: a failure during cooling
    #: is currently indistinguishable from a normal oven entry.
    closed_tracks: list[Track] = field(default_factory=list)

    #: Analysis interval in seconds that produced this frame. Needed by
    #: anything rate-based - the cadence switches between 45 s and 10 s, so a
    #: per-frame delta is not a per-second rate.
    interval_s: float = 0.0

    # ---------------------------------------------------------------- sugar
    def report_for(self, track_id: int) -> VialReport | None:
        for r in self.reports:
            if r.track_id == track_id:
                return r
        return None

    def track_for(self, track_id: int) -> Track | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def in_stage(self, stage: str) -> list[Track]:
        """Confirmed vials whose committed stage is `stage`."""
        return [t for t in self.tracks if t.stage == stage]

    def feature_column(self, key: str, stage: str | None = None
                       ) -> tuple[list[int], list[float]]:
        """(track_ids, values) for one feature across the batch.

        The shape batch-median scoring wants. Restrict to one stage when the
        comparison should only be against peers at the same point in the
        process - vials in heating and vials still in filling are not
        comparable and pooling them widens the median's spread for nothing.
        """
        ids: list[int] = []
        values: list[float] = []
        for r in self.reports:
            if stage is not None:
                t = self.track_for(r.track_id)
                if t is None or t.stage != stage:
                    continue
            if key in r.features:
                ids.append(r.track_id)
                values.append(float(r.features[key]))
        return ids, values

    def previous_crop(self, track_id: int) -> np.ndarray | None:
        """This vial's previous crop, resized to match its current one."""
        current = self.crops.get(track_id)
        if current is None:
            return None
        return self.history.previous_crop_matched(track_id, current)


class Detector(ABC):
    """One failure mode. Instantiated once, called once per analysis frame."""

    #: Registry key. Must match an entry in DETECTION.enabled to be loaded.
    name: str = "unset"
    #: Shown on the dashboard next to the detector's health.
    description: str = ""
    #: Consecutive raises before the runner disables this detector.
    max_consecutive_errors: int = 5
    #: Set False by the shipped stubs so the dashboard can say plainly that a
    #: detector is registered but not yet looking at anything.
    implemented: bool = True

    def start(self) -> None:
        """Called once before the first frame."""

    def stop(self) -> None:
        """Called once at shutdown. Must be safe to call twice."""

    @abstractmethod
    def check(self, ctx: DetectionContext) -> list[Event]:
        """Return any events this frame warrants. Empty list is normal."""

    # ------------------------------------------------------------- helper
    def event(self, ctx: DetectionContext, kind: str, message: str,
              severity: str = "warning", track_id: int | None = None,
              zone: str | None = None, **data) -> Event:
        """Build an Event stamped with this detector and this frame.

        Keyword arguments land in Event.data and are stored as JSON, so keep
        them to plain types - the numbers that justified the event, so a
        human reading the log later can see why it fired.
        """
        return Event(
            kind=kind,
            severity=severity,
            message=message,
            timestamp=ctx.timestamp or time.time(),
            frame_id=ctx.frame_id,
            track_id=track_id,
            detector=self.name,
            zone=zone,
            data=data,
        )


class NotImplementedDetector(Detector):
    """Base for the stubs: announces itself once, then stays quiet.

    Subclasses that have not been written yet inherit this so the pipeline
    runs end to end with the full detector set loaded, and the dashboard shows
    honestly that five detectors are registered and none of them is looking at
    anything.
    """

    implemented = False

    def __init__(self) -> None:
        self._announced = False

    def check(self, ctx: DetectionContext) -> list[Event]:
        if not self._announced:
            self._announced = True
            log.info("detector %r is a stub - loaded but not detecting anything",
                     self.name)
        return []
