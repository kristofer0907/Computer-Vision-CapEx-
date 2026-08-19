"""Vial identity across analysis frames.

Approach is settled: global Hungarian assignment on ground distance, gated in
millimetres, with the stage machine running inside zone polygons. DeepSORT was
evaluated and rejected - its motion model assumes near-continuous frames and
there are 30-60 seconds between ours, so its appearance-plus-Kalman machinery
would be predicting from a state that is already meaningless. Do not
reintroduce it.

The Tracker ABC exists so that decision stays reversible for a *different*
reason: if the platform ever gets a continuous-video zone, that zone can run a
different Tracker implementation without the rest of the pipeline noticing.

No detection logic here. This module answers "which vial is this", never
"is this vial in trouble".
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


class HungarianTracker(Tracker):
    """Global nearest-vial assignment with a physical-plausibility gate.

    Cost is ground distance in millimetres. Anything beyond the gate is not
    matched at all: at the slow cadence a vial can legitimately cross most of
    the platform between frames, but it cannot appear on the other side of a
    zone it never entered, and a gate in mm is the only honest way to say so.
    """

    def __init__(self, zone_map: ZoneMap | None = None,
                 stage_tracker: StageTracker | None = None,
                 max_assignment_mm: float | None = None,
                 max_missed_frames: int | None = None,
                 min_hits_to_confirm: int | None = None) -> None:
        self.zones = zone_map or ZoneMap()
        self.stages = stage_tracker or StageTracker(self.zones)
        self.gate_mm = (TRACKING.max_assignment_mm
                        if max_assignment_mm is None else max_assignment_mm)
        self.max_missed = (TRACKING.max_missed_frames
                           if max_missed_frames is None else max_missed_frames)
        self.min_hits = (TRACKING.min_hits_to_confirm
                         if min_hits_to_confirm is None else min_hits_to_confirm)
        self._tracks: list[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------ interface
    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def set_gate_mm(self, gate_mm: float) -> None:
        """Tighten the gate when the cadence speeds up.

        At the 10 s conveyor cadence a vial covers roughly a quarter of what
        it can cover in 45 s, so keeping the slow gate would let the solver
        cheerfully swap two neighbouring vials' identities.
        """
        self.gate_mm = gate_mm

    # -------------------------------------------------------------- update
    def update(self, detections: list[Detection], timestamp: float
               ) -> tuple[list[Track], list[Track]]:
        matched, lost_idx, new_idx = self._associate(detections)

        for t_idx, d_idx in matched:
            self._absorb(self._tracks[t_idx], detections[d_idx], timestamp)

        for d_idx in new_idx:
            self._tracks.append(self._spawn(detections[d_idx], timestamp))

        closed: list[Track] = []
        for t_idx in lost_idx:
            track = self._tracks[t_idx]
            track.missed += 1
            if track.missed > self.max_missed:
                track.closed_reason = self.stages.close_reason(track)
                closed.append(track)

        if closed:
            gone = {id(t) for t in closed}
            self._tracks = [t for t in self._tracks if id(t) not in gone]

        return self.tracks, closed

    # -------------------------------------------------------------- helpers
    def _associate(self, detections: list[Detection]):
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        cost = np.zeros((len(self._tracks), len(detections)), dtype=np.float64)
        for i, track in enumerate(self._tracks):
            for j, det in enumerate(detections):
                cost[i, j] = distance_mm(track.center, det.center)
        return assign_with_gate(cost, self.gate_mm)

    def _spawn(self, det: Detection, timestamp: float) -> Track:
        track = Track(track_id=self._next_id, cx=det.cx, cy=det.cy,
                      radius=det.radius, first_seen_ts=timestamp,
                      last_seen_ts=timestamp)
        self._next_id += 1
        track.confirmed = self.min_hits <= 1
        # A brand-new track commits its first stage immediately. Hysteresis
        # guards against flapping between stages, not against the initial
        # observation, and making a vial wait N frames for its first stage
        # would leave it unstaged through most of a short zone.
        stage = self.zones.zone_at(track.cx, track.cy)
        if stage is not None:
            track.stage = stage
            track.stage_since_ts = timestamp
            track.stage_log.append((stage, timestamp))
        return track

    def _absorb(self, track: Track, det: Detection, timestamp: float) -> None:
        track.cx, track.cy = det.cx, det.cy
        track.radius = det.radius
        track.last_seen_ts = timestamp
        track.hits += 1
        track.missed = 0
        if track.hits >= self.min_hits:
            track.confirmed = True
        self.stages.update(track, timestamp)


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
