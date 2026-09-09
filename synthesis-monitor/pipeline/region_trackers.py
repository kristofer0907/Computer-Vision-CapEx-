"""Per-zone tracking for the real crucible platform.

Two mechanisms, one per zone shape:

    SlotTracker   storing / heating / collection - fixed hand-marked slot
                  positions (tools/mark_slots.py). Identity = nearest
                  unclaimed slot, gated by that layout's slot_radius_px.
    FifoTracker   injection - a single-file lane with no fixed positions
                  (tools/mark_slots.py --lane). Identity = order along the
                  lane axis: a crucible can't pass another one on a narrow
                  lane, so an ordered diff against the live queue is enough -
                  no bipartite matching needed.

RegionCoordinator owns one tracker per zone, routes each frame's detections
by zone membership, and reconciles identity across zone boundaries: a track
closing in one zone can hand its id to a new track spawning in a
*neighbouring* zone (per REGION_TRACKING.region_sequence, a single fixed
process order) within a short time window, instead of that new track getting
a fresh id.

These are the trackers the pipeline runs (main.py). They match the real
hardware layout: fixed slots, a single-file lane and a small fixed process
order, rather than free movement across a whole platform. They implement the
Tracker ABC in pipeline/tracking.py and use its Track/Detection types.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np

from config import DATA_DIR, GEOMETRY, REGION_TRACKING
from pipeline.assignment import assign_with_gate
from pipeline.tracking import Tracker
from pipeline.types import Detection, Track
from pipeline.zones import ZoneMap, px_to_mm

log = logging.getLogger(__name__)

_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        log.warning(message)


def _scale_for(ref_size: tuple[int, int], frame_size: tuple[int, int],
               what: str) -> tuple[float, float]:
    """Rescale factor if a hand-marked layout's reference image size differs
    from the frame actually being processed. Mirrors
    pipeline.localize.ManualLocalizer._scale_for exactly, duplicated (not
    imported) since that one is module-private to pipeline/localize.py.
    """
    rw, rh = ref_size
    w, h = frame_size
    if (w, h) == (rw, rh):
        return 1.0, 1.0
    _warn_once(f"{what}: marked on {rw}x{rh} but this frame is {w}x{h} - "
              "scaling, which is only valid if the framing is otherwise "
              "identical")
    return w / rw, h / rh


class SlotTracker(Tracker):
    """Nearest-fixed-slot identity for storing / heating / collection.

    Loads a hand-marked slot layout and matches each frame's detections to
    the nearest unclaimed slot via the existing Hungarian solver, gated by
    that layout's slot_radius_px (solve globally, then reject - same
    "gate after solving" reasoning as pipeline/assignment.py's own docstring).

    Identity is the slot itself: a track's id is stable for as long as
    something occupies that slot. There is no hit-count hysteresis before a
    track is confirmed - a slot match, gated by a hand-marked radius, is
    already strong identity evidence.
    """

    def __init__(self, zone_name: str, frame_size: tuple[int, int],
                 slots_path: str | Path | None = None,
                 id_source: itertools.count | None = None,
                 max_missed_frames: int | None = None) -> None:
        self.zone_name = zone_name
        self.frame_size = frame_size
        self.slots_path = Path(slots_path) if slots_path else (
            DATA_DIR / REGION_TRACKING.slot_files[zone_name])
        self._id_source = id_source or itertools.count(1)
        self.max_missed = (REGION_TRACKING.slot_max_missed_frames
                           if max_missed_frames is None else max_missed_frames)

        self._slot_ids: list[int] = []
        self._slot_xy = np.zeros((0, 2))
        self._slot_radius_px = 0.0
        self._track_by_slot: dict[int, Track] = {}

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if not self.slots_path.exists():
            raise FileNotFoundError(
                f"no slot layout at {self.slots_path}. Create it with "
                f"`python -m tools.mark_slots --stage <name>`.")
        raw = json.loads(self.slots_path.read_text())
        slots = raw.get("slots") or []
        if not slots:
            raise ValueError(f"{self.slots_path} has no slots")
        if "slot_radius_px" not in raw:
            raise ValueError(
                f"{self.slots_path} has no slot_radius_px - re-mark with "
                "tools.mark_slots using +/- to set one before it can gate a match")

        ref_size = tuple(raw.get("image_size", list(self.frame_size)))
        sx, sy = _scale_for(ref_size, self.frame_size, f"slots[{self.zone_name}]")

        self._slot_ids = [s["id"] for s in slots]
        self._slot_xy = np.array([[s["x"] * sx, s["y"] * sy] for s in slots],
                                 dtype=np.float64)
        self._slot_radius_px = raw["slot_radius_px"] * (sx + sy) * 0.5

    def reset(self) -> None:
        self._track_by_slot.clear()

    @property
    def tracks(self) -> list[Track]:
        return list(self._track_by_slot.values())

    def slot_status(self) -> dict[int, int | None]:
        """slot_id -> occupying track_id, or None. For reporting/overlays."""
        occupied = {sid: t.track_id for sid, t in self._track_by_slot.items()}
        return {sid: occupied.get(sid) for sid in self._slot_ids}

    def slots(self) -> list[tuple[int, float, float]]:
        """(slot_id, x, y) for every known slot, in frame pixels. For overlays."""
        return [(sid, float(x), float(y)) for sid, (x, y)
                in zip(self._slot_ids, self._slot_xy)]

    @property
    def slot_radius_px(self) -> float:
        return self._slot_radius_px

    # -------------------------------------------------------------- update
    def update(self, detections: list[Detection], timestamp: float
               ) -> tuple[list[Track], list[Track]]:
        n_det, n_slot = len(detections), len(self._slot_ids)
        matched: list[tuple[int, int]] = []
        unmatched_slots = list(range(n_slot))

        if n_det and n_slot:
            cost = np.zeros((n_det, n_slot), dtype=np.float64)
            for i, d in enumerate(detections):
                cost[i, :] = np.hypot(self._slot_xy[:, 0] - d.cx,
                                      self._slot_xy[:, 1] - d.cy)
            matched, _unmatched_det, unmatched_slots = assign_with_gate(
                cost, self._slot_radius_px)

        matched_by_slot = {slot_idx: det_idx for det_idx, slot_idx in matched}
        for slot_idx, det_idx in matched_by_slot.items():
            slot_id = self._slot_ids[slot_idx]
            det = detections[det_idx]
            track = self._track_by_slot.get(slot_id)
            if track is None:
                track = Track(
                    track_id=next(self._id_source), cx=det.cx, cy=det.cy,
                    radius=det.radius, first_seen_ts=timestamp,
                    last_seen_ts=timestamp, confirmed=True,
                    stage=self.zone_name, stage_since_ts=timestamp,
                    stage_log=[(self.zone_name, timestamp)], slot_id=slot_id,
                )
                self._track_by_slot[slot_id] = track
            else:
                track.cx, track.cy, track.radius = det.cx, det.cy, det.radius
                track.last_seen_ts = timestamp
                track.hits += 1
                track.missed = 0

        closed: list[Track] = []
        for slot_idx in unmatched_slots:
            slot_id = self._slot_ids[slot_idx]
            track = self._track_by_slot.get(slot_id)
            if track is None:
                continue
            track.missed += 1
            if track.missed > self.max_missed:
                track.closed_reason = "vacated"
                closed.append(track)
                del self._track_by_slot[slot_id]

        matched_det_idx = {det_idx for det_idx in matched_by_slot.values()}
        for det_idx in range(n_det):
            if det_idx not in matched_det_idx:
                d = detections[det_idx]
                _warn_once(
                    f"{self.zone_name}: detection at ({d.cx:.0f},{d.cy:.0f}) "
                    "is not within slot_radius_px of any known slot")

        return self.tracks, closed


class FifoTracker(Tracker):
    """Order-based identity for a single-file lane with no fixed positions.

    A crucible on a narrow lane can't pass another one, so identity comes
    from where a detection falls in the sorted order along the lane axis
    (defined by two hand-clicked entry/exit points), not from matching
    against any stored position. This is an ordered two-pointer diff, not
    Hungarian - much cheaper, and correct precisely because the sequence
    can't invert.
    """

    def __init__(self, zone_name: str, frame_size: tuple[int, int],
                 lane_path: str | Path | None = None,
                 id_source: itertools.count | None = None,
                 max_missed_frames: int | None = None,
                 step_gate_mm: float | None = None) -> None:
        self.zone_name = zone_name
        self.frame_size = frame_size
        self.lane_path = Path(lane_path) if lane_path else (
            DATA_DIR / REGION_TRACKING.lane_files[zone_name])
        self._id_source = id_source or itertools.count(1)
        self.max_missed = (REGION_TRACKING.fifo_max_missed_frames
                           if max_missed_frames is None else max_missed_frames)
        self.gate_mm = (REGION_TRACKING.fifo_step_gate_mm
                        if step_gate_mm is None else step_gate_mm)

        self._entry = np.zeros(2)
        self._exit = np.zeros(2)
        self._tracks: list[Track] = []

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if not self.lane_path.exists():
            raise FileNotFoundError(
                f"no lane layout at {self.lane_path}. Create it with "
                f"`python -m tools.mark_slots --stage {self.zone_name} --lane`.")
        raw = json.loads(self.lane_path.read_text())
        if "entry" not in raw or "exit" not in raw:
            raise ValueError(f"{self.lane_path} is missing entry/exit")

        ref_size = tuple(raw.get("image_size", list(self.frame_size)))
        sx, sy = _scale_for(ref_size, self.frame_size, f"lane[{self.zone_name}]")

        self._entry = np.array([raw["entry"][0] * sx, raw["entry"][1] * sy])
        self._exit = np.array([raw["exit"][0] * sx, raw["exit"][1] * sy])

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def lane_endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(entry, exit) in frame pixels. For overlays."""
        return (float(self._entry[0]), float(self._entry[1])), \
               (float(self._exit[0]), float(self._exit[1]))

    def queue_order(self) -> list[int]:
        """track_ids sorted entry -> exit by current position. Recomputed,
        not stored - the queue is whatever the tracks' positions say it is."""
        return [t.track_id for t in
                sorted(self._tracks, key=lambda t: self._project_mm(t.cx, t.cy))]

    def _project_mm(self, cx: float, cy: float) -> float:
        vec = self._exit - self._entry
        length = float(np.hypot(*vec))
        if length == 0:
            return 0.0
        unit = vec / length
        rel = np.array([cx, cy]) - self._entry
        return px_to_mm(float(np.dot(rel, unit)))

    # -------------------------------------------------------------- update
    def update(self, detections: list[Detection], timestamp: float
               ) -> tuple[list[Track], list[Track]]:
        proj = sorted(((self._project_mm(d.cx, d.cy), d) for d in detections),
                     key=lambda p: p[0])
        live = sorted(((self._project_mm(t.cx, t.cy), t) for t in self._tracks),
                     key=lambda p: p[0])

        i = j = 0
        matched: list[tuple[Track, Detection]] = []
        missing: list[Track] = []
        spawned: list[Detection] = []
        while i < len(live) and j < len(proj):
            ds = proj[j][0] - live[i][0]
            if abs(ds) <= self.gate_mm:
                matched.append((live[i][1], proj[j][1]))
                i += 1
                j += 1
            elif ds < -self.gate_mm:
                # a detection sits well before this track along the axis
                spawned.append(proj[j][1])
                j += 1
            else:
                # this track has no detection near it
                missing.append(live[i][1])
                i += 1
        missing.extend(t for _, t in live[i:])
        spawned.extend(d for _, d in proj[j:])

        for track, det in matched:
            track.cx, track.cy, track.radius = det.cx, det.cy, det.radius
            track.last_seen_ts = timestamp
            track.hits += 1
            track.missed = 0

        exit_s = self._project_mm(*self._exit)
        closed: list[Track] = []
        for track in missing:
            track.missed += 1
            if track.missed > self.max_missed:
                last_s = self._project_mm(track.cx, track.cy)
                # near the exit end: a normal dequeue. anywhere else: lost -
                # occlusion, or lifted out mid-lane rather than exiting the
                # marked end. Both are "missing"; only the label differs.
                track.closed_reason = (
                    "advanced" if abs(last_s - exit_s) <= self.gate_mm else "lost")
                closed.append(track)
                self._tracks.remove(track)

        entry_s = self._project_mm(*self._entry)
        for det in spawned:
            det_s = self._project_mm(det.cx, det.cy)
            if abs(det_s - entry_s) > self.gate_mm:
                _warn_once(
                    f"{self.zone_name}: new detection at "
                    f"({det.cx:.0f},{det.cy:.0f}) appeared away from the "
                    "marked entry point")
            track = Track(
                track_id=next(self._id_source), cx=det.cx, cy=det.cy,
                radius=det.radius, first_seen_ts=timestamp,
                last_seen_ts=timestamp, confirmed=True,
                stage=self.zone_name, stage_since_ts=timestamp,
                stage_log=[(self.zone_name, timestamp)],
            )
            self._tracks.append(track)

        return self.tracks, closed


class RegionCoordinator:
    """Owns one tracker per zone, routes detections by zone membership, and
    reconciles a track's identity across zone boundaries so a crucible keeps
    one id for its whole visible life.

    Not itself a Tracker - it drives several of them and stitches their
    outputs together, which the single-frame Tracker.update() signature
    can't express.

    Known, explicit simplification: concurrent same-frame handoffs are
    resolved greedily (nearest-in-time first, donor removed from the
    candidate pool), not via a global optimal match. Acceptable given
    expected concurrency is low - heating only ever holds 2 crucibles,
    batches move slowly - not worth the extra complexity here.
    """

    def __init__(self, zone_map: ZoneMap, trackers: dict[str, Tracker],
                 region_sequence: tuple[str, ...] | None = None,
                 handoff_window_s: float | None = None) -> None:
        self.zone_map = zone_map
        self.trackers = trackers
        self.region_sequence = region_sequence or REGION_TRACKING.region_sequence
        self.handoff_window_s = (REGION_TRACKING.handoff_window_s
                                 if handoff_window_s is None else handoff_window_s)

        # (track, zone it closed in, when it closed)
        self._recently_closed: list[tuple[Track, str, float]] = []
        self._prev_active_ids: dict[str, set[int]] = {z: set() for z in trackers}
        self._last_handoffs: list[tuple[int, str, str]] = []
        #: zone -> detections seen there this frame that carry no identity
        self.untracked: dict[str, list[Detection]] = {}

    def start(self) -> None:
        for tracker in self.trackers.values():
            start = getattr(tracker, "start", None)
            if start is not None:
                start()

    def _neighbors(self, zone: str) -> list[str]:
        """Zones a track in `zone` may have come from.

        Any of them. The arm lifts a crucible and puts it down wherever it
        is going - it does not walk it along the process order - so storing
        to collection in one move is normal, not a skipped step. Restricting
        this to adjacent entries in region_sequence meant a crucible taken
        out of storing and set down in heating could never inherit its id,
        and a six-crucible run ended up numbering into the teens.

        region_sequence therefore says which zones exist and in what order to
        report them, not what moves are allowed. What actually constrains a
        handoff is that a track has to have closed, recently
        (handoff_window_s), for a spawn to claim its id at all.
        """
        return [z for z in self.region_sequence if z != zone]

    def update(self, detections: list[Detection], timestamp: float
               ) -> dict[str, tuple[list[Track], list[Track]]]:
        buckets: dict[str, list[Detection]] = {name: [] for name in self.trackers}
        self.untracked = {}
        for d in detections:
            zone = self.zone_map.zone_at(d.cx, d.cy)
            if zone is None:
                _warn_once(
                    f"detection at ({d.cx:.0f},{d.cy:.0f}) is outside every "
                    "zone polygon")
                continue
            if zone not in self.trackers:
                # a real crucible, in a zone that does not carry identity
                # (REGION_TRACKING.id_zones) - reported and drawn, not numbered
                self.untracked.setdefault(zone, []).append(d)
                continue
            buckets[zone].append(d)

        results: dict[str, tuple[list[Track], list[Track]]] = {}
        for zone, tracker in self.trackers.items():
            active, closed = tracker.update(buckets[zone], timestamp)
            results[zone] = (active, closed)
            for t in closed:
                self._recently_closed.append((t, zone, timestamp))

        self._recently_closed = [
            (t, z, ts) for (t, z, ts) in self._recently_closed
            if timestamp - ts <= self.handoff_window_s
        ]

        self._last_handoffs = []
        for zone, (active, _closed) in results.items():
            prev_ids = self._prev_active_ids.get(zone, set())
            new_spawns = [t for t in active if t.track_id not in prev_ids]
            eligible = {zone, *self._neighbors(zone)}
            for spawn in new_spawns:
                candidates = [c for c in self._recently_closed if c[1] in eligible]
                if not candidates:
                    continue
                donor, donor_zone, donor_ts = min(
                    candidates, key=lambda c: timestamp - c[2])
                self._recently_closed.remove((donor, donor_zone, donor_ts))
                old_id = spawn.track_id
                spawn.track_id = donor.track_id
                spawn.first_seen_ts = donor.first_seen_ts
                spawn.hits += donor.hits
                spawn.confirmed = True
                spawn.stage_log = donor.stage_log + spawn.stage_log
                self._last_handoffs.append((spawn.track_id, donor_zone, zone))
                log.info(
                    "handoff: track %d moved %s -> %s (was spawned as %d)",
                    spawn.track_id, donor_zone, zone, old_id)

        self._prev_active_ids = {
            z: {t.track_id for t in active} for z, (active, _c) in results.items()}
        return results

    @property
    def tracks(self) -> list[Track]:
        out: list[Track] = []
        for tracker in self.trackers.values():
            out.extend(tracker.tracks)
        return out

    def handoffs_this_frame(self) -> list[tuple[int, str, str]]:
        return list(self._last_handoffs)

    def reset(self) -> None:
        """Forget everything, in every zone. Used when a run ends or restarts."""
        for tracker in self.trackers.values():
            tracker.reset()
        self._recently_closed.clear()
        self._prev_active_ids = {z: set() for z in self.trackers}
        self._last_handoffs = []


def create_region_coordinator(frame_size: tuple[int, int] | None = None,
                              zone_map: ZoneMap | None = None) -> RegionCoordinator:
    """One tracker per configured zone, all sharing one id source.

    frame_size defaults to the configured geometry, the same assumption
    ZoneMap already makes when constructed with no size. Slot and lane
    layouts rescale themselves to it if they were marked on a differently
    sized image.
    """
    if frame_size is None:
        frame_size = (GEOMETRY.frame_width_px, GEOMETRY.frame_height_px)
    zm = zone_map or ZoneMap(frame_size[0], frame_size[1])

    id_source = itertools.count(1)
    trackers: dict[str, Tracker] = {}
    for zone in REGION_TRACKING.region_sequence:
        if zone not in REGION_TRACKING.id_zones:
            continue    # detected and reported, but not numbered
        if zone in REGION_TRACKING.slot_files:
            trackers[zone] = SlotTracker(zone, frame_size, id_source=id_source)
        elif zone in REGION_TRACKING.lane_files:
            trackers[zone] = FifoTracker(zone, frame_size, id_source=id_source)
    return RegionCoordinator(zm, trackers)


class RegionTracker(Tracker):
    """Tracker-ABC adapter over RegionCoordinator.

    RegionCoordinator deliberately is not a Tracker: its update() returns
    per-zone results, which is what tools/track_regions.py needs to draw
    overlays and report slot occupancy. The loop in main.py wants the flat
    (active, closed) pair every Tracker returns, so this
    flattens it, and leaves `.coordinator` reachable for anything that wants
    the zone-level detail back.
    """

    def __init__(self, coordinator: RegionCoordinator | None = None,
                 frame_size: tuple[int, int] | None = None,
                 zone_map: ZoneMap | None = None) -> None:
        self.coordinator = coordinator or create_region_coordinator(
            frame_size, zone_map)

    def start(self) -> None:
        self.coordinator.start()

    def reset(self) -> None:
        self.coordinator.reset()

    @property
    def tracks(self) -> list[Track]:
        return self.coordinator.tracks

    def update(self, detections: list[Detection], timestamp: float
               ) -> tuple[list[Track], list[Track]]:
        results = self.coordinator.update(detections, timestamp)

        # A track whose id was handed to a spawn in another zone did not end -
        # the same crucible is still on the bench under that id, one zone
        # along. Reporting it as closed would make the pipeline fire a
        # disappearance event and drop the id's feature history every single
        # time a crucible moved between zones.
        handed_off = {tid for tid, _from, _to in self.coordinator.handoffs_this_frame()}

        active: list[Track] = []
        closed: list[Track] = []
        seen: set[int] = set()
        for zone_active, zone_closed in results.values():
            for t in zone_active:
                if t.track_id not in seen:
                    seen.add(t.track_id)
                    active.append(t)
            closed.extend(t for t in zone_closed if t.track_id not in handed_off)
        return active, closed

    def handoffs_this_frame(self) -> list[tuple[int, str, str]]:
        return self.coordinator.handoffs_this_frame()
