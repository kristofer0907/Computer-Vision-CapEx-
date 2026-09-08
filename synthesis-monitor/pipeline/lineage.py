"""Heater to cooling vial lineage, by occupancy transitions alone.

One question only: which vial that was on a heater is now the one on the
cooling pad. Nothing here tracks a vial before it reaches a heater, and
nothing here looks at pixels.

The mechanism is deliberately not the one in pipeline/region_trackers.py.
That matches detections to slots by distance, which answers "is this slot
occupied" and was then read as "this is the same vial as last frame" - so a
jar lifted off a heater and a different one set down in the same slot kept
the first one's id, because the slot never read empty in between. Identity
here is minted by an empty->occupied edge instead, and a slot that stays
occupied is never re-examined.

Matching is by event order, not position: the most recent unmatched heater
vacate is the vial that turns up next on the cooling pad. That holds because
vials move one at a time. When they do not, this will mismatch, and nothing
in this file would notice - the assumption carries the result, so it is
stated rather than defended.

Known blind spot, inherent to occupancy: a vial swapped out and replaced
between two captures leaves occupancy True on both, so no edge is seen and
the first vial's id stays on the slot. At a 30 s cadence an arm doing a
lift-then-place will usually leave one frame showing the slot empty, but
"usually" is doing real work in that sentence. Closing it needs either a
cadence faster than the swap, or something that tells the two jars apart.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from config import REGION_TRACKING

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LineageRecord:
    """One vial's passage from a heater slot to a cooling slot."""

    vial_id: str
    heater_slot: int
    heater_entry_ts: float
    heater_vacate_ts: float
    cooling_slot: int
    cooling_fill_ts: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "vial_id": self.vial_id,
            "heater_slot": self.heater_slot,
            "heater_entry_ts": self.heater_entry_ts,
            "heater_vacate_ts": self.heater_vacate_ts,
            "cooling_slot": self.cooling_slot,
            "cooling_fill_ts": self.cooling_fill_ts,
        }


@dataclass
class _Vacate:
    vial_id: str
    heater_slot: int
    heater_entry_ts: float
    vacate_ts: float


class VialLineage:
    """Tracks vials from heater entry to cooling arrival.

    Feed it per-frame occupancy - which heater slots and which cooling slots
    are full - and it emits the transitions and the links between them.

    There is no timeout on an unmatched vacate: a vial that left a heater
    stays claimable indefinitely. Deliberate for now, and the obvious thing
    to revisit once real per-stage timings exist.
    """

    def __init__(self) -> None:
        #: heater slot -> (vial on it, when it arrived)
        self._on_heater: dict[int, tuple[str, float]] = {}
        #: heater slot -> where its vial was last detected, for spotting a
        #: swap that never leaves the slot reading empty
        self._heater_pos: dict[int, tuple[float, float]] = {}
        #: cooling slots reading full, whether or not a vial is known for
        #: them - occupancy is what edges are computed from, and it has to be
        #: recorded even when the fill could not be linked, or the same slot
        #: reports itself newly filled on every later frame
        self._cooling_full: set[int] = set()
        #: cooling slot -> vial on it, for the ones that were linked
        self._on_cooling: dict[int, str] = {}
        #: heater vacates not yet claimed by a cooling fill, oldest first
        self._pending: list[_Vacate] = []

        self.records: list[LineageRecord] = []
        self.warnings: list[str] = []

    # ------------------------------------------------------------- queries
    def vial_on_heater(self, slot: int) -> str | None:
        entry = self._on_heater.get(slot)
        return entry[0] if entry else None

    def vial_on_cooling(self, slot: int) -> str | None:
        return self._on_cooling.get(slot)

    @property
    def pending_vacates(self) -> list[str]:
        return [v.vial_id for v in self._pending]

    # -------------------------------------------------------------- update
    def update(self, heater_occupied: dict[int, bool],
               cooling_occupied: dict[int, bool], timestamp: float,
               heater_pos: dict[int, tuple[float, float]] | None = None) -> None:
        """Absorb one frame's occupancy.

        Heater edges are processed before cooling edges, so a vial that
        leaves a heater and lands on the pad within one frame interval is
        still matched - at this cadence that is the normal case, not a race.

        `heater_pos` is optional: where each occupied heater slot's vial was
        detected. Given it, a slot whose detection jumps by more than
        REGION_TRACKING.heater_replacement_move_px is read as a swap - the
        vial that was there vacates and a new one arrives - which occupancy
        on its own cannot see.
        """
        self._heater_edges(heater_occupied, timestamp, heater_pos or {})
        self._cooling_edges(cooling_occupied, timestamp)

    def _heater_edges(self, occupied: dict[int, bool], timestamp: float,
                      positions: dict[int, tuple[float, float]]) -> None:
        for slot, is_occupied in sorted(occupied.items()):
            was_occupied = slot in self._on_heater

            if is_occupied and was_occupied and slot in positions:
                moved = self._moved(slot, positions[slot])
                if moved is not None and moved > REGION_TRACKING.heater_replacement_move_px:
                    old_id, entry_ts = self._on_heater.pop(slot)
                    self._pending.append(
                        _Vacate(old_id, slot, entry_ts, timestamp))
                    log.info("heater slot %d: detection moved %.1f px - reading "
                             "that as %s being replaced", slot, moved, old_id)
                    was_occupied = False        # fall through and mint a new one

            if is_occupied and not was_occupied:
                # Millisecond resolution, not seconds: two vials can occupy
                # the same slot inside one second when replaying captures
                # faster than they were taken, and second-resolution ids then
                # collide - two different vials sharing one id, each linked to
                # a different cooling slot.
                vial_id = f"heater-{slot}-{int(timestamp * 1000)}"
                self._on_heater[slot] = (vial_id, timestamp)
                log.info("heater slot %d: %s arrived", slot, vial_id)
            elif was_occupied and not is_occupied:
                vial_id, entry_ts = self._on_heater.pop(slot)
                self._pending.append(_Vacate(vial_id, slot, entry_ts, timestamp))
                log.info("heater slot %d: %s left, awaiting a cooling slot",
                         slot, vial_id)

            if is_occupied and slot in positions:
                self._heater_pos[slot] = positions[slot]
            elif not is_occupied:
                self._heater_pos.pop(slot, None)

    def _moved(self, slot: int, now: tuple[float, float]) -> float | None:
        was = self._heater_pos.get(slot)
        if was is None:
            return None
        return math.hypot(now[0] - was[0], now[1] - was[1])

    def _cooling_edges(self, occupied: dict[int, bool], timestamp: float) -> None:
        for slot, is_occupied in sorted(occupied.items()):
            was_occupied = slot in self._cooling_full
            if is_occupied and not was_occupied:
                self._cooling_full.add(slot)
                self._claim(slot, timestamp)
            elif was_occupied and not is_occupied:
                self._cooling_full.discard(slot)
                vial_id = self._on_cooling.pop(slot, None)
                log.info("cooling slot %d: %s left", slot, vial_id or "(unlinked)")

    def _claim(self, cooling_slot: int, timestamp: float) -> None:
        if not self._pending:
            # Left unlinked rather than linked to a guess: a vial on the pad
            # that no heater accounts for is a real thing to look at, whether
            # that is a missed vacate, a hand-placed vial or a false detection.
            msg = (f"cooling slot {cooling_slot} filled at {timestamp:.0f} with "
                   "no unmatched heater vacate pending - not linked")
            self.warnings.append(msg)
            log.warning(msg)
            return

        vacate = self._pending.pop()          # most recent
        self._on_cooling[cooling_slot] = vacate.vial_id
        self.records.append(LineageRecord(
            vial_id=vacate.vial_id,
            heater_slot=vacate.heater_slot,
            heater_entry_ts=vacate.heater_entry_ts,
            heater_vacate_ts=vacate.vacate_ts,
            cooling_slot=cooling_slot,
            cooling_fill_ts=timestamp,
        ))
        log.info("lineage: %s heater slot %d -> cooling slot %d",
                 vacate.vial_id, vacate.heater_slot, cooling_slot)

    # --------------------------------------------------------------- output
    def as_dict(self) -> dict[str, Any]:
        return {
            "on_heater": {s: v for s, (v, _t) in self._on_heater.items()},
            "on_cooling": dict(self._on_cooling),
            "pending_vacates": self.pending_vacates,
            "records": [r.as_dict() for r in self.records],
            "warnings": list(self.warnings),
        }
