"""Left-heater to cold-storage lineage, driven by arrivals not departures.

One heater slot is watched. A crucible on it gets an id; the next crucible
to appear in cold storage is taken to be that one, and inherits the id.

The point of driving it from the destination is that the destination is the
reliable signal. Measured over capture/tracking_practice: cold storage filled
six times, one slot per frame, monotonic, never flickering. The heater over
the same run showed exactly one empty->occupied edge, because a crucible
lifted off and another set down between two captures leaves the slot reading
occupied throughout - so watching the heater for departures misses most of
them, while watching cold storage for arrivals misses none.

That inverts the usual framing: we do not detect that a crucible left the
heater and then look for where it went. We notice something arrive, and
conclude what must have left.

Two things this leans on, both stated rather than defended:

  * one crucible in flight at a time - an arrival is attributable to the one
    id currently held at the heater, with nothing to disambiguate if there
    were two;
  * everything reaching cold storage came via the watched heater. An arrival
    with no id waiting is reported rather than guessed at, so if that is
    wrong it shows up in the output instead of quietly mislabelling.

Appearance was measured as an alternative and rejected: in CIELAB mean
colour, the same crop compared across the two positions differs by 12.9-55.2
while different crucibles at cold storage differ by only 1.1-19.1. Moving a
crucible changes how it looks more than the crucibles differ from each
other, so matching on cached crops would be matching on where a thing is,
not which thing it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandoffRecord:
    crucible_id: str
    heater_slot: int
    heater_entry_ts: float
    storage_slot: int
    storage_fill_ts: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "crucible_id": self.crucible_id,
            "heater_slot": self.heater_slot,
            "heater_entry_ts": self.heater_entry_ts,
            "storage_slot": self.storage_slot,
            "storage_fill_ts": self.storage_fill_ts,
        }


class HeaterHandoff:
    """Watches one heater slot and the cold storage rack."""

    def __init__(self, heater_slot: int) -> None:
        self.heater_slot = heater_slot

        #: the id held by whatever is on the heater, waiting to be handed on
        self._held: str | None = None
        self._held_since: float | None = None
        #: storage slots reading full, for edge detection
        self._storage_full: set[int] = set()
        #: storage slot -> the id of the crucible in it
        self.storage_ids: dict[int, str] = {}

        self.records: list[HandoffRecord] = []
        self.warnings: list[str] = []

    @property
    def held_id(self) -> str | None:
        return self._held

    def update(self, heater_occupied: bool, storage_occupied: dict[int, bool],
               timestamp: float) -> None:
        """One frame.

        Storage is read before the heater. A crucible that moved to storage
        and was immediately replaced on the heater shows both at once, and
        handing the old id on first is what leaves the heater free to mint a
        new one in the same frame rather than a frame late.
        """
        self._storage_edges(storage_occupied, timestamp)
        self._heater_edge(heater_occupied, timestamp)

    def _storage_edges(self, occupied: dict[int, bool], timestamp: float) -> None:
        for slot, is_full in sorted(occupied.items()):
            was_full = slot in self._storage_full
            if is_full and not was_full:
                self._storage_full.add(slot)
                self._arrive(slot, timestamp)
            elif was_full and not is_full:
                self._storage_full.discard(slot)
                gone = self.storage_ids.pop(slot, None)
                log.info("storage slot %d: %s removed", slot, gone or "(unlabelled)")

    def _arrive(self, slot: int, timestamp: float) -> None:
        if self._held is None:
            msg = (f"storage slot {slot} filled at {timestamp:.0f} with no id "
                   "waiting on the heater - left unlabelled")
            self.warnings.append(msg)
            log.warning(msg)
            return

        self.storage_ids[slot] = self._held
        self.records.append(HandoffRecord(
            crucible_id=self._held, heater_slot=self.heater_slot,
            heater_entry_ts=self._held_since or timestamp,
            storage_slot=slot, storage_fill_ts=timestamp))
        log.info("handoff: %s heater -> storage slot %d", self._held, slot)
        self._held = None
        self._held_since = None

    def _heater_edge(self, occupied: bool, timestamp: float) -> None:
        if occupied and self._held is None:
            # Occupied with nothing held covers both a slot that was empty and
            # one whose crucible was just handed on and replaced between
            # frames - the second never reads empty, so waiting for an
            # empty->occupied edge would miss it.
            self._held = f"crucible-{int(timestamp * 1000)}"
            self._held_since = timestamp
            log.info("heater: %s picked up an id", self._held)
        elif not occupied and self._held is not None:
            # Left the heater without anything arriving in storage yet; keep
            # holding, the arrival is expected on a later frame.
            log.info("heater empty, still holding %s for the next arrival",
                     self._held)

    def as_dict(self) -> dict[str, Any]:
        return {
            "held_id": self._held,
            "storage_ids": dict(self.storage_ids),
            "records": [r.as_dict() for r in self.records],
            "warnings": list(self.warnings),
        }
