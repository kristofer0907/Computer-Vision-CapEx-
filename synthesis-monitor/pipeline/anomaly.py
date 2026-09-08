"""Shared anomaly scoring.  ***YOURS, IF YOU WANT IT.***

Left empty on purpose. All five detectors are planned around batch-median
comparison, so there is an obvious temptation to write one scorer here and
have every detector call it. That may well be right - but which features go
in, whether peers are restricted by stage, how many frames a divergence must
persist and where the threshold sits are all decisions that differ per failure
mode, and factoring them together before any of them has been calibrated
against real chemistry would lock in a shape that has not been tested.

If a common scorer does emerge, this is where it goes: import it from the
detectors rather than growing a second copy in each.

The raw material is already available:

    pipeline.stats.median / mad / robust_z / iqr
    ctx.feature_column(key, stage=...)   the batch's values for one feature
    ctx.history.series(tid, key)         one vial's trajectory over time
    DETECTION.robust_z_threshold         placeholder threshold, uncalibrated
    DETECTION.min_vials_for_batch_stats  minimum peers before the median means
                                         anything

The standing constraint, worth repeating here because this is the file where
it is easiest to forget: no threshold in this system has been calibrated. The
pipeline has been exercised against synthetic frames only, never against real
chemistry, and a threshold crossing today demonstrates that the plumbing works
and nothing whatsoever about the batch.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from config import DETECTION, REGION_TRACKING
from pipeline.features import lid_score
from pipeline.types import Event

log = logging.getLogger(__name__)

class Turbidity():
    """
    Detect change in cloudiness or haziness of the liquid
    """
    def __init__(self):
        pass

class SolGelTrans():
    """
    Detect change in matter from sol to gel 
    """
    def __init__(self):
        pass

class ColorChange():
    """
    Detect change in color of the crucible liquid when in heating stage
    """
    def __init__(self):
        pass

class FallenCrucible():
    """
    Detect if a crucible has fallen over
    """
    def __init__(self):
        pass

class MissingLid:
    """A crucible put on a heater without a lid. Latching, and stops the run.

    Checked once per arrival, when a crucible first appears in a heater slot:
    that is when the decision was made and when it can still be acted on.
    The verdict then latches: the slot stays flagged in `open_slots` so a
    standing hazard keeps showing rather than scrolling past, while only the
    one event is raised so the log does not fill with repeats. Because the
    first flag also stops the run, that state is effectively frozen at the
    moment of the stop - it is the record of why everything halted.

    Latching also protects the verdict. lid_score reads an overlapping
    distribution, so re-scoring the same jar every frame would eventually
    produce a frame that says "lid" and silently clear a real hazard. Asking
    once and holding the answer is the safer shape.

    Once any slot is flagged, `stop_requested` is set and further frames are
    not scored: an open crucible on a live heater is not a condition to keep
    measuring through.

    WHAT THIS STOPS, AND WHAT IT DOES NOT. It stops this pipeline. There is
    no control channel from this system to the synthesis platform - it is a
    camera and a process that looks at pictures, and stopping the robot is
    out of scope for this phase (see CLAUDE.md). So this raises the alarm
    and halts our own analysis; acting on it is a person's job, or a
    controller integration that does not exist yet. `on_stop` is where that
    integration attaches when it does. Reading this class as an interlock
    that makes the bench safe would be a serious misreading.
    """

    #: Registry-style name, so this reads the same as pipeline/detectors/.
    name = "missing_lid"
    description = "crucible placed on a heater without a lid"

    def __init__(self, heater_zone: str | None = None,
                 threshold: float | None = None,
                 on_stop: Callable[[Event], None] | None = None) -> None:
        self.heater_zone = heater_zone or REGION_TRACKING.heater_zone
        self.threshold = (DETECTION.lid_score_threshold
                          if threshold is None else threshold)
        #: called once when the stop is raised - the seam a real platform
        #: stop would attach to. Nothing is wired to it today.
        self.on_stop = on_stop

        self._seen: set[int] = set()
        self._open: dict[int, Event] = {}
        self._stopped = False

    # ------------------------------------------------------------- queries
    @property
    def stop_requested(self) -> bool:
        """True once an unlidded crucible has been seen on a heater."""
        return self._stopped

    @property
    def open_slots(self) -> dict[int, Event]:
        """Heater slots currently believed to hold an unlidded crucible."""
        return dict(self._open)

    def reset(self) -> None:
        self._seen.clear()
        self._open.clear()
        self._stopped = False

    # -------------------------------------------------------------- update
    def check(self, image: np.ndarray, on_heater: dict[int, tuple[float, float]],
              timestamp: float, frame_id: int = 0) -> list[Event]:
        """Score crucibles that just arrived on a heater.

        `on_heater` maps heater slot -> the crucible's (cx, cy) this frame.
        Returns the events raised by this frame, which is at most one per
        slot that just filled, and nothing at all once stopped.
        """
        if self._stopped:
            return []

        present = set(on_heater)
        events: list[Event] = []
        for slot in sorted(present - self._seen):
            cx, cy = on_heater[slot]
            score = lid_score(image, cx, cy)
            if score >= self.threshold:
                log.info("heater slot %d: lid present (%.1f)", slot, score)
                continue
            event = Event(
                kind="missing_lid", severity="alert",
                message=(f"crucible placed on heater slot {slot} without a "
                         f"lid (score {score:.1f}, below {self.threshold:.1f})"
                         " - stopping"),
                timestamp=timestamp, frame_id=frame_id,
                detector=self.name, zone=self.heater_zone,
                data={"heater_slot": slot, "lid_score": round(score, 1),
                      "threshold": self.threshold, "cx": cx, "cy": cy,
                      "stop_requested": True},
            )
            self._open[slot] = event
            events.append(event)
            log.error("heater slot %d: NO LID (%.1f) - crucible is about to be "
                      "heated open", slot, score)

        self._seen = present

        if events and not self._stopped:
            self._stopped = True
            log.error("STOP requested by %s. This halts the monitoring "
                      "pipeline only - nothing here can stop the platform.",
                      self.name)
            if self.on_stop is not None:
                self.on_stop(events[0])

        return events
