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
    """A crucible put on a heater without a lid.

    Checked once, when a crucible first appears on a heater slot, because
    that is the moment the decision was made and the moment it can still be
    acted on. Re-checking every frame afterwards would mostly re-report the
    same jar, and lid_score is a per-frame verdict on an overlapping
    distribution - given enough frames one of them will read wrong, and a
    detector that cries wolf on a correctly lidded crucible is worse than one
    that speaks once.

    Raised at "alert", the highest severity: heating an open crucible is a
    safety matter, not a process-quality observation like the other classes
    in this file.

    The verdict is only as good as pipeline.features.lid_score, which reads
    the middle of a crucible and asks whether it is busy or smooth - see that
    function for what it actually measures and where it fails. Two things
    inherited from it matter here:

      * it needs the detection to be centred, since it samples a fixed
        window on the reported centre;
      * lid and open overlap (56.0-82.7 against 4.8-67.7 on the labelled
        set), so a single frame's answer is worth doubting. Every labelled
        lid is caught, and the errors that remain are open jars called
        lidded - which for this detector is the quiet direction: it stays
        silent rather than raising a false alarm.
    """

    #: Registry-style name, so this reads the same as pipeline/detectors/.
    name = "missing_lid"
    description = "crucible placed on a heater without a lid"

    def __init__(self, heater_zone: str | None = None,
                 threshold: float | None = None) -> None:
        self.heater_zone = heater_zone or REGION_TRACKING.heater_zone
        self.threshold = (DETECTION.lid_score_threshold
                          if threshold is None else threshold)
        #: heater slots occupied on the previous frame, to fire on arrival only
        self._seen: set[int] = set()

    def reset(self) -> None:
        self._seen.clear()

    def check(self, image: np.ndarray, on_heater: dict[int, tuple[float, float]],
              timestamp: float, frame_id: int = 0) -> list[Event]:
        """Score crucibles that just arrived on a heater.

        `on_heater` maps heater slot -> the crucible's (cx, cy) this frame.
        Slots already occupied last frame are skipped; slots that emptied are
        forgotten, so the same slot filling again is checked again.
        """
        events: list[Event] = []
        present = set(on_heater)

        for slot in sorted(present - self._seen):
            cx, cy = on_heater[slot]
            score = lid_score(image, cx, cy)
            if score >= self.threshold:
                log.info("heater slot %d: lid present (%.1f)", slot, score)
                continue
            events.append(Event(
                kind="missing_lid", severity="alert",
                message=(f"crucible placed on heater slot {slot} without a "
                         f"lid (score {score:.1f}, below {self.threshold:.1f})"),
                timestamp=timestamp, frame_id=frame_id,
                detector=self.name, zone=self.heater_zone,
                data={"heater_slot": slot, "lid_score": round(score, 1),
                      "threshold": self.threshold, "cx": cx, "cy": cy},
            ))
            log.error("heater slot %d: NO LID (%.1f) - crucible is about to be "
                      "heated open", slot, score)

        self._seen = present
        return events
