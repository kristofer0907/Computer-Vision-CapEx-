"""Fail-safe checks, run once per cycle.

One shape for every failure mode. A cycle hands in what the frame produced -
the image, where the crucibles are, optionally the zone polygons - and each
check hands back one `CheckResult`: did it fail, and by how much.

    cycle = Cycle(image=frame.image, positions=..., timestamp=..., ...)
    for r in run_checks(cycle):
        if r.failed:
            log.error("%s: %s", r.name, r.message)

Why functions and not classes: a check that looks at one frame and answers a
question about it has no state to keep, and wrapping it in an object only
hides that. State belongs to whatever is doing something with the answer -
latching a hazard, counting frames before committing, deciding to stop. That
is `MissingLid` at the bottom of this file: the stateless score is
`check_missing_lid`, the latch and the run-stop are the class. New checks
start as functions and only grow a class if they actually need to remember
something between cycles.

`score` is the number the verdict came from and is per-check, not comparable
across checks - a lid gradient of 55 and a turbidity z of 3.5 mean nothing to
each other. `subjects` carries the same number per crucible, so an overlay or
a log can point at the one that failed rather than at the frame.

Three of the five checks are not written yet. They return
`implemented=False, failed=False` so the whole set can be wired into the
runner now and read honestly on a dashboard: registered, looking at nothing.

The standing constraint: no threshold in this file has been calibrated. The
pipeline has been exercised against synthetic and bench frames, never against
real chemistry, and a threshold crossing today demonstrates that the plumbing
works and nothing whatsoever about the batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import DETECTION, REGION_TRACKING
from pipeline.features import lid_score, rim_circularity, rim_gradient
from pipeline.types import Event

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# What one cycle hands to every check
# --------------------------------------------------------------------------
@dataclass
class Cycle:
    """One analysis frame's worth of input, shared by every check.

    `positions` is the only field a check can usually count on: crucible id
    (heater slot, or track id - the caller decides, the checks only ever pass
    it back out) mapped to its centre in frame pixels. `zones` is optional
    because most checks look at a crucible, not at a region; a check that
    needs a zone and did not get one returns not-implemented rather than
    guessing where the heater is.
    """

    image: np.ndarray
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    #: zone name -> polygon in frame pixels, as ZoneMap.polygons_px gives it.
    zones: dict[str, np.ndarray] | None = None
    timestamp: float = 0.0
    frame_id: int = 0

    def zone(self, name: str) -> np.ndarray | None:
        return None if self.zones is None else self.zones.get(name)


@dataclass
class CheckResult:
    """One check's answer for one cycle."""

    name: str
    #: The verdict. False whenever the check is not implemented.
    failed: bool = False
    #: The number the verdict came from. Per-check units, None if unscored.
    score: float | None = None
    message: str = ""
    #: crucible id -> that crucible's score, for the ones this check looked at.
    subjects: dict[int, float] = field(default_factory=dict)
    #: The ids that tripped the threshold - a subset of `subjects`.
    offenders: list[int] = field(default_factory=list)
    implemented: bool = True

    def __bool__(self) -> bool:
        return self.failed


def _todo(name: str, what: str) -> CheckResult:
    """Placeholder result for a check that has not been written."""
    return CheckResult(name=name, failed=False, implemented=False,
                       message=f"not implemented: {what}")


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------
def check_missing_lid(cycle: Cycle,
                      threshold: float | None = None) -> CheckResult:
    """Any crucible in `positions` without a lid.

    `lid_score` is a mean gradient magnitude in a fixed window on the
    crucible centre - a lid breaks up the smooth interior whether it reads
    bright or dark. Low score means open.

    Stateless: it answers for this frame only. The latch, and the decision to
    stop the run, live in `MissingLid` below.
    """
    limit = DETECTION.lid_score_threshold if threshold is None else threshold
    scores = {cid: lid_score(cycle.image, cx, cy)
              for cid, (cx, cy) in cycle.positions.items()}
    open_jars = sorted(cid for cid, s in scores.items() if s < limit)
    if open_jars:
        ids = ", ".join(str(c) for c in open_jars)
        message = f"crucible(s) {ids} have no lid (below {limit:.1f})"
    elif scores:
        message = f"all {len(scores)} crucible(s) lidded"
    else:
        message = "no crucibles to check"
    return CheckResult(
        name="missing_lid",
        failed=bool(open_jars),
        score=min(scores.values()) if scores else None,
        message=message,
        subjects={cid: round(s, 1) for cid, s in scores.items()},
        offenders=open_jars,
    )


def check_fallen_crucible(cycle: Cycle,
                          threshold: float | None = None) -> CheckResult:
    """Any crucible in `positions` lying on its side rather than standing.

    `rim_circularity` is the measure: an upright crucible closes a full
    circle of rim about its own centre, a tipped one does not, because what
    is under the sampling circle is the cylinder wall. Low score means
    tipped. Stateless, single frame, no batch reference needed - the score
    is absolute in a way the colour and turbidity features are not, because
    it is measuring a shape and not an appearance.

    Two limits worth knowing before trusting an all-clear:

    - It only ever sees the positions it is handed, and those must be where
      a crucible is *expected* - a slot centre - not where one was detected.
      detect_crucibles() finds none of the tipped crucibles in
      capture/fail_safe: a crucible on its side stops being a circle.

    - It is blind off the plate, and that is a real gap, not a rounding
      error. capture/fail_safe holds five tipped crucibles, not the three on
      the plate: two more lie on the pegboard between the injector and the
      plate. rim_circularity scores them 0.58 and 0.80, well under the
      threshold - the measure is right, there is simply no anchor pointing at
      them. See "off-plate tip-over" in CLAUDE.md for why nothing cheap
      fixes that.
    - The threshold is a mid-gap guess over four objects in one scene. See
      DETECTION.fallen_circularity_threshold.
    """
    limit = (DETECTION.fallen_circularity_threshold if threshold is None
             else threshold)
    # One Sobel pass for the whole frame rather than one per crucible.
    mag = rim_gradient(cycle.image) if cycle.positions else None
    scores = {cid: rim_circularity(cycle.image, cx, cy, _mag=mag)
              for cid, (cx, cy) in cycle.positions.items()}
    tipped = sorted(cid for cid, s in scores.items() if s < limit)
    if tipped:
        ids = ", ".join(str(c) for c in tipped)
        message = f"crucible(s) {ids} have tipped over (below {limit:.2f})"
    elif scores:
        message = f"all {len(scores)} crucible(s) upright"
    else:
        message = "no crucibles to check"
    return CheckResult(
        name="fallen_crucible",
        failed=bool(tipped),
        score=min(scores.values()) if scores else None,
        message=message,
        subjects={cid: round(s, 3) for cid, s in scores.items()},
        offenders=tipped,
    )


def check_turbidity(cycle: Cycle) -> CheckResult:
    """Liquid gone cloudy or hazy.

    Wants texture variance inside the liquid disc, compared against the
    batch median rather than an absolute value.
    """
    return _todo("turbidity", "needs real cloudy-vs-clear captures")


def check_solgel(cycle: Cycle) -> CheckResult:
    """Sol-to-gel transition - liquid setting to solid.

    Wants frame-to-frame differencing inside the disc: a set gel stops
    showing the small surface motion a liquid does. Rate-based, so it needs
    the cycle interval, which is why this one will likely be the first to
    need more than a single frame.
    """
    return _todo("solgel", "needs a per-crucible trajectory, not one frame")


def check_color_change(cycle: Cycle) -> CheckResult:
    """Gross colour change of the liquid, during heating.

    Wants mean HSV inside the disc against the batch median, restricted to
    crucibles at the same stage. Qualitative only - this is not colorimetry,
    and the flat-field correction the LED panel needs is not in place.
    """
    return _todo("color_change", "needs flat-field correction and a baseline")


#: Every fail-safe, in the order they run. Add new checks here.
CHECKS: tuple[Callable[[Cycle], CheckResult], ...] = (
    check_missing_lid,
    check_fallen_crucible,
    check_turbidity,
    check_solgel,
    check_color_change,
)


def run_checks(cycle: Cycle,
               checks: tuple[Callable[[Cycle], CheckResult], ...] = CHECKS,
               ) -> list[CheckResult]:
    """Run every check over one cycle and return all their results.

    A check that raises is caught and logged and comes back as a failed=False
    result, so one broken check cannot take the cycle - or the other four -
    down with it.
    """
    results: list[CheckResult] = []
    for fn in checks:
        name = getattr(fn, "__name__", "check").removeprefix("check_")
        try:
            results.append(fn(cycle))
        except Exception:
            log.exception("check %r raised - skipping it this cycle", name)
            results.append(CheckResult(name=name, failed=False,
                                       implemented=False,
                                       message="raised, see log"))
    return results


def failures(results: list[CheckResult]) -> list[CheckResult]:
    """Just the checks that tripped."""
    return [r for r in results if r.failed]


def to_events(results: list[CheckResult], cycle: Cycle,
              severity: str = "alert") -> list[Event]:
    """Failed checks as Events, for the log and the dashboard."""
    return [
        Event(kind=r.name, severity=severity, message=r.message,
              timestamp=cycle.timestamp, frame_id=cycle.frame_id,
              detector=r.name,
              data={"score": r.score, "offenders": r.offenders,
                    "subjects": r.subjects})
        for r in results if r.failed
    ]


# --------------------------------------------------------------------------
# The one check that needs memory
# --------------------------------------------------------------------------
class MissingLid:
    """A crucible put on a heater without a lid. Latching, and stops the run.

    Checked once per arrival, when a crucible first appears in a heater slot:
    that is when the decision was made and when it can still be acted on.
    The verdict then latches: the slot stays flagged in `open_slots` so a
    standing hazard keeps showing rather than scrolling past, while only the
    one event is raised so the log does not fill with repeats. Because the
    first flag also stops the run, that state is effectively frozen at the
    moment of the stop - it is the record of why everything halted.

    The scoring itself is `check_missing_lid` above - this class is only the
    memory around it: which slots have already been asked, what the answer
    was, and whether that answer stopped the run.

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

    def __init__(self, heating_zone: str | None = None,
                 threshold: float | None = None,
                 on_stop: Callable[[Event], None] | None = None) -> None:
        self.heating_zone = heating_zone or REGION_TRACKING.heating_zone
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
        arrived = sorted(present - self._seen)
        scored = check_missing_lid(
            Cycle(image=image, positions={s: on_heater[s] for s in arrived},
                  timestamp=timestamp, frame_id=frame_id),
            threshold=self.threshold)

        events: list[Event] = []
        for slot in arrived:
            cx, cy = on_heater[slot]
            score = scored.subjects[slot]
            if slot not in scored.offenders:
                log.info("heater slot %d: lid present (%.1f)", slot, score)
                continue
            event = Event(
                kind="missing_lid", severity="alert",
                message=(f"crucible placed on heater slot {slot} without a "
                         f"lid (score {score:.1f}, below {self.threshold:.1f})"
                         " - stopping"),
                timestamp=timestamp, frame_id=frame_id,
                detector=self.name, zone=self.heating_zone,
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
