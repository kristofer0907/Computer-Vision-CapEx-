"""Detector registry and the guarded call that isolates their failures.

Adding a detector is two lines: write the module, add its class to REGISTRY,
and put its name in DETECTION.enabled. Nothing else in the codebase changes.
"""

from __future__ import annotations

import logging

from config import DETECTION
from pipeline.detectors.base import (Detector, DetectionContext, ZoneView,
                                     NotImplementedDetector)
from pipeline.detectors.color_change import ColorChangeDetector
from pipeline.detectors.solgel import SolGelDetector
from pipeline.detectors.spill import SpillDetector
from pipeline.detectors.turbidity import TurbidityDetector
from pipeline.detectors.vial_presence import VialPresenceDetector
from pipeline.types import Event

log = logging.getLogger(__name__)

__all__ = ["Detector", "DetectionContext", "ZoneView", "NotImplementedDetector",
           "REGISTRY", "load_detectors", "DetectorHost"]

REGISTRY: dict[str, type[Detector]] = {
    TurbidityDetector.name: TurbidityDetector,
    SolGelDetector.name: SolGelDetector,
    ColorChangeDetector.name: ColorChangeDetector,
    SpillDetector.name: SpillDetector,
    VialPresenceDetector.name: VialPresenceDetector,
}


def load_detectors(names: tuple[str, ...] | list[str] | None = None
                   ) -> list[Detector]:
    """Instantiate the enabled detectors, in the order given.

    An unknown name is a hard error rather than a warning: a typo in the
    enabled list would otherwise silently switch off a failure mode the
    system is supposed to be watching, and it would look identical to that
    failure mode simply never occurring.
    """
    wanted = list(DETECTION.enabled if names is None else names)
    unknown = [n for n in wanted if n not in REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown detector(s) {unknown}; registered: {sorted(REGISTRY)}")
    return [REGISTRY[n]() for n in wanted]


class DetectorHost:
    """Runs a set of detectors so that one of them failing is survivable.

    A detector that raises is caught, counted and reported; the rest of the
    frame still processes and still gets stored. A detector that raises
    max_consecutive_errors times in a row is disabled for the life of the
    process - a detector broken by a refactor should cost one log entry, not
    one per frame forever. One success resets its counter.
    """

    def __init__(self, detectors: list[Detector] | None = None) -> None:
        self.detectors = detectors if detectors is not None else load_detectors()
        self._errors: dict[str, int] = {d.name: 0 for d in self.detectors}
        self._total_errors: dict[str, int] = {d.name: 0 for d in self.detectors}
        self._disabled: set[str] = set()
        self._started = False

    def start(self) -> None:
        for d in self.detectors:
            try:
                d.start()
            except Exception:
                log.exception("detector %r failed to start - disabling it", d.name)
                self._disabled.add(d.name)
        self._started = True

    def stop(self) -> None:
        for d in self.detectors:
            try:
                d.stop()
            except Exception:
                log.warning("detector %r failed to stop", d.name, exc_info=True)
        self._started = False

    def run(self, ctx: DetectionContext) -> tuple[list[Event], list[str]]:
        """(events from every healthy detector, warnings about the unhealthy)."""
        events: list[Event] = []
        warnings: list[str] = []

        for d in self.detectors:
            if d.name in self._disabled:
                continue
            try:
                produced = d.check(ctx) or []
            except Exception as exc:
                self._errors[d.name] += 1
                self._total_errors[d.name] += 1
                log.exception("detector %r raised", d.name)
                warnings.append(f"{d.name}: {type(exc).__name__}: {exc}")
                if self._errors[d.name] >= d.max_consecutive_errors:
                    self._disabled.add(d.name)
                    msg = (f"{d.name}: disabled after "
                           f"{d.max_consecutive_errors} consecutive failures")
                    log.error(msg)
                    warnings.append(msg)
                continue

            self._errors[d.name] = 0
            events.extend(produced)

        return events, warnings

    def health(self) -> list[dict]:
        """Per-detector status, for the dashboard."""
        return [
            {
                "name": d.name,
                "description": d.description,
                "implemented": d.implemented,
                "enabled": d.name not in self._disabled,
                "errors": self._total_errors.get(d.name, 0),
            }
            for d in self.detectors
        ]
