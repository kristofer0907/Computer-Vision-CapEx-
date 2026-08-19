"""Data types passed between pipeline stages and across process boundaries.

Everything here must stay picklable and cheap: PipelineResult crosses a
multiprocessing.Queue on every analysis frame. That is why results carry
numbers and an already-encoded JPEG, never raw numpy images. Raw images stay
inside the processing process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Severities, ordered. Used for dashboard colouring and for deciding whether
# an event is worth saving a crop for.
# --------------------------------------------------------------------------
SEVERITIES = ("info", "warning", "alert")


def severity_rank(name: str) -> int:
    try:
        return SEVERITIES.index(name)
    except ValueError:
        return 0


@dataclass(frozen=True)
class Detection:
    """One candidate vial in one frame, in pixel coordinates.

    This is the localiser's output and the tracker's input. It carries no
    identity: assigning identity across frames is the tracker's job.
    """

    cx: float
    cy: float
    radius: float
    confidence: float = 1.0
    # Free-form, for whatever the localiser wants to pass through
    # (contour area, YOLO class, circularity...).
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return self.cx, self.cy

    def bbox(self, scale: float = 2.0) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) square box of `scale` * radius, unclipped."""
        half = self.radius * scale * 0.5
        return (int(round(self.cx - half)), int(round(self.cy - half)),
                int(round(self.cx + half)), int(round(self.cy + half)))


@dataclass
class Track:
    """A vial followed across analysis frames.

    Mutable by design - the tracker updates it in place. `stage` is the
    committed stage; `pending_stage`/`pending_count` are the hysteresis
    machinery and are not published.
    """

    track_id: int
    cx: float
    cy: float
    radius: float
    first_seen_ts: float
    last_seen_ts: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False

    stage: str | None = None
    pending_stage: str | None = None
    pending_count: int = 0
    stage_since_ts: float | None = None
    # Every committed stage transition, oldest first: (stage, timestamp).
    stage_log: list[tuple[str, float]] = field(default_factory=list)

    # Set when the track ends. "oven" means inferred oven entry, "lost" means
    # it vanished from somewhere it should not have.
    closed_reason: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        return self.cx, self.cy

    @property
    def age_s(self) -> float:
        return self.last_seen_ts - self.first_seen_ts

    def time_in_stage_s(self, now: float | None = None) -> float | None:
        if self.stage_since_ts is None:
            return None
        return (now if now is not None else time.time()) - self.stage_since_ts


@dataclass(frozen=True)
class Event:
    """Something a detector wants a human to know about.

    `kind` is the detector-defined event type, e.g. "turbidity_divergence".
    `data` is free-form and is stored as JSON, so keep it to plain types.
    """

    kind: str
    severity: str
    message: str
    timestamp: float
    frame_id: int
    track_id: int | None = None
    detector: str = ""
    zone: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"unknown severity {self.severity!r}, expected one of {SEVERITIES}")


@dataclass
class VialReport:
    """Per-vial state published for one analysis frame.

    This is what the dashboard renders and what storage writes as a row.
    `features` is whatever pipeline.features returned - it is passed through
    untouched so adding a feature needs no change here, in storage, or in the
    dashboard.
    """

    track_id: int
    cx: float
    cy: float
    radius: float
    stage: str | None
    hits: int
    missed: int
    age_s: float
    time_in_stage_s: float | None
    features: dict[str, float] = field(default_factory=dict)
    # Filled by detectors that want to attach a score without raising an
    # event, e.g. a per-vial anomaly score that is below threshold.
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """One analysis frame, fully processed. Crosses the queue to the dashboard."""

    frame_id: int
    timestamp: float
    source: str
    simulated: bool
    vials: list[VialReport] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    # stage name -> number of confirmed tracks currently in it
    stage_counts: dict[str, int] = field(default_factory=dict)
    # Wall-clock milliseconds per pipeline stage, for spotting what is slow.
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Annotated preview, already JPEG encoded. None when overlay is disabled.
    overlay_jpeg: bytes | None = None
    # Non-fatal problems from this frame (a detector raised, the localiser
    # is a stub, ...). Surfaced on the dashboard rather than only in the log.
    warnings: list[str] = field(default_factory=list)

    @property
    def n_vials(self) -> int:
        return len(self.vials)

    def worst_severity(self) -> str | None:
        if not self.events:
            return None
        return max((e.severity for e in self.events), key=severity_rank)
