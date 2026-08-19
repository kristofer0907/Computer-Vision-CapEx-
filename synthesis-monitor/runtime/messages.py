"""Payloads that cross process boundaries.

Everything here is a plain dataclass of primitives and bytes, because it gets
pickled onto a multiprocessing.Queue. Two consequences worth stating plainly:

  * No numpy images on the preview or thermal queues. A 1280x720 BGR frame is
    2.7 MB pickled; the same frame JPEG-encoded is around 80 kB. The encode
    happens in the process that owns the camera, where there is time to spare
    between captures, rather than in the dashboard where it would block a
    request.

  * The one exception is the analysis queue, which carries raw Frames on
    purpose - the pipeline needs real pixels, and at one frame per 45 s the
    2.7 MB is about 60 kB/s. Do not widen that queue to hide a slow pipeline;
    the cost lands in RAM on a Pi 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreviewMessage:
    """A JPEG for the live view, plus enough context to say how stale it is."""

    jpeg: bytes
    timestamp: float
    frame_id: int
    source: str
    simulated: bool


@dataclass
class ThermalMessage:
    """A colourised thermal JPEG and the three numbers worth displaying.

    Passive logging only: no model, no algorithm and no part of the anomaly
    logic runs on thermal data. At 2.4 cm/px a vial is one or two pixels.
    """

    jpeg: bytes
    timestamp: float
    frame_id: int
    source: str
    simulated: bool
    min_c: float
    mean_c: float
    max_c: float


@dataclass
class WorkerStatus:
    """Heartbeat from a worker process, so the dashboard can tell 'idle' from
    'dead'. A source that has published nothing for a minute is a real
    condition and looks exactly like a slow cadence unless the worker says so."""

    worker: str
    alive: bool = True
    source: str | None = None
    simulated: bool | None = None
    frames: int = 0
    errors: int = 0
    dropped: int = 0
    last_frame_ts: float | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def copy(self) -> "WorkerStatus":
        """Snapshot for queueing.

        Workers keep one long-lived status object and mutate it. Queueing it
        directly would be fine today because pickling copies it, but it makes
        the correctness depend on the transport - and the moment anything
        keeps a local reference instead, the dashboard starts showing counters
        that change under it.
        """
        return WorkerStatus(**{**self.__dict__, "extra": dict(self.extra)})
