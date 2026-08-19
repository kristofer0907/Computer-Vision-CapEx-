"""Source interfaces shared by real hardware and simulated backends.

Import rule for this whole package: no module-level import of a Raspberry Pi
library. picamera2 / libcamera / board / busio / adafruit_mlx90640 are system
packages tied to the Pi camera stack and do not resolve on a laptop. They are
imported lazily inside start(), so every module here imports cleanly on any
machine and the failure - if there is one - happens at start-up, where it can
be reported and handled, instead of at import time.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class HardwareUnavailable(RuntimeError):
    """Raised by a real backend when its device or driver is not present."""


@dataclass(frozen=True)
class Frame:
    """One RGB capture.

    image: HxWx3, uint8, BGR channel order (what cv2 expects).
    """

    image: np.ndarray
    timestamp: float          # unix seconds, wall clock
    frame_id: int             # monotonic per source
    source: str               # backend name, e.g. "picamera2" / "mock"
    simulated: bool = False
    # Ground truth, populated by simulated backends only. A real camera leaves
    # this None and nothing in the pipeline may require it - it exists so the
    # plumbing can be exercised end to end before a localiser is written.
    truth: dict[str, Any] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]


@dataclass(frozen=True)
class ThermalFrame:
    """One thermal capture. celsius: 24x32 float32, degrees C."""

    celsius: np.ndarray
    timestamp: float
    frame_id: int
    source: str
    simulated: bool = False

    @property
    def stats(self) -> tuple[float, float, float]:
        a = self.celsius
        return float(a.min()), float(a.max()), float(a.mean())


class _Source(ABC):
    """Common lifecycle. capture() is deliberately blocking - it matches the
    multiprocessing capture-loop design and needs no async machinery at a
    30-60 s cadence."""

    name: str = "unset"
    simulated: bool = False

    def __init__(self) -> None:
        self._frame_id = 0
        self._started = False

    @abstractmethod
    def start(self) -> None:
        """Open the device. Raises HardwareUnavailable if it is not there."""

    @abstractmethod
    def stop(self) -> None:
        """Release the device. Must be safe to call twice."""

    def _next_id(self) -> int:
        self._frame_id += 1
        return self._frame_id

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def describe(self) -> str:
        kind = "SIMULATED" if self.simulated else "hardware"
        return f"{self.name} ({kind})"


class CameraSource(_Source):
    @abstractmethod
    def capture(self) -> Frame:
        """Block until one frame is available and return it."""


class ThermalSource(_Source):
    @abstractmethod
    def capture(self) -> ThermalFrame:
        """Block until one thermal frame is available and return it."""


def sleep_remaining(started_at: float, target_s: float) -> None:
    """Sleep only for whatever is left of target_s since started_at.

    Used by simulated backends so they impose a realistic capture latency
    without double-charging for work already done.
    """
    remaining = target_s - (time.monotonic() - started_at)
    if remaining > 0:
        time.sleep(remaining)
