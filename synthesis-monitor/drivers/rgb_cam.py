"""RGB camera backends.

Three interchangeable implementations of CameraSource:

    PiCameraSource   - IMX477 via picamera2 (Raspberry Pi only)
    MockCameraSource - synthetic platform renderer, runs anywhere
    FileCameraSource - replay of a recorded video or a folder of stills

picamera2 is imported inside start(), never at module level, so this file
imports on a laptop exactly as it does on the Pi. create_camera() picks a
backend once at start-up and everything downstream talks to the interface.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from config import DASHBOARD, GEOMETRY, SOURCES
from drivers.base import CameraSource, Frame, HardwareUnavailable, sleep_remaining
from drivers.scene import SyntheticPlatform

log = logging.getLogger(__name__)


class PiCameraSource(CameraSource):
    """IMX477 + Arducam varifocal through picamera2."""

    name = "picamera2"
    simulated = False

    def __init__(self, width: int | None = None, height: int | None = None) -> None:
        super().__init__()
        self.width = width or GEOMETRY.frame_width_px
        self.height = height or GEOMETRY.frame_height_px
        self._cam = None

    def start(self) -> None:
        if self._started:
            return
        try:
            from picamera2 import Picamera2  # Pi-only, imported lazily on purpose
        except Exception as exc:  # ModuleNotFoundError on any non-Pi machine
            raise HardwareUnavailable(
                f"picamera2 not importable ({exc}). It is a Raspberry Pi system "
                "package and cannot be pip-installed elsewhere."
            ) from exc

        try:
            cam = Picamera2()
            cam.configure(
                cam.create_preview_configuration(
                    main={"format": SOURCES.picamera2_format,
                          "size": (self.width, self.height)}
                )
            )
            cam.start()
            time.sleep(1.0)  # let AE/AWB settle before the first analysed frame
        except Exception as exc:
            raise HardwareUnavailable(f"IMX477 open failed: {exc}") from exc

        self._cam = cam
        self._started = True
        log.info("picamera2 started at %dx%d %s",
                 self.width, self.height, SOURCES.picamera2_format)

    def capture(self) -> Frame:
        if not self._started:
            raise RuntimeError("capture() before start()")
        # RGB888 in picamera2 lands in memory as BGR, which is what cv2 wants.
        img = self._cam.capture_array()
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return Frame(img, time.time(), self._next_id(), self.name, False)

    def stop(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                log.warning("picamera2 stop failed", exc_info=True)
            self._cam = None
        self._started = False


class MockCameraSource(CameraSource):
    """Synthetic 18-vial platform. Behaves like the real thing on the wire."""

    name = "mock"
    simulated = True

    def __init__(self, time_scale: float | None = None,
                 latency_s: float | None = None) -> None:
        super().__init__()
        self.time_scale = SOURCES.mock_time_scale if time_scale is None else time_scale
        self.latency_s = SOURCES.mock_rgb_latency_s if latency_s is None else latency_s
        self.scene = SyntheticPlatform()
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._wall_t0 = time.time()
        self._started = True
        log.info("mock camera started (time scale x%.1f, latency %.0f ms)",
                 self.time_scale, self.latency_s * 1000)

    @property
    def sim_time(self) -> float:
        return (time.monotonic() - self._t0) * self.time_scale

    def capture(self) -> Frame:
        if not self._started:
            raise RuntimeError("capture() before start()")
        t_begin = time.monotonic()
        sim_t = self.sim_time
        img = self.scene.render(sim_t)
        truth = self.scene.truth_at(sim_t)
        # Real capture latency, on purpose: an instant mock hides the queue
        # backpressure bugs that only surface against hardware.
        sleep_remaining(t_begin, self.latency_s)
        # Timestamp on the SIMULATED clock, not the wall clock. The scene
        # advances at time_scale, so wall-clock stamps would tell the tracker
        # that vials teleport - and any dt-dependent logic (the association
        # gate, dwell times, stall thresholds) would be tested against a lie.
        return Frame(img, self._wall_t0 + sim_t, self._next_id(), self.name,
                     True, truth)

    def stop(self) -> None:
        self._started = False


class FileCameraSource(CameraSource):
    """Replay a recorded run. Same interface, so the pipeline cannot tell."""

    name = "file"
    simulated = True

    def __init__(self, path: str, loop: bool = True,
                 latency_s: float | None = None) -> None:
        super().__init__()
        self.path = Path(path)
        self.loop = loop
        self.latency_s = SOURCES.mock_rgb_latency_s if latency_s is None else latency_s
        self._cap = None
        self._stills: list[Path] = []
        self._idx = 0

    def start(self) -> None:
        if not self.path.exists():
            raise HardwareUnavailable(f"replay path does not exist: {self.path}")
        if self.path.is_dir():
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            self._stills = sorted(p for p in self.path.iterdir()
                                  if p.suffix.lower() in exts)
            if not self._stills:
                raise HardwareUnavailable(f"no images in {self.path}")
        else:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise HardwareUnavailable(f"cannot open video {self.path}")
            self._cap = cap
        self._started = True
        log.info("file source started: %s", self.path)

    def capture(self) -> Frame:
        if not self._started:
            raise RuntimeError("capture() before start()")
        t_begin = time.monotonic()

        # Which file this frame came from, carried on Frame.truth. It is the
        # only way anything downstream can tell one still from another, and
        # ManualLocalizer keys its hand-marked vial positions off it.
        origin: dict[str, object] = {}

        if self._stills:
            if self._idx >= len(self._stills):
                if not self.loop:
                    raise StopIteration("end of stills")
                self._idx = 0
            still = self._stills[self._idx]
            img = cv2.imread(str(still))
            if img is None:
                raise HardwareUnavailable(f"could not decode {still}")
            origin = {"name": still.name, "path": str(still),
                      "index": self._idx}
            self._idx += 1
        else:
            ok, img = self._cap.read()
            if not ok:
                if not self.loop:
                    raise StopIteration("end of video")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
                if not ok:
                    raise HardwareUnavailable("replay source produced no frames")
            origin = {"name": self.path.name, "path": str(self.path),
                      "index": int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))}

        sleep_remaining(t_begin, self.latency_s)
        return Frame(img, time.time(), self._next_id(), self.name, True, origin)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._started = False


def create_camera(backend: str | None = None) -> CameraSource:
    """Build and start a camera. "auto" probes hardware, then falls back.

    The caller gets a started source and never has to know which one it is.
    """
    backend = (backend or SOURCES.rgb_backend).lower()

    if backend == "picamera2":
        cam = PiCameraSource()
        cam.start()
        return cam
    if backend == "mock":
        cam = MockCameraSource()
        cam.start()
        return cam
    if backend == "file":
        cam = FileCameraSource(SOURCES.rgb_file_path)
        cam.start()
        return cam
    if backend != "auto":
        raise ValueError(f"unknown rgb backend: {backend!r}")

    try:
        cam = PiCameraSource()
        cam.start()
        log.info("RGB source: %s", cam.describe())
        return cam
    except HardwareUnavailable as exc:
        log.warning("RGB hardware unavailable (%s) - falling back to simulator", exc)

    cam = MockCameraSource()
    cam.start()
    log.info("RGB source: %s", cam.describe())
    return cam


def encode_jpeg(image: np.ndarray, quality: int | None = None) -> bytes:
    q = DASHBOARD.jpeg_quality if quality is None else quality
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
