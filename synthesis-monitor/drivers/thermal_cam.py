"""Thermal camera backends: PureThermal/Lepton over USB, and the MLX90640.

Scope reminder: the thermal stream is a passive logger for researchers to look
at. No model, no algorithm and no part of the anomaly logic runs on it. It is
written to SQLite and displayed, nothing more.

Why two backends. The MLX90640 is 32x24 over a 110 deg FOV, measured at
~2.4 cm/px at 800 mm, so a 27 mm crucible spans 1-2 pixels - enough to see a
heater pad glow, not enough to see which crucible is on it. The PureThermal
board carries a Lepton 3.x at 160x120, which at the same 800 mm gives 1.09
cm/px on a 95 deg part (2.5 px per crucible) or 0.54 cm/px on a 57 deg part
(5.0 px). Mounted low over the heater pad alone at 150 mm those become 0.20
and 0.10 cm/px. Either way it is still a viewing aid, not a measurement the
pipeline reads.

As in rgb_cam.py, board/busio/adafruit_mlx90640 are imported inside start(),
so this module imports cleanly on a laptop. The Lepton needs only OpenCV's
V4L2 backend, so it works on the laptop and the Pi alike.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from config import SOURCES
from drivers.base import (HardwareUnavailable, ThermalFrame, ThermalSource,
                          sleep_remaining)
from drivers.scene import SyntheticPlatform

log = logging.getLogger(__name__)

THERMAL_ROWS, THERMAL_COLS = 24, 32

# Lepton 3.x. The board appends 2 telemetry rows, so a delivered frame is
# 122 rows; see LeptonSource._to_celsius.
LEPTON_ROWS, LEPTON_COLS = 120, 160


class MLX90640Source(ThermalSource):
    name = "mlx90640"
    simulated = False

    def __init__(self, max_read_retries: int = 5) -> None:
        super().__init__()
        self.max_read_retries = max_read_retries
        self._mlx = None
        self._buf = [0.0] * (THERMAL_ROWS * THERMAL_COLS)

    def start(self) -> None:
        if self._started:
            return
        try:
            import adafruit_mlx90640
            import board
            import busio
        except Exception as exc:
            raise HardwareUnavailable(
                f"MLX90640 libraries not importable ({exc})"
            ) from exc

        try:
            # 400 kHz, not the 100 kHz default: needed to sustain the refresh
            # rate. Requires dtparam=i2c_arm_baudrate=400000 in /boot/config.txt.
            i2c = busio.I2C(board.SCL, board.SDA, frequency=SOURCES.mlx90640_i2c_hz)
            mlx = adafruit_mlx90640.MLX90640(i2c)  # I2C address 0x33
            rate = {1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
                    2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                    4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                    8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ}
            mlx.refresh_rate = rate.get(SOURCES.mlx90640_refresh_hz,
                                        adafruit_mlx90640.RefreshRate.REFRESH_2_HZ)
        except Exception as exc:
            # No I2C bus on a laptop; a Pi with bad wiring lands here too.
            raise HardwareUnavailable(f"MLX90640 open failed: {exc}") from exc

        self._mlx = mlx
        self._started = True
        log.info("MLX90640 started at %d Hz on %d kHz I2C",
                 SOURCES.mlx90640_refresh_hz, SOURCES.mlx90640_i2c_hz // 1000)

    def capture(self) -> ThermalFrame:
        if not self._started:
            raise RuntimeError("capture() before start()")
        last: Exception | None = None
        for _ in range(self.max_read_retries):
            try:
                self._mlx.getFrame(self._buf)
                arr = np.reshape(np.asarray(self._buf, dtype=np.float32),
                                 (THERMAL_ROWS, THERMAL_COLS))
                return ThermalFrame(arr, time.time(), self._next_id(), self.name, False)
            except (ValueError, OSError) as exc:
                # Checksum / I2C hiccups are routine on this part. Retry a
                # bounded number of times instead of spinning forever.
                last = exc
                time.sleep(0.05)
        raise HardwareUnavailable(
            f"MLX90640 failed {self.max_read_retries} consecutive reads: {last}"
        )

    def stop(self) -> None:
        self._mlx = None
        self._started = False


def find_purethermal() -> int | None:
    """/dev/videoN index of the PureThermal board, by V4L2 name.

    Never guess an index. On a laptop /dev/video0 is the built-in webcam, and
    opening it returns a perfectly valid 8-bit picture of the room that will
    be read as centi-Kelvin and reported as temperatures.

    The board exposes two nodes; only the first is a capture device.
    """
    try:
        nodes = sorted(Path("/sys/class/video4linux").glob("video*"),
                       key=lambda p: int(p.name[5:]))
    except OSError:
        return None
    for node in nodes:
        try:
            if "purethermal" in (node / "name").read_text().lower():
                return int(node.name[5:])
        except (OSError, ValueError):
            continue
    return None


class LeptonSource(ThermalSource):
    """PureThermal 3 + FLIR Lepton 3.x over USB, 160x120 radiometric.

    Three things the obvious VideoCapture(0) loop gets wrong, all of which
    produce plausible-looking garbage rather than an error:

      * the index. See find_purethermal() above.
      * CAP_V4L2. Without the explicit backend the Y16 request is quietly
        ignored on some builds and the frames come back 8-bit BGR, i.e. a
        picture, not a measurement.
      * telemetry. With telemetry enabled the board delivers 122 rows, not
        120: the last two are the Lepton's own status words (uptime, FPA
        temperature, serial). Left in, they push the reported maximum to
        several hundred degrees. They are cropped here.

    Radiometry: TLinear is on, so each pixel is centi-Kelvin, hence /100 -
    273.15. Measured on the bench 2026-09-09 - once warm the field sits at a
    16 C median across a 13-23 C room, wobbling +-0.5 C as the flat-field
    shutter cycles.

    IT NEEDS TO WARM UP. From a cold start the same room reads about 13 C low
    and keeps sliding down ~2 C a minute (a scene minimum of -14 C, which
    nothing indoors is) before settling after a few minutes. Anything that
    treats the first minute of a run as calibrated will be wrong, and wrong in
    a way that looks plausible. start() logs the scene median and warns when
    it is outside an indoor range, which catches exactly this.

    Absolute accuracy beyond that is still unverified against a reference -
    the part is specified at about +-5 C. Ice water at 0 C and a hand at ~33 C
    is a two-minute check if the numbers ever need to be trusted rather than
    compared.
    """

    name = "lepton"
    simulated = False

    def __init__(self, device: int | None = None,
                 max_read_retries: int = 5) -> None:
        super().__init__()
        self.device = SOURCES.lepton_device if device is None else device
        self.max_read_retries = max_read_retries
        self._cap = None

    def start(self) -> None:
        if self._started:
            return
        index = self.device if self.device is not None else find_purethermal()
        if index is None:
            raise HardwareUnavailable(
                "no PureThermal board found in /sys/class/video4linux. Is it "
                "plugged in? Pass SOURCES.lepton_device to override.")

        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise HardwareUnavailable(f"/dev/video{index} would not open")

        # Order matters: the pixel format first, then disable the automatic
        # conversion to RGB that would throw the high byte away.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ' '))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise HardwareUnavailable(f"/dev/video{index} opened but read() failed")
        if frame.dtype != np.uint16 or frame.shape[1] != LEPTON_COLS:
            cap.release()
            raise HardwareUnavailable(
                f"/dev/video{index} gave {frame.shape} {frame.dtype}, expected "
                f"a {LEPTON_COLS}-wide uint16 Y16 frame - this is probably not "
                "the Lepton, or Y16 was refused")

        self._cap = cap
        self._started = True

        celsius = self._to_celsius(frame)
        median = float(np.median(celsius))
        log.info("Lepton started on /dev/video%d, %dx%d, scene median %.1f C",
                 index, LEPTON_COLS, LEPTON_ROWS, median)
        # An indoor scene sits in the teens or twenties. Below that the sensor
        # is almost certainly still warming up - see the class docstring.
        if not 10.0 < median < 40.0:
            log.warning(
                "Lepton scene median is %.1f C, which is not an indoor room. "
                "The sensor is probably still warming up; give it a few "
                "minutes before trusting the numbers.", median)

    @staticmethod
    def _to_celsius(frame: np.ndarray) -> np.ndarray:
        """Crop the telemetry rows, then centi-Kelvin -> degC."""
        image = frame[:LEPTON_ROWS] if frame.shape[0] > LEPTON_ROWS else frame
        return image.astype(np.float32) / 100.0 - 273.15

    def capture(self) -> ThermalFrame:
        if not self._started:
            raise RuntimeError("capture() before start()")
        last = "no attempt"
        for _ in range(self.max_read_retries):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return ThermalFrame(self._to_celsius(frame), time.time(),
                                    self._next_id(), self.name, False)
            # A dropped frame over USB is routine, especially across the
            # Lepton's periodic flat-field correction shutter.
            last = "read() returned no frame"
            time.sleep(0.05)
        raise HardwareUnavailable(
            f"Lepton failed {self.max_read_retries} consecutive reads: {last}")

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._started = False


class MockThermalSource(ThermalSource):
    """Synthetic 24x32 field with a heater-pad hot spot."""

    name = "mock"
    simulated = True

    def __init__(self, scene: SyntheticPlatform | None = None,
                 time_scale: float | None = None,
                 latency_s: float | None = None) -> None:
        super().__init__()
        self.scene = scene or SyntheticPlatform()
        self.time_scale = SOURCES.mock_time_scale if time_scale is None else time_scale
        self.latency_s = (SOURCES.mock_thermal_latency_s
                          if latency_s is None else latency_s)
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._wall_t0 = time.time()
        self._started = True
        log.info("mock thermal started (latency %.0f ms)", self.latency_s * 1000)

    def capture(self) -> ThermalFrame:
        if not self._started:
            raise RuntimeError("capture() before start()")
        t_begin = time.monotonic()
        sim_t = (time.monotonic() - self._t0) * self.time_scale
        arr = self.scene.thermal(sim_t)
        sleep_remaining(t_begin, self.latency_s)  # MLX90640 is genuinely slow
        # Simulated clock, matching MockCameraSource - see the note there.
        return ThermalFrame(arr, self._wall_t0 + sim_t, self._next_id(),
                            self.name, True)

    def stop(self) -> None:
        self._started = False


def create_thermal(backend: str | None = None) -> ThermalSource:
    """Build and start a thermal source. "auto" probes, then falls back."""
    backend = (backend or SOURCES.thermal_backend).lower()

    if backend == "lepton":
        src = LeptonSource()
        src.start()
        return src
    if backend == "mlx90640":
        src = MLX90640Source()
        src.start()
        return src
    if backend == "mock":
        src = MockThermalSource()
        src.start()
        return src
    if backend != "auto":
        raise ValueError(f"unknown thermal backend: {backend!r}")

    # Lepton first: 160x120 against the MLX's 32x24, and it is USB, so it
    # works on the laptop as well as the Pi.
    for candidate in (LeptonSource, MLX90640Source):
        try:
            src = candidate()
            src.start()
            log.info("thermal source: %s", src.describe())
            return src
        except HardwareUnavailable as exc:
            log.warning("%s unavailable (%s)", candidate.__name__, exc)

    src = MockThermalSource()
    src.start()
    log.info("thermal source: %s", src.describe())
    return src


def colorize(celsius: np.ndarray, size: tuple[int, int] = (640, 480),
             span: tuple[float, float] | None = None) -> np.ndarray:
    """A degC field -> upscaled INFERNO BGR image for display.

    Takes either sensor's shape: 24x32 from the MLX90640, 120x160 from a
    Lepton.

    span fixes the colour scale to (min_c, max_c). Leave it None for
    per-frame auto-scaling, which looks livelier but makes frames
    non-comparable to each other - a hand entering the frame rescales
    everything behind it.
    """
    if span is None:
        lo, hi = float(celsius.min()), float(celsius.max())
    else:
        lo, hi = span
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((celsius - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    # INTER_NEAREST, not a smooth interpolation: at 1-2 px per vial, smoothing
    # invents detail that the sensor did not measure.
    return cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)
