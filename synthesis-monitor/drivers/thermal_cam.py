"""Thermal camera backends (MLX90640).

Scope reminder: the thermal stream is a passive logger for researchers to look
at. No model, no algorithm and no part of the anomaly logic runs on it. It is
written to SQLite and displayed, nothing more.

Known limitation, already measured: 32x24 px over a 110 deg FOV at 800 mm is
~2.4 cm/px, so a vial spans 1-2 pixels. The planned fix is to remount the
sensor low over the heater pad only (~1.3 cm/px at h=150 mm) before spending
anything on a Lepton.

As in rgb_cam.py, board/busio/adafruit_mlx90640 are imported inside start(),
so this module imports cleanly on a laptop.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from config import SOURCES
from drivers.base import (HardwareUnavailable, ThermalFrame, ThermalSource,
                          sleep_remaining)
from drivers.scene import SyntheticPlatform

log = logging.getLogger(__name__)

THERMAL_ROWS, THERMAL_COLS = 24, 32


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

    try:
        src = MLX90640Source()
        src.start()
        log.info("thermal source: %s", src.describe())
        return src
    except HardwareUnavailable as exc:
        log.warning("thermal hardware unavailable (%s) - falling back to simulator",
                    exc)

    src = MockThermalSource()
    src.start()
    log.info("thermal source: %s", src.describe())
    return src


def colorize(celsius: np.ndarray, size: tuple[int, int] = (640, 480),
             span: tuple[float, float] | None = None) -> np.ndarray:
    """24x32 degC -> upscaled INFERNO BGR image for display.

    span fixes the colour scale to (min_c, max_c). Leave it None for
    per-frame auto-scaling, which looks livelier but makes frames
    non-comparable to each other.
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
