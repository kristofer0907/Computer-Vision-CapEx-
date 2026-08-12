"""Configuration for the CapEx synthesis monitor.

Scope note: this file only covers what currently exists — camera/thermal
sources, the synthetic scene, and the live dashboard. Localisation, tracking,
stage and anomaly settings belong with the pipeline code when you write it.

Nothing here imports a hardware library, so it is importable on any machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
# Platform            780 x 260 mm
# Camera height       800 mm (hard ceiling 850 mm)
# IMX477 @ f=5 mm     65.1 deg H  ->  2 * 800 * tan(32.55) = 1021 mm horizontal
# Capture 1280x720    1021 mm / 1280 px = 0.798 mm/px  => 1.253 px/mm
# Vertical at 720 px  720 * 0.798 = 574 mm, platform needs 260 mm
#                     -> platform occupies 260 * 1.253 = 326 px of 720 (45%)
# That wasted vertical field is the known open item about cropping in software.
@dataclass(frozen=True)
class GeometryConfig:
    platform_length_mm: float = 780.0
    platform_width_mm: float = 260.0
    camera_height_mm: float = 800.0
    frame_width_px: int = 1280
    frame_height_px: int = 720
    horizontal_coverage_mm: float = 1021.0
    vial_diameter_mm: float = 27.0
    n_vials: int = 18

    @property
    def px_per_mm(self) -> float:
        return self.frame_width_px / self.horizontal_coverage_mm

    @property
    def vial_radius_px(self) -> float:
        return 0.5 * self.vial_diameter_mm * self.px_per_mm

    @property
    def platform_band_px(self) -> tuple[int, int]:
        """(y_top, y_bottom) of the platform inside the frame, centred."""
        band = self.platform_width_mm * self.px_per_mm
        top = 0.5 * (self.frame_height_px - band)
        return int(round(top)), int(round(top + band))


GEOMETRY = GeometryConfig()


# --------------------------------------------------------------------------
# Zone polygons  (normalised frame coordinates, 0..1, clockwise)
# --------------------------------------------------------------------------
# PLACEHOLDER GEOMETRY, used only to lay out the synthetic scene. Real
# coordinates must be traced from a calibration capture of the actual platform.
def _band_rect(x0: float, x1: float, y0: float = 0.27, y1: float = 0.73):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@dataclass(frozen=True)
class ZoneConfig:
    # Ordered along the physical process flow.
    polygons: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {
            "filling": _band_rect(0.02, 0.32),
            "conveyor": _band_rect(0.32, 0.56),
            "lidding": _band_rect(0.56, 0.68),
            "heating": _band_rect(0.68, 0.82),
            "cooling": _band_rect(0.82, 0.97),
        }
    )


ZONES = ZoneConfig()


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceConfig:
    """Backend selection.

    "auto" probes for real hardware once at start-up and falls back to the
    simulator if it is not present. Code above the drivers is written against
    the abstract interface and never branches on which one won.
    """

    rgb_backend: str = os.environ.get("CAPEX_RGB_BACKEND", "auto")  # auto|picamera2|mock|file
    thermal_backend: str = os.environ.get("CAPEX_THERMAL_BACKEND", "auto")  # auto|mlx90640|mock

    # file backend: replay a recorded video or a directory of stills
    rgb_file_path: str = os.environ.get("CAPEX_RGB_FILE", "")

    # Simulated capture latency. Deliberately non-zero: instant mocks hide
    # timing and backpressure bugs that only appear against real hardware.
    mock_rgb_latency_s: float = 0.09     # picamera2 still capture ~60-120 ms
    mock_thermal_latency_s: float = 0.55  # MLX90640 at 2 Hz refresh

    # Mock scene behaviour
    mock_time_scale: float = 12.0        # 1 real second = 12 simulated seconds
    mock_anomalous_vials: tuple[int, ...] = (7,)
    mock_noise_sigma: float = 3.0

    # Real hardware
    picamera2_format: str = "RGB888"     # not XRGB8888: avoids alpha/BGR confusion
    mlx90640_i2c_hz: int = 400_000
    mlx90640_refresh_hz: int = 2


SOURCES = SourceConfig()


# --------------------------------------------------------------------------
# Capture cadence
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CadenceConfig:
    preview_interval_s: float = 0.2      # live camera feed, ~5 fps
    thermal_interval_s: float = 2.0      # thermal poll


CADENCE = CadenceConfig()


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DashboardConfig:
    host: str = os.environ.get("CAPEX_DASHBOARD_HOST", "0.0.0.0")
    port: int = int(os.environ.get("CAPEX_DASHBOARD_PORT", "5000"))
    # Flask(debug=True) spawns a duplicate process via the reloader, which
    # would open the camera twice. Never enable this with capture running.
    debug: bool = False
    jpeg_quality: int = 80


DASHBOARD = DashboardConfig()
