"""Synthetic platform renderer used by the simulated camera backends.

This is not a toy noise generator. It renders 18 vials moving through the real
zone sequence (filling -> conveyor -> lidding -> heating -> cooling -> oven)
with the real geometry, so the localiser, tracker, feature extractor and
batch-median scorer all get exercised end to end with no hardware attached.
One vial can be flagged anomalous (hue shift + turbidity ramp) to verify the
scoring actually fires.

Everything here is fake. It validates plumbing and logic, never chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import GEOMETRY, SOURCES, ZONES

# Simulated-seconds spent in each stage after leaving the filling rack.
# The first 40 % of each dwell is travel, the rest is stationary.
STAGE_DWELL_S: dict[str, float] = {
    "conveyor": 40.0,
    "lidding": 25.0,
    "heating": 60.0,
    "cooling": 50.0,
}
DEPARTURE_PACE_S = 35.0   # one vial leaves the filling rack every 35 sim-seconds
TRAVEL_FRACTION = 0.4


def _poly_center(poly) -> tuple[float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return sum(xs) / len(xs), sum(ys) / len(ys)


@dataclass
class VialState:
    index: int
    x: float          # normalised frame coords
    y: float
    stage: str
    hue: float        # OpenCV hue, 0..179
    saturation: float
    value: float
    turbidity: float  # 0..1, drives speckle + desaturation
    anomalous: bool


class SyntheticPlatform:
    """Deterministic-in-time scene: state is a pure function of sim-time."""

    def __init__(self, n_vials: int | None = None) -> None:
        self.n_vials = n_vials or GEOMETRY.n_vials
        self.w = GEOMETRY.frame_width_px
        self.h = GEOMETRY.frame_height_px
        self.r_px = int(round(GEOMETRY.vial_radius_px))
        self._rng = np.random.default_rng(0xCA9E)  # fixed: vials stay themselves
        # Fixed per-vial cosmetic jitter so vials are not clones.
        self._hue_jitter = self._rng.uniform(-3.0, 3.0, self.n_vials)
        self._sat_jitter = self._rng.uniform(-12.0, 12.0, self.n_vials)
        self._fill_slots = self._build_fill_slots()
        self._zone_centers = {k: _poly_center(v) for k, v in ZONES.polygons.items()}
        # Transport lane, level with the downstream zone centres and clear of
        # both rack rows.
        self._lane_y = self._zone_centers["conveyor"][1]
        self._background = self._build_background()
        self._noise_rng = np.random.default_rng()

    # ---------------------------------------------------------------- layout
    def _build_fill_slots(self) -> list[tuple[float, float]]:
        """18 vials in 2 rows x 9 inside the filling polygon."""
        poly = ZONES.polygons["filling"]
        x0, x1 = poly[0][0], poly[1][0]
        y0, y1 = poly[0][1], poly[2][1]
        cols, rows = 9, 2
        slots = []
        for r in range(rows):
            for c in range(cols):
                fx = x0 + (x1 - x0) * (c + 0.5) / cols
                fy = y0 + (y1 - y0) * (r + 0.5) / rows
                slots.append((fx, fy))
        return slots[: self.n_vials]

    def _build_background(self) -> np.ndarray:
        """Platform band, LED falloff over the uncovered ~20 cm, vignetting."""
        img = np.full((self.h, self.w, 3), 28, dtype=np.uint8)
        top, bottom = GEOMETRY.platform_band_px
        img[top:bottom, :] = (74, 72, 70)  # brushed metal, slightly cool

        # Zone seams, so the rendered scene is readable by eye.
        for poly in ZONES.polygons.values():
            x = int(poly[1][0] * self.w)
            cv2.line(img, (x, top), (x, bottom), (95, 93, 91), 1)

        illum = np.ones((self.h, self.w), np.float32)

        # One 595 mm LED panel covers ~600 mm of the 780 mm platform. The last
        # ~180 mm is the documented dim zone (which stage sits there is still
        # an open question - assumed here to be the far end).
        lit_mm = 600.0
        lit_px = lit_mm * GEOMETRY.px_per_mm
        xs = np.arange(self.w, dtype=np.float32)
        falloff = np.clip(1.0 - 0.45 * (xs - lit_px) / max(self.w - lit_px, 1), 0.55, 1.0)
        illum *= falloff[None, :]

        # Cos^4-ish lens vignetting: this is what flat-field correction removes.
        yy, xx = np.mgrid[0 : self.h, 0 : self.w].astype(np.float32)
        rr = np.sqrt(((xx - self.w / 2) / (self.w / 2)) ** 2
                     + ((yy - self.h / 2) / (self.h / 2)) ** 2)
        illum *= np.clip(1.0 - 0.22 * rr**2, 0.0, 1.0)

        return np.clip(img.astype(np.float32) * illum[..., None], 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- state
    def states_at(self, t: float) -> list[VialState]:
        """Vial states at simulated time t (seconds since batch start)."""
        out: list[VialState] = []
        for i in range(self.n_vials):
            depart = i * DEPARTURE_PACE_S
            if t < depart:
                st = self._state_in_filling(i, t)
            else:
                st = self._state_in_transit(i, t - depart)
            if st is not None:
                out.append(st)
        return out

    def _base_colour(self, i: int) -> tuple[float, float, float]:
        return (
            24.0 + self._hue_jitter[i],   # amber precursor solution
            185.0 + self._sat_jitter[i],
            205.0,
        )

    def _state_in_filling(self, i: int, t: float) -> VialState:
        hue, sat, val = self._base_colour(i)
        fx, fy = self._fill_slots[i]
        # Fill level rises during the shared filling step -> brightness ramp.
        fill = min(1.0, t / max(DEPARTURE_PACE_S * self.n_vials * 0.35, 1.0))
        return VialState(i, fx, fy, "filling", hue, sat * (0.55 + 0.45 * fill),
                         val * (0.7 + 0.3 * fill), 0.0, i in SOURCES.mock_anomalous_vials)

    def _state_in_transit(self, i: int, tau: float) -> VialState | None:
        """tau = simulated seconds since this vial left the filling rack."""
        origin = self._fill_slots[i]
        elapsed = 0.0
        for k, (stage, dwell) in enumerate(STAGE_DWELL_S.items()):
            target = self._zone_centers[stage]
            if tau < elapsed + dwell:
                local = tau - elapsed
                travel = dwell * TRAVEL_FRACTION
                f = min(1.0, local / travel) if travel > 0 else 1.0
                f = f * f * (3 - 2 * f)  # smoothstep
                x, y = self._path_point(origin, target, f, first_leg=(k == 0))
                return self._colour_for_stage(i, stage, local / dwell, x, y)
            elapsed += dwell
            origin = target
        return None  # past cooling -> entered the oven, out of frame

    def _path_point(self, origin, target, f: float, first_leg: bool):
        """Position a fraction f along the transport path.

        Leaving the filling rack is routed out to the transport lane first,
        then along it. A straight line from a rack slot to the conveyor would
        drive the vial through the slots to its right - two vials sharing the
        same pixels, which is not something the real platform does and which
        would make the scene a test of occlusion handling rather than of the
        pipeline.
        """
        if not first_leg:
            return (origin[0] + (target[0] - origin[0]) * f,
                    origin[1] + (target[1] - origin[1]) * f)

        exit_frac = 0.3                       # rack -> lane, then lane -> zone
        if f <= exit_frac:
            g = f / exit_frac
            return origin[0], origin[1] + (self._lane_y - origin[1]) * g
        g = (f - exit_frac) / (1.0 - exit_frac)
        return origin[0] + (target[0] - origin[0]) * g, self._lane_y

    def _colour_for_stage(self, i: int, stage: str, progress: float,
                          x: float, y: float) -> VialState:
        hue, sat, val = self._base_colour(i)
        turb = 0.0
        anomalous = i in SOURCES.mock_anomalous_vials

        if stage == "lidding":
            val *= 0.97
        elif stage == "heating":
            # Normal, batch-wide evolution: the whole cohort darkens and
            # reddens together, so batch-median scoring should NOT flag it.
            hue -= 6.0 * progress
            sat += 25.0 * progress
            val -= 35.0 * progress
            turb = 0.12 * progress
        elif stage == "cooling":
            hue -= 6.0
            sat += 25.0
            val -= 35.0
            turb = 0.12 + 0.10 * progress

        if anomalous and stage in ("heating", "cooling"):
            # Injected fault: off-hue plus a turbidity/gelation ramp.
            ramp = progress if stage == "heating" else 1.0
            hue += 26.0 * ramp
            sat -= 55.0 * ramp
            turb = min(1.0, turb + 0.75 * ramp)

        return VialState(i, x, y, stage, hue % 180, np.clip(sat, 0, 255),
                         np.clip(val, 0, 255), turb, anomalous)

    # --------------------------------------------------------------- render
    def render(self, t: float) -> np.ndarray:
        img = self._background.copy()
        for st in self.states_at(t):
            self._draw_vial(img, st)

        noise = self._noise_rng.normal(0.0, SOURCES.mock_noise_sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _draw_vial(self, img: np.ndarray, st: VialState) -> None:
        cx, cy = int(st.x * self.w), int(st.y * self.h)
        r = self.r_px
        bgr = cv2.cvtColor(
            np.uint8([[[st.hue, st.saturation, st.value]]]), cv2.COLOR_HSV2BGR
        )[0][0].tolist()

        cv2.circle(img, (cx, cy), r, (46, 46, 50), -1)          # glass shadow
        cv2.circle(img, (cx, cy), r - 2, bgr, -1)               # liquid
        cv2.circle(img, (cx, cy), r - 1, (168, 170, 172), 1)    # rim

        if st.turbidity > 0.02:
            # Turbidity / precipitate: raises texture variance and edge
            # density, which is exactly what the features key on.
            patch = img[cy - r : cy + r, cx - r : cx + r]
            if patch.size:
                mask = np.zeros(patch.shape[:2], np.uint8)
                cv2.circle(mask, (r, r), r - 3, 255, -1)
                speck = self._noise_rng.normal(0, 40 * st.turbidity, patch.shape)
                blended = np.clip(patch.astype(np.float32) + speck, 0, 255)
                patch[mask > 0] = blended[mask > 0].astype(np.uint8)

        # Faint residual specular highlight. Cross-polarisation is meant to
        # remove most of this; the leftover is what the real optics leave.
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(1, r // 6), (215, 215, 215), -1)

    # -------------------------------------------------------------- thermal
    def thermal(self, t: float) -> np.ndarray:
        """24x32 degC field: ambient, heater pad hot spot, warm cooling zone."""
        rows, cols = 24, 32
        field = np.full((rows, cols), 22.0, np.float32)
        yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)

        def blob(nx: float, ny: float, amp: float, sigma: float) -> None:
            cx, cy = nx * cols, ny * rows
            field[:] += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))

        hx, hy = self._zone_centers["heating"]
        blob(hx, hy, 62.0, 2.6)                       # heater pad
        cx_, cy_ = self._zone_centers["cooling"]
        blob(cx_, cy_, 14.0, 3.0)                     # residual heat on cooling pad

        for st in self.states_at(t):
            if st.stage == "heating":
                blob(st.x, st.y, 10.0, 1.0)

        field += self._noise_rng.normal(0.0, 0.35, field.shape)  # MLX90640 NETD
        return field.astype(np.float32)
