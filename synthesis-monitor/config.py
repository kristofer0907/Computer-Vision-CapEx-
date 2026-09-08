"""Configuration for the CapEx synthesis monitor.

Nothing here imports a hardware library, so it is importable on any machine.

Two kinds of setting live in this file and they are NOT equivalent:

  * Plumbing values (cadences, queue depths, image sizes, DB paths). These are
    real decisions and are safe to trust.
  * Detection values (anything under DETECTION). These are placeholders with
    no empirical basis. HSV/texture thresholds and anomaly scoring cannot be
    pre-tuned without real captured vial images — they are here so the
    detector modules have somewhere to read from, not because the numbers
    mean anything yet.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CAPEX_DATA_DIR", ROOT / "data"))

log = logging.getLogger(__name__)


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
    # Full sensor readout, matching data/camera_calibration.npz and every
    # capture in capture/. Not a preview size: detect_crucibles' radius
    # bounds, the lid score threshold and the hand-marked slot layouts are
    # all calibrated at this resolution, and at half of it detection finds
    # nothing at all (crucibles fall under min_r).
    frame_width_px: int = 4056
    frame_height_px: int = 3040
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


_DEFAULT_POLYGONS: dict[str, list[tuple[float, float]]] = {
    "filling": _band_rect(0.02, 0.32),
    "conveyor": _band_rect(0.32, 0.56),
    "lidding": _band_rect(0.56, 0.68),
    "heating": _band_rect(0.68, 0.82),
    "cooling": _band_rect(0.82, 0.97),
}

# Process order.
STAGE_ORDER: tuple[str, ...] = (
    "storing", "injection", "heating", "collection",
)

ZONES_FILE = DATA_DIR / "zones.json"

# Hand-marked vial positions, produced by tools/mark_vials.py and read by
# pipeline.localize.ManualLocalizer. Lets a folder of real captured images run
# through the whole pipeline before a real localiser exists.
VIALS_FILE = Path(os.environ.get("CAPEX_VIALS_FILE", DATA_DIR / "vials.json"))


def _load_polygons() -> dict[str, list[tuple[float, float]]]:
    """Prefer traced polygons from tools/edit_zones.py over the defaults."""
    if not ZONES_FILE.exists():
        return dict(_DEFAULT_POLYGONS)
    try:
        raw = json.loads(ZONES_FILE.read_text())
        return {k: [tuple(p) for p in v] for k, v in raw.items()}
    except Exception:
        log.warning("could not read %s, using placeholder zones", ZONES_FILE,
                    exc_info=True)
        return dict(_DEFAULT_POLYGONS)


@dataclass(frozen=True)
class ZoneConfig:
    # Ordered along the physical process flow.
    polygons: dict[str, list[tuple[float, float]]] = field(
        default_factory=_load_polygons
    )

    @property
    def calibrated(self) -> bool:
        """False while the polygons are the hard-coded placeholders."""
        return ZONES_FILE.exists()

    def order(self) -> list[str]:
        """Zone names in process order, ignoring any not in STAGE_ORDER."""
        known = [s for s in STAGE_ORDER if s in self.polygons]
        return known + [k for k in self.polygons if k not in STAGE_ORDER]


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

    # Analysis cadence. The capture loop always runs at preview rate; a frame
    # is forwarded for analysis only this often, so the pipeline cost is
    # decoupled from the preview smoothness.
    #
    # Overridable by environment variable rather than by patching the object:
    # the worker processes are spawned, not forked, so they re-import this
    # module from scratch and would never see a mutation made in the parent.
    # Turning the cadence down is the only practical way to exercise the full
    # stack in under a minute.
    analysis_interval_s: float = float(
        os.environ.get("CAPEX_ANALYSIS_INTERVAL", "45"))       # 30-60 s band
    analysis_interval_busy_s: float = float(
        os.environ.get("CAPEX_ANALYSIS_INTERVAL_BUSY", "10"))  # conveyor moving

    # A track that moved further than this between two analysis frames is
    # taken as "the conveyor is running" and the cadence drops to busy.
    busy_motion_mm: float = 15.0
    # Frames of quiet before dropping back to the slow cadence. Without it the
    # cadence flaps between 10 s and 45 s on a single noisy centroid.
    busy_release_frames: int = 3


CADENCE = CadenceConfig()


# --------------------------------------------------------------------------
# IPC
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QueueConfig:
    """Queue depths and drop policy.

    Every queue here is shallow and drop-oldest on overflow. A monitoring
    system that buffers is worse than one that skips: a backlog means the
    dashboard shows a frame from four minutes ago while claiming it is live.
    Dropping is visible in the dropped counters; lag is not.
    """

    preview_depth: int = 2       # capture -> dashboard, JPEG bytes
    analysis_depth: int = 2      # capture -> processing, full raw frames
    result_depth: int = 8        # processing -> dashboard, small dicts
    thermal_depth: int = 2       # thermal -> dashboard

    # Full-resolution frames are the one expensive payload on a queue
    # (1280x720x3 = 2.7 MB pickled per frame). At the analysis cadence that is
    # ~60 kB/s, which is fine. Do not raise analysis_depth to hide a slow
    # pipeline - the cost lands in RAM on a Pi 5.
    join_timeout_s: float = 5.0  # per-process wait on shutdown


QUEUES = QueueConfig()


# --------------------------------------------------------------------------
# Tracking and stage assignment
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackingConfig:
    """Hungarian assignment within zone polygons.

    DeepSORT was evaluated and rejected: its motion model assumes near
    continuous frames and there is a 30-60 s gap between ours. Gating is
    therefore done in millimetres of platform, not in pixels of predicted
    motion.
    """

    # Maximum plausible displacement between two analysis frames. A vial on
    # the conveyor covers the ~250 mm from lidding to heating in one slow
    # frame, so the gate has to be generous or every handover breaks the ID.
    max_assignment_mm: float = 320.0
    # Same, at the fast cadence.
    max_assignment_busy_mm: float = 90.0

    # A track survives this many consecutive analysis frames without a
    # detection before it is closed. Covers a vial briefly occluded by the
    # lidding head, and the dim ~180 mm the LED panel does not cover.
    max_missed_frames: int = 3

    # Frames a track must be seen in before it is published as a real vial.
    # Suppresses one-frame localisation noise from creating ghost vials.
    min_hits_to_confirm: int = 2

    # Hysteresis: consecutive frames a track must read as a new stage before
    # the transition is committed. PLACEHOLDER - the right value depends on
    # per-stage timings, which are still an open question with the
    # researchers. At the 45 s cadence, 2 means a stage change is confirmed
    # ~90 s after it happens.
    stage_hysteresis_n: int = 2

    # A track last confirmed in "cooling" that then disappears for
    # max_missed_frames is recorded as having entered the oven rather than
    # as lost. KNOWN BLIND SPOT: a real failure during cooling looks
    # identical to normal oven entry. Not solved, deliberately not papered
    # over - both paths raise the same inference and it is flagged as
    # inferred, never observed.
    oven_entry_from: str = "cooling"


TRACKING = TrackingConfig()


# --------------------------------------------------------------------------
# Region tracking -- real-hardware crucible zones (storing/injection/heating/
# collection), independent of the vial-flow TrackingConfig above.
# PLACEHOLDER VALUES pending real captured sequences to tune against, same
# caveat as TRACKING.stage_hysteresis_n.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionTrackingConfig:
    """Per-zone tracking for the real crucible platform.

    storing/heating/collection have fixed hand-marked slot positions (see
    tools/mark_slots.py, pipeline.region_trackers.SlotTracker). injection has
    no fixed positions - it is a narrow lane crucibles move through in a
    fixed order, tracked by pipeline.region_trackers.FifoTracker against a
    hand-marked entry/exit axis (tools/mark_slots.py --lane).
    """

    # zone name -> slot JSON filename under DATA_DIR. heating/collection
    # predate the slots_<zone>.json naming and were produced one at a time
    # by tools/mark_slots.py --stage filling. storing was re-derived by
    # clustering real detect_crucibles() output across a full capture batch
    # (see data/slots_storing.json's "note" field) - the original
    # hand-marked slots_filling_beginning.json was off by 40-77px per hole,
    # consistent per hole but not a single global offset, which meant
    # something closer to a re-mark than a translation fix.
    slot_files: dict[str, str] = field(default_factory=lambda: {
        "storing": "slots_storing.json",
        "heating": "slots_filling_heaters.json",
        "collection": "slots_filling_end_zone.json",
    })

    # zone name -> lane JSON filename under DATA_DIR, produced by
    # tools/mark_slots.py --stage <zone> --lane.
    lane_files: dict[str, str] = field(default_factory=lambda: {
        "injection": "lane_injection.json",
    })

    # The physical process order, used for reporting and to bound which
    # zones may hand a track's identity to which others.
    region_sequence: tuple[str, ...] = (
        "storing", "injection", "heating", "collection",
    )

    # Only these zones carry identity. A crucible is not numbered while it
    # sits in the storing rack or crosses the injection lane - it gets an id
    # when it reaches a heater, and keeps it through to the collection rack.
    #
    # Before this, every zone assigned ids and every move between zones had
    # to be stitched back together by the handoff logic. Storing holds twelve
    # near-identical jars that the arm lifts in an arbitrary order, so those
    # stitches were guesses, and a six-crucible run numbered into the teens.
    # Numbering only what the process actually distinguishes removes the
    # guessing rather than tuning it.
    id_zones: tuple[str, ...] = ("heating", "collection")

    # Which zones the heater->cooling lineage watches (pipeline/lineage.py).
    # "collection" is the cooling/end rack on this bench; rename here if the
    # zones are ever re-traced under different names.
    heater_zone: str = "heating"
    cooling_zone: str = "collection"

    # Only this heater slot is watched by pipeline/handoff.py. Slot 1 is the
    # left pad, the one the arm actually cycles; slot 0 sits untouched for a
    # whole run and is deliberately ignored.
    handoff_heater_slot: int = 1

    # A heater slot that stays occupied but whose detection jumps further
    # than this is treated as a different vial, not the same one wobbling.
    # A swap between two captures never shows the slot empty, so occupancy
    # alone cannot see it - but a replaced jar does not land back on the same
    # pixels. Measured on capture/tracking_practice: an untouched jar moved
    # 0.00-6.40 px between frames (identical on two pairs, mean 2.12), while
    # the slot being worked moved 4.24-28.79 px, 14+ on four of five steps.
    # 12 px sits above the detector's own noise on a static jar and below a
    # typical replacement. The two ranges do overlap, so a slow careful swap
    # can still pass as one vial - this narrows the blind spot rather than
    # closing it.
    heater_replacement_move_px: float = 12.0

    # Frames a slot may sit unmatched before its track is closed as vacated.
    slot_max_missed_frames: int = 2

    # Frames a FIFO queue entry may go unmatched before its track is closed.
    fifo_max_missed_frames: int = 2

    # Per-frame gate on how far a detection may have moved along the lane
    # axis and still be considered the same queue entry, in mm.
    fifo_step_gate_mm: float = 120.0

    # Seconds a closed track stays eligible to donate its id to a new track
    # in a neighbouring (or the same) zone before the handoff window expires.
    handoff_window_s: float = 120.0


REGION_TRACKING = RegionTrackingConfig()


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StorageConfig:
    db_path: Path = DATA_DIR / "monitor.sqlite3"
    snapshot_dir: Path = DATA_DIR / "snapshots"

    # SQLite in WAL mode tolerates the thermal writer and the pipeline writer
    # running concurrently. busy_timeout makes a collision wait instead of
    # raising "database is locked".
    busy_timeout_ms: int = 5_000

    # Keep a full-frame JPEG every so often so a run can be reviewed later.
    # 0 disables. At 45 s cadence, every 4th frame is one image per 3 minutes.
    snapshot_every_n_frames: int = 4
    snapshot_jpeg_quality: int = 85

    # Per-vial crops are what the colour-change comparison needs to diff
    # against. Kept in memory by pipeline.history; written to disk only when
    # an event fires, so a normal run does not fill the SD card.
    save_crop_on_event: bool = True

    retention_days: int = 30     # 0 disables pruning


STORAGE = StorageConfig()


# --------------------------------------------------------------------------
# Detection  -- PLACEHOLDER VALUES, NOT CALIBRATED
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectionConfig:
    """Read by the detector modules. Every number here is a guess.

    None of it can be calibrated without real chemistry baseline runs, which
    need lab batch access. Treat a threshold crossing on these defaults as
    "the plumbing works", never as "the batch is failing".
    """

    # Which detector modules pipeline.detectors loads. Names match the
    # `name` attribute on each Detector subclass.
    enabled: tuple[str, ...] = ("turbidity", "solgel", "color_change",
                                "spill", "vial_presence")

    # pipeline.features.lid_score() at or above this reads as a lid.
    # UNLIKE the rest of this dataclass this one IS calibrated: swept against
    # the 225 hand-labelled crucibles in data/lid_review.json (tools/
    # review_lids.py), where it scores 96.4% - every one of the 120 lids
    # caught, 8 of 105 open jars wrongly called lidded. The classes do
    # overlap (open runs 40-1107, lid 627-1087), so a single frame's verdict
    # is worth doubting; a crucible that flips between frames is the case to
    # expect, not a surprise.
    lid_score_threshold: float = 585.0

    # Size of the square crop taken around each tracked centroid, as a
    # multiple of the vial radius. 2.0 would be exactly the rim; the margin
    # keeps the rim and a little bench either side inside the crop.
    roi_scale: float = 2.6

    # How many past analysis frames of per-vial features are retained in
    # memory for temporal comparisons.
    history_length: int = 40

    # Crops retained per vial for the previous-image comparison.
    crop_history_length: int = 4

    # Placeholder scoring knobs, unused until the detectors are written.
    robust_z_threshold: float = 3.5
    min_vials_for_batch_stats: int = 5


DETECTION = DetectionConfig()


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
    preview_width_px: int = 960   # downscale before sending to the browser
    events_page_size: int = 50


DASHBOARD = DashboardConfig()


# --------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create the writable directories. Called by entry points, not on import.

    Importing a config module must not touch the filesystem: tests and the
    calibration tool import it, and a read-only checkout should not fail.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE.snapshot_dir.mkdir(parents=True, exist_ok=True)
