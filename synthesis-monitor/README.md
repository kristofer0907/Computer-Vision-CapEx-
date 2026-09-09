# CapEx synthesis monitor

Process monitoring for the automated sol-gel perovskite synthesis platform.
Camera and thermal capture, crucible tracking, staging, persistence and a live
dashboard — everything around the detection logic.

**Detection is not implemented.** Localisation, feature extraction and all five
detectors are stubs with fixed interfaces. Nothing in this repository has been
validated against real chemistry, and no threshold in it has been calibrated.
Anything written for an external audience must say "architected and exercised
against mock interfaces", never "validated".

## Run it

```bash
pip install -r requirements.txt
python main.py                  # auto-detect hardware, simulate what is missing
python main.py --rgb mock       # force the synthetic platform
python -m pytest tests/ -q
```

Then open `http://localhost:5000/`. From the laptop over the ICS link that is
the Pi's address on `192.168.137.x`.

Runs on a laptop exactly as on the Pi. `picamera2`, the MLX90640 library and
the Lepton's V4L2 access are all opened inside `start()`, never imported at
module level, so a machine without that hardware reports it unavailable and
falls back to the simulator instead of failing to import.

Common flags (`python main.py --help` for the rest):

```bash
python main.py --rgb file --file captures/         # replay a folder or video
python main.py --thermal lepton                    # PureThermal/Lepton, not the MLX90640
python main.py --no-thermal                        # skip the thermal logger entirely
python main.py --rgb-interval 15 --thermal-interval 3   # pin both cadences, in seconds
python main.py --no-persist                        # don't write to SQLite
python main.py --note "batch 12, new polariser"     # tag this run in the runs table
```

`--rgb-interval`/`--thermal-interval` override the adaptive default (45s
still / 10s moving for the RGB loop, 3s fixed for thermal) with a fixed
number of seconds between frames each — useful for comparing runs at a known
rate, or slowing things down while watching one stage by eye.

## What is yours to write

Four files and one directory. Everything else is wired, tested and running.

| File | What it owes the rest of the system |
|---|---|
| `pipeline/localize.py` | `locate(frame) -> list[Detection]`, pixel coordinates |
| `pipeline/features.py` | `extract(crop, mask, track, prev) -> dict[str, float]` |
| `pipeline/detectors/*.py` | `check(ctx) -> list[Event]`, one file per failure mode |
| `pipeline/anomaly.py` | empty; a shared scorer, if one turns out to be shared |

Each of those files opens with a docstring describing what the plan for it
needs and what the context already hands it. Read those before writing code —
several of the traps are mechanical (illumination gradients, crop size
mismatches, batch-wide changes reading as per-crucible anomalies) rather than
chemical, and they are documented where they bite.

Nothing else needs to change to add a feature or a detector. Features are
stored as JSON so a new one needs no migration; detectors are registered in
one dict in `pipeline/detectors/__init__.py`.

While detection is unwritten the system still runs: `GroundTruthLocalizer`
reads the simulator's ground truth so tracking, staging, storage and the
dashboard can be exercised end to end. It returns nothing on a real camera
frame, deliberately — it is a test oracle, not an algorithm.

## Layout

```
main.py                 entry point: start workers, serve dashboard
config.py               all configuration; DETECTION values are uncalibrated
drivers/                camera and thermal backends behind CameraSource/ThermalSource
  scene.py              synthetic 18-crucible platform, with ground truth
runtime/                the four processes and the queues between them
  capture.py            owns the camera -> preview JPEG + raw analysis frames
  processing.py         owns the pipeline and its SQLite writes
  thermal.py            owns the MLX90640; passive logging only
  supervisor.py         process lifecycle, queue draining, shutdown
pipeline/
  runner.py             localise -> track -> stage -> crop -> features -> detect
  tracking.py           Hungarian assignment, gated in millimetres
  assignment.py         the solver itself, no scipy dependency
  zones.py              polygons, px/mm conversion, stage hysteresis
  roi.py                per-crucible and per-zone crops and masks
  history.py            per-crucible feature and crop memory across frames
  stats.py              median / MAD helpers; optional, delete if unwanted
storage/                SQLite for numbers, files on disk for images
dashboard/              Flask; reads only, owns no device
tools/
  edit_zones.py         drag the zone polygons onto a real capture
  mark_vials.py         click the crucibles in your captures -> data/vials.json
  inspect_roi.py        render crops, masks and zones for visual checking
  storage_policy.py     set snapshot/retention limits, clear stored data
```

## Working against your own captured images

Put your captures anywhere — a folder of stills, or a video. Three steps, once:

```bash
# 1. drag the zone polygons onto a real frame
python -m tools.edit_zones --image captures/capture_00.jpg

# 2. click the crucibles, so something can find them before a localiser exists
python -m tools.mark_vials --images captures/

# 3. look at what the pipeline actually sees
python -m tools.inspect_roi --images captures/ --all-crucibles --show
```

After that your images run through the whole pipeline:

```bash
python main.py --rgb file --file captures/ --localizer manual
```

`tools/inspect_roi.py` is the visual loop while tuning ROIs: `--scale` sets
the crop size as a multiple of crucible radius, `--all-crucibles` shows all 18 crops
with the disc boundary drawn over them, `--show` opens a window (falls back to
writing files when there is no display).

Once you write a real localiser, register it in `create_localizer()` and swap
`--localizer manual` for `--localizer yours`. Running both over the same folder
and diffing the centroids gives an actual localisation error, with the hand
marks as the reference.

## Storage: how much is kept, and clearing it

A run writes a SQLite row per frame and, by default, a full-frame JPEG every
4th analysis frame (`data/monitor.sqlite3`, `data/snapshots/`). At the
default 45s cadence that snapshot rate is roughly 400 MB/day at ~850 kB a
frame, which fills a 32 GB card in about ten weeks if nothing is pruned.

```bash
python -m tools.storage_policy
```

Interactive: shows what's on disk (file counts, MB, rows per table), lets you
set how often a frame is saved, a maximum snapshot count (recommended — a day
count alone says nothing about how much a day costs, and that changes with
the cadence you picked), and a retention window in days. Also deletes a date
range or everything, both requiring you to type `DELETE` to confirm since
neither is recoverable. Run it against a stopped monitor; deleting rows
underneath a live run won't corrupt anything, but the numbers on screen will
be wrong the moment they're printed.

Settings are written to `data/storage_policy.json`, which `main.py` reads on
its next start — restart it to pick up a change. `anomaly_events` (the alarm
log) is excluded from both the retention window and a range delete unless you
opt in: it's small, and a failure worth recording is worth keeping past the
window that governs routine measurements.

## Before this measures anything real

1. **Place the zone polygons.** `python -m tools.edit_zones --image cap.jpg`. Until this
   is run, every stage assignment is against placeholder rectangles. It will
   look like it works, which is what makes it worth doing early. No display on
   the Pi: use `--save-frame`, copy the image over, and trace it with
   `--image`.
2. **Tune crucible localisation** — classical Hough circles. This is
   the biggest open architecture decision and it blocks all real detection
   work. It does not block the hardware bring-up, and it does not block
   writing the tracker tests, which already exist.
3. **Capture baseline runs on real chemistry.** Every threshold depends on
   them. This is not in the student's control and is the long pole.

## Things that are settled

- **Tracker: global assignment within zone polygons**, gated in millimetres
  of platform. Appearance-plus-Kalman trackers were evaluated and rejected —
  their motion models assume near-continuous frames and there are 30–60
  seconds between ours. Not open for revisiting.
- **Localisation: classical CV.** Hough circles, `CrucibleLocalizer`. No
  learned detector and no dependency for one.
- **Thermal is a passive log.** No model, no algorithm, no part of the anomaly
  logic. At 2.4 cm/px a crucible spans one or two pixels; there is nothing there to
  run anything on. `DetectionContext` deliberately has no thermal field.
- **`Flask(debug=True)` must stay off.** The reloader forks a second
  interpreter, which would start a second set of worker processes and open the
  camera twice.
- **Queues drop, never buffer.** A backlog means the dashboard shows a
  four-minute-old frame while claiming it is live. Dropping is visible in the
  counters; staleness is not.

## Known blind spot

A crucible that disappears after cooling is inferred to have entered the oven,
which is outside the camera's view. A genuine failure during cooling — knocked
over, removed by hand — produces exactly the same observation. The tracker
records the inference as an inference (`oven_entry_inferred`, severity info)
so nothing downstream can mistake it for something observed, but it cannot
currently be distinguished. Resolving it needs a signal from outside this
camera: an oven door sensor, or a scheduler event from the platform
controller. More image processing will not fix it.
