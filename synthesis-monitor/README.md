# CapEx synthesis monitor

Process monitoring for the automated sol-gel perovskite synthesis platform.
Camera and thermal capture, vial tracking, staging, persistence and a live
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
python -m tools.replay --frames 40      # pipeline only, no Flask, no processes
python -m pytest tests/ -q
```

Then open `http://localhost:5000/`. From the laptop over the ICS link that is
the Pi's address on `192.168.137.x`.

Runs on a laptop exactly as on the Pi. `picamera2` and the MLX90640 libraries
are imported inside `start()`, never at module level, so a machine without the
Pi camera stack reports the hardware as unavailable and falls back to the
simulator instead of failing to import.

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
mismatches, batch-wide changes reading as per-vial anomalies) rather than
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
  scene.py              synthetic 18-vial platform, with ground truth
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
  roi.py                per-vial and per-zone crops and masks
  history.py            per-vial feature and crop memory across frames
  stats.py              median / MAD helpers; optional, delete if unwanted
storage/                SQLite for numbers, files on disk for images
dashboard/              Flask; reads only, owns no device
tools/
  edit_zones.py         drag the zone polygons onto a real capture
  mark_vials.py         click the vials in your captures -> data/vials.json
  inspect_roi.py        render crops, masks and zones for visual checking
  replay.py             run the pipeline over a recording or the simulator
```

## Working against your own captured images

Put your captures anywhere — a folder of stills, or a video. Three steps, once:

```bash
# 1. drag the zone polygons onto a real frame
python -m tools.edit_zones --image captures/capture_00.jpg

# 2. click the vials, so something can find them before a localiser exists
python -m tools.mark_vials --images captures/

# 3. look at what the pipeline actually sees
python -m tools.inspect_roi --images captures/ --all-vials --show
```

After that your images run through the whole pipeline:

```bash
python -m tools.replay --rgb file --file captures/ --localizer manual
python main.py --rgb file --file captures/          # with the dashboard
```

`tools/inspect_roi.py` is the visual loop while tuning ROIs: `--scale` sets
the crop size as a multiple of vial radius, `--all-vials` shows all 18 crops
with the disc boundary drawn over them, `--show` opens a window (falls back to
writing files when there is no display).

Once you write a real localiser, register it in `create_localizer()` and swap
`--localizer manual` for `--localizer yours`. Running both over the same folder
and diffing the centroids gives an actual localisation error, with the hand
marks as the reference.

## Before this measures anything real

1. **Place the zone polygons.** `python -m tools.edit_zones --image cap.jpg`. Until this
   is run, every stage assignment is against placeholder rectangles. It will
   look like it works, which is what makes it worth doing early. No display on
   the Pi: use `--save-frame`, copy the image over, and trace it with
   `--image`.
2. **Resolve vial localisation** — classical CV or a fine-tuned YOLO. This is
   the biggest open architecture decision and it blocks all real detection
   work. It does not block the hardware bring-up, and it does not block
   writing the tracker tests, which already exist.
3. **Capture baseline runs on real chemistry.** Every threshold depends on
   them. This is not in the student's control and is the long pole.

## Things that are settled

- **Tracker: Hungarian assignment within zone polygons.** DeepSORT was
  evaluated and rejected — its motion model assumes near-continuous frames and
  there are 30–60 seconds between ours. Not open for revisiting.
- **Thermal is a passive log.** No model, no algorithm, no part of the anomaly
  logic. At 2.4 cm/px a vial spans one or two pixels; there is nothing there to
  run anything on. `DetectionContext` deliberately has no thermal field.
- **`Flask(debug=True)` must stay off.** The reloader forks a second
  interpreter, which would start a second set of worker processes and open the
  camera twice.
- **Queues drop, never buffer.** A backlog means the dashboard shows a
  four-minute-old frame while claiming it is live. Dropping is visible in the
  counters; staleness is not.

## Known blind spot

A vial that disappears after cooling is inferred to have entered the oven,
which is outside the camera's view. A genuine failure during cooling — knocked
over, removed by hand — produces exactly the same observation. The tracker
records the inference as an inference (`oven_entry_inferred`, severity info)
so nothing downstream can mistake it for something observed, but it cannot
currently be distinguished. Resolving it needs a signal from outside this
camera: an oven door sensor, or a scheduler event from the platform
controller. More image processing will not fix it.
