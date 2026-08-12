# CapEx Vision System — Project Context for Claude Code

## What this is
Computer vision process-monitoring system for an automated sol-gel perovskite
synthesis platform (DTU, green-transition materials research). Detects failures
across discrete synthesis stages: spills, overflows, gross color changes,
turbidity, liquid→solid (sol-to-gel) transitions, misplaced/missing labware.

**Explicitly out of scope for this phase:** robot-arm / pick-and-place guidance.
That's deferred to a later phase. Do not conflate the two.

**Detection philosophy:** qualitative, gross-state detection — not fine
colorimetry. The 18 vials run the same protocol in parallel, so the batch is
its own ground truth: a vial statistically diverging from its peers (batch-
median scoring) is anomalous, with no labeled training data required.

## Physical setup
- Platform: 78–80cm long × 26cm wide, 3–5cm tall
- Camera mounting height: 80–85cm above platform — **hard ceiling constraint**,
  cannot go higher
- Windowed box, ambient office lighting present (not a dark/enclosed box)
- Two levels of metal support bars: LED panel hangs from upper bars, camera
  bridge mounts to lower bars
- Vial flow: filling (all 18 together) → conveyor → lidding → heating pad
  (max 2 vials at a time) → cooling pad → oven (enclosed, **outside camera
  view** — oven entry is inferred, not observed)

## Hardware (finalized)
- Raspberry Pi 5, high-endurance microSD (Pi username: `capex-vision`)
- Pi HQ Camera (IMX477) + Arducam 2.8–12mm varifocal C-mount lens, target
  ~5mm focal length for full-platform coverage; varifocal gives margin
- LED panel: 595×595mm, 30W, 4000K. One panel covers ~60cm of the 80cm
  platform length — the remaining ~20cm is a low-priority dim zone (which
  process stage sits there is still unresolved, see Open Questions)
  - Driver: dimmable Mean Well, 800mA/36VDC — confirmed, settled decision.
    (Supersedes an earlier non-dimmable-preferred stance from before the panel
    spec was finalized.)
  - Flat-field correction required (SDCM color uniformity
    variation across the panel)
  - CRI 80 vs 90+ is an open question pending researcher input — depends on
    whether subtle color changes matter for this specific perovskite chemistry
- Cross-polarization: polarizing film (K&F/Nitto) on both panel (source side)
  and lens (imaging side) to kill specular glare on glass vials, tuned by
  rotating one film until glare disappears. Must be **linear, not circular
  (CPL)**. Lens-side film quality matters more (its defects degrade the image
  directly; source-side defects don't project into frame). Film must mount on
  a fixed non-rotating collar/hood, not on a rotating zoom/focus ring.
- Thermal: MLX90640, 110° FOV, I2C 0x33 @ 400kHz (bumped from default
  100kHz via `dtparam=i2c_arm_baudrate=400000` — assumed necessary for
  refresh rate, not fully confirmed) — passive data logger for researchers'
  visual observation only, **no models/algorithms run on it, not part of
  detection/anomaly logic** (this scope is now explicitly confirmed, not just
  assumed)
  - Bring-up complete: wiring, I2C, bus scan, library, live Flask MJPEG
    stream all working (`cv2.COLORMAP_INFERNO` + `INTER_NEAREST` upscale)
  - Resolution confirmed insufficient at 80–85cm mounting height: 32×24px
    over 110° FOV at 80cm ≈ 2.4cm/pixel → a vial spans only ~1–2 pixels
  - **Decision path**: try the free option first — remount the MLX90640
    lower, dedicated to just the heater-pad zone (max 2 vials) instead of
    sharing the RGB camera's full-platform height. At h=15cm this gets to
    ~1.3cm/pixel, no hardware cost. Only escalate to the Lepton upgrade if
    that's still insufficient.
  - Lepton part decision finalized (if needed): 500-0758-03 (Lepton 3.1R:
    160×120, 95° FOV, radiometric, ~$66–142, short lead time) over
    500-0771-01 (Lepton 3.5: same resolution, narrower 57° FOV — geometrically
    a better fit for a small dedicated heater-zone camera, but its 16-week
    lead time isn't worth it for a viewing-only use case). Plan if this path
    is needed: mount lower to compensate for the wider FOV.
- Storage: USB SSD or high-endurance microSD
- A Pi Camera v3 was bought by mistake and is being returned; may be
  repurposed later as a side-view fill-level camera (not currently in scope)

## Software architecture
Decoupled Python multiprocessing, communicating via `multiprocessing.Queue`:

1. **Capture loop** — camera + thermal acquisition
2. **Processing pipeline** — vial localization → Hungarian-assignment
   tracking within zone polygons → per-vial feature extraction (mean HSV,
   texture variance, brightness, frame differencing) → batch-median anomaly
   scoring → zone-polygon stage classification
3. **Thermal logger** — independent, writes to SQLite
4. **Flask dashboard**

Cadence: every 30–60s normally, ~10s during conveyor operation.

- **Tracker: confirmed.** DeepSORT was evaluated and rejected — its motion
  model assumes near-continuous frames, not 30–60s gaps. Hungarian assignment
  within known zone polygons is the actual approach, tracker implementation
  kept behind a swappable interface. Not open for debate; don't reintroduce
  DeepSORT.
- **⚠️ OPEN — vial localization method still unresolved**: classical CV
  (contour/Hough-circle) vs. fine-tuned YOLO. Per the most recent working
  session this is still called "the biggest unresolved architecture
  decision, unchanged from earlier sessions" — despite earlier notes
  elsewhere referring to YOLOv8n pretrained as if it were settled. Treat
  YOLOv8n as *a candidate*, not a locked-in decision, until this is actually
  resolved.
- Camera/Thermal source interface formalized (not yet implemented):
  `CameraSource`/`ThermalSource` ABCs with `start()` / `capture()` / `stop()`,
  returning `Frame`/`ThermalFrame` dataclasses (image or thermal array +
  timestamp + monotonic `frame_id`).
  - `capture()` is deliberately **blocking** — matches the multiprocessing
    capture-loop design; no async/callback complexity needed at a 30–60s
    cadence.
  - **Mock implementations must simulate realistic capture latency**
    (`time.sleep()`), not return instantly — otherwise real-hardware queue
    timing/backpressure bugs stay invisible until hardware integration, which
    defeats the point of building against mocks first.
- Confirmed integration sequence, in order:
  1. Implement `RealCamera`/`RealThermal` against the formalized interface
     (clean swap-in test against the mocks).
  2. Validate real capture latency doesn't break queue assumptions built
     against instant mocks.
  3. Resolve vial localization approach (see open item above) — this blocks
     real detection work, independent of hardware readiness.
  4. Hungarian tracker assignment logic can be unit-tested against synthetic
     centroid data independent of camera readiness — doesn't need to wait on
     1–3.
- Explicitly flagged: HSV/texture thresholds and anomaly-scoring calibration
  **cannot be meaningfully pre-tuned without real captured vial images** —
  don't guess these values ahead of real data.
- **Oven inference rule**: a vial disappearing after its last confirmed
  cooling-stage detection is inferred to have entered the oven; tracking
  stops. Requires an N-consecutive-frame hysteresis threshold before
  committing a stage transition. **Known gap**: a real failure during
  cooling is currently indistinguishable from normal oven entry — this is
  an acknowledged blind spot, not yet solved.

## Current hardware bring-up state
- SSH over Windows ICS Ethernet working (Pi at `192.168.137.x`, Windows
  gateway `192.168.137.1`)
- MLX90640: bring-up complete (see Hardware section for details)
- IMX477/Arducam: **bring-up complete.** The earlier zero-cameras
  `IndexError` is resolved (camera now confirmed working), but the exact fix
  wasn't logged this session — if it recurs, standard triage is CSI ribbon
  orientation/seating, testing with the short stock cable before the planned
  500mm extension, and checking `dmesg | grep -i imx477` for kernel
  detection. Worth confirming/documenting properly if hit again.
  - Live Flask MJPEG viewer working at 1280×720, `RGB888` format (not
    `XRGB8888` — avoids unused-alpha-channel confusion against cv2's BGR
    expectation)
  - FOV geometry calculated and validated against the height constraint:
    sensor 6.29×4.71mm (4056×3040px, 1.55µm pixel pitch), 7.857mm diagonal.
    At 5mm focal length: ~65.1°H / ~50.5°V. At h=80cm: horizontal coverage
    ≈102cm (need 78–80cm, ~22–28cm margin), vertical coverage ≈75cm (need
    only 26cm width — ~3x oversized). Confirms the mounting height clears
    the platform requirement — **assuming the varifocal is actually set to
    5mm**, which hasn't been separately verified.
  - **New open item**: because vertical FOV (75cm) so drastically overshoots
    the 26cm platform width, effective vertical pixel density on the
    platform is only ~1/3 of the sensor's nominal resolution. Decide whether
    to crop in software or accept the loss — relevant if per-vial pixel
    density becomes limiting during feature-extraction tuning.
- **New unresolved error**: `ModuleNotFoundError: No module named
  'libcamera'` when running `main.py` — traced to running from what looks
  like a non-Pi path (`/home/kkristjansson/DTU/CapEx/...`, different
  username than the Pi's `capex-vision`, likely the WSL/laptop dev
  environment). `libcamera`/`picamera2` are Pi-specific system packages tied
  to the Raspberry Pi camera stack — they will not resolve via pip or
  generic apt on a non-Pi machine. Next step: confirm which machine this
  code is actually meant to run on; if the Pi, confirm venv activation
  (`--system-site-packages`) and `apt update` before reinstalling
  `python3-libcamera python3-picamera2`.
- venv at `~/venvs/vision`, built with `--system-site-packages` (required for
  Picamera2 access)

## CAD (Onshape)
- Camera enclosure: ~50×50×105mm external, 2.5mm PETG walls, Ø35mm bottom
  aperture placeholder, M2 standoffs on 32×32mm pattern, 20×5mm CSI cable
  exit slot, M4 beam mounting at top. Separate enclosure from the Pi (geometry,
  vibration coupling, and thermal stress on the sensor all argue against
  combining them).
- Pi enclosure: passive chimney venting (inlet low at one end, outlet high
  over heatsink), Official Active Cooler installed. Diagonal flow-path
  geometry matters more than hitting a precise vent-area number; over-venting
  has no real downside.
- Thermal camera aperture needs a chamfered/flared opening to avoid
  vignetting its 110° FOV.
- **Blocking item**: polarizer holder outer radius not yet designed. This
  blocks finalizing camera-to-thermal separation distance and the enclosure
  bottom aperture. Minimum separation for full 110° FOV protection: 
  `separation ≥ R_lens + L·tanθ`, with L = 11.6mm, R_lens = 23mm → ≥40mm.
- IMX477 FOV at 5mm: ~65°H × ~50°V → ~102×75cm ground coverage at 80cm
  height. Fully covers the platform horizontally; roughly 2/3 of vertical
  pixels are wasted on background.
- 500mm CSI ribbon flagged as a signal-integrity risk — deferred, not solved.

## Known constraints / gotchas
- `Flask(debug=True)` spawns duplicate processes via the reloader — must be
  `False` when hardware capture loops are running.
- No accuracy metrics exist yet. Pipeline is validated against mock/synthetic
  data only, not real chemistry. Any output framed for external audiences
  (resume, reports) must distinguish "architected against mock interfaces"
  from "validated on real chemistry."
- Anomaly threshold calibration requires real chemistry baseline runs —
  cannot be done until lab batch-run access is available (not in the
  student's control).
- Camera exposure should use integer multiples of 10ms to average out
  possible 100Hz mains flicker — needs empirical verification once the panel
  arrives.

## Open questions (unresolved, need researcher/owner input)
- Do subtle color changes matter for this specific perovskite chemistry?
  (Determines CRI 80 vs 90+ requirement.)
- Which process stage sits in the dim ~20cm zone the LED panel doesn't cover?
- Expected per-stage timings (needed to set temporal anomaly thresholds)
- Zone polygon coordinates — not yet defined
- Hysteresis N (frames before a stage transition is committed) — not yet
  chosen
- Vial localization method: classical CV (contour/Hough-circle) vs.
  fine-tuned YOLO — unresolved, see Software architecture section
- Vertical pixel density loss on IMX477 (~2/3 wasted on background) — crop
  in software or accept it?
- Which machine `main.py` is actually meant to run on (Pi vs. WSL/laptop dev
  env) — resolve the `libcamera` import error at the source, not by patching
  around it
- Exact root cause of the earlier Picamera2 zero-camera detection — resolved
  itself but undocumented; confirm if it recurs

## How to work with this person
- CAD: intermediate Onshape user (assemblies, mate connectors, Part Studios,
  in-context sketching, Derive) — comfortable but not expert.
- Python: comfortable. Git/GitHub via SSH.
- Wants direct engineer-to-engineer communication: lead with the answer,
  explain the mechanism, skip exhaustive methodology.
- Wants pushback when reasoning is unclear, overstated, or inconsistent —
  not agreement for its own sake.
- Prefers simple/fast solutions over robust-but-complex ones in mechanical
  contexts (e.g. nut trap over heat-set inserts).
- Prefers being shown the actual numbers/geometry over being told a
  conclusion.
- Gets overwhelmed by long dense answers — one focused step at a time over
  exhaustive coverage.
