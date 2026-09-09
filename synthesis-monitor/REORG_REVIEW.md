# Reorg review — closed out

Your answers from `fae8d11` are applied. This file is now the record of what was
removed and why, plus the two items still open.

Test suite: **145 passed, 0 failed** (was 162 passed + 1 pre-existing failure).
The 18 tests that disappeared tested code you asked to delete — itemised below.
The pre-existing `test_edit_zones` failure is **fixed**, not deleted (see §6).

Recovery for anything here: `git checkout 19aaccb -- <path>`.

---

## 1. Lid detection — KEPT ✅

> ANSWER: Lid detection should be in the anomaly

Done. `MissingLidDetector` (`pipeline/detectors/missing_lid.py`) is registered in
`DETECTION.enabled` alongside the five stubs and returns `AnomalyResult` like the rest.
Scoring stays in `pipeline/features.lid_score`, the latch in `pipeline/anomaly.MissingLid`.
It is the only detector in the set that is `implemented = True`.

A latched hazard now also **halts the capture loop** (`main.py`), with the dashboard left
serving so the frame that stopped the run is the one on screen.

---

## 2. Multiprocessing scaffolding — REMOVED (3 of 4) ✅

> ANSWER: That's fine

Deleted: `runtime/supervisor.py` (236), `runtime/capture.py` (127),
`runtime/processing.py` (224). `runtime/__init__.py` rewritten.

**`runtime/messages.py` had to stay.** `runtime/thermal.py` imports `ThermalMessage` and
`WorkerStatus` from it, and the thermal logger is on your keep-exactly-as-is list. `main.py`
uses `PreviewMessage` for the same reason. It holds three plain dataclasses and no
`multiprocessing` import, so nothing multiprocess survives in it.

`runtime/bus.py` kept as planned (threading only, `LatestSlot`/`RingBuffer`).

---

## 3. HungarianTracker and the old orchestrator — REMOVED ✅

> ANSWER: Remove any thing involving HungarianTracker

Deleted:
- `HungarianTracker` class, cut out of `pipeline/tracking.py` (106 lines)
- `pipeline/runner.py` (376) — its default tracker was `HungarianTracker`
- `tools/replay.py` (132) — existed only to drive that runner; `tools/track_regions.py`
  already does headless replay for the real trackers
- `pipeline/stats.py` (81) — dead, zero importers

`create_tracker()` now has one implementation: `RegionTracker` (slots + lane) for every name.

**`pipeline/assignment.py` is NOT removed, on purpose.** The function is called
`hungarian`, but it is the assignment *solver*, and `SlotTracker` uses it on every frame
to match detections to marked slots. Removing it would break the tracker you kept. The
`HungarianTracker` *class* is what's gone.

**Kept: `CadenceController`** (in `pipeline/tracking.py`) — drives the 15–60 s cadence,
`main.py` depends on it. And `StageTracker` (in `pipeline/zones.py`) — holds stage
hysteresis and the oven-entry inference.

### Tests removed with the code — 18 total
- `tests/test_tracking.py`: 6 tests of `HungarianTracker` identity/gating/ghost
  suppression. The other 7 were **rewritten to drive `StageTracker` and
  `CadenceController` directly** — hysteresis, boundary flicker, oven inference and
  cadence are all still covered.
- `tests/test_pipeline.py`: 12 `PipelineRunner` tests. Detector isolation, the
  registry, all four `ManualLocalizer` tests, the file source and the scene-render test
  were **kept and rewritten** against `DetectorHost` directly.

⚠️ **Coverage genuinely lost:** end-to-end "synthetic frames in, result out". The mock
camera draws the old whole-platform bench, not the marked crucible rack, so those
assertions could only ever have tested a drawing. End-to-end proof now comes from
`python main.py --rgb file` over a real capture folder (run each time, see below).

⚠️ **`StageTracker` is now unused by production code** — the region trackers set `stage`
from zone membership directly. Its oven-inference rule is still documented in CLAUDE.md
as a requirement. Left in place rather than silently dropping that capability.
**Open: wire it into the region trackers, or drop the oven rule.** ← see §9

---

## 4. Scratch / duplicate — LEFT ALONE ✅

> ANSWER: That's fine

No action, as flagged. `tools/acquired_regions.py`, `capture/capture_images.py` and the
two committed virtualenvs (`venv/`, `capture/capex/`) are untouched.
The `.gitignore` recommendation for the virtualenvs still stands and is **not** done.

---

## 5. Dashboard routes — KEPT ✅

> ANSWER: That's fine

All routes kept: three `/feed/*` MJPEG streams plus `/`, `/api/state`, `/api/events`,
`/api/crucibles`, `/api/crucibles/<id>/series`, `/api/detectors`, `/healthz`,
`/snapshot/<path>`. `tests/test_runtime.py` still covers them.

---

## 6. One naming convention — DONE ✅

> ANSWER: Make sure there is only one naming convention, and it should be
> storing / injection / heating / collection

The retired vocabulary (`filling / conveyor / lidding / cooling`) is gone from code:

- `config._DEFAULT_POLYGONS` renamed to the four real stages (was the source of the drift)
- `TRACKING.oven_entry_from`: `"cooling"` → `"collection"`
- `REGION_TRACKING.cooling_zone` → `collection_zone`; `heater_zone` → `heating_zone`
- `tools/mark_slots.py` docstring examples, `tests/test_storage.py` fixture,
  `tests/test_edit_zones.py` fixture

**This fixed the long-standing `test_edit_zones` failure** — it expected `"filling"` to
sort first, and `"filling"` had stopped being a stage.

Note `data/slots_filling_*.json` filenames still say "filling". `data/` is do-not-touch and
`config.REGION_TRACKING.slot_files` maps the real stage names onto them. Renaming the
files is a one-line config change plus a `git mv` whenever you want it.

---

## 7. Learned-detector / motion-model tracker references — DONE ✅

> ANSWER: That's fine

Zero hits repo-wide. Decisions preserved in neutral wording in `CLAUDE.md`.

---

## 8. `vial` → `crucible` — DONE ✅

> ANSWER: That's fine, just don't break anything.

`VialLineage` → `CrucibleLineage`, `vial_on_heater` → `crucible_on_heater`,
`vial_on_cooling` → `crucible_on_cooling`, `vial_id` → `crucible_id`, across
`pipeline/lineage.py`, `tests/test_lineage.py`, `main.py`, `tools/track_regions.py`.
All 17 lineage tests still pass — nothing broken.

"heater" and "cooling" survive as *hardware* words (the pads). They are not stage names;
the four stage names are storing/injection/heating/collection and none of them is affected.

Remaining `vial` hits, all deliberate:
- `data/vials.json` and the path strings pointing at it (`data/` is do-not-touch)
- `tools/mark_vials.py` — filename only; its contents were swept
- `drivers/thermal_cam.py`, `runtime/thermal.py` — 3 comment lines. Reverted on purpose so
  those two files stay **byte-identical**, which is your keep-as-is instruction.
- `capture/`, `testing/`, `run01/`, `venv/` — data and vendored code

---

## 9. STILL OPEN — two decisions

### 9a. `pipeline/handoff.py` (144 lines, 8 tests)
Your §3 answer named `HungarianTracker`; it did not mention this. It is the older
heater→storage linker that `lineage.py` supersedes, still imported by
`tools/track_regions.py`. **Not removed.**

- [ ] Remove it (costs 8 tests + one edit to `tools/track_regions.py`)
- [ ] Keep it

### 9b. `StageTracker` / the oven-entry rule
Now unused by production code (see §3). The rule — "a crucible last seen on the end rack
that disappears entered the oven" — is a documented requirement with a known blind spot.

- [ ] Wire `StageTracker` into the region trackers so the rule actually runs
- [ ] Drop the oven rule and delete `StageTracker`
