# Reorg review — nothing here has been deleted

Everything below is **still on disk and still working**. This file is the verify gate:
tick a box, and the removal happens in a separate pass.

Baseline before the reorg: `python -m pytest -q` → **1 failed, 162 passed**. The one
failure is pre-existing (`tests/test_edit_zones.py:34` expects the old `filling` stage
name; `config.STAGE_ORDER` is now `storing/injection/heating/collection`).

Recovery for anything removed later: `git checkout <commit> -- <path>`.

---

## 1. DECIDE FIRST — lid detection

`pipeline/anomaly.py` (353 lines) + `lid_score`/`has_lid` in `pipeline/features.py`.

The **only implemented detection in the repo**. `MissingLid` latches an open crucible on
a heater, requests a run stop, and freezes state. Ground truth: 225 hand-labelled
crucibles in `data/lid_review.json`, zero missed lids. Tests: `test_anomaly.py` (13),
`test_lid.py` (4). Your last three commits are this feature.

It is **not** in the MVP `anomaly/` list you gave me.

- [x] **Keep** — wrapped as `MissingLidDetector`, a sixth `check() -> AnomalyResult`
      detector. *(assumed; this is what I implemented)*
- [ ] Remove — the detector set then ships as five stubs with zero working detection.

Note `anomaly.py:257`: `on_stop` stops *this pipeline*. There is no control channel to
the synthesis platform. It is not a safety interlock.

---

## 2. Multiprocessing scaffolding — replaced by the single-process loop

`main.py` no longer imports any of it. Nothing else imports them except each other.

| File | LOC | What it does | Tests lost |
|---|---|---|---|
| `runtime/supervisor.py` | 236 | spawns 3 daemon processes + a drain thread | 0 direct |
| `runtime/processing.py` | 224 | pipeline worker, owns SQLite writes | 0 direct |
| `runtime/capture.py` | 127 | camera worker, preview + analysis queues | 0 direct |
| `runtime/messages.py` | 78 | pickle-safe dataclasses for queue transport | 3 in `test_runtime.py` |

- [ ] Remove all four

**`runtime/bus.py` (143) is NOT in this list — keep it.** It is `threading.Lock` only, no
`multiprocessing`. The capture loop and the dashboard both use `LatestSlot`, and
`test_runtime.py` covers it.

---

## 3. Superseded by the MVP path

| File | LOC | Why off-path | Tests lost |
|---|---|---|---|
| `pipeline/runner.py` | 376 | old whole-platform orchestrator; `main.py` is now the loop | **25** (`test_pipeline.py`) |
| `pipeline/handoff.py` | 144 | heater→storage linking; `lineage.py` supersedes it | **8** (`test_handoff.py`) |
| `pipeline/stats.py` | 81 | dead — zero importers, zero tests | 0 |

`pipeline/runner.py` is also used by `tools/replay.py`; `pipeline/handoff.py` by
`tools/track_regions.py`. Both tools are do-not-touch, so removing either file means
editing a tool.

- [ ] Remove `pipeline/stats.py` (zero risk)
- [ ] Remove `pipeline/runner.py` (costs 25 tests + `tools/replay.py`)
- [ ] Remove `pipeline/handoff.py` (costs 8 tests + edits `tools/track_regions.py`)

**`HungarianTracker` in `pipeline/tracking.py` is off-path but the file stays** —
`CadenceController` lives in it and drives the 15–60 s cadence. Splitting the file is a
later cleanup, not this pass.

---

## 4. Scratch / duplicate — flagged only, no action taken

Your do-not-touch set covers `tools/` and `capture/`, so these are listed, not touched.

- `tools/acquired_regions.py` (46) — click-to-print coords, hardcoded absolute path to a
  file that is currently `git rm`'d. Scratch.
- `capture/capture_images.py` (154) — standalone picamera2 + Flask MJPEG script.
  Duplicates what `main.py` + `dashboard/app.py` now do.
- **`venv/` and `capture/capex/` are committed virtualenvs.** Thousands of vendored
  third-party files in git. They are why a repo-wide grep returns noise. Recommend
  `.gitignore` + `git rm -r --cached`. Not done.

---

## 5. Dashboard routes beyond MJPEG

Your spec says "Flask MJPEG stream only". The non-MJPEG routes are kept because
`tests/test_runtime.py` asserts them directly and tests are do-not-touch:

`/`, `/api/state`, `/api/events`, `/api/crucibles`, `/api/crucibles/<id>/series`,
`/healthz`, `/snapshot/<path>`.

- [ ] Trim to the three `/feed/*` MJPEG routes (costs ~8 tests in `test_runtime.py`)

---

## 6. Stage-name drift in `config.py`

Two vocabularies coexist:

- `config.py` `ZONES` polygons — `filling / conveyor / lidding / heating / cooling`.
  Old whole-platform placeholders, `ZONES.calibrated` is false.
- `config.py` `STAGE_ORDER` — `storing / injection / heating / collection`. The real
  bench, matching `data/slots_*.json` and `data/lane_injection.json`.

The second is authoritative and the data files are do-not-touch, so both were left alone.
This mismatch is the cause of the pre-existing `test_edit_zones.py` failure.

- [ ] Re-trace zone polygons under the real stage names (an operator task with
      `tools/edit_zones.py` on a real frame, not a code change)

---

## 7. Learned-detector and motion-model tracker references — nothing existed to delete

Confirmed by grep: **zero code, zero imports, zero dependencies**. Every hit was prose.
The rejection notes were rewritten in neutral wording so the decisions survive but the
banned terms do not: `pipeline/tracking.py`, `pipeline/region_trackers.py`,
`pipeline/localize.py`, `pipeline/types.py`, `runtime/processing.py`, `config.py`,
`requirements.txt`, `README.md`, `CLAUDE.md`.

Both decisions are stated as settled in `CLAUDE.md` under Software architecture.

---

## 8. `vial` → `crucible`: four accepted residues

A repo-wide zero-hit grep is not reachable while `lineage.py`, `tools/`, `data/` and
`capture/` are do-not-touch. Remaining hits, all deliberate:

1. **`pipeline/lineage.py`** — `VialLineage`, `vial_on_heater`, `vial_on_cooling`.
   "Keep as-is, do not modify" won over the rename. `tests/test_lineage.py` stays with it.
2. **`data/vials.json`, `tools/mark_vials.py`, `short_vial_id`** — the on-disk
   artifact name and the tool that writes it. `data/` and `tools/` are do-not-touch, so
   the filenames stayed; the vocabulary inside the tools was swept.
3. **`capture/`, `testing/`, `run01/`** — captured images and generated JSON.
4. **`venv/`, `capture/capex/`** — vendored third-party code (see §4).

- [ ] Rename `VialLineage` → `CrucibleLineage` anyway (pure rename, touches
      `pipeline/lineage.py` + 55 refs in `tests/test_lineage.py`)
