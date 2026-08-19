"""Storage: schema, round-tripping, and the concurrency the design assumes."""

from __future__ import annotations

import json
import time

import pytest

from pipeline.types import Event, PipelineResult, VialReport
from storage.db import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.sqlite3")
    d.start_run(rgb_source="mock", thermal_source="mock", simulated=True)
    yield d
    d.close()


def result_with(events=(), vials=(), frame_id=1) -> PipelineResult:
    return PipelineResult(
        frame_id=frame_id, timestamp=time.time(), source="mock", simulated=True,
        vials=list(vials), events=list(events),
        stage_counts={"filling": len(list(vials))},
        timings_ms={"localize": 1.2},
    )


def vial(track_id=1, **features) -> VialReport:
    return VialReport(track_id=track_id, cx=100.0, cy=200.0, radius=17.0,
                      stage="filling", hits=3, missed=0, age_s=90.0,
                      time_in_stage_s=90.0, features=features)


def event(kind="turbidity", severity="warning", **data) -> Event:
    return Event(kind=kind, severity=severity, message="something diverged",
                 timestamp=time.time(), frame_id=1, track_id=1,
                 detector="turbidity", zone="heating", data=data)


def test_schema_is_created(db):
    tables = {r["name"] for r in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "frames", "vial_samples", "events", "thermal_samples"} <= tables


def test_wal_is_enabled(db):
    mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal", "two processes write to this file"


def test_round_trips_a_result(db):
    db.log_result(result_with(vials=[vial(1, brightness=12.5)],
                              events=[event(z=3.9)]))
    counts = db.counts()
    assert counts["frames"] == 1
    assert counts["vial_samples"] == 1
    assert counts["events"] == 1


def test_features_survive_as_json(db):
    db.log_result(result_with(vials=[vial(1, hue=24.5, texture_var=8.0)]))
    row = db.conn.execute("SELECT features FROM vial_samples").fetchone()
    assert json.loads(row["features"]) == {"hue": 24.5, "texture_var": 8.0}


def test_json_extract_queries_a_feature(db):
    """The schema's whole premise: new features need no migration."""
    for i in range(3):
        db.log_result(result_with(vials=[vial(1, brightness=float(i))],
                                  frame_id=i + 1))
    points = db.vial_series(1, "brightness")
    assert [p["value"] for p in points] == [0.0, 1.0, 2.0]


def test_event_data_round_trips(db):
    db.log_result(result_with(events=[event(z=4.2, peers=17)]))
    got = db.recent_events()[0]
    assert got["data"] == {"z": 4.2, "peers": 17}
    assert got["severity"] == "warning"


def test_severity_filter(db):
    db.log_result(result_with(events=[
        event(severity="info"), event(severity="warning"),
        event(severity="alert")]))
    assert len(db.recent_events(min_severity="warning")) == 2
    assert len(db.recent_events(min_severity="alert")) == 1
    assert len(db.recent_events()) == 3


def test_a_failed_write_rolls_back_the_whole_frame(db):
    """A frame row without its vials would be a lie that is painful later.

    Provoked with a real constraint violation rather than a mock: cx is NOT
    NULL, so a vial row missing a centroid fails inside the transaction after
    the frame row has already been inserted.
    """
    broken = vial(1)
    broken.cx = None
    with pytest.raises(Exception):
        db.log_result(result_with(vials=[broken], events=[event()]))

    assert db.counts()["frames"] == 0
    assert db.counts()["vial_samples"] == 0
    assert db.counts()["events"] == 0

    # And the connection is still usable afterwards - a rollback must not
    # leave the writer wedged mid-transaction.
    db.log_result(result_with(vials=[vial(2)], frame_id=2))
    assert db.counts()["frames"] == 1


def test_thermal_logging(db):
    db.log_thermal(time.time(), 1, 21.0, 24.5, 63.2, simulated=True)
    series = db.thermal_series()
    assert len(series) == 1
    assert series[0]["max_c"] == 63.2


def test_two_connections_can_write_concurrently(tmp_path):
    """The thermal logger and the pipeline are separate processes."""
    path = tmp_path / "concurrent.sqlite3"
    a = Database(path)
    run_id = a.start_run(rgb_source="mock")
    b = Database(path)
    b.attach_run(run_id)
    try:
        a.log_result(result_with(vials=[vial(1)]))
        b.log_thermal(time.time(), 1, 20.0, 22.0, 25.0)
        a.log_result(result_with(frame_id=2))
        assert a.counts()["frames"] == 2
        assert b.counts()["thermal_samples"] == 1
    finally:
        a.close()
        b.close()


def test_prune_drops_old_rows_only(db):
    old = result_with(frame_id=1, vials=[vial(1)])
    old.timestamp = time.time() - 40 * 86400
    for v in old.vials:
        pass
    db.log_result(old)
    db.conn.execute("UPDATE vial_samples SET timestamp=?", (old.timestamp,))
    db.log_result(result_with(frame_id=2, vials=[vial(2)]))

    db.prune(retention_days=30)
    assert db.counts()["frames"] == 1
    assert db.counts()["vial_samples"] == 1


def test_prune_is_a_noop_when_disabled(db):
    db.log_result(result_with(vials=[vial(1)]))
    assert db.prune(retention_days=0) == {}
    assert db.counts()["frames"] == 1


def test_log_result_before_start_run_is_an_error(tmp_path):
    d = Database(tmp_path / "norun.sqlite3")
    try:
        with pytest.raises(RuntimeError):
            d.log_result(result_with())
    finally:
        d.close()


def test_snapshot_store_paths_are_relative(tmp_path):
    import numpy as np

    from storage.images import SnapshotStore

    store = SnapshotStore(tmp_path / "snaps")
    path = store.save_frame(np.zeros((20, 20, 3), np.uint8), 7)
    assert path is not None
    assert not path.startswith("/"), "paths in the DB must survive a move"
    assert store.resolve(path).is_file()
