"""The storage policy: the file, the deletes, and the snapshot cap."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from pipeline.types import AnomalyResult, Event, CrucibleReport, PipelineResult
from storage.db import Database
from storage.images import SnapshotStore


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite3")
    d.start_run(rgb_source="test")
    yield d
    d.close()


def result_at(ts: float, frame_id: int = 1) -> PipelineResult:
    return PipelineResult(
        frame_id=frame_id, timestamp=ts, source="test", simulated=True,
        crucibles=[CrucibleReport(track_id=1, cx=1.0, cy=2.0, radius=3.0,
                                  stage="heating", hits=1, missed=0, age_s=0.0,
                                  time_in_stage_s=0.0)],
        events=[Event(kind="k", severity="info", message="m", timestamp=ts,
                      frame_id=frame_id)])


def write_snapshot(store: SnapshotStore, day: date, frame_id: int) -> str:
    ts = datetime.combine(day, datetime.min.time()).timestamp() + 3600
    img = np.zeros((8, 8, 3), np.uint8)
    return store.save_frame(img, frame_id, ts)


# --------------------------------------------------------------------- policy
def test_policy_file_overrides_the_defaults(tmp_path, monkeypatch):
    import config

    policy = tmp_path / "storage_policy.json"
    policy.write_text(json.dumps({"snapshot_every_n_frames": 9,
                                  "max_snapshots": 500,
                                  "retention_days": 7}))
    monkeypatch.setattr(config, "STORAGE_POLICY_FILE", policy)
    assert config._policy_value("snapshot_every_n_frames", 4) == 9
    assert config._policy_value("max_snapshots", 0) == 500
    assert config._policy_value("retention_days", 30) == 7


def test_a_missing_policy_file_leaves_the_defaults(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "STORAGE_POLICY_FILE", tmp_path / "absent.json")
    assert config.load_storage_policy() == {}
    assert config._policy_value("retention_days", 30) == 30


@pytest.mark.parametrize("bad", ["not-a-number", -5, None])
def test_a_bad_policy_value_falls_back_rather_than_keeping_nothing(
        tmp_path, monkeypatch, bad):
    """A negative or junk retention must not become "delete everything"."""
    import config

    policy = tmp_path / "storage_policy.json"
    policy.write_text(json.dumps({"retention_days": bad}))
    monkeypatch.setattr(config, "STORAGE_POLICY_FILE", policy)
    assert config._policy_value("retention_days", 30) == 30


def test_unreadable_policy_file_falls_back(tmp_path, monkeypatch):
    import config

    policy = tmp_path / "storage_policy.json"
    policy.write_text("{ this is not json")
    monkeypatch.setattr(config, "STORAGE_POLICY_FILE", policy)
    assert config.load_storage_policy() == {}


# ------------------------------------------------------------- date deletes
def test_prune_between_takes_only_the_window(db):
    now = time.time()
    for i, ts in enumerate((now - 3 * 86400, now - 86400, now)):
        db.log_result(result_at(ts, frame_id=i))
    assert db.counts_by_table()["frames"] == 3

    # The middle day only.
    deleted = db.prune_between(now - 2 * 86400, now - 3600)
    assert deleted["frames"] == 1
    assert db.counts_by_table()["frames"] == 2


def test_prune_between_is_half_open_at_the_end(db):
    now = time.time()
    db.log_result(result_at(now))
    # A window ending exactly at the row's timestamp must not take it.
    assert db.prune_between(now - 100, now)["frames"] == 0
    assert db.prune_between(now, now + 100)["frames"] == 1


def test_prune_between_rejects_a_backwards_range(db):
    with pytest.raises(ValueError):
        db.prune_between(time.time(), time.time() - 100)


def test_anomalies_are_kept_unless_asked_for(db):
    now = datetime.now()
    db.write_event(AnomalyResult(True, "heating", "missing_lid", now, "a.jpg"))
    db.log_result(result_at(now.timestamp()))

    span = (now.timestamp() - 3600, now.timestamp() + 3600)
    db.prune_between(*span)
    assert db.counts_by_table()["anomaly_events"] == 1, "alarms are not history"

    db.prune_between(*span, include_anomalies=True)
    assert db.counts_by_table()["anomaly_events"] == 0


def test_clear_all_empties_everything(db):
    now = datetime.now()
    db.log_result(result_at(now.timestamp()))
    db.write_event(AnomalyResult(True, "heating", "missing_lid", now, "a.jpg"))

    db.clear_all()
    assert sum(db.counts_by_table().values()) == 0


# ----------------------------------------------------------------- snapshots
def test_delete_between_removes_whole_days(tmp_path):
    store = SnapshotStore(tmp_path / "snaps")
    today = date.today()
    for offset in range(4):
        write_snapshot(store, today - timedelta(days=offset), offset)
    assert store.usage()[0] == 4

    removed = store.delete_between(today - timedelta(days=2),
                                   today - timedelta(days=1))
    assert removed == 2
    assert store.usage()[0] == 2


def test_delete_between_rejects_a_backwards_range(tmp_path):
    store = SnapshotStore(tmp_path / "snaps")
    with pytest.raises(ValueError):
        store.delete_between(date.today(), date.today() - timedelta(days=1))


def test_clear_all_removes_every_snapshot(tmp_path):
    store = SnapshotStore(tmp_path / "snaps")
    for offset in range(3):
        write_snapshot(store, date.today() - timedelta(days=offset), offset)
    assert store.clear_all() == 3
    assert store.usage() == (0, 0)
    assert store.root.exists(), "the root itself stays"


def test_enforce_max_keeps_the_newest(tmp_path):
    store = SnapshotStore(tmp_path / "snaps")
    today = date.today()
    for offset in range(5):
        write_snapshot(store, today - timedelta(days=offset), offset)

    assert store.enforce_max(2) == 3
    count, _ = store.usage()
    assert count == 2
    # The two surviving days are the most recent ones.
    days = sorted(p.name for p in store.root.iterdir() if p.is_dir())
    assert days == [str(today - timedelta(days=1)), str(today)]


def test_a_cap_of_zero_means_no_cap(tmp_path):
    store = SnapshotStore(tmp_path / "snaps")
    write_snapshot(store, date.today(), 0)
    assert store.enforce_max(0) == 0
    assert store.usage()[0] == 1


def test_usage_reports_files_and_bytes(tmp_path):
    store = SnapshotStore(tmp_path / "snaps")
    write_snapshot(store, date.today(), 0)
    count, total = store.usage()
    assert count == 1
    assert total > 0
