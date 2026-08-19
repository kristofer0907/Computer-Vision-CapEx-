"""SQLite persistence for runs, frames, per-vial rows, events and thermal.

Two processes write to this file concurrently - the pipeline and the thermal
logger - which is why WAL mode and a busy timeout are set on every connection.
WAL lets one writer and any number of readers proceed at once; busy_timeout
turns the writer-versus-writer collision into a short wait instead of an
immediate "database is locked". Neither is optional on a Pi with an SD card,
where an fsync can take a surprisingly long time.

Connections are per-process and per-thread. sqlite3 objects are not safe to
share across either, so Database is constructed where it is used rather than
passed across a Queue.

Schema notes:

  * Per-vial features are stored as a JSON blob, not as columns. The feature
    set is not settled and will not be for a while - adding a feature must not
    mean a migration. Query with json_extract() when a specific one is needed;
    SQLite has had it built in since 3.38 and the Pi ships newer.
  * timestamps are unix seconds as REAL, always. Local time formatting is the
    dashboard's problem, and storing a formatted string would make any
    duration query a parsing exercise.
  * A "run" groups everything from one start of the monitor. Nothing tries to
    guess where a chemistry batch starts and ends - that needs a signal from
    the platform controller that does not exist yet.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from config import STORAGE
from pipeline.types import Event, PipelineResult

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    rgb_source      TEXT,
    thermal_source  TEXT,
    simulated       INTEGER NOT NULL DEFAULT 0,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    frame_id    INTEGER NOT NULL,
    timestamp   REAL NOT NULL,
    n_vials     INTEGER NOT NULL,
    stage_counts TEXT,            -- JSON {stage: count}
    timings_ms   TEXT,            -- JSON {stage: ms}
    warnings     TEXT,            -- JSON [str]
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_frames_run_ts ON frames(run_id, timestamp);

CREATE TABLE IF NOT EXISTS vial_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    frame_row   INTEGER NOT NULL REFERENCES frames(id),
    timestamp   REAL NOT NULL,
    track_id    INTEGER NOT NULL,
    stage       TEXT,
    cx          REAL NOT NULL,
    cy          REAL NOT NULL,
    radius      REAL NOT NULL,
    age_s       REAL,
    time_in_stage_s REAL,
    features    TEXT NOT NULL DEFAULT '{}',   -- JSON {name: number}
    scores      TEXT NOT NULL DEFAULT '{}'    -- JSON {name: number}
);
CREATE INDEX IF NOT EXISTS idx_vial_run_track ON vial_samples(run_id, track_id, timestamp);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    timestamp   REAL NOT NULL,
    frame_id    INTEGER,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    detector    TEXT,
    track_id    INTEGER,
    zone        TEXT,
    message     TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}',   -- JSON
    crop_path   TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity, timestamp DESC);

CREATE TABLE IF NOT EXISTS thermal_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(id),
    timestamp   REAL NOT NULL,
    frame_id    INTEGER,
    min_c       REAL NOT NULL,
    mean_c      REAL NOT NULL,
    max_c       REAL NOT NULL,
    simulated   INTEGER NOT NULL DEFAULT 0,
    grid        BLOB               -- optional raw 24x32 float32, see log_thermal
);
CREATE INDEX IF NOT EXISTS idx_thermal_ts ON thermal_samples(timestamp DESC);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with the pragmas this project needs.

    Every caller must go through here. A connection opened without WAL will
    make the other writer's transactions fail, and the failure will look like
    a random intermittent bug rather than a missing pragma.
    """
    target = Path(path or STORAGE.db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=STORAGE.busy_timeout_ms / 1000.0,
                           isolation_level=None)   # autocommit; we manage BEGINs
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={STORAGE.busy_timeout_ms}")
    # NORMAL rather than FULL: with WAL this is durable across process crashes
    # and only loses the last transactions on an OS crash or power cut. On an
    # SD card the difference in write amplification is worth having, and a
    # monitoring log is not financial data.
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=float)
    except (TypeError, ValueError):
        log.warning("could not serialise %r for storage", type(value), exc_info=True)
        return "{}"


class Database:
    """Per-process handle. Not shared across processes or threads."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or STORAGE.db_path)
        self.conn = connect(self.path)
        self._lock = threading.Lock()   # guards this connection only
        self.run_id: int | None = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),))

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                log.warning("closing database failed", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ runs
    def start_run(self, rgb_source: str | None = None,
                  thermal_source: str | None = None,
                  simulated: bool = False, note: str | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO runs(started_at, rgb_source, thermal_source, "
                "simulated, note) VALUES(?,?,?,?,?)",
                (time.time(), rgb_source, thermal_source, int(simulated), note))
            self.run_id = int(cur.lastrowid)
        log.info("run %d started (rgb=%s thermal=%s%s)", self.run_id, rgb_source,
                 thermal_source, ", simulated" if simulated else "")
        return self.run_id

    def end_run(self, run_id: int | None = None) -> None:
        rid = run_id or self.run_id
        if rid is None:
            return
        with self._lock:
            self.conn.execute("UPDATE runs SET ended_at=? WHERE id=? AND "
                              "ended_at IS NULL", (time.time(), rid))

    def attach_run(self, run_id: int) -> None:
        """Adopt a run started by another process (the thermal logger does this)."""
        self.run_id = run_id

    def latest_run(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()

    # ----------------------------------------------------------- pipeline IO
    def log_result(self, result: PipelineResult,
                   snapshot_path: str | None = None) -> int:
        """Write one analysis frame, its vials and its events in one transaction.

        One transaction on purpose: a frame row with no vial rows, or events
        pointing at a frame that was not written, would both be lies that are
        painful to notice weeks later while reading a run back.
        """
        if self.run_id is None:
            raise RuntimeError("log_result() before start_run()")

        with self._lock:
            self.conn.execute("BEGIN")
            try:
                cur = self.conn.execute(
                    "INSERT INTO frames(run_id, frame_id, timestamp, n_vials, "
                    "stage_counts, timings_ms, warnings, snapshot_path) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (self.run_id, result.frame_id, result.timestamp,
                     result.n_vials, _json(result.stage_counts),
                     _json(result.timings_ms), _json(result.warnings),
                     snapshot_path))
                frame_row = int(cur.lastrowid)

                self.conn.executemany(
                    "INSERT INTO vial_samples(run_id, frame_row, timestamp, "
                    "track_id, stage, cx, cy, radius, age_s, time_in_stage_s, "
                    "features, scores) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(self.run_id, frame_row, result.timestamp, v.track_id,
                      v.stage, v.cx, v.cy, v.radius, v.age_s,
                      v.time_in_stage_s, _json(v.features), _json(v.scores))
                     for v in result.vials])

                self._insert_events(result.events)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return frame_row

    def _insert_events(self, events: Iterable[Event],
                       crop_paths: dict[int, str] | None = None) -> None:
        """Caller holds the lock."""
        rows = []
        for e in events:
            rows.append((self.run_id, e.timestamp, e.frame_id, e.kind,
                         e.severity, e.detector, e.track_id, e.zone,
                         e.message, _json(e.data),
                         (crop_paths or {}).get(e.track_id)))
        if rows:
            self.conn.executemany(
                "INSERT INTO events(run_id, timestamp, frame_id, kind, severity, "
                "detector, track_id, zone, message, data, crop_path) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)

    def log_events(self, events: Iterable[Event],
                   crop_paths: dict[int, str] | None = None) -> None:
        with self._lock:
            self._insert_events(events, crop_paths)

    def set_event_crop(self, event_id: int, path: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE events SET crop_path=? WHERE id=?",
                              (path, event_id))

    # ------------------------------------------------------------- thermal IO
    def log_thermal(self, timestamp: float, frame_id: int, min_c: float,
                    mean_c: float, max_c: float, simulated: bool = False,
                    grid: bytes | None = None) -> None:
        """One thermal sample.

        `grid` is the optional raw 24x32 float32 array as bytes - 3 kB per
        sample, which at the 2 s poll is about 130 MB a day. Off by default;
        the three summary numbers are what the dashboard plots. Pass it only
        when a researcher actually wants the field back for offline review,
        and prune afterwards.
        """
        with self._lock:
            self.conn.execute(
                "INSERT INTO thermal_samples(run_id, timestamp, frame_id, "
                "min_c, mean_c, max_c, simulated, grid) VALUES(?,?,?,?,?,?,?,?)",
                (self.run_id, timestamp, frame_id, min_c, mean_c, max_c,
                 int(simulated), grid))

    # -------------------------------------------------------------- querying
    def recent_events(self, limit: int = 50, run_id: int | None = None,
                      min_severity: str | None = None) -> list[dict]:
        sql = ["SELECT * FROM events WHERE 1=1"]
        args: list[Any] = []
        rid = run_id if run_id is not None else self.run_id
        if rid is not None:
            sql.append("AND run_id=?")
            args.append(rid)
        if min_severity:
            from pipeline.types import SEVERITIES, severity_rank
            allowed = [s for s in SEVERITIES
                       if severity_rank(s) >= severity_rank(min_severity)]
            sql.append(f"AND severity IN ({','.join('?' * len(allowed))})")
            args.extend(allowed)
        sql.append("ORDER BY timestamp DESC, id DESC LIMIT ?")
        args.append(limit)
        rows = self.conn.execute(" ".join(sql), args).fetchall()
        return [self._event_row(r) for r in rows]

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["data"] = json.loads(d.get("data") or "{}")
        except json.JSONDecodeError:
            d["data"] = {}
        return d

    def vial_series(self, track_id: int, feature: str,
                    run_id: int | None = None, limit: int = 500) -> list[dict]:
        """One feature's history for one vial, oldest first.

        json_extract does the work in SQLite rather than pulling every blob
        into Python, which matters once a run has a few hundred thousand vial
        rows.
        """
        rid = run_id if run_id is not None else self.run_id
        rows = self.conn.execute(
            "SELECT timestamp, stage, json_extract(features, '$.' || ?) AS value "
            "FROM vial_samples WHERE run_id=? AND track_id=? "
            "ORDER BY timestamp DESC LIMIT ?",
            (feature, rid, track_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def latest_frame(self, run_id: int | None = None) -> dict | None:
        rid = run_id if run_id is not None else self.run_id
        row = self.conn.execute(
            "SELECT * FROM frames WHERE run_id=? ORDER BY timestamp DESC LIMIT 1",
            (rid,)).fetchone()
        return dict(row) if row else None

    def thermal_series(self, limit: int = 240,
                       run_id: int | None = None) -> list[dict]:
        rid = run_id if run_id is not None else self.run_id
        rows = self.conn.execute(
            "SELECT timestamp, min_c, mean_c, max_c FROM thermal_samples "
            "WHERE (? IS NULL OR run_id=?) ORDER BY timestamp DESC LIMIT ?",
            (rid, rid, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def counts(self, run_id: int | None = None) -> dict[str, int]:
        rid = run_id if run_id is not None else self.run_id
        out = {}
        for table in ("frames", "vial_samples", "events", "thermal_samples"):
            row = self.conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE run_id=?", (rid,)
            ).fetchone()
            out[table] = int(row["n"]) if row else 0
        return out

    # -------------------------------------------------------------- pruning
    def prune(self, retention_days: int | None = None) -> dict[str, int]:
        """Delete rows older than the retention window. 0 days disables.

        Does not VACUUM: reclaiming the space rewrites the whole file, which
        on an SD card is exactly the kind of write burst worth avoiding. The
        freed pages get reused by subsequent inserts anyway.
        """
        days = STORAGE.retention_days if retention_days is None else retention_days
        if days <= 0:
            return {}
        cutoff = time.time() - days * 86400
        deleted: dict[str, int] = {}
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                for table in ("vial_samples", "events", "thermal_samples", "frames"):
                    cur = self.conn.execute(
                        f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
                    deleted[table] = cur.rowcount
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        log.info("pruned rows older than %d days: %s", days, deleted)
        return deleted
