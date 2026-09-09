"""Set the storage policy, and clear what you no longer want.

    python -m tools.storage_policy

Two jobs, both interactive:

  * Edit the policy. How often a full frame is saved, how many to keep, and
    how many days of history to hold. Answers are written to
    data/storage_policy.json, which config.StorageConfig loads, so main.py
    picks them up on its next start with nothing else to change.

  * Delete. A date range, or everything. Both say what they are about to
    remove and ask before doing it, because neither is recoverable.

Run it against a stopped monitor. Deleting rows under a live run will not
corrupt anything - SQLite is transactional and the snapshots are plain files -
but the run will keep writing rows behind you and the numbers on screen will
be wrong the moment they are printed.

Why a script and not a dashboard button: this deletes the record of what the
bench did. It should take a deliberate act on the machine, not a stray click
in a browser someone left open.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta

from config import STORAGE, STORAGE_POLICY_FILE, load_storage_policy
from storage.db import Database
from storage.images import SnapshotStore

# One full frame off this bench is about this big. Used only to turn a policy
# into a number of megabytes, so a rough figure is the honest kind.
TYPICAL_SNAPSHOT_BYTES = 850_000


def _mb(n_bytes: float) -> str:
    if n_bytes < 1e6:
        return f"{n_bytes / 1e3:,.0f} kB"
    return f"{n_bytes / 1e6:,.0f} MB"


def effective() -> dict[str, int]:
    """The policy as it stands on disk right now.

    Not read off config.STORAGE: that is a frozen dataclass built when the
    module was imported, so within this process it still holds whatever was
    in force at startup. Editing the file and then being shown the old
    numbers - or worse, applying them - is exactly the confusion to avoid.
    """
    policy = load_storage_policy()
    return {
        "snapshot_every_n_frames": int(policy.get(
            "snapshot_every_n_frames", STORAGE.snapshot_every_n_frames)),
        "max_snapshots": int(policy.get("max_snapshots",
                                        STORAGE.max_snapshots)),
        "retention_days": int(policy.get("retention_days",
                                         STORAGE.retention_days)),
    }


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default or ""
    return answer or (default or "")


def ask_int(prompt: str, default: int, minimum: int = 0) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print(f"  not a whole number: {raw!r}")
            continue
        if value < minimum:
            print(f"  must be {minimum} or more")
            continue
        return value


def ask_date(prompt: str) -> date | None:
    while True:
        raw = ask(f"{prompt} (YYYY-MM-DD, blank to cancel)")
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            print(f"  not a date: {raw!r}")


def confirm(what: str) -> bool:
    """Deleting is not undoable, so require the word, not a keystroke."""
    return ask(f"{what}\n  type DELETE to confirm") == "DELETE"


# --------------------------------------------------------------------------
def show(db: Database, snaps: SnapshotStore) -> None:
    policy = effective()
    source = ("data/storage_policy.json" if load_storage_policy()
              else "built-in defaults")
    print(f"\nPolicy (from {source})")
    print(f"  save one frame every       "
          f"{policy['snapshot_every_n_frames'] or 'never -'} analysis frames")
    print(f"  keep at most               "
          f"{policy['max_snapshots'] or 'unlimited'} snapshots")
    print(f"  keep history for           "
          f"{policy['retention_days'] or 'forever'} days")

    count, total = snaps.usage()
    print(f"\nOn disk")
    print(f"  snapshots                  {count:,} files, {_mb(total)}")
    if policy["max_snapshots"]:
        print(f"  cap                        {policy['max_snapshots']:,} files, "
              f"about {_mb(policy['max_snapshots'] * TYPICAL_SNAPSHOT_BYTES)}")
    db_bytes = STORAGE.db_path.stat().st_size if STORAGE.db_path.exists() else 0
    print(f"  database                   {_mb(db_bytes)}")
    print("\nRows")
    for table, n in db.counts_by_table().items():
        print(f"  {table:<18}         {n:,}")


def edit_policy() -> None:
    policy = load_storage_policy()
    current = effective()
    print("\nHow much to keep. Blank keeps the current value.")

    every_n = ask_int(
        "  save a full frame every N analysis frames (0 = never)",
        current["snapshot_every_n_frames"])
    max_snaps = ask_int(
        "  maximum snapshots to keep, oldest deleted first (0 = no limit)",
        current["max_snapshots"])
    days = ask_int(
        "  days of history to keep (0 = keep forever)",
        current["retention_days"])

    policy.update({"snapshot_every_n_frames": every_n,
                   "max_snapshots": max_snaps,
                   "retention_days": days})
    STORAGE_POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
    STORAGE_POLICY_FILE.write_text(json.dumps(policy, indent=2) + "\n")

    print(f"\n  written to {STORAGE_POLICY_FILE}")
    if max_snaps:
        print(f"  a cap of {max_snaps:,} snapshots is about "
              f"{_mb(max_snaps * TYPICAL_SNAPSHOT_BYTES)} at ~850 kB a frame")
    else:
        print("  no snapshot cap: only the retention window bounds the disk")
    print("  main.py reads this at startup - restart it to apply")


def clear_range(db: Database, snaps: SnapshotStore) -> None:
    start = ask_date("\n  from")
    if start is None:
        return
    end = ask_date("  to")
    if end is None:
        return
    if end < start:
        print("  end is before start")
        return

    # Snapshots are stored per day, so the range is inclusive of both ends;
    # the database is queried on the same span in epoch seconds.
    start_ts = datetime.combine(start, datetime.min.time()).timestamp()
    end_ts = datetime.combine(end + timedelta(days=1),
                              datetime.min.time()).timestamp()

    also_alarms = ask("  delete anomaly_events in that range too? (y/N)",
                      "n").lower().startswith("y")
    span = f"{start} to {end} inclusive"
    if not confirm(f"  this deletes history and snapshots for {span}."):
        print("  cancelled")
        return

    rows = db.prune_between(start_ts, end_ts, include_anomalies=also_alarms)
    files = snaps.delete_between(start, end)
    print(f"  deleted {sum(rows.values()):,} rows and {files:,} snapshots")
    for table, n in rows.items():
        print(f"    {table:<18} {n:,}")


def clear_everything(db: Database, snaps: SnapshotStore) -> None:
    count, total = snaps.usage()
    rows = sum(db.counts_by_table().values())
    if not confirm(f"\n  this deletes ALL {rows:,} rows and all {count:,} "
                   f"snapshots ({_mb(total)})."):
        print("  cancelled")
        return
    deleted = db.clear_all()
    files = snaps.clear_all()
    print(f"  deleted {sum(deleted.values()):,} rows and {files:,} snapshots")


def apply_policy_now(db: Database, snaps: SnapshotStore) -> None:
    """Run the policy against what is already on disk."""
    policy = effective()
    rows = db.prune(policy["retention_days"])
    by_age = snaps.prune(policy["retention_days"])
    over_cap = snaps.enforce_max(policy["max_snapshots"])
    if not rows and not by_age and not over_cap:
        print("  nothing to do: everything on disk is inside the policy")
        return
    print(f"  removed {sum(rows.values()):,} rows, {by_age:,} snapshots past "
          f"the retention window, {over_cap:,} over the cap")


MENU = """
1  show what is stored
2  edit the policy
3  apply the policy now
4  clear a date range
5  clear everything
q  quit
"""


def main() -> int:
    db = Database()
    snaps = SnapshotStore()
    print(f"Storage policy - {STORAGE.db_path.parent}")
    try:
        show(db, snaps)
        while True:
            print(MENU)
            choice = ask("choose").lower()
            if choice in ("q", "quit", "exit", ""):
                return 0
            if choice == "1":
                show(db, snaps)
            elif choice == "2":
                edit_policy()
            elif choice == "3":
                apply_policy_now(db, snaps)
            elif choice == "4":
                clear_range(db, snaps)
            elif choice == "5":
                clear_everything(db, snaps)
            else:
                print(f"  no such option: {choice!r}")
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
