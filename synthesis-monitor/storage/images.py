"""Snapshot store: JPEGs on disk, paths in the database.

Images are not put in SQLite. A full frame is ~150 kB and a run is thousands
of them; storing them as blobs makes the database slow to open, expensive to
back up, and awkward to look at with any normal tool. Files on disk can be
browsed, rsynced and deleted individually.

Layout is date-sharded:

    data/snapshots/2026-08-14/frame_000123.jpg
    data/snapshots/2026-08-14/event_000123_t07_turbidity.jpg

One directory per day keeps any single directory small enough that listing it
stays fast on an SD card, and makes "delete everything before the 3rd" a
directory removal rather than a scan.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import STORAGE

log = logging.getLogger(__name__)


class SnapshotStore:
    """Writes JPEGs under a root directory and returns their relative paths.

    Paths returned are relative to the root so the database stays portable:
    copying data/ somewhere else, or mounting it at a different path on
    another machine, must not invalidate every row.
    """

    def __init__(self, root: Path | None = None, quality: int | None = None) -> None:
        self.root = Path(root or STORAGE.snapshot_dir)
        self.quality = (STORAGE.snapshot_jpeg_quality
                        if quality is None else quality)
        self.root.mkdir(parents=True, exist_ok=True)

    def _day_dir(self, timestamp: float | None = None) -> Path:
        day = datetime.fromtimestamp(timestamp or time.time()).strftime("%Y-%m-%d")
        path = self.root / day
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, image: np.ndarray, name: str,
               timestamp: float | None = None) -> str | None:
        target = self._day_dir(timestamp) / name
        try:
            ok = cv2.imwrite(str(target), image,
                             [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        except Exception:
            log.exception("snapshot write failed: %s", target)
            return None
        if not ok:
            log.warning("snapshot encode failed: %s", target)
            return None
        return str(target.relative_to(self.root))

    def write_bytes(self, jpeg: bytes, name: str,
                    timestamp: float | None = None) -> str | None:
        """Store an already-encoded JPEG. Avoids a decode/re-encode round trip."""
        target = self._day_dir(timestamp) / name
        try:
            target.write_bytes(jpeg)
        except OSError:
            log.exception("snapshot write failed: %s", target)
            return None
        return str(target.relative_to(self.root))

    # ------------------------------------------------------------------ API
    def save_frame(self, image: np.ndarray, frame_id: int,
                   timestamp: float | None = None) -> str | None:
        return self._write(image, f"frame_{frame_id:06d}.jpg", timestamp)

    def save_frame_jpeg(self, jpeg: bytes, frame_id: int,
                        timestamp: float | None = None) -> str | None:
        return self.write_bytes(jpeg, f"frame_{frame_id:06d}.jpg", timestamp)

    def save_crop(self, crop: np.ndarray, frame_id: int, track_id: int,
                  kind: str, timestamp: float | None = None) -> str | None:
        """The close-up that justified an event.

        Worth having for exactly one reason: when a researcher asks two weeks
        later why vial 7 was flagged, "here is the picture it was flagged on"
        is an answer and a robust z-score is not.
        """
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in kind)[:32]
        return self._write(crop, f"event_{frame_id:06d}_t{track_id:02d}_{safe}.jpg",
                           timestamp)

    def resolve(self, relative: str) -> Path:
        """Relative path from the database back to an absolute one."""
        return self.root / relative

    def prune(self, retention_days: int | None = None) -> int:
        """Remove whole day directories older than the retention window."""
        days = STORAGE.retention_days if retention_days is None else retention_days
        if days <= 0:
            return 0
        cutoff = datetime.fromtimestamp(time.time() - days * 86400).date()
        removed = 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                day = datetime.strptime(child.name, "%Y-%m-%d").date()
            except ValueError:
                continue    # not one of ours, leave it alone
            if day < cutoff:
                for f in child.iterdir():
                    f.unlink(missing_ok=True)
                    removed += 1
                child.rmdir()
        if removed:
            log.info("pruned %d snapshot files older than %d days", removed, days)
        return removed
