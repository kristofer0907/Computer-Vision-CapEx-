"""Persistence: SQLite for numbers, files on disk for images."""

from storage.db import Database, connect
from storage.images import SnapshotStore

__all__ = ["Database", "SnapshotStore", "connect"]
