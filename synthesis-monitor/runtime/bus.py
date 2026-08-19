"""Queue helpers and the latest-value slot the dashboard reads from.

One policy runs through all of this: never buffer, always drop the oldest.

A monitoring system that queues is worse than one that skips. If the pipeline
falls behind, a deep queue means the dashboard shows a four-minute-old frame
while presenting it as live, and the operator trusts it. Dropping instead
means the dashboard shows the newest frame available and the dropped counter
says how much was skipped - visibly wrong beats invisibly stale.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


def put_drop_oldest(q, item: Any) -> bool:
    """Enqueue, evicting the oldest item if the queue is full.

    Returns False if something had to be dropped to make room.

    The get/put pair is not atomic - another consumer can take the slot in
    between - so the second put is also guarded. On a full queue with a fast
    consumer that just means this item is the one dropped, which is the
    correct outcome anyway.
    """
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        pass

    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        return False
    return False


def drain_latest(q, limit: int = 32) -> Any | None:
    """Take everything waiting and return only the newest, or None.

    `limit` bounds the work per call so a caller draining in a UI loop cannot
    be pinned by a producer that is faster than it is.
    """
    newest = None
    for _ in range(limit):
        try:
            newest = q.get_nowait()
        except queue.Empty:
            break
    return newest


def drain_all(q, limit: int = 256) -> list:
    """Take everything waiting, oldest first.

    Used for results and events, where every item matters - unlike preview
    frames, where only the newest does.
    """
    out = []
    for _ in range(limit):
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


class LatestSlot:
    """A thread-safe one-value box. Last write wins.

    The dashboard's view of each source. Deliberately not a queue: a viewer
    that fell behind wants the newest frame, not a backlog of stale ones.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Any = None
        self._ts: float = 0.0
        self._writes: int = 0

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._ts = time.time()
            self._writes += 1

    def get(self) -> tuple[Any, float]:
        """(value, wall-clock time it was set). (None, 0.0) if never set."""
        with self._lock:
            return self._value, self._ts

    @property
    def age_s(self) -> float | None:
        with self._lock:
            return None if not self._ts else time.time() - self._ts

    @property
    def writes(self) -> int:
        with self._lock:
            return self._writes


class RingBuffer:
    """Bounded, thread-safe history for the dashboard's event list.

    Kept in memory as well as in SQLite so that rendering the dashboard does
    not hit the database on every poll, and so a database write failing does
    not also blank the live view.
    """

    def __init__(self, capacity: int = 200) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        self._items: list = []

    def extend(self, items) -> None:
        with self._lock:
            self._items.extend(items)
            if len(self._items) > self.capacity:
                del self._items[: len(self._items) - self.capacity]

    def latest(self, n: int | None = None) -> list:
        """Newest first."""
        with self._lock:
            items = list(reversed(self._items))
        return items[:n] if n else items

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
