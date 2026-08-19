"""Per-vial memory across analysis frames.

Two things are kept, both bounded:

  * a rolling window of past feature dicts, for temporal comparisons,
  * a rolling window of past crops, so a detector can diff a vial against a
    picture of *itself* from earlier rather than against its neighbours.

The crop cache is here because the colour-change plan needs exactly it: cache
the previous close-up of that beaker, compare the new one against it. Holding
the images in the processing process and only writing them to disk when
something fires keeps a normal 30-day run from filling the SD card.

Memory cost is bounded and worth stating: 18 vials x 4 crops x roughly
(2.6 * 34 px)^2 * 3 bytes is about half a megabyte. Raising
DETECTION.crop_history_length scales that linearly.

No decisions are made here. This module remembers; it does not compare.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from config import DETECTION
from pipeline.roi import resize_like


class VialHistory:
    """Bounded per-track history of features and crops."""

    def __init__(self, feature_len: int | None = None,
                 crop_len: int | None = None) -> None:
        self.feature_len = (DETECTION.history_length
                            if feature_len is None else feature_len)
        self.crop_len = (DETECTION.crop_history_length
                         if crop_len is None else crop_len)
        # track_id -> deque of (timestamp, features)
        self._features: dict[int, deque[tuple[float, dict[str, float]]]] = {}
        # track_id -> deque of (timestamp, BGR crop)
        self._crops: dict[int, deque[tuple[float, np.ndarray]]] = {}

    # ------------------------------------------------------------- writing
    def record(self, track_id: int, timestamp: float,
               features: dict[str, float] | None = None,
               crop: np.ndarray | None = None) -> None:
        if features is not None:
            self._features.setdefault(
                track_id, deque(maxlen=self.feature_len)
            ).append((timestamp, dict(features)))
        if crop is not None:
            # Copy: crop() hands back a view into the frame, and the frame is
            # freed as soon as this analysis pass ends.
            self._crops.setdefault(
                track_id, deque(maxlen=self.crop_len)
            ).append((timestamp, np.ascontiguousarray(crop)))

    def forget(self, track_id: int) -> None:
        """Drop a finished vial. Called when a track closes."""
        self._features.pop(track_id, None)
        self._crops.pop(track_id, None)

    def clear(self) -> None:
        self._features.clear()
        self._crops.clear()

    # ------------------------------------------------------------- reading
    def features(self, track_id: int) -> list[tuple[float, dict[str, float]]]:
        """Every retained feature dict for a vial, oldest first."""
        return list(self._features.get(track_id, ()))

    def last_features(self, track_id: int, back: int = 1
                      ) -> tuple[float, dict[str, float]] | None:
        """The Nth-most-recent feature dict, 1 = the previous frame."""
        window = self._features.get(track_id)
        if not window or len(window) < back:
            return None
        return window[-back]

    def series(self, track_id: int, key: str) -> list[float]:
        """One feature's values over time, oldest first, missing frames skipped."""
        return [f[key] for _, f in self._features.get(track_id, ())
                if key in f]

    def last_crop(self, track_id: int, back: int = 1
                  ) -> tuple[float, np.ndarray] | None:
        """The Nth-most-recent crop of a vial, 1 = the previous frame."""
        window = self._crops.get(track_id)
        if not window or len(window) < back:
            return None
        return window[-back]

    def previous_crop_matched(self, track_id: int, current: np.ndarray
                              ) -> np.ndarray | None:
        """The previous crop, resized to `current` so the two can be diffed.

        Crops are clipped independently at the frame edge and a vial's
        apparent radius wobbles slightly between frames, so two crops of the
        same vial are frequently a pixel or two different in size. Resizing
        here means no caller has to remember that.
        """
        prev = self.last_crop(track_id)
        if prev is None:
            return None
        return resize_like(prev[1], current)

    def known_ids(self) -> set[int]:
        return set(self._features) | set(self._crops)

    def prune(self, live_ids: set[int]) -> None:
        """Forget everything not in `live_ids`. Cheap insurance against a leak."""
        for tid in self.known_ids() - live_ids:
            self.forget(tid)
