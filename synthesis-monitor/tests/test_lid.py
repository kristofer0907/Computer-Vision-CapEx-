"""Lid detection against the hand-labelled set.

data/lid_review.json is 225 crucibles labelled by hand (tools/review_lids.py).
These assert the feature still separates them, so a change to lid_score() or
its threshold cannot quietly regress on the only ground truth there is.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from config import DETECTION
from pipeline.features import lid_score

LABELS = "data/lid_review.json"
IMAGES = "capture/second_iteration/crucibles/undistored"


def _labelled():
    if not (os.path.exists(LABELS) and os.path.isdir(IMAGES)):
        pytest.skip("labelled lid set not present in this checkout")
    out, cache = [], {}
    for fname, entries in json.load(open(LABELS)).items():
        if fname.startswith("_"):
            continue
        path = os.path.join(IMAGES, fname)
        if not os.path.exists(path):
            continue
        if path not in cache:
            cache[path] = cv2.imread(path)
        for c in entries:
            out.append((cache[path], c["cx"], c["cy"], c["r"], c["label"] == "lid"))
    if not out:
        pytest.skip("no labelled crucibles resolved to images")
    return out


def test_every_labelled_lid_is_caught():
    """Missing a lid is the costlier error - a jar recorded as open when it
    was sealed. The threshold is set so this stays at zero."""
    data = _labelled()
    missed = [1 for img, cx, cy, r, is_lid in data
              if is_lid and lid_score(img, cx, cy, r) < DETECTION.lid_score_threshold]
    assert not missed, f"{len(missed)} lidded crucibles scored as open"


def test_accuracy_over_the_labelled_set_holds():
    data = _labelled()
    correct = sum((lid_score(img, cx, cy, r) >= DETECTION.lid_score_threshold) == is_lid
                  for img, cx, cy, r, is_lid in data)
    acc = correct / len(data)
    assert acc >= 0.98, f"lid accuracy fell to {acc:.1%} over {len(data)} labels"


def test_the_score_does_not_track_the_detected_radius():
    """Why the window is fixed rather than a fraction of r.

    Hough's radius wanders for jars of one physical size. When the sampled
    window scaled with it, a jar that measured large averaged in more of its
    own smooth rim and scored lower - which is what read one lidded crucible
    as open while its identical neighbours passed.
    """
    data = _labelled()
    lids = [(r, lid_score(img, cx, cy, r))
            for img, cx, cy, r, is_lid in data if is_lid]
    radii = np.array([r for r, _ in lids])
    scores = np.array([s for _, s in lids])
    corr = abs(float(np.corrcoef(radii, scores)[0, 1]))
    assert corr < 0.35, f"lid score still correlates with radius (r={corr:.2f})"


def test_ignoring_the_radius_argument_changes_nothing():
    data = _labelled()
    img, cx, cy, r, _ = data[0]
    assert lid_score(img, cx, cy, r) == lid_score(img, cx, cy, r * 1.5)
