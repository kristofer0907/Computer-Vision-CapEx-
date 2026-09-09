"""Tip-over detection against the fail-safe frames.

capture/fail_safe is seven frames of a plate holding three upright crucibles
and three lying on their sides, labelled in tools/review_fallen.py. These
assert the feature still separates them, so a change to rim_circularity(),
its radial band, its edge threshold or its centre search cannot quietly
regress on the only pictures of a tip-over this project has.

Six objects, one scene, one lighting setup. Passing here means the measure
has not regressed, not that it is calibrated.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from config import DETECTION
from pipeline.anomaly import Cycle, check_fallen_crucible
from pipeline.features import rim_circularity, rim_gradient
from tools.review_fallen import (FAIL_SAFE_SLOTS, IMAGE_DIR, frame_paths,
                                 label_for)


def _scored():
    """(label, score) for all 15 plate positions in all 7 frames."""
    if not IMAGE_DIR.is_dir():
        pytest.skip("capture/fail_safe not present in this checkout")
    out = []
    for index, path in enumerate(frame_paths(IMAGE_DIR)):
        image = cv2.imread(str(path))
        if image is None:
            continue
        mag = rim_gradient(image)
        for slot, (cx, cy) in FAIL_SAFE_SLOTS.items():
            out.append((label_for(slot, index),
                        rim_circularity(image, cx, cy, _mag=mag)))
    if not out:
        pytest.skip("no fail-safe frames resolved")
    return out


def test_the_two_classes_do_not_overlap():
    """The whole claim, stated once: no tipped crucible scores as high as
    the worst-scoring standing position."""
    data = _scored()
    tipped = [s for label, s in data if label == "fallen"]
    standing = [s for label, s in data if label != "fallen"]
    assert max(tipped) < min(standing), (
        f"classes overlap: tipped up to {max(tipped):.3f}, "
        f"standing down to {min(standing):.3f}")


def test_the_threshold_sits_in_the_gap():
    data = _scored()
    tipped = [s for label, s in data if label == "fallen"]
    standing = [s for label, s in data if label != "fallen"]
    limit = DETECTION.fallen_circularity_threshold
    assert max(tipped) < limit < min(standing), (
        f"threshold {limit} is outside the gap "
        f"[{max(tipped):.3f}, {min(standing):.3f}]")


def test_every_tipped_crucible_is_caught_in_every_frame():
    """Missing a tip-over is the costlier error - spilled precursor recorded
    as a normal run. Across 7 frames that is 21 chances to miss one."""
    limit = DETECTION.fallen_circularity_threshold
    missed = [s for label, s in _scored() if label == "fallen" and s >= limit]
    assert not missed, f"{len(missed)} tipped crucibles scored as standing"


def test_an_empty_slot_reads_as_standing_not_tipped():
    """The measure says 'upright or nothing', never 'occupied'. An empty
    slot's own rim is a circle, so it must land on the standing side -
    otherwise every gap in the plate is a false alarm."""
    limit = DETECTION.fallen_circularity_threshold
    empties = [s for label, s in _scored() if label == "empty"]
    assert min(empties) >= limit, (
        f"an empty slot scored {min(empties):.3f}, below {limit}")


def test_the_centre_search_is_what_makes_it_work():
    """Why rim_circularity searches a window instead of sampling one point.

    The rim sits ~40 mm above the plate, so away from the optical axis it
    projects outward from the slot it stands in. Anchored exactly on the slot
    centre the two classes are indistinguishable; the search is not a
    tolerance, it is the measurement.
    """
    data = []
    for index, path in enumerate(frame_paths(IMAGE_DIR)):
        image = cv2.imread(str(path))
        mag = rim_gradient(image)
        for slot, (cx, cy) in FAIL_SAFE_SLOTS.items():
            label = label_for(slot, index)
            if label == "empty":
                continue
            data.append((label, rim_circularity(image, cx, cy, search_px=0,
                                                _mag=mag)))
    tipped = [s for label, s in data if label == "fallen"]
    upright = [s for label, s in data if label == "upright"]
    assert max(tipped) >= min(upright), (
        "without the centre search the classes now separate - if that is "
        "real, RIM_SEARCH_PX and this test are both out of date")


def test_check_reports_the_offenders_and_their_scores():
    path = frame_paths(IMAGE_DIR)[0]
    image = cv2.imread(str(path))
    positions = {slot: pos for slot, pos in FAIL_SAFE_SLOTS.items()}
    result = check_fallen_crucible(Cycle(image=image, positions=positions))
    assert result.failed
    assert result.offenders == [2, 9, 12]
    assert set(result.subjects) == set(positions)
    # subjects are rounded for logging, score is not
    assert round(result.score, 3) == min(result.subjects.values())


def test_check_is_quiet_when_there_is_nothing_to_look_at():
    image = np.zeros((400, 400, 3), np.uint8)
    result = check_fallen_crucible(Cycle(image=image, positions={}))
    assert not result.failed
    assert result.implemented
    assert result.score is None


def test_check_honours_an_explicit_threshold():
    path = frame_paths(IMAGE_DIR)[0]
    image = cv2.imread(str(path))
    positions = {11: FAIL_SAFE_SLOTS[11]}
    assert not check_fallen_crucible(Cycle(image=image, positions=positions))
    assert check_fallen_crucible(Cycle(image=image, positions=positions),
                                 threshold=1.01)
