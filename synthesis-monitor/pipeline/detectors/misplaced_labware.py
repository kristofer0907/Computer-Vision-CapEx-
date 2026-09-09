"""Labware in the wrong place.  ***YOURS TO IMPLEMENT.***

Split out of missing_crucible.py: "the slot is empty" and "there is something
where nothing belongs" are different questions with different evidence, even
though both come from comparing detections against the marked slot geometry.

What this one has to work with:

    ctx.zones[name].track_ids       what the tracker put in each zone
    ctx.tracks[i].slot_id           the marked slot a track occupies, or None
                                    when it matched no slot at all
    zone_map.distance_mm(a, b)      ground millimetres between two image
                                    points, valid because the camera looks
                                    straight down at a flat platform

The signal is `slot_id is None`. SlotTracker gates every detection against
the hand-marked slot positions in data/slots_*.json, so a crucible sitting
between two holes, or on the bench beside the rack, is already visible as a
track the tracker could not seat. That is the cheap version and it should be
written first.

Rules worth encoding rather than learning: the heater pad takes at most two
crucibles, and the process order is fixed. A third crucible on the heater is
misplaced by definition, not by statistics.

Not solved: the arm itself passing through frame looks like an object in the
wrong place. A hysteresis of N frames is the obvious defence, but N cannot be
chosen without knowing the arm's cycle time against the analysis cadence, and
that timing is still open with the researchers.
"""

from __future__ import annotations

from pipeline.detectors.base import NotImplementedDetector


class MisplacedLabwareDetector(NotImplementedDetector):
    name = "misplaced_labware"
    description = "A crucible seated in no marked slot, or where none belongs"
