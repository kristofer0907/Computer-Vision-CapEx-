"""Missing crucibles.  ***YOURS TO IMPLEMENT.***

Plan on record: batch-median again, to derive the general cadence and spacing
between beakers and compare against it. Wrong-position crucibles are a
separate module, misplaced_labware.py.

The geometry this needs is already computed and available:

    ctx.tracks[i].cx / .cy          centroids in pixels
    zones.distance_mm(a, b)         ground distance in millimetres between
                                    two image points, valid because the camera
                                    looks straight down at a flat platform
    ctx.zones[name].track_ids       which crucibles are in each zone
    ctx.in_stage("heating")         confirmed crucibles at one stage
    track.stage_log                 every committed transition with timestamps,
                                    which is where per-crucible cadence comes from
    ctx.interval_s                  seconds since the previous analysis frame

Physical constraints worth encoding rather than learning: the filling step
loads all 18 crucibles at once, the heater pad takes at most 2 at a time, and the
process order is fixed. A crucible in heating while 17 sit in filling is normal; a
third crucible on the heater is not, and that is a rule, not a statistic.

The hard case, and it is worth being honest that it is not solved:

    A crucible that disappears is either an ordinary oven entry or a real failure.
    The tracker already distinguishes them as far as it can - a track last
    confirmed in cooling that vanishes is closed with closed_reason "oven",
    anything else closes as "lost", and both arrive in ctx.closed_tracks. But
    the oven is outside the camera's view, so "entered the oven" is an
    inference, never an observation. A crucible knocked over during cooling
    produces exactly the same pixels as a crucible correctly removed to the oven.

    Nothing in this module can fix that; it needs a signal from outside this
    camera - an oven door sensor, or a scheduler event from the platform
    controller saying a crucible was taken. Until one exists, "lost" is
    actionable and "oven" is not, and treating "oven" as clean is a known
    accepted blind spot rather than a solved problem.

Expected per-stage timings are still open with the researchers, and they are
what would turn "this crucible has been in lidding a long time" from a batch
comparison into an absolute one.
"""

from __future__ import annotations

from pipeline.detectors.base import NotImplementedDetector


class MissingCrucibleDetector(NotImplementedDetector):
    name = "missing_crucible"
    description = "A crucible absent from a slot the process says is filled"
