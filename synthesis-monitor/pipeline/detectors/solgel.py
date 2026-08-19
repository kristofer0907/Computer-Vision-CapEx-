"""Sol-to-gel transition.  ***YOURS TO IMPLEMENT.***

Plan on record: same approach as turbidity - batch-median comparison.

The one structural difference from turbidity is worth building around: a
sol-gel transition is expected. Every vial in the batch is supposed to gel.
The anomaly is not "this vial gelled", it is "this vial gelled early, late, or
not at all relative to its peers", which makes this a comparison of *timing*
rather than of instantaneous state.

That means the useful comparison is probably over trajectories, not values:

    ctx.history.series(tid, "<key>")   the vial's own trajectory, oldest first
    ctx.feature_column("<key>", stage) the batch's current values
    track.time_in_stage_s()            how long this vial has been where it is
    track.stage_log                    every committed stage transition, with
                                       timestamps, so elapsed time since a
                                       specific stage began is available

Note that the sample interval is not constant - the cadence switches between
CADENCE.analysis_interval_s and analysis_interval_busy_s when the conveyor
moves. ctx.interval_s carries the interval for the current frame. A rate
computed as "change per frame" will be wrong by 4.5x across that switch;
divide by the interval if you want anything per-second.

Expected per-stage timings are still an open question with the researchers,
and they are the thing that would let a timing anomaly be scored against an
absolute expectation rather than only against the batch.
"""

from __future__ import annotations

from pipeline.detectors.base import NotImplementedDetector


class SolGelDetector(NotImplementedDetector):
    name = "solgel"
    description = "Liquid-to-solid transition timing versus the batch"
