"""Turbidity.  ***YOURS TO IMPLEMENT.***

Plan on record: batch-median comparison. The 18 vials run the same protocol in
parallel, so the batch is its own ground truth and a vial that diverges
statistically from its peers is the anomaly - no labelled data needed.

What the context already gives you:

    ctx.feature_column("<key>", stage="heating")
        -> (track_ids, values) for one feature across vials at the same stage.
           Restricting by stage matters: a vial still in filling is not a
           peer of one that has been on the heater for four minutes, and
           pooling them inflates the spread the median is measured against.

    ctx.crops[tid] / ctx.masks[tid]
        -> the pixels, if you want to compute something here rather than
           adding it to features.py. Prefer features.py: anything computed
           there is stored, plotted and available to the other detectors,
           whereas anything computed here is thrown away after the frame.

    pipeline.stats.robust_z / mad
        -> median and MAD-based scoring, if you want it. Plain functions, no
           opinions, delete them if you would rather write your own.

Two things worth deciding before writing the comparison:

  * Minimum peer count. DETECTION.min_vials_for_batch_stats exists for this.
    A median over three vials is not a batch consensus, and early in a run
    (or late, as vials leave for the oven) that is exactly what you have.
  * Whether a divergence has to persist. ctx.history.series(tid, key) gives
    the vial's own trajectory - one frame of divergence on an uncalibrated
    threshold is a coin flip, several consecutive frames is a signal.

Nothing here is calibratable without real baseline runs on real chemistry.
"""

from __future__ import annotations

from pipeline.detectors.base import NotImplementedDetector


class TurbidityDetector(NotImplementedDetector):
    name = "turbidity"
    description = "Batch-median divergence in cloudiness / scattering"
