"""Shared anomaly scoring.  ***YOURS, IF YOU WANT IT.***

Left empty on purpose. All five detectors are planned around batch-median
comparison, so there is an obvious temptation to write one scorer here and
have every detector call it. That may well be right - but which features go
in, whether peers are restricted by stage, how many frames a divergence must
persist and where the threshold sits are all decisions that differ per failure
mode, and factoring them together before any of them has been calibrated
against real chemistry would lock in a shape that has not been tested.

If a common scorer does emerge, this is where it goes: import it from the
detectors rather than growing a second copy in each.

The raw material is already available:

    pipeline.stats.median / mad / robust_z / iqr
    ctx.feature_column(key, stage=...)   the batch's values for one feature
    ctx.history.series(tid, key)         one vial's trajectory over time
    DETECTION.robust_z_threshold         placeholder threshold, uncalibrated
    DETECTION.min_vials_for_batch_stats  minimum peers before the median means
                                         anything

The standing constraint, worth repeating here because this is the file where
it is easiest to forget: no threshold in this system has been calibrated. The
pipeline has been exercised against synthetic frames only, never against real
chemistry, and a threshold crossing today demonstrates that the plumbing works
and nothing whatsoever about the batch.
"""

from __future__ import annotations
