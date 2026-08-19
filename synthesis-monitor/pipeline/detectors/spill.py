"""Spills and overflows.  ***YOURS TO IMPLEMENT.***

Plan on record: see whether wetted aluminium is detectable at all; fall back
to filter paper that has to be replaced after every spill if it is not.

This is the only detector whose subject is the bench rather than a vial, so
the context gives it a different view:

    ctx.zones["filling"].image        BGR crop of the zone's bounding box
    ctx.zones["filling"].bench_mask   the zone polygon with a disc punched out
                                      around every vial in it - bare surface
                                      only, so a vial's own colour cannot read
                                      as a wet patch
    ctx.zones["filling"].zone_mask    the polygon without the vials removed
    ctx.zones[...].track_ids          which vials are in that zone right now

A reference frame of the clean bench is the missing piece and this module is
where it belongs. Nothing upstream keeps one, because what counts as "clean"
is a detection decision: the first frame of a run, a rolling median over the
last N frames, or a stored calibration image are all defensible and they
behave differently when someone leaves a glove on the bench.

On the aluminium-versus-filter-paper question, what the optics already imply:

  * Cross-polarisation is fitted to kill specular glare off the glass vials.
    Bare aluminium is a specular surface and a thin wet film on it shows up
    largely *as* a change in specularity - so the polarisers that make the
    vials readable are actively working against the easiest spill signal.
    Rotating them is not an option; the vials are the primary subject.

  * That pushes towards the residual cues: a wet film changes the diffuse
    colour and the local texture slightly, and a pooled drop has an edge.
    At roughly 0.8 mm/px a droplet a few millimetres across is a handful of
    pixels, so this will be a marginal signal on aluminium and the filter
    paper is a reasonable fallback rather than a defeat. Paper wicks, so the
    wetted patch is far larger than the drop and reads as a strong diffuse
    darkening with no specular component at all.

  * Test this empirically before building on either. It is a twenty-minute
    experiment with a pipette once the panel and polarisers are mounted, and
    the answer decides which of the two this module is written against.

Solvent evaporation cuts both ways and is worth knowing before tuning any
persistence rule: on aluminium a spill can vanish between two analysis frames
45 s apart, so a rule requiring N consecutive frames may never fire. On paper
the stain persists, which is the same property that makes replacement tedious.
"""

from __future__ import annotations

from pipeline.detectors.base import NotImplementedDetector


class SpillDetector(NotImplementedDetector):
    name = "spill"
    description = "Liquid on the bench surface between the vials"
