"""Gross colour change.  ***YOURS TO IMPLEMENT.***

Plan on record: cache the previous close-up of that beaker, compare it to the
new one, and compare the colours between them. That cache exists and is wired
up - this detector does not have to build it:

    ctx.previous_crop(tid)
        -> that vial's previous crop, already resized to match the current
           one so the two can be differenced directly. None on the vial's
           first sighting. Crops are clipped independently at the frame edge
           and the apparent radius wobbles between frames, which is why the
           resize is not optional and why it is done for you.

    ctx.crops[tid], ctx.masks[tid]
        -> the current crop and the liquid-disc mask. Compare inside the mask:
           the rim carries glass, meniscus and whatever specular highlight
           cross-polarisation did not kill, and none of that is the liquid.

    ctx.history.last_crop(tid, back=N)
        -> further back than one frame, up to DETECTION.crop_history_length.

Three traps specific to this one, all mechanical rather than chemical:

  * Illumination, not chemistry. One 595 mm LED panel covers about 600 mm of
    the 780 mm platform; the far ~180 mm is a documented dim zone. A vial that
    changed colour and a vial that moved into the dark end look identical to a
    raw before/after diff. Either compare only within a stage where the vial
    is stationary, or flat-field correct upstream in features.py.

  * The detection philosophy is qualitative and gross-state, not fine
    colorimetry. Whether subtle colour shifts even matter for this chemistry
    is still an open question with the researchers - it is the same question
    that decides CRI 80 versus 90+ on the panel. Do not build something that
    depends on the answer being "yes" before the answer exists.

  * A batch-wide change is normal. Every vial darkening together during
    heating is the protocol working. If this detector fires per-vial on an
    absolute before/after delta it will flag all 18 at once every run - which
    is why the same batch-median idea used for turbidity is worth applying to
    the *deltas* here too, not only to the absolute values.
"""

from __future__ import annotations

from pipeline.detectors.base import NotImplementedDetector


class ColorChangeDetector(NotImplementedDetector):
    name = "color_change"
    description = "Gross colour shift against this vial's own previous image"
