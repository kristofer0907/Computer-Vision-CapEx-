"""A crucible put on a heater without a lid. The one implemented detector.

This module is only the adapter. The scoring is pipeline.features.lid_score,
the latch is pipeline.anomaly.MissingLid, and both are calibrated against the
225 hand-labelled crucibles in data/lid_review.json - unlike every other
detector here, which is a stub.

What the adapter does: pull the crucibles currently seated on heater slots out
of the detection context, hand them to MissingLid, and translate its Events
into AnomalyResults. MissingLid keeps the memory of which slots it has already
asked about, so it is checked once per arrival rather than every frame.

`stop_requested` is exposed so the orchestrator loop can halt. That stops this
pipeline and nothing else - there is no control channel to the platform.
"""

from __future__ import annotations

from config import REGION_TRACKING
from pipeline.anomaly import MissingLid
from pipeline.detectors.base import DetectionContext, Detector
from pipeline.types import AnomalyResult


class MissingLidDetector(Detector):
    name = "missing_lid"
    description = "Crucible placed on a heater without a lid"

    def __init__(self, heater_zone: str | None = None) -> None:
        self.heater_zone = heater_zone or REGION_TRACKING.heater_zone
        self.latch = MissingLid(heater_zone=self.heater_zone)

    @property
    def stop_requested(self) -> bool:
        return self.latch.stop_requested

    def _on_heater(self, ctx: DetectionContext) -> dict[int, tuple[float, float]]:
        """Heater slot -> crucible centre, for the crucibles seated this frame.

        Keyed by slot rather than track id because the question is about the
        heater position, and a slot that has already been asked must not be
        re-asked just because the track behind it was renumbered.
        """
        view = ctx.zones.get(self.heater_zone)
        ids = set(view.track_ids) if view is not None else set()
        return {t.slot_id: t.center for t in ctx.tracks
                if t.slot_id is not None and t.track_id in ids}

    def check(self, ctx: DetectionContext) -> list[AnomalyResult]:
        events = self.latch.check(ctx.frame.image, self._on_heater(ctx),
                                  ctx.timestamp, ctx.frame_id)
        return [self.anomaly(ctx, zone=self.heater_zone) for _ in events]
