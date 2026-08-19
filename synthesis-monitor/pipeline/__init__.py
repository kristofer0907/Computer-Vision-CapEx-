"""Analysis pipeline.

Wired and working: tracking, zone/stage assignment with hysteresis, ROI
extraction, per-vial history, the detector host, and the runner that sequences
them.

Deliberately empty, and yours: localize.py, features.py, anomaly.py and every
module under detectors/. The interfaces they must satisfy are fixed; what they
decide is not, and cannot be until there are real captured vial images to
calibrate against.
"""

from pipeline.types import (Detection, Event, PipelineResult, Track,
                            VialReport, SEVERITIES)

__all__ = ["Detection", "Event", "PipelineResult", "Track", "VialReport",
           "SEVERITIES"]
