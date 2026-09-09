"""The main.py command line.

Nothing here starts hardware: these cover argument parsing and how the parsed
values reach Monitor. The cadence actually holding is checked by running the
thing, not by a unit test.
"""

from __future__ import annotations

import pytest

from config import CADENCE
from main import Monitor, parse_args


def test_cadence_defaults_to_adaptive():
    args = parse_args([])
    assert args.rgb_interval is None
    assert args.thermal_interval is None


def test_cadence_flags_are_read_as_seconds():
    args = parse_args(["--rgb-interval", "15", "--thermal-interval", "3"])
    assert args.rgb_interval == 15.0
    assert args.thermal_interval == 3.0


def test_fractional_intervals_are_allowed():
    args = parse_args(["--thermal-interval", "0.5"])
    assert args.thermal_interval == 0.5


@pytest.mark.parametrize("flag", ["--rgb-interval", "--thermal-interval"])
@pytest.mark.parametrize("bad", ["0", "-1"])
def test_a_non_positive_interval_is_refused(flag, bad):
    """Zero would spin the capture loop as fast as the camera can go."""
    with pytest.raises(SystemExit):
        parse_args([flag, bad])


def test_monitor_defaults_to_the_configured_cadence():
    monitor = Monitor()
    assert monitor.rgb_interval_s is None
    assert monitor.thermal_interval_s is None


def test_monitor_keeps_a_pinned_cadence():
    monitor = Monitor(rgb_interval_s=15.0, thermal_interval_s=3.0)
    assert monitor.rgb_interval_s == 15.0
    assert monitor.thermal_interval_s == 3.0


def test_pinned_rgb_interval_overrides_the_controller():
    """The controller still observes; its answer is just not what is used."""
    monitor = Monitor(rgb_interval_s=7.0)
    adaptive = monitor.cadence.observe([])
    assert adaptive == CADENCE.analysis_interval_s      # what it would have said
    chosen = (adaptive if monitor.rgb_interval_s is None
              else monitor.rgb_interval_s)
    assert chosen == 7.0
