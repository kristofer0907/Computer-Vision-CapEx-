"""Heater to cooling lineage, driven by occupancy alone.

No images, no coordinates - the whole mechanism is which slots read full on
which frame, so the tests say exactly that and nothing else.
"""

from __future__ import annotations

from pipeline.lineage import CrucibleLineage


def occ(*slots: int) -> dict[int, bool]:
    """Occupancy for a 2-slot heater / small pad: listed slots are full."""
    return {i: (i in slots) for i in range(4)}


def test_id_is_minted_when_a_heater_slot_fills():
    lin = CrucibleLineage()
    lin.update(occ(), occ(), 100.0)
    lin.update(occ(0), occ(), 130.0)

    assert lin.crucible_on_heater(0) == "heater-0-130000"
    assert lin.crucible_on_heater(1) is None


def test_a_slot_that_stays_full_is_not_re_minted():
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0)
    first = lin.crucible_on_heater(0)
    for t in (130.0, 160.0, 190.0):
        lin.update(occ(0), occ(), t)
    assert lin.crucible_on_heater(0) == first


def test_vacate_then_cooling_fill_links_them():
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0)
    crucible = lin.crucible_on_heater(0)

    lin.update(occ(), occ(), 130.0)            # lifted off the heater
    assert lin.pending_vacates == [crucible]

    lin.update(occ(), occ(2), 160.0)           # lands on the pad
    assert lin.pending_vacates == []
    assert lin.crucible_on_cooling(2) == crucible

    (rec,) = lin.records
    assert rec.crucible_id == crucible
    assert (rec.heater_slot, rec.cooling_slot) == (0, 2)
    assert rec.heater_entry_ts == 100.0
    assert rec.heater_vacate_ts == 130.0
    assert rec.cooling_fill_ts == 160.0


def test_the_heater_id_carries_forward_rather_than_being_renamed():
    """The pad shows the id the crucible was given at the heater, not a new one."""
    lin = CrucibleLineage()
    lin.update(occ(1), occ(), 100.0)
    crucible = lin.crucible_on_heater(1)
    lin.update(occ(), occ(), 130.0)
    lin.update(occ(), occ(0), 160.0)

    assert lin.crucible_on_cooling(0) == crucible
    assert crucible.startswith("heater-1-")


def test_leaving_and_landing_within_one_frame_still_links():
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0)
    crucible = lin.crucible_on_heater(0)
    lin.update(occ(), occ(3), 130.0)           # both edges, same frame
    assert lin.crucible_on_cooling(3) == crucible
    assert len(lin.records) == 1


def test_a_replaced_crucible_gets_a_new_id():
    """The bug this mechanism exists for.

    Slot-occupancy tracking kept one id on a heater slot across a change of
    jar, because the slot never read empty. An empty frame between the two
    is what distinguishes them, and it has to produce a different id.
    """
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0)
    first = lin.crucible_on_heater(0)

    lin.update(occ(), occ(), 130.0)            # slot empties
    lin.update(occ(0), occ(), 160.0)           # a different jar is set down
    second = lin.crucible_on_heater(0)

    assert second != first
    assert second == "heater-0-160000"


def test_most_recent_vacate_is_the_one_claimed():
    lin = CrucibleLineage()
    lin.update(occ(0, 1), occ(), 100.0)
    first, second = lin.crucible_on_heater(0), lin.crucible_on_heater(1)

    lin.update(occ(1), occ(), 130.0)           # slot 0 leaves
    lin.update(occ(), occ(), 160.0)            # slot 1 leaves
    assert lin.pending_vacates == [first, second]

    lin.update(occ(), occ(2), 190.0)
    assert lin.crucible_on_cooling(2) == second    # the later vacate

    lin.update(occ(), occ(2, 3), 220.0)
    assert lin.crucible_on_cooling(3) == first


def test_unexplained_cooling_fill_warns_and_does_not_link():
    lin = CrucibleLineage()
    lin.update(occ(), occ(), 100.0)
    lin.update(occ(), occ(1), 130.0)           # appears with no heater vacate

    assert lin.records == []
    assert lin.crucible_on_cooling(1) is None
    assert len(lin.warnings) == 1
    assert "no unmatched heater vacate" in lin.warnings[0]


def test_a_cooling_slot_emptying_is_not_an_error():
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0)
    lin.update(occ(), occ(), 130.0)
    lin.update(occ(), occ(2), 160.0)
    lin.update(occ(), occ(), 190.0)            # taken off the pad

    assert lin.crucible_on_cooling(2) is None
    assert lin.warnings == []
    assert len(lin.records) == 1               # the record stands


def test_as_dict_round_trips_the_state():
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0)
    lin.update(occ(), occ(1), 130.0)
    d = lin.as_dict()

    assert d["on_cooling"] == {1: "heater-0-100000"}
    assert d["records"][0]["crucible_id"] == "heater-0-100000"
    assert d["pending_vacates"] == []


def test_an_unlinked_fill_warns_once_not_every_frame():
    """Occupancy has to be remembered even when identity could not be.

    The first version only recorded a cooling slot once it was linked to a
    heater vacate, so an unlinked slot kept looking newly-filled and warned
    again on every later frame - on a real run, hundreds of times for one
    crucible.
    """
    lin = CrucibleLineage()
    for t in (100.0, 130.0, 160.0, 190.0):
        lin.update(occ(), occ(1), t)

    assert len(lin.warnings) == 1


def test_an_unlinked_slot_can_still_be_claimed_after_it_empties():
    lin = CrucibleLineage()
    lin.update(occ(), occ(1), 100.0)           # unexplained, warns
    lin.update(occ(), occ(), 130.0)            # emptied again
    lin.update(occ(0), occ(), 160.0)           # a crucible reaches a heater
    lin.update(occ(), occ(), 190.0)            # and leaves it
    lin.update(occ(), occ(1), 220.0)           # same pad slot, now explained

    assert len(lin.warnings) == 1
    assert lin.crucible_on_cooling(1) == "heater-0-160000"


# --------------------------------------------------------------------------
# Replacement spotted by movement, not by the slot emptying
# --------------------------------------------------------------------------
def test_a_static_jar_keeps_its_id_through_detector_jitter():
    """Real numbers from capture/tracking_practice slot 0, an untouched jar:
    it repeats exact pixels and never moves more than 6.4 px."""
    lin = CrucibleLineage()
    track = [(2017.5, 2503.5), (2017.5, 2503.5), (2017.5, 2502.5),
             (2016.5, 2500.5), (2021.5, 2504.5), (2019.5, 2506.5),
             (2020.5, 2502.5), (2022.5, 2503.5), (2022.5, 2503.5)]
    for i, pos in enumerate(track):
        lin.update(occ(0), occ(), 100.0 + 30 * i, heater_pos={0: pos})

    assert lin.crucible_on_heater(0) == "heater-0-100000"
    assert lin.records == []
    assert lin.pending_vacates == []


def test_a_jump_while_still_occupied_is_read_as_a_replacement():
    """Slot 1's real steps: 28.3, 28.8, 4.2, 14.0, 15.0 px, occupancy True
    throughout. Occupancy alone sees one crucible; the jumps say otherwise."""
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0, heater_pos={0: (1804.5, 2459.5)})
    first = lin.crucible_on_heater(0)

    lin.update(occ(0), occ(), 130.0, heater_pos={0: (1800.5, 2431.5)})  # 28.3
    second = lin.crucible_on_heater(0)

    assert second != first
    assert lin.pending_vacates == [first]      # the old one is claimable


def test_a_replacement_spotted_by_movement_still_reaches_the_cooling_pad():
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0, heater_pos={0: (1804.0, 2459.0)})
    first = lin.crucible_on_heater(0)
    lin.update(occ(0), occ(), 130.0, heater_pos={0: (1830.0, 2459.0)})  # 26 px
    lin.update(occ(0), occ(2), 160.0, heater_pos={0: (1830.0, 2459.0)})

    assert lin.crucible_on_cooling(2) == first
    assert len(lin.records) == 1


def test_movement_is_ignored_when_no_positions_are_given():
    """Occupancy-only callers keep the original behaviour."""
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.0)
    first = lin.crucible_on_heater(0)
    lin.update(occ(0), occ(), 130.0)
    assert lin.crucible_on_heater(0) == first


def test_two_crucibles_in_one_slot_within_a_second_get_distinct_ids():
    """Replaying captures faster than they were taken put two crucibles in one
    slot inside the same second. At second resolution their ids collided and
    one id ended up linked to two different cooling slots."""
    lin = CrucibleLineage()
    lin.update(occ(0), occ(), 100.10, heater_pos={0: (1800.0, 2400.0)})
    first = lin.crucible_on_heater(0)
    lin.update(occ(0), occ(), 100.35, heater_pos={0: (1840.0, 2400.0)})
    second = lin.crucible_on_heater(0)

    assert first != second
    lin.update(occ(0), occ(1), 100.60, heater_pos={0: (1840.0, 2400.0)})
    lin.update(occ(), occ(1, 2), 100.85, heater_pos={})
    assert len({r.crucible_id for r in lin.records}) == len(lin.records)
