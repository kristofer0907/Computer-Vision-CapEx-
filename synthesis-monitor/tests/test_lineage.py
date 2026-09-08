"""Heater to cooling lineage, driven by occupancy alone.

No images, no coordinates - the whole mechanism is which slots read full on
which frame, so the tests say exactly that and nothing else.
"""

from __future__ import annotations

from pipeline.lineage import VialLineage


def occ(*slots: int) -> dict[int, bool]:
    """Occupancy for a 2-slot heater / small pad: listed slots are full."""
    return {i: (i in slots) for i in range(4)}


def test_id_is_minted_when_a_heater_slot_fills():
    lin = VialLineage()
    lin.update(occ(), occ(), 100.0)
    lin.update(occ(0), occ(), 130.0)

    assert lin.vial_on_heater(0) == "heater-0-130"
    assert lin.vial_on_heater(1) is None


def test_a_slot_that_stays_full_is_not_re_minted():
    lin = VialLineage()
    lin.update(occ(0), occ(), 100.0)
    first = lin.vial_on_heater(0)
    for t in (130.0, 160.0, 190.0):
        lin.update(occ(0), occ(), t)
    assert lin.vial_on_heater(0) == first


def test_vacate_then_cooling_fill_links_them():
    lin = VialLineage()
    lin.update(occ(0), occ(), 100.0)
    vial = lin.vial_on_heater(0)

    lin.update(occ(), occ(), 130.0)            # lifted off the heater
    assert lin.pending_vacates == [vial]

    lin.update(occ(), occ(2), 160.0)           # lands on the pad
    assert lin.pending_vacates == []
    assert lin.vial_on_cooling(2) == vial

    (rec,) = lin.records
    assert rec.vial_id == vial
    assert (rec.heater_slot, rec.cooling_slot) == (0, 2)
    assert rec.heater_entry_ts == 100.0
    assert rec.heater_vacate_ts == 130.0
    assert rec.cooling_fill_ts == 160.0


def test_the_heater_id_carries_forward_rather_than_being_renamed():
    """The pad shows the id the vial was given at the heater, not a new one."""
    lin = VialLineage()
    lin.update(occ(1), occ(), 100.0)
    vial = lin.vial_on_heater(1)
    lin.update(occ(), occ(), 130.0)
    lin.update(occ(), occ(0), 160.0)

    assert lin.vial_on_cooling(0) == vial
    assert vial.startswith("heater-1-")


def test_leaving_and_landing_within_one_frame_still_links():
    lin = VialLineage()
    lin.update(occ(0), occ(), 100.0)
    vial = lin.vial_on_heater(0)
    lin.update(occ(), occ(3), 130.0)           # both edges, same frame
    assert lin.vial_on_cooling(3) == vial
    assert len(lin.records) == 1


def test_a_replaced_vial_gets_a_new_id():
    """The bug this mechanism exists for.

    Slot-occupancy tracking kept one id on a heater slot across a change of
    jar, because the slot never read empty. An empty frame between the two
    is what distinguishes them, and it has to produce a different id.
    """
    lin = VialLineage()
    lin.update(occ(0), occ(), 100.0)
    first = lin.vial_on_heater(0)

    lin.update(occ(), occ(), 130.0)            # slot empties
    lin.update(occ(0), occ(), 160.0)           # a different jar is set down
    second = lin.vial_on_heater(0)

    assert second != first
    assert second == "heater-0-160"


def test_most_recent_vacate_is_the_one_claimed():
    lin = VialLineage()
    lin.update(occ(0, 1), occ(), 100.0)
    first, second = lin.vial_on_heater(0), lin.vial_on_heater(1)

    lin.update(occ(1), occ(), 130.0)           # slot 0 leaves
    lin.update(occ(), occ(), 160.0)            # slot 1 leaves
    assert lin.pending_vacates == [first, second]

    lin.update(occ(), occ(2), 190.0)
    assert lin.vial_on_cooling(2) == second    # the later vacate

    lin.update(occ(), occ(2, 3), 220.0)
    assert lin.vial_on_cooling(3) == first


def test_unexplained_cooling_fill_warns_and_does_not_link():
    lin = VialLineage()
    lin.update(occ(), occ(), 100.0)
    lin.update(occ(), occ(1), 130.0)           # appears with no heater vacate

    assert lin.records == []
    assert lin.vial_on_cooling(1) is None
    assert len(lin.warnings) == 1
    assert "no unmatched heater vacate" in lin.warnings[0]


def test_a_cooling_slot_emptying_is_not_an_error():
    lin = VialLineage()
    lin.update(occ(0), occ(), 100.0)
    lin.update(occ(), occ(), 130.0)
    lin.update(occ(), occ(2), 160.0)
    lin.update(occ(), occ(), 190.0)            # taken off the pad

    assert lin.vial_on_cooling(2) is None
    assert lin.warnings == []
    assert len(lin.records) == 1               # the record stands


def test_as_dict_round_trips_the_state():
    lin = VialLineage()
    lin.update(occ(0), occ(), 100.0)
    lin.update(occ(), occ(1), 130.0)
    d = lin.as_dict()

    assert d["on_cooling"] == {1: "heater-0-100"}
    assert d["records"][0]["vial_id"] == "heater-0-100"
    assert d["pending_vacates"] == []


def test_an_unlinked_fill_warns_once_not_every_frame():
    """Occupancy has to be remembered even when identity could not be.

    The first version only recorded a cooling slot once it was linked to a
    heater vacate, so an unlinked slot kept looking newly-filled and warned
    again on every later frame - on a real run, hundreds of times for one
    vial.
    """
    lin = VialLineage()
    for t in (100.0, 130.0, 160.0, 190.0):
        lin.update(occ(), occ(1), t)

    assert len(lin.warnings) == 1


def test_an_unlinked_slot_can_still_be_claimed_after_it_empties():
    lin = VialLineage()
    lin.update(occ(), occ(1), 100.0)           # unexplained, warns
    lin.update(occ(), occ(), 130.0)            # emptied again
    lin.update(occ(0), occ(), 160.0)           # a vial reaches a heater
    lin.update(occ(), occ(), 190.0)            # and leaves it
    lin.update(occ(), occ(1), 220.0)           # same pad slot, now explained

    assert len(lin.warnings) == 1
    assert lin.vial_on_cooling(1) == "heater-0-160"
