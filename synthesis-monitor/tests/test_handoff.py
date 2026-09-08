"""Left-heater to cold-storage handoff, driven by arrivals.

The mechanism is small enough that these say the whole of it: what is on the
heater, which storage slots read full, and what that implies.
"""

from __future__ import annotations

from pipeline.handoff import HeaterHandoff


def store(*slots: int) -> dict[int, bool]:
    return {i: (i in slots) for i in range(4)}


def test_a_crucible_on_the_heater_picks_up_an_id():
    h = HeaterHandoff(heater_slot=1)
    h.update(False, store(), 100.0)
    assert h.held_id is None

    h.update(True, store(), 130.0)
    assert h.held_id == "vial-130000"


def test_the_next_storage_arrival_inherits_that_id():
    h = HeaterHandoff(heater_slot=1)
    h.update(True, store(), 100.0)
    vial = h.held_id

    h.update(False, store(2), 130.0)
    assert h.storage_ids == {2: vial}
    assert h.held_id is None
    (rec,) = h.records
    assert rec.vial_id == vial and rec.storage_slot == 2


def test_a_replacement_that_never_leaves_the_slot_empty_still_works():
    """The case occupancy alone cannot see.

    The crucible moves to storage and another is put on the heater between
    two frames, so the heater reads occupied on both. Handing the id on
    first leaves the heater free to mint a new one in the same frame.
    """
    h = HeaterHandoff(heater_slot=1)
    h.update(True, store(), 100.0)
    first = h.held_id

    h.update(True, store(2), 130.0)      # arrived in storage, heater refilled
    assert h.storage_ids[2] == first
    second = h.held_id
    assert second is not None and second != first

    h.update(True, store(2, 3), 160.0)
    assert h.storage_ids[3] == second


def test_a_run_of_arrivals_each_take_the_current_heater_id():
    h = HeaterHandoff(heater_slot=1)
    seen = []
    filled = []
    for i in range(4):
        h.update(True, store(*filled), 100.0 + 30 * i)
        seen.append(h.held_id)
        filled.append(i)
        h.update(True, store(*filled), 115.0 + 30 * i)
    assert len(h.records) == 4
    assert [r.vial_id for r in h.records] == seen
    assert len({r.storage_slot for r in h.records}) == 4


def test_an_arrival_with_nothing_held_is_reported_not_guessed():
    h = HeaterHandoff(heater_slot=1)
    h.update(False, store(1), 100.0)

    assert h.records == []
    assert h.storage_ids == {}
    assert len(h.warnings) == 1
    assert "no id waiting" in h.warnings[0]


def test_an_unlabelled_arrival_warns_once_not_every_frame():
    h = HeaterHandoff(heater_slot=1)
    for t in (100.0, 130.0, 160.0):
        h.update(False, store(1), t)
    assert len(h.warnings) == 1


def test_the_id_survives_the_heater_going_empty_before_the_arrival():
    """Lifted off on one frame, put down in storage on a later one."""
    h = HeaterHandoff(heater_slot=1)
    h.update(True, store(), 100.0)
    vial = h.held_id
    h.update(False, store(), 130.0)      # in transit, nothing anywhere
    assert h.held_id == vial

    h.update(False, store(0), 160.0)
    assert h.storage_ids[0] == vial


def test_removing_something_from_storage_drops_its_label():
    h = HeaterHandoff(heater_slot=1)
    h.update(True, store(), 100.0)
    h.update(False, store(1), 130.0)
    assert h.storage_ids
    h.update(False, store(), 160.0)
    assert h.storage_ids == {}
    assert len(h.records) == 1           # the record still stands
