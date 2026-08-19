"""The zone editor's logic, driven without a display.

The cv2 window is a thin shell over ZoneEditor, so everything that can be
wrong about dragging - grabbing the wrong vertex, moving the wrong zone,
saving coordinates that do not round-trip - is testable here.
"""

from __future__ import annotations

import json

import pytest

from tools.edit_zones import ZoneEditor, save

W, H = 1000, 500


def rect(x0, x1, y0=0.2, y1=0.8):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.fixture
def editor():
    return ZoneEditor(W, H, {
        "filling": rect(0.0, 0.3),
        "heating": rect(0.5, 0.8),
    })


def test_polygons_load_into_pixels(editor):
    assert editor.zones["filling"][0] == [0.0, 100.0]
    assert editor.zones["heating"][1] == [800.0, 100.0]
    assert editor.selected == "filling"          # process order, not dict order


def test_dragging_a_vertex_moves_only_that_vertex(editor):
    before = [list(v) for v in editor.zones["filling"]]
    assert editor.begin_drag(300.0, 100.0)       # the (0.3, 0.2) corner
    editor.drag_to(320.0, 90.0)
    editor.end_drag()

    assert editor.zones["filling"][1] == [320.0, 90.0]
    assert editor.zones["filling"][0] == before[0]
    assert editor.zones["filling"][2] == before[2]
    assert editor.dirty


def test_dragging_inside_moves_the_whole_zone(editor):
    before = [list(v) for v in editor.zones["heating"]]
    assert editor.begin_drag(650.0, 250.0)       # interior of heating
    editor.drag_to(660.0, 260.0)
    editor.end_drag()

    for old, new in zip(before, editor.zones["heating"]):
        assert new == [old[0] + 10.0, old[1] + 10.0]
    assert editor.selected == "heating"


def test_clicking_empty_space_grabs_nothing(editor):
    assert editor.begin_drag(450.0, 250.0) is False   # gap between zones
    assert not editor.dragging
    editor.drag_to(999.0, 999.0)                      # must be a no-op
    assert editor.zones["filling"][0] == [0.0, 100.0]


def test_vertex_wins_over_interior(editor):
    """A click on a corner drags the corner, not the whole zone."""
    editor.begin_drag(300.0, 400.0)      # the (0.3, 0.8) corner of filling
    editor.drag_to(310.0, 400.0)
    editor.end_drag()
    moved = [v for v in editor.zones["filling"] if v == [310.0, 400.0]]
    assert len(moved) == 1
    assert editor.zones["filling"][0] == [0.0, 100.0], "other corners unmoved"


def test_selected_zone_wins_a_shared_vertex():
    """Two zones meeting on an edge: you grab the one you can see selected."""
    ed = ZoneEditor(W, H, {"filling": rect(0.0, 0.5), "heating": rect(0.5, 1.0)})
    ed.select("heating")
    hit = ed.vertex_at(500.0, 100.0)
    assert hit is not None and hit[0] == "heating"

    ed.select("filling")
    hit = ed.vertex_at(500.0, 100.0)
    assert hit[0] == "filling"


def test_vertices_are_clamped_to_the_frame(editor):
    editor.begin_drag(0.0, 100.0)
    editor.drag_to(-500.0, -500.0)
    editor.end_drag()
    x, y = editor.zones["filling"][0]
    assert 0.0 <= x < W and 0.0 <= y < H


def test_nudge_moves_the_selection_only(editor):
    editor.select("heating")
    before = [list(v) for v in editor.zones["filling"]]
    editor.nudge(5, -3)
    assert editor.zones["filling"] == before
    assert editor.zones["heating"][0] == [505.0, 97.0]


def test_add_and_delete_vertex(editor):
    n = len(editor.zones["filling"])
    assert editor.add_vertex(150.0, 100.0)      # on the top edge
    assert len(editor.zones["filling"]) == n + 1
    assert editor.delete_vertex(150.0, 100.0)
    assert len(editor.zones["filling"]) == n


def test_cannot_delete_below_three_corners(editor):
    editor.zones["filling"] = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    assert editor.delete_vertex(0.0, 0.0) is False
    assert len(editor.zones["filling"]) == 3


def test_scale_keeps_the_centre(editor):
    poly = editor.zones["filling"]
    cx = sum(v[0] for v in poly) / len(poly)
    editor.scale_selected(0.5)
    poly = editor.zones["filling"]
    assert sum(v[0] for v in poly) / len(poly) == pytest.approx(cx)


def test_add_and_delete_zone(editor):
    editor.add_zone("lidding")
    assert "lidding" in editor.zones
    assert editor.selected == "lidding"
    editor.delete_zone()
    assert "lidding" not in editor.zones
    assert editor.selected in editor.zones


def test_selection_cycles(editor):
    first = editor.selected
    editor.select_next()
    assert editor.selected != first
    editor.select_next()
    assert editor.selected == first


def test_overlap_detection(editor):
    assert editor.overlaps() == []
    editor.select("heating")
    editor.nudge(-400, 0)                # slide it onto filling
    pairs = [tuple(sorted(o[:2])) for o in editor.overlaps()]
    assert ("filling", "heating") in pairs


def test_shared_edges_are_not_reported_as_overlap():
    """Adjacent zones touch. Warning on that would train you to ignore warnings."""
    ed = ZoneEditor(W, H, {"filling": rect(0.0, 0.5), "conveyor": rect(0.5, 1.0)})
    assert ed.overlaps() == []


def test_normalised_round_trips(editor):
    norm = editor.normalised()
    assert norm["filling"][0] == [0.0, 0.2]
    reloaded = ZoneEditor(W, H, {k: [tuple(p) for p in v] for k, v in norm.items()})
    assert reloaded.zones["filling"] == editor.zones["filling"]


def test_normalised_survives_a_resolution_change(editor):
    """A layout drawn on one capture size must mean the same on another."""
    norm = editor.normalised()
    bigger = ZoneEditor(2000, 1000, {k: [tuple(p) for p in v]
                                     for k, v in norm.items()})
    assert bigger.zones["filling"][0] == [0.0, 200.0]
    assert bigger.normalised()["filling"] == norm["filling"]


def test_save_writes_loadable_json(editor, tmp_path):
    out = tmp_path / "zones.json"
    assert save(editor, out) == 0
    data = json.loads(out.read_text())
    assert set(data) == {"filling", "heating"}
    assert all(0.0 <= c <= 1.0 for poly in data.values() for pt in poly for c in pt)


def test_editor_handles_no_zones():
    ed = ZoneEditor(W, H, {})
    assert ed.selected is None
    assert ed.begin_drag(10.0, 10.0) is False
    ed.nudge(5, 5)                       # must not raise
    ed.add_zone("filling")
    assert ed.selected == "filling"
    assert len(ed.zones["filling"]) == 4
