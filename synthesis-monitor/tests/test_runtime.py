"""Transport and dashboard wiring, without starting real processes."""

from __future__ import annotations

import queue
import time

import pytest

from runtime.bus import LatestSlot, RingBuffer, drain_all, drain_latest, put_drop_oldest
from runtime.messages import WorkerStatus


def test_put_drop_oldest_keeps_the_newest():
    q = queue.Queue(maxsize=2)
    assert put_drop_oldest(q, 1) is True
    assert put_drop_oldest(q, 2) is True
    assert put_drop_oldest(q, 3) is False, "should report the drop"
    assert list(q.queue) == [2, 3], "the oldest goes, not the newest"


def test_drain_latest_skips_the_backlog():
    q = queue.Queue()
    for i in range(5):
        q.put(i)
    assert drain_latest(q) == 4
    assert q.empty()


def test_drain_latest_on_empty():
    assert drain_latest(queue.Queue()) is None


def test_drain_all_preserves_order():
    q = queue.Queue()
    for i in range(4):
        q.put(i)
    assert drain_all(q) == [0, 1, 2, 3]


def test_drain_all_respects_its_limit():
    q = queue.Queue()
    for i in range(10):
        q.put(i)
    assert len(drain_all(q, limit=3)) == 3


def test_latest_slot():
    slot = LatestSlot()
    assert slot.get() == (None, 0.0)
    slot.set("a")
    slot.set("b")
    value, ts = slot.get()
    assert value == "b"
    assert ts == pytest.approx(time.time(), abs=5)
    assert slot.writes == 2


def test_ring_buffer_is_bounded_and_newest_first():
    ring = RingBuffer(capacity=3)
    ring.extend([1, 2])
    ring.extend([3, 4])
    assert ring.latest() == [4, 3, 2]
    assert ring.latest(2) == [4, 3]


def test_worker_status_copy_is_detached():
    status = WorkerStatus(worker="capture", frames=1)
    snapshot = status.copy()
    status.frames = 99
    status.extra["x"] = 1
    assert snapshot.frames == 1
    assert snapshot.extra == {}


# --------------------------------------------------------------------------
# Dashboard, against a stand-in supervisor. Flask's test client, no processes.
# --------------------------------------------------------------------------
class FakeSupervisor:
    def __init__(self):
        self.preview = LatestSlot()
        self.thermal = LatestSlot()
        self.result = LatestSlot()
        self.events = RingBuffer()
        self.status = {}
        self.run_id = None
        self.persist = False

    def health(self):
        return {"run_id": None, "uptime_s": 1.0, "persisting": False,
                "analysis_interval_s": 45.0, "workers": {}}

    def alive(self):
        return True


@pytest.fixture
def client():
    from dashboard.app import create_app

    app = create_app(FakeSupervisor())
    app.config["TESTING"] = True
    return app.test_client()


def test_index_renders_before_any_frame(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"CapEx Synthesis Monitor" in page.data
    assert b"Detection is not implemented" in page.data, \
        "the page must not imply the detectors work"


def test_state_endpoint_is_valid_with_no_data(client):
    body = client.get("/api/state").get_json()
    assert body["pipeline"]["n_crucibles"] == 0
    assert body["events"] == []
    assert "health" in body


def test_healthz_is_503_without_frames(client):
    assert client.get("/healthz").status_code == 503


def test_crucibles_endpoint_empty(client):
    assert client.get("/api/crucibles").get_json()["crucibles"] == []


def test_series_requires_a_feature(client):
    assert client.get("/api/crucibles/1/series").status_code == 400


def test_snapshot_path_traversal_is_refused(client):
    assert client.get("/snapshot/../../config.py").status_code in (301, 404)


# --------------------------------------------------------------------------
# The hazard banner. Its logic is client-side JS (severity=='alert' among
# recent events, plus whether the vision worker actually halted) - not
# runnable here. What's tested at this layer is the data contract it reads:
# that a tripped detector's event really does arrive at /api/state with
# severity "alert", and that the page ships the wiring to read it.
# --------------------------------------------------------------------------
def test_index_ships_the_hazard_banner_element_and_its_renderer(client):
    page = client.get("/").data.decode()
    assert 'id="hazard-banner"' in page
    assert "renderHazardBanner" in page
    assert "renderHazardBanner(st.events" in page, \
        "the banner must be driven by the same events the table below shows"


def test_state_reports_an_alert_event_for_the_banner_to_read():
    from pipeline.types import Event

    from dashboard.app import create_app

    sup = FakeSupervisor()
    sup.events.extend([Event(
        kind="missing_lid", severity="alert",
        message="missing_lid in heating", timestamp=1_800_000_000.0,
        frame_id=3, detector="missing_lid", zone="heating")])
    app = create_app(sup)
    app.config["TESTING"] = True
    client = app.test_client()

    body = client.get("/api/state").get_json()
    assert body["events"][0]["severity"] == "alert"
    assert body["events"][0]["message"] == "missing_lid in heating"


def test_state_with_no_alerts_has_none_at_alert_severity(client):
    body = client.get("/api/state").get_json()
    assert not any(e["severity"] == "alert" for e in body["events"])
