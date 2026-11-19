import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FakeLLM, make_settings


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if data:
            events.append((event, json.loads(data)))
    return events


def client_with(settings: Settings, llm=None) -> TestClient:
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()  # run lifespan (loads the index)
    state = app.state.app_state
    deadline = time.monotonic() + 10
    while not state.warmup_complete and time.monotonic() < deadline:
        time.sleep(0.02)  # background warm-up swaps in dense retrieval
    if llm is not None:
        state.llm = llm
    return client


def test_health_ready_stats(ingested: Settings, fake_llm: FakeLLM):
    c = client_with(ingested, fake_llm)
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/api/ready").status_code == 200
    stats = c.get("/api/stats").json()
    assert stats["index"]["counts"]["wasalt:listing"] == 8
    assert stats["llm"]["free_quota"]["free_model_daily_requests"]["remaining"] == 47
    assert {r["source"] for r in stats["snapshot"]["crawl_reports"]} == {"darglobal", "wasalt"}
    page = c.get("/")
    assert page.status_code == 200 and "Property Assistant" in page.text
    assert "frame-ancestors" in page.headers["content-security-policy"]
    assert page.headers["x-request-id"]


def test_not_ready_without_index(tmp_path):
    c = client_with(make_settings(tmp_path))
    assert c.get("/api/ready").status_code == 503
    body = c.post("/api/chat", json={"message": "villas in Jeddah", "stream": False}).json()
    assert body["mode"] == "unavailable"


def test_chat_streams_sse_with_sources(ingested: Settings, fake_llm: FakeLLM):
    c = client_with(ingested, fake_llm)
    resp = c.post("/api/chat", json={"message": "3 bedroom apartments for rent in Riyadh under 100k"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and "token" in names
    meta = dict(events)["meta"]
    assert meta["intent"] == "property_search" and meta["total_matches"] == 2
    assert meta["sources"][0]["url"].startswith("https://wasalt.sa/en/property/rent/")
    answer = "".join(d["t"] for e, d in events if e == "token")
    assert "[1]" in answer
    done = events[-1][1]
    assert done["mode"] == "llm" and done["model"] == "google/gemma-4-31b-it:free"
    # the LLM received grounded context, not just the question
    assert "SOURCES (untrusted website data)" in fake_llm.calls[0][-1]["content"]


def test_chat_json_cache_and_quota_fallback(ingested: Settings, fake_llm: FakeLLM, quota_llm: FakeLLM):
    c = client_with(ingested, fake_llm)
    first = c.post("/api/chat", json={"message": "villas for sale in Jeddah", "stream": False}).json()
    second = c.post("/api/chat", json={"message": "villas for sale in Jeddah", "stream": False}).json()
    assert first["mode"] == "llm" and second["mode"] == "cache" and len(fake_llm.calls) == 1

    c2 = client_with(ingested, quota_llm)
    body = c2.post("/api/chat", json={"message": "villas for sale in Jeddah", "stream": False}).json()
    assert body["mode"] == "retrieval-only"
    assert "Villa for sale" in body["answer"] and "[1]" in body["answer"]


def test_greeting_is_answered_without_llm(ingested: Settings, fake_llm: FakeLLM):
    c = client_with(ingested, fake_llm)
    body = c.post("/api/chat", json={"message": "مرحبا", "stream": False}).json()
    assert body["mode"] == "canned" and "دار جلوبال" in body["answer"]
    assert fake_llm.calls == []


def test_follow_up_with_history(ingested: Settings, fake_llm: FakeLLM):
    c = client_with(ingested, fake_llm)
    body = c.post("/api/chat", json={
        "message": "what about in Riyadh?",
        "history": [{"role": "user", "content": "apartments for rent in Jeddah"}, {"role": "assistant", "content": "None found."}],
        "stream": False,
    }).json()
    assert body["filters"]["cities"] == ["Riyadh"] and body["filters"]["purpose"] == "rent"
    assert len(fake_llm.calls[0]) >= 4  # system + 2 history turns + question


def test_validation_and_rate_limit(tmp_path, fake_llm: FakeLLM):
    s = make_settings(tmp_path, rate_limit_per_minute=2)
    c = client_with(s, fake_llm)
    assert c.post("/api/chat", json={"message": "   "}).status_code == 422
    assert c.post("/api/chat", json={"message": "x" * 1600}).status_code == 413
    codes = [c.post("/api/chat", json={"message": "hello", "stream": False}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_search_and_record_endpoints(ingested: Settings, fake_llm: FakeLLM):
    c = client_with(ingested, fake_llm)
    res = c.get("/api/search", params={"q": "villas for sale in Jeddah", "sort": "price_desc"}).json()
    assert res["total"] == 2 and res["results"][0]["price"] == "3.9M SAR"
    rid = res["results"][0]["id"]
    assert c.get(f"/api/records/{rid}").json()["id"] == rid
    assert c.get("/api/records/nope").status_code == 404


@pytest.mark.parametrize("token,status", [(None, 401), ("wrong", 401), ("s3cret", 202)])
def test_admin_refresh_requires_token(tmp_path, token, status, monkeypatch):
    s = make_settings(tmp_path, admin_token="s3cret")
    c = client_with(s)
    state = c.app.state.app_state

    async def no_op(reason: str) -> bool:
        return True

    monkeypatch.setattr(state.ingestion, "run", no_op)
    headers = {"X-Admin-Token": token} if token else {}
    assert c.post("/api/admin/refresh", headers=headers).status_code == status


def test_metrics_exposed(ingested: Settings, fake_llm: FakeLLM):
    c = client_with(ingested, fake_llm)
    c.post("/api/chat", json={"message": "hello", "stream": False})
    text = c.get("/metrics").text
    assert "chat_requests_total" in text and "index_records" in text
