"""Tests for the token, latency and retrieval-quality improvements."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.llm.openrouter import OpenRouterClient, StreamResult
from app.models import Record, comparable_usd
from app.rag.embeddings import HashingEmbedder
from app.rag.index import SearchIndex, build_index
from app.rag.prompt import SYSTEM_PROMPT, build_messages
from app.rag.query import QueryFilters
from app.rag.retriever import Retriever
from app.services.state import AppState
from app.textutil import light_stem_ar, search_terms
from tests.conftest import make_settings


def _index(tmp_path: Path, records: list[Record]) -> SearchIndex:
    out = tmp_path / "build"
    build_index(records, out, HashingEmbedder())
    return SearchIndex(out, HashingEmbedder())


def _listing(i: int, price: float, period: str, text: str = "Apartment for rent") -> Record:
    return Record(id=f"wasalt:listing:{i}", source="wasalt", kind="listing", url=f"https://wasalt.sa/x/{i}",
                  title=f"Apartment {i}", purpose="rent", property_type="apartment", property_types=["apartment"],
                  price=price, currency="SAR", rent_period=period, city="Riyadh", text=text,
                  updated_at=datetime(2026, 11, 1, tzinfo=UTC))


def test_arabic_light_stemming():
    assert light_stem_ar("الشقق") == "شقق"
    assert light_stem_ar("بالرياض") == "رياض"
    assert light_stem_ar("للايجار") == "ايجار"
    assert light_stem_ar("Riyadh") == "Riyadh"
    assert "شقق" in search_terms("الشقق في الرياض")


def test_arabic_keyword_search_matches_across_prefixes(tmp_path):
    rec = Record(id="wasalt:guide:1", source="wasalt", kind="guide", url="u", title="دليل",
                 text="أفضل الشقق للإيجار بالرياض قرب المدارس")
    idx = _index(tmp_path, [rec])
    hits = idx.search_chunks("شقق ايجار رياض", None, k=3)
    assert hits and hits[0].record_id == "wasalt:guide:1" and hits[0].keyword_match


def test_rents_are_compared_annually(tmp_path):
    assert comparable_usd(5000, "SAR", "rent", "month") == pytest.approx(5000 * 12 / 3.75)
    idx = _index(tmp_path, [_listing(1, 5_000, "month"), _listing(2, 45_000, "year")])
    # Budget 48,000 SAR/year: the 5,000/month (= 60,000/year) flat must NOT match.
    res = idx.search_records(QueryFilters(purpose="rent", max_price=48_000, currency="SAR"), kinds=["listing"])
    assert [r.id for r in res.records] == ["wasalt:listing:2"]
    cheapest = idx.search_records(QueryFilters(purpose="rent", sort="price_asc"), kinds=["listing"], limit=2)
    assert [r.id for r in cheapest.records] == ["wasalt:listing:2", "wasalt:listing:1"]


def test_dense_only_passages_below_floor_are_dropped(tmp_path):
    idx = _index(tmp_path, [Record(id="a", source="darglobal", kind="article", url="u", title="Golf", text="Golf course villas in Oman")])
    qvec = HashingEmbedder().embed(["zzzz qqqq"])[0]
    assert idx.search_chunks("zzzz qqqq", qvec, k=3, min_similarity=0.99) == []
    assert idx.search_chunks("golf villas", qvec, k=3, min_similarity=0.99)  # keyword match bypasses the floor


def test_prompt_is_lean(ingested: Settings):
    from app.rag.index import current_build_dir

    ret = Retriever(SearchIndex(current_build_dir(ingested.index_root), HashingEmbedder()))
    r = ret.retrieve("villas for sale in Jeddah")
    history = [{"role": "user", "content": "flats in Riyadh"},
               {"role": "assistant", "content": "Two flats [1] and [2, 3]. Source: Wasalt.sa"}] * 3
    msgs = build_messages("villas for sale in Jeddah", history, r, snapshot_date="2026-11-18", history_messages=4)
    assert msgs[0]["content"] == SYSTEM_PROMPT  # static -> cache-friendly prefix
    assert len(msgs) == 1 + 4 + 1
    assert all("[1]" not in m["content"] and "Source: Wasalt.sa" not in m["content"] for m in msgs[1:-1])
    user = msgs[-1]["content"]
    assert "https://" not in user  # URLs are not needed by the model
    assert "[Wasalt.sa listing]" not in user  # chunk headers stripped
    assert len(SYSTEM_PROMPT) < 1600


async def _collect(client: OpenRouterClient) -> tuple[str, StreamResult]:
    result = StreamResult()
    text = ""
    async for delta in client.stream_chat([{"role": "user", "content": "hi"}], result):
        text += delta
    return text, result


async def test_first_token_deadline_fails_over():
    calls: list[list[str]] = []

    async def slow_body():
        for _ in range(20):
            yield b": OPENROUTER PROCESSING\n\n"
            await asyncio.sleep(0.02)

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body["models"])
        assert body["usage"] == {"include": True}
        if body["models"][0] == "a/one:free":
            return httpx.Response(200, content=slow_body(), headers={"content-type": "text/event-stream"})
        chunks = [
            {"model": "c/three:free", "choices": [{"delta": {"content": "fast"}}]},
            {"model": "c/three:free", "choices": [], "usage": {"prompt_tokens": 812, "completion_tokens": 64}},
        ]
        data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=data.encode())

    s = Settings(openrouter_api_key="k", llm_discover_free_models=False, llm_models_per_request=2,
                 llm_models=["a/one:free", "b/two:free", "c/three:free", "d/four:free"], llm_first_token_timeout_s=0.1)
    client = OpenRouterClient(s, transport=httpx.MockTransport(handler))
    text, result = await _collect(client)
    assert text == "fast" and calls[1][0] == "c/three:free"
    assert result.usage == {"prompt_tokens": 812, "completion_tokens": 64}
    await client.aclose()


def test_old_index_format_is_rebuilt_from_snapshot(ingested: Settings):
    from app.rag.index import INDEX_FORMAT, current_build_dir

    build = current_build_dir(ingested.index_root)
    meta = json.loads((build / "meta.json").read_text())
    meta["format"] = 1
    (build / "meta.json").write_text(json.dumps(meta))
    state = AppState(ingested)
    state.embedder = HashingEmbedder()
    assert state.load_index() and state.index.meta["format"] == INDEX_FORMAT
    assert state.index.vectors is not None and state.index.vectors.backend == "chroma"


def test_sse_keepalive_and_token_usage(ingested: Settings, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api.routes as routes
    from app.main import create_app
    from tests.conftest import FakeLLM

    class SlowLLM(FakeLLM):
        async def stream_chat(self, messages, result):
            await asyncio.sleep(0.3)
            result.model = "google/gemma-4-26b-a4b-it:free"
            result.usage = {"prompt_tokens": 950, "completion_tokens": 80}
            yield "Answer [1]"

    monkeypatch.setattr(routes, "KEEPALIVE_S", 0.05)
    app = create_app(ingested)
    with TestClient(app) as client:
        state = app.state.app_state
        while not state.warmup_complete:
            pass
        state.llm = SlowLLM()
        resp = client.post("/api/chat", json={"message": "villas for sale in Jeddah"})
    assert ": keep-alive" in resp.text
    done = json.loads(resp.text.strip().split("\n\n")[-1].split("data:", 1)[1])
    assert done["tokens"] == {"prompt_tokens": 950, "completion_tokens": 80}
    assert done["ttft_ms"] is not None and done["ttft_ms"] >= 250


def test_port_opens_before_index_loads(tmp_path):
    """Lifespan must not block on loading data (tiny-CPU hosts would fail port scans)."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
