"""Semantic search on a 512 MB host: compact model profile, Arabic bridging,
warm-up behaviour and deployment configuration that stays consistent."""

import asyncio
import re
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.api.routes import process_memory
from app.config import Settings
from app.main import _warm_up
from app.rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    MULTILINGUAL_EMBEDDING_MODEL,
    PROFILES,
    FastEmbedder,
    HashingEmbedder,
    profile_for,
)
from app.rag.index import SearchIndex, current_build_dir
from app.rag.retriever import Retriever
from app.scraping.pipeline import ingest
from app.services.state import AppState
from tests.conftest import make_settings

ROOT = Path(__file__).resolve().parents[1]


class EnglishOnlyEmbedder(HashingEmbedder):
    """Test double with the default (English-only BGE) profile."""

    profile = PROFILES[DEFAULT_EMBEDDING_MODEL]

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return super().embed_query(text)


@pytest.fixture
def english_index(tmp_path: Path, sites) -> SearchIndex:
    settings = make_settings(tmp_path)
    asyncio.run(ingest(settings, EnglishOnlyEmbedder(), transport=sites.transport, min_records=5))
    return SearchIndex(current_build_dir(settings.index_root), EnglishOnlyEmbedder())


# ------------------------------------------------------------------ deployment
def test_default_model_is_the_same_everywhere():
    assert Settings.model_fields["embedding_model"].default == DEFAULT_EMBEDDING_MODEL
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert f"ARG EMBEDDING_MODEL={DEFAULT_EMBEDDING_MODEL}" in dockerfile
    assert "EMBEDDING_MODEL=${EMBEDDING_MODEL}" in dockerfile  # the ARG reaches the runtime ENV
    compose = (ROOT / "docker-compose.yml").read_text()
    assert f"${{EMBEDDING_MODEL:-{DEFAULT_EMBEDDING_MODEL}}}" in compose
    assert f"EMBEDDING_MODEL={DEFAULT_EMBEDDING_MODEL}" in (ROOT / ".env.example").read_text()


def test_render_blueprint_keeps_semantic_search_on_the_free_plan():
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text())
    (service,) = blueprint["services"]
    assert service["type"] == "web" and service["runtime"] == "docker" and service["plan"] == "free"
    assert service["healthCheckPath"] == "/api/ready"
    env = {e["key"]: e for e in service["envVars"]}
    assert env["OPENROUTER_API_KEY"] == {"key": "OPENROUTER_API_KEY", "sync": False}  # prompted, never committed
    assert env["EMBEDDING_MODEL"]["value"] == DEFAULT_EMBEDDING_MODEL
    assert env["ENABLE_DENSE"]["value"] == "true" and env["VECTOR_BACKEND"]["value"] == "chroma"
    assert env["REFRESH_INTERVAL_HOURS"]["value"] == "0" and env["REEMBED_IF_OUTDATED"]["value"] == "false"
    assert not re.search(r"sk-or-|sk-[A-Za-z0-9]{20}", (ROOT / "render.yaml").read_text())


# -------------------------------------------------------------------- profiles
def test_model_profiles():
    bge = profile_for(DEFAULT_EMBEDDING_MODEL)
    assert not bge.multilingual and bge.query_prefix.startswith("Represent this sentence") and bge.min_similarity == 0.5
    assert profile_for(MULTILINGUAL_EMBEDDING_MODEL).multilingual
    assert profile_for("intfloat/multilingual-e5-small").multilingual  # unknown, but named multilingual
    assert profile_for("hashing").multilingual


def test_bge_instruction_is_added_to_queries_only(monkeypatch, tmp_path: Path):
    seen: list[str] = []

    class StubModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed(self, texts, batch_size=16):
            seen.extend(texts)
            for text in texts:
                yield np.ones(4, dtype=np.float32) * (1 + len(text) % 3)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=StubModel))
    emb = FastEmbedder(DEFAULT_EMBEDDING_MODEL, tmp_path, threads=1)
    assert emb.dim == 4 and emb._model.kwargs["threads"] == 1
    seen.clear()
    emb.embed(["Sea-view villa in Jeddah"])
    vec = emb.embed_query("villa with a sea view")
    assert seen == ["Sea-view villa in Jeddah", PROFILES[DEFAULT_EMBEDDING_MODEL].query_prefix + "villa with a sea view"]
    assert vec.shape == (4,) and np.isclose(np.linalg.norm(vec), 1.0)


def test_relevance_floor_follows_the_model(tmp_path: Path, sites):
    settings = make_settings(tmp_path)
    asyncio.run(ingest(settings, EnglishOnlyEmbedder(), transport=sites.transport, min_records=5))
    state = AppState(settings)
    state.embedder = EnglishOnlyEmbedder()
    assert state.load_index() and state.retriever.min_sim == 0.5  # BGE's own scale
    state = AppState(make_settings(tmp_path, min_passage_similarity=0.42))
    state.embedder = EnglishOnlyEmbedder()
    assert state.load_index() and state.retriever.min_sim == 0.42  # explicit setting wins


# --------------------------------------------------------- Arabic, English model
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("شقق للإيجار في جدة", "apartment for rent in Jeddah"),
        ("فيلا 4 غرف نوم للبيع في الرياض", "4 bedroom villa for sale in Riyadh"),
        ("ما هي مميزات السكن هنا", None),  # nothing understood: no vector search for this one
        ("villas for sale in Jeddah", "villas for sale in Jeddah"),
    ],
)
def test_english_only_model_gets_an_english_query(english_index: SearchIndex, question, expected):
    retriever = Retriever(english_index)
    assert retriever.dense_query_text(retriever.parser.parse(question, [])) == expected


def test_arabic_question_still_finds_listings_with_an_english_model(english_index: SearchIndex):
    retriever = Retriever(english_index)
    result = retriever.retrieve("شقق للإيجار في الرياض")
    assert english_index.embedder.queries == ["apartment for rent in Riyadh"]  # what the vector search saw
    assert result.sources and all(s.record.city == "Riyadh" for s in result.sources if s.record.kind == "listing")


def test_multilingual_model_embeds_arabic_as_is(ingested: Settings):
    index = SearchIndex(current_build_dir(ingested.index_root), HashingEmbedder())
    retriever = Retriever(index)
    parsed = retriever.parser.parse("شقق للإيجار في جدة", [])
    assert retriever.dense_query_text(parsed) == parsed.search_text


# ---------------------------------------------------------------------- warm-up
def test_first_question_waits_briefly_for_warm_up(tmp_path: Path):
    state = AppState(make_settings(tmp_path))
    threading.Timer(0.3, lambda: setattr(state, "warmup_complete", True)).start()
    started = asyncio.run(_timed(state.wait_for_warmup(5)))
    assert state.warmup_complete and 0.25 <= started < 2
    state.warmup_complete = False
    assert asyncio.run(_timed(state.wait_for_warmup(0.3))) < 1  # never waits longer than allowed


async def _timed(coro) -> float:
    loop = asyncio.get_running_loop()
    start = loop.time()
    await coro
    return loop.time() - start


def test_outdated_vectors_are_left_alone_when_reembedding_is_off(tmp_path: Path, sites, caplog):
    settings = make_settings(tmp_path, reembed_if_outdated=False)
    asyncio.run(ingest(settings, None, transport=sites.transport, min_records=5))  # no vectors
    state = AppState(settings)
    asyncio.run(_warm_up(state))
    assert state.ready and not state.index.dense_enabled and state.ingestion.status["state"] == "idle"
    assert "semantic search is off until the image is rebuilt" in caplog.text


def test_stats_reports_memory_use():
    mem = process_memory()
    assert mem["rss_mb"] and mem["rss_mb"] > 10
    assert mem["limit_mb"] is None or mem["limit_mb"] > 0


def test_warm_up_adds_semantic_search_without_reloading_records(ingested: Settings):
    state = AppState(ingested)
    assert state.load_index() and not state.index.dense_enabled  # keyword search first, as at start-up
    first = state.index
    state.embedder = HashingEmbedder()
    assert state.load_index()
    assert state.index is first and first.dense_enabled and state.retriever.index is first  # upgraded in place
