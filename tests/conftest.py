from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.config import Settings
from app.llm.openrouter import LLMQuotaExceeded, StreamResult
from app.rag.embeddings import HashingEmbedder
from app.scraping.pipeline import ingest
from tests.fixtures.sites import MockSites


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        crawl_min_interval_s=0,
        crawl_max_retries=1,
        embedding_model="hashing",
        openrouter_api_key=None,
        ingest_on_startup="never",
        refresh_interval_hours=0,
        rate_limit_per_minute=1000,
        rate_limit_per_day=10000,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def sites() -> MockSites:
    return MockSites()


@pytest.fixture
def ingested(settings: Settings, sites: MockSites) -> Settings:
    """A settings object whose data_dir contains a snapshot + active index built from the mock sites."""
    asyncio.run(ingest(settings, HashingEmbedder(), transport=sites.transport, min_records=5))
    return settings


class FakeLLM:
    """Stand-in for OpenRouterClient used by API/chat tests."""

    def __init__(self, reply: str = "Here are two options [1] and [2]. Source: Wasalt.sa", fail: Exception | None = None):
        self.reply = reply
        self.fail = fail
        self.calls: list[list[dict[str, str]]] = []
        self.configured = True

    async def stream_chat(self, messages, result: StreamResult) -> AsyncIterator[str]:
        self.calls.append(messages)
        if self.fail:
            raise self.fail
        result.model = "google/gemma-4-31b-it:free"
        for word in self.reply.split(" "):
            yield word + " "

    async def refresh_models(self, force: bool = False):
        return ["google/gemma-4-31b-it:free"]

    async def key_status(self):
        return {"free_model_daily_requests": {"used": 3, "limit": 50, "remaining": 47}}

    def status(self):
        return {"configured": True, "model_chain": ["google/gemma-4-31b-it:free"], "last_model": None}

    async def aclose(self):
        return None


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def quota_llm() -> FakeLLM:
    return FakeLLM(fail=LLMQuotaExceeded())
