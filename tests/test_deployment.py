import asyncio
from copy import deepcopy

import httpx
import pytest

from app.cli import main
from app.coverage import source_coverage
from app.deployment import verify_deployment
from app.rag.embeddings import HashingEmbedder
from app.rag.index import current_build_dir
from app.scraping.pipeline import ingest, snapshot_paths
from tests.fixtures.sites import MockSites
from tests.test_api import client_with


def test_coverage_requires_property_records_not_just_blog_articles():
    coverage = source_coverage({"darglobal:project": 24, "wasalt:article": 50})
    assert not coverage["complete"] and coverage["missing_sources"] == ["wasalt"]
    assert coverage["sources"]["wasalt"]["status"] == "missing_property_data"
    coverage = source_coverage({"darglobal:project": 24, "wasalt:listing": 8}, ["wasalt"])
    assert coverage["complete"] and coverage["sources"]["wasalt"]["status"] == "stale"
    assert source_coverage({})["sources"]["wasalt"]["status"] == "unavailable"


def test_strict_ingestion_accepts_both_sources(settings, sites):
    summary = asyncio.run(ingest(
        settings, HashingEmbedder(), transport=sites.transport, min_records=5, require_all_sources=True,
    ))
    assert summary["index"]["counts"]["wasalt:listing"] == 8
    assert summary["index"]["counts"]["darglobal:project"] == 2


def test_strict_ingestion_rejection_preserves_snapshot_and_index(ingested):
    paths = snapshot_paths(ingested)
    before = [path.read_bytes() for path in paths]
    build = current_build_dir(ingested.index_root)
    with pytest.raises(RuntimeError, match="unavailable or incomplete: wasalt"):
        asyncio.run(ingest(
            ingested, HashingEmbedder(), transport=MockSites(wasalt_blocked=True).transport,
            min_records=5, require_all_sources=True,
        ))
    assert [path.read_bytes() for path in paths] == before
    assert current_build_dir(ingested.index_root) == build


def test_strict_ingestion_cannot_be_combined_with_allow_empty(settings, monkeypatch):
    monkeypatch.setattr("app.cli.get_settings", lambda: settings)
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--require-all-sources", "--allow-empty"])
    assert exc.value.code == 2


def test_deployment_verification_through_real_routes(ingested, fake_llm):
    c = client_with(ingested, fake_llm)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=c.app), base_url="http://test/") as client:
            stats = (await client.get("api/stats")).json()
            assert stats["coverage"]["complete"]
            return await verify_deployment(client)

    try:
        result = asyncio.run(run())
        assert result["ok"], result
        assert result["chat_checked"] and len(fake_llm.calls) == 2
    finally:
        c.__exit__(None, None, None)


@pytest.mark.parametrize("problem", ["missing_source", "stale_source", "no_vectors", "no_key", "fallback", "paid_model", "uncited"])
def test_verifier_rejects_incomplete_deployment(problem):
    stats = {
        "index": {"counts": {"darglobal:project": 24, "wasalt:listing": 8}, "dense_retrieval": True,
                  "vector_store": {"loaded": True, "vectors": 32}},
        "snapshot": {"stale_sources": []}, "llm": {"configured": True},
    }
    if problem == "missing_source":
        stats["index"]["counts"].pop("wasalt:listing")
    elif problem == "stale_source":
        stats["snapshot"]["stale_sources"] = ["wasalt"]
    elif problem == "no_vectors":
        stats["index"]["vector_store"]["loaded"] = False
    elif problem == "no_key":
        stats["llm"]["configured"] = False
    calls = []

    def handler(request):
        if request.url.path == "/api/ready":
            return httpx.Response(200, json={"ready": True})
        if request.url.path == "/api/stats":
            return httpx.Response(200, json=deepcopy(stats))
        calls.append(request)
        return httpx.Response(200, json={
            "answer": "An available property." if problem == "uncited" else "An available property [1] [2].",
            "mode": "fallback" if problem == "fallback" else "llm",
            "model": "paid/model" if problem == "paid_model" else "openrouter/free",
            "sources": [
                {"source": "darglobal", "kind": "project", "n": 1, "url": "https://darglobal.co.uk/dg1"},
                {"source": "wasalt", "kind": "listing", "n": 2, "url": "https://wasalt.sa/en/property/sale/1"},
            ],
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test/") as client:
            return await verify_deployment(client)

    result = asyncio.run(run())
    assert not result["ok"] and result["errors"]
    if problem in {"missing_source", "stale_source", "no_vectors", "no_key"}:
        assert not calls  # do not spend free requests when prerequisites already failed
