"""Ingestion pipeline: crawl -> normalise -> snapshot -> index -> activate.

Resilience: each source runs with its own polite client and failure domain.
If a source fails (blocked, outage), its records from the previous snapshot
are carried forward and flagged as stale, so the assistant keeps working.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from app import __version__
from app.config import Settings
from app.coverage import REQUIRED_KINDS, source_coverage
from app.models import CrawlReport, Record, utcnow
from app.net import tls_verify
from app.rag.embeddings import Embedder
from app.rag.index import activate, build_index, new_build_dir
from app.scraping.darglobal import DarGlobalScraper
from app.scraping.http import HostPolicy, PoliteClient
from app.scraping.pii import sanitize_record
from app.scraping.wasalt import WasaltScraper

log = logging.getLogger(__name__)

SCRAPERS = {"darglobal": DarGlobalScraper, "wasalt": WasaltScraper}


def make_client(
    settings: Settings, source: str, transport: httpx.AsyncBaseTransport | None = None
) -> PoliteClient:
    base = settings.crawl_min_interval_s  # default 1.0 s
    policies = {
        # DarGlobal: no crawl-delay requested; 2 concurrent, base/2 apart (0.5 s by default).
        "darglobal.co.uk": HostPolicy(min_interval_s=base / 2, concurrency=2),
        "www.darglobal.co.uk": HostPolicy(min_interval_s=base / 2, concurrency=2),
        # Wasalt's llms.txt asks for a "moderate" rate: strictly sequential, base apart (1 s).
        "wasalt.sa": HostPolicy(min_interval_s=base, concurrency=1),
        "cdn.wasalt.sa": HostPolicy(min_interval_s=base, concurrency=1),
        "blog.wasalt.sa": HostPolicy(min_interval_s=base, concurrency=1),
    }
    return PoliteClient(
        user_agent=settings.user_agent,
        default_policy=HostPolicy(min_interval_s=base, concurrency=1),
        host_policies=policies,
        timeout_s=settings.crawl_timeout_s,
        max_retries=settings.crawl_max_retries,
        max_bytes=settings.crawl_max_bytes,
        failure_budget=settings.crawl_failure_budget,
        transport=transport,
        verify=tls_verify(settings.tls_trust),
    )


async def crawl_source(
    settings: Settings, source: str, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[list[Record], CrawlReport]:
    started = time.perf_counter()
    async with make_client(settings, source, transport) as client:
        scraper = SCRAPERS[source](client, settings)
        try:
            records, report = await scraper.run()
        except Exception as exc:  # a bug in one adapter must not take down ingestion
            log.exception("scraper crashed", extra={"source": source})
            report = scraper.report
            report.ok = False
            report.notes.append(f"crashed: {exc!r}"[:300])
            records = []
        for host, reason in client.robots_problems.items():
            report.notes.append(f"robots.txt of {host} unreachable ({reason}); host skipped"[:300])
        report.requests = int(client.stats.get("requests", 0))
        report.blocked_by_robots = int(client.stats.get("robots_blocked", 0))
        report.status_codes = dict(client.status_counts)
        report.notes.append(f"duration {time.perf_counter() - started:.0f}s")
    log.info(
        "crawl finished",
        extra={"source": source, "ok": report.ok, "records": len(records), "requests": report.requests},
    )
    return records, report


def snapshot_paths(settings: Settings) -> tuple[Path, Path]:
    return settings.snapshot_dir / "records.jsonl", settings.snapshot_dir / "manifest.json"


def load_snapshot(settings: Settings) -> tuple[list[Record], dict[str, Any]]:
    records_path, manifest_path = snapshot_paths(settings)
    if not records_path.exists():
        return [], {}
    records: list[Record] = []
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(Record.model_validate_json(line))
                except ValueError:
                    log.warning("skipping corrupt snapshot line")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return records, manifest


def write_snapshot(settings: Settings, records: list[Record], manifest: dict[str, Any]) -> None:
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    records_path, manifest_path = snapshot_paths(settings)
    tmp = records_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
    os.replace(tmp, records_path)
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp_manifest, manifest_path)


async def crawl_all(
    settings: Settings, sources: list[str] | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    """Crawl sources concurrently and merge with the previous snapshot on failure."""
    sources = [s for s in (sources or settings.sources) if s in SCRAPERS]
    previous, _ = load_snapshot(settings)
    results = await asyncio.gather(*(crawl_source(settings, s, transport) for s in sources))

    merged: dict[str, Record] = {}
    reports: list[dict[str, Any]] = []
    stale: list[str] = []
    for source, (records, report) in zip(sources, results, strict=True):
        prev_for_source = [r for r in previous if r.source == source]
        if not report.ok:
            # Flag every coverage gap, including sources with no prior snapshot.
            stale.append(source)
        if not report.ok and prev_for_source:
            # Keep previously collected data for this source (stale > nothing).
            fresh_ids = {r.id for r in records}
            carried = [r for r in prev_for_source if r.id not in fresh_ids]
            records = records + carried
            report.notes.append(f"carried forward {len(carried)} records from previous snapshot")
        for rec in records:
            merged[rec.id] = sanitize_record(rec)
        reports.append(report.model_dump(mode="json"))
    # Sources not crawled this run are kept as-is.
    for rec in previous:
        if rec.source not in sources:
            merged.setdefault(rec.id, rec)

    records = sorted(merged.values(), key=lambda r: (r.source, r.kind, r.id))
    manifest = {
        "app_version": __version__,
        "collected_at": utcnow().isoformat(),
        "records": len(records),
        "stale_sources": stale,
        "reports": reports,
    }
    return {"records": records, "manifest": manifest}


def build_and_activate(settings: Settings, records: list[Record], embedder: Embedder | None) -> tuple[Path, dict[str, Any]]:
    settings.index_root.mkdir(parents=True, exist_ok=True)
    build_dir = new_build_dir(settings.index_root)
    meta = build_index(
        records,
        build_dir,
        embedder,
        backend=settings.vector_backend,
        ef_search=settings.chroma_ef_search,
        batch_size=settings.embedding_batch_size,
    )
    activate(settings.index_root, build_dir)
    return build_dir, meta


async def ingest(
    settings: Settings,
    embedder: Embedder | None,
    *,
    sources: list[str] | None = None,
    min_records: int = 20,
    require_all_sources: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Full refresh. Refuses to activate a suspiciously small dataset."""
    started = time.perf_counter()
    crawl = await crawl_all(settings, sources, transport)
    records: list[Record] = crawl["records"]
    manifest: dict[str, Any] = crawl["manifest"]
    if require_all_sources:
        coverage = source_coverage(Counter(f"{r.source}:{r.kind}" for r in records))
        successful = {r["source"] for r in manifest["reports"] if r["ok"]}
        gaps = sorted(set(coverage["missing_sources"]) | (REQUIRED_KINDS.keys() - successful))
        if gaps:
            raise RuntimeError(
                "release requires a successful crawl and property data from both sources; "
                f"unavailable or incomplete: {', '.join(gaps)}. Keeping the current snapshot and index. "
                "Retry later or use a source-approved data feed; anti-bot challenges are never bypassed"
            )
    if len(records) < min_records:
        raise RuntimeError(
            f"ingestion produced only {len(records)} records; keeping the current index. "
            "If the sites were unreachable, run `python -m app.cli doctor` to check this machine's connection"
        )
    write_snapshot(settings, records, manifest)
    build_dir, meta = await asyncio.to_thread(build_and_activate, settings, records, embedder)
    summary = {
        "records": len(records),
        "build": build_dir.name,
        "index": meta,
        "stale_sources": manifest["stale_sources"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    log.info("ingestion complete", extra=summary)
    return summary


def rebuild_from_snapshot(settings: Settings, embedder: Embedder | None) -> dict[str, Any]:
    records, _manifest = load_snapshot(settings)
    if not records:
        raise RuntimeError("no snapshot to index; run ingestion first")
    build_dir, meta = build_and_activate(settings, records, embedder)
    return {"records": len(records), "build": build_dir.name, "index": meta}
