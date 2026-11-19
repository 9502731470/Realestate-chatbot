"""Process-wide application state: index (hot-swappable), LLM, ingestion jobs."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.llm.openrouter import OpenRouterClient
from app.models import Record
from app.rag.embeddings import Embedder, load_embedder
from app.rag.index import IndexFormatError, SearchIndex, current_build_dir
from app.rag.retriever import Retriever
from app.scraping.pipeline import ingest, load_snapshot, rebuild_from_snapshot
from app.services.cache import TTLCache
from app.services.metrics import INDEX_RECORDS, INGESTIONS
from app.services.ratelimit import RateLimiter

log = logging.getLogger(__name__)


class IngestionManager:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self._lock = asyncio.Lock()
        self.status: dict[str, Any] = {"state": "idle"}

    @property
    def running(self) -> bool:
        return self._lock.locked()

    async def run(self, reason: str) -> bool:
        if self._lock.locked():
            return False
        async with self._lock:
            self.status = {"state": "running", "reason": reason, "started_at": datetime.now(UTC).isoformat()}
            log.info("ingestion started", extra={"reason": reason})
            try:
                summary = await ingest(self.state.settings, self.state.embedder)
                await asyncio.to_thread(self.state.load_index)
                self.status = {
                    "state": "succeeded",
                    "reason": reason,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "summary": {k: summary[k] for k in ("records", "build", "stale_sources", "seconds")},
                }
                INGESTIONS.labels("success").inc()
            except Exception as exc:
                log.exception("ingestion failed")
                self.status = {
                    "state": "failed",
                    "reason": reason,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "error": repr(exc)[:300],
                }
                INGESTIONS.labels("failure").inc()
            return True

    async def reindex(self, reason: str) -> bool:
        """Rebuild the index (and its vector store) from the current snapshot,
        without crawling; the active index keeps serving until the swap."""
        if self._lock.locked():
            return False
        async with self._lock:
            self.status = {"state": "running", "reason": reason, "started_at": datetime.now(UTC).isoformat()}
            log.info("reindex started", extra={"reason": reason})
            try:
                summary = await asyncio.to_thread(rebuild_from_snapshot, self.state.settings, self.state.embedder)
                await asyncio.to_thread(self.state.load_index)
                self.status = {
                    "state": "succeeded",
                    "reason": reason,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "summary": {k: summary[k] for k in ("records", "build")},
                }
            except Exception as exc:
                log.exception("reindex failed")
                self.status = {
                    "state": "failed",
                    "reason": reason,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "error": repr(exc)[:300],
                }
            return True

    async def periodic(self, interval_hours: float) -> None:
        if interval_hours <= 0:
            return
        while True:
            await asyncio.sleep(interval_hours * 3600)
            await self.run("scheduled refresh")


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder: Embedder | None = None
        self.index: SearchIndex | None = None
        self.retriever: Retriever | None = None
        self.manifest: dict[str, Any] = {}
        self.llm = OpenRouterClient(settings)
        self.cache: TTLCache[tuple[str, str | None]] = TTLCache(settings.cache_max_entries, settings.cache_ttl_s)
        self.rate_limiter = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_per_day)
        self.ingestion = IngestionManager(self)
        self._swap_lock = threading.Lock()
        self.started_at = datetime.now(UTC)
        self.warmup_complete = False

    def load_embedder(self) -> None:
        self.embedder = load_embedder(
            self.settings.embedding_model,
            self.settings.model_cache_dir,
            enabled=self.settings.enable_dense,
            threads=self.settings.embedding_threads,
            local_only=self.settings.embedding_local_only,
        )

    def _make_retriever(self, index: SearchIndex) -> Retriever:
        floor = self.settings.min_passage_similarity
        if floor is None:  # each embedding model has its own cosine scale
            floor = index.embedder.profile.min_similarity if index.embedder else 0.25
        return Retriever(
            index,
            top_k_listings=self.settings.top_k_listings,
            top_k_passages=self.settings.top_k_passages,
            min_passage_similarity=floor,
        )

    def load_index(self) -> bool:
        build = current_build_dir(self.settings.index_root)
        if build is None:
            return False
        current = self.index
        if current is not None and current.dir == build and self.embedder is not None and not current.dense_enabled:
            # Same data, model now loaded (start-up warm-up): add semantic search in
            # place instead of loading every record a second time.
            if current.attach_embedder(self.embedder):
                retriever = self._make_retriever(current)
                with self._swap_lock:
                    self.retriever = retriever
                self.cache.clear()  # answers cached before semantic search was on
                log.info("semantic search enabled", extra={"build": build.name, "model": self.embedder.name})
            return True
        try:
            new_index = SearchIndex(build, self.embedder)
        except IndexFormatError:
            # Index written by an older version (e.g. a persisted volume): rebuild it
            # from the format-stable snapshot instead of failing.
            log.warning("index format changed; rebuilding from snapshot", extra={"build": build.name})
            rebuild_from_snapshot(self.settings, self.embedder)
            build = current_build_dir(self.settings.index_root)
            assert build is not None
            new_index = SearchIndex(build, self.embedder)
        new_retriever = self._make_retriever(new_index)
        _records, manifest = load_snapshot(self.settings)
        with self._swap_lock:
            old = self.index
            self.index, self.retriever, self.manifest = new_index, new_retriever, manifest
        if old is None or old.dir != new_index.dir:
            self.cache.clear()  # new data: cached answers may be outdated
        if old is not None:
            # In-flight requests may still be searching the old index: close it later.
            timer = threading.Timer(30.0, old.close)
            timer.daemon = True
            timer.start()
        for key, count in new_index.stats()["counts"].items():
            source, kind = key.split(":", 1)
            INDEX_RECORDS.labels(source, kind).set(count)
        log.info("index loaded", extra={"build": build.name, "records": len(new_index.records)})
        return True

    def vectors_outdated(self) -> bool:
        """True when the embedding model is loaded but the active build has no
        usable vector store for it (built without embeddings, with another model,
        or with a different ``VECTOR_BACKEND``): a re-embed from the snapshot fixes it."""
        if self.embedder is None or self.index is None or not self.index.records:
            return False
        meta = self.index.meta
        return (
            not self.index.dense_enabled
            or meta.get("embedding_model") != self.embedder.name
            or meta.get("vector_backend") != self.settings.vector_backend
        )

    async def wait_for_warmup(self, max_wait_s: float) -> None:
        """Give a request that arrives during start-up a moment for the index and
        embedding model to load. Polls a flag instead of using an asyncio.Event,
        which would be bound to one event loop (the state outlives loops in the CLI/tests)."""
        deadline = time.monotonic() + max_wait_s
        while not self.warmup_complete and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.2)

    @property
    def ready(self) -> bool:
        return self.index is not None and self.retriever is not None

    def index_age_hours(self) -> float | None:
        if not self.index:
            return None
        built = self.index.meta.get("built_at")
        if not built:
            return None
        return (datetime.now(UTC) - datetime.fromisoformat(built)).total_seconds() / 3600

    def snapshot_date(self) -> str:
        collected = self.manifest.get("collected_at") or (self.index.meta.get("built_at") if self.index else None)
        return str(collected)[:10] if collected else "unknown"

    def record(self, record_id: str) -> Record | None:
        return self.index.records.get(record_id) if self.index else None
