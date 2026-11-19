"""HTTP routes."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.api.schemas import ChatRequest, ChatResponse
from app.coverage import source_coverage
from app.rag.query import QueryFilters
from app.services.chat import ChatService
from app.services.metrics import RATE_LIMITED, REGISTRY
from app.services.state import AppState
from app.textutil import truncate
from app.vocab import canonical_property_type

router = APIRouter()
KEEPALIVE_S = 10.0


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("/api/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe: the process is up."""
    return {"status": "ok", "version": __version__}


@router.get("/api/ready", tags=["ops"])
async def ready(request: Request) -> JSONResponse:
    """Readiness probe: an index is loaded and answers can be served."""
    state = _state(request)
    body = {"ready": state.ready, "ingestion": state.ingestion.status.get("state")}
    return JSONResponse(body, status_code=200 if state.ready else 503)


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _windows_working_set_mb() -> float | None:
    """Current resident bytes from Windows, without an extra dependency."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        memory_info = kernel.K32GetProcessMemoryInfo
        memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        memory_info.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if memory_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / 2**20, 1)
    except (OSError, AttributeError):
        pass  # Memory metrics must not make /api/stats unavailable.
    return None


def process_memory() -> dict[str, float | None]:
    """Resident memory on Linux/Windows and the Linux container limit, in MB."""
    if os.name == "nt":
        return {"rss_mb": _windows_working_set_mb(), "limit_mb": None}
    rss = None
    statm = _read("/proc/self/statm")
    if statm:
        rss = round(int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    limit = None
    raw = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if raw and raw.isdigit() and int(raw) < 2**50:  # "max" / huge values mean no limit
        limit = round(int(raw) / 2**20)
    return {"rss_mb": rss, "limit_mb": limit}


@router.get("/api/stats", tags=["ops"])
async def stats(request: Request) -> dict[str, Any]:
    """Data coverage, freshness, crawl reports and model status."""
    state = _state(request)
    key_status = await state.llm.key_status()
    index_stats = state.index.stats() if state.index else None
    reports = [
        {k: r.get(k) for k in ("source", "ok", "records", "requests", "errors", "blocked_by_robots", "notes", "status_codes")}
        for r in state.manifest.get("reports", [])
    ]
    return {
        "version": __version__,
        "ready": state.ready,
        "coverage": source_coverage(
            (index_stats or {}).get("counts", {}), state.manifest.get("stale_sources", [])
        ),
        "snapshot": {
            "collected_at": state.manifest.get("collected_at"),
            "stale_sources": state.manifest.get("stale_sources", []),
            "crawl_reports": reports,
        },
        "index": index_stats,
        "index_age_hours": round(state.index_age_hours() or 0, 2) if state.index else None,
        "ingestion": state.ingestion.status,
        "llm": {**state.llm.status(), "free_quota": key_status},
        "cache": {"entries": len(state.cache), "hits": state.cache.hits, "misses": state.cache.misses},
        "uptime_since": state.started_at.isoformat(),
        "process": process_memory(),
    }


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.post("/api/chat", tags=["chat"], response_model=None)
async def chat(request: Request, body: ChatRequest) -> Response:
    """Ask a question. Streams Server-Sent Events by default (``stream=false`` for JSON).

    Events: ``meta`` (intent, filters, sources), ``token`` (answer text deltas),
    ``done`` (mode, model, latency), ``error``.
    """
    state = _state(request)
    settings = state.settings
    if len(body.message) > settings.max_message_chars:
        raise HTTPException(413, f"message too long (max {settings.max_message_chars} characters)")
    decision = state.rate_limiter.check(_client_id(request))
    if not decision.allowed:
        RATE_LIMITED.inc()
        return JSONResponse(
            {"detail": f"Rate limit: {decision.reason}. Please retry in {decision.retry_after_s}s."},
            status_code=429,
            headers={"Retry-After": str(decision.retry_after_s)},
        )
    history = [t.model_dump() for t in body.history][-settings.max_history_messages:]
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    service = ChatService(state)
    events = service.stream(body.message, history, body.previous_source_ids, request_id)

    if not body.stream:
        answer: list[str] = []
        meta: dict[str, Any] = {}
        done: dict[str, Any] = {}
        async for ev in events:
            if ev["event"] == "meta":
                meta = ev["data"]
            elif ev["event"] == "token":
                answer.append(ev["data"]["t"])
            elif ev["event"] == "done":
                done = ev["data"]
        return JSONResponse(
            ChatResponse(
                answer="".join(answer).strip(),
                mode=done.get("mode", "unknown"),
                model=done.get("model"),
                intent=meta.get("intent"),
                filters=meta.get("filters") or {},
                total_matches=meta.get("total_matches"),
                relaxed=meta.get("relaxed") or [],
                sources=meta.get("sources") or [],
                latency_ms=done.get("latency_ms"),
            ).model_dump()
        )

    async def gen() -> AsyncIterator[bytes]:
        # While a free model is queued nothing flows for a while; some proxies
        # (Render, Cloud Run, HF) close idle connections, so send SSE comment
        # keep-alives every KEEPALIVE_S. The browser parser ignores comments.
        iterator = events.__aiter__()
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(iterator.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=KEEPALIVE_S)
                if not done:
                    if await request.is_disconnected():
                        break
                    yield b": keep-alive\n\n"
                    continue
                try:
                    ev = pending.result()
                except StopAsyncIteration:
                    break
                finally:
                    pending = None
                yield _sse(ev["event"], ev["data"])
        except Exception:  # never leak stack traces to clients
            logging.getLogger(__name__).exception("chat stream failed")
            yield _sse("error", {"message": "Something went wrong while answering. Please try again."})
        finally:
            if pending is not None:
                pending.cancel()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/api/search", tags=["data"])
async def search(
    request: Request,
    q: str | None = Query(None, max_length=500, description="Free text; filters are extracted automatically"),
    purpose: str | None = Query(None, pattern="^(sale|rent)$"),
    type: str | None = Query(None, description="apartment, villa, land, ..."),
    city: str | None = None,
    district: str | None = None,
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    bedrooms: int | None = Query(None, ge=0, le=20),
    source: str | None = Query(None, pattern="^(darglobal|wasalt)$"),
    kind: str | None = Query(None, pattern="^(listing|project)$"),
    sort: str | None = Query(None, pattern="^(price_asc|price_desc|area_desc|newest)$"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Structured search over listings and projects (transparent, no LLM)."""
    state = _state(request)
    if not state.ready or state.retriever is None or state.index is None:
        raise HTTPException(503, "index not ready")
    parsed_filters = state.retriever.parser.parse(q).filters if q else QueryFilters()
    overrides = {
        "purpose": purpose,
        "property_types": [canonical_property_type(type)] if type and canonical_property_type(type) else None,
        "cities": [city] if city else None,
        "districts": [district] if district else None,
        "min_price": min_price,
        "max_price": max_price,
        "bedrooms": bedrooms,
        "sources": [source] if source else None,
        "sort": sort,
    }
    f = replace(parsed_filters, **{k: v for k, v in overrides.items() if v is not None})
    kinds = [kind] if kind else (f.kinds or ["listing", "project"])
    qvec = state.index.embed_query(q) if q else None
    res = state.index.search_records(f, kinds=kinds, qvec=qvec, limit=limit)
    from app.rag.retriever import Source

    return {
        "filters": f.to_dict(),
        "total": res.total,
        "results": [Source(n=i + 1, record=r).card() for i, r in enumerate(res.records)],
    }


@router.get("/api/records/{record_id:path}", tags=["data"])
async def record(request: Request, record_id: str) -> dict[str, Any]:
    state = _state(request)
    rec = state.record(record_id)
    if rec is None:
        raise HTTPException(404, "record not found")
    data = rec.model_dump(mode="json")
    data["text"] = truncate(data["text"], 6000)
    return data


@router.post("/api/admin/refresh", tags=["ops"], status_code=202)
async def refresh(
    request: Request,
    background: BackgroundTasks,
    x_admin_token: str | None = Header(None),
) -> dict[str, Any]:
    """Re-scrape both sources and hot-swap the index (requires ``X-Admin-Token``)."""
    state = _state(request)
    expected = state.settings.admin_token.get_secret_value() if state.settings.admin_token else ""
    if not expected:
        raise HTTPException(404, "admin endpoint disabled (set ADMIN_TOKEN to enable)")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "invalid admin token")
    if state.ingestion.running:
        return {"started": False, "detail": "an ingestion is already running", "status": state.ingestion.status}
    background.add_task(state.ingestion.run, "admin request")
    return {"started": True}


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
