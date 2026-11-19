"""FastAPI application factory, lifespan management and middleware."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import router
from app.config import Settings, get_settings
from app.logging_setup import configure_logging, request_id_var
from app.services.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.services.state import AppState

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; "
    "frame-ancestors 'self' https://huggingface.co https://*.hf.space"
)


def _should_ingest(state: AppState) -> bool:
    mode = state.settings.ingest_on_startup
    if mode == "never":
        return False
    if mode == "always":
        return True
    if not state.ready:
        return True
    if mode == "if-stale":
        age = state.index_age_hours()
        return age is not None and age > state.settings.stale_after_hours
    return False


async def _warm_up(state: AppState) -> None:
    """Runs after the server is accepting traffic, so health checks pass immediately
    even on tiny instances (e.g. 0.1 vCPU), where loading takes a while.

    1. load the existing index with keyword (BM25) search;
    2. load the embedding model and hot-swap the index to hybrid search;
    3. then, if the data is missing or stale, refresh it in the background;
       otherwise, if the vector store is missing or was built for another
       backend/model, re-embed the snapshot in the background.
    """
    try:
        await asyncio.to_thread(state.load_index)
    except Exception:
        log.exception("failed to load existing index; a fresh ingestion will be attempted")
    try:
        await asyncio.to_thread(state.load_embedder)
        if state.embedder is not None and state.index is not None:
            await asyncio.to_thread(state.load_index)
    except Exception:
        log.exception("embedding warm-up failed; continuing with keyword search")
    finally:
        state.warmup_complete = True
    if _should_ingest(state):
        await state.ingestion.run("startup")
    elif state.vectors_outdated():
        if state.settings.reembed_if_outdated:
            await state.ingestion.reindex("vector store missing or outdated")
        else:
            log.warning(
                "the index's vectors don't match EMBEDDING_MODEL/VECTOR_BACKEND; semantic search is off "
                "until the image is rebuilt (REEMBED_IF_OUTDATED=false)",
                extra={"model": state.settings.embedding_model, "backend": state.settings.vector_backend},
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: AppState = app.state.app_state
    settings = state.settings
    tasks: list[asyncio.Task] = []
    if state.llm.configured:
        tasks.append(asyncio.create_task(state.llm.refresh_models()))
    else:
        log.warning("OPENROUTER_API_KEY not set: answers will be retrieval-only until it is configured")
    tasks.append(asyncio.create_task(_warm_up(state)))
    tasks.append(asyncio.create_task(state.ingestion.periodic(settings.refresh_interval_hours)))
    log.info("service started", extra={"version": __version__, "ready": state.ready})
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await state.llm.aclose()
        if state.index:
            state.index.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="DarGlobal × Wasalt Property Assistant API",
        version=__version__,
        description=(
            "Retrieval-augmented chatbot over public data from darglobal.co.uk and wasalt.sa. "
            "Answers are grounded in scraped records and cite their sources."
        ),
        lifespan=lifespan,
    )
    app.state.app_state = AppState(settings)

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def observability(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        route = request.url.path if request.url.path.startswith("/api") else "static"
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            HTTP_LATENCY.labels(route).observe(time.perf_counter() - started)
            HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
            response.headers["Content-Security-Policy"] = CSP
        return response

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
