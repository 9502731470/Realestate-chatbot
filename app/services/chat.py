"""Chat orchestration: retrieve -> (cache) -> generate (stream) -> fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.llm.openrouter import LLMError, LLMNotConfigured, LLMQuotaExceeded, StreamResult
from app.rag.prompt import build_messages, canned_answer, fallback_answer, source_cards
from app.rag.retriever import RetrievalResult
from app.services.metrics import CHAT_LATENCY, CHAT_REQUESTS, LLM_REQUESTS, LLM_TOKENS, LLM_TTFT, RETRIEVAL_LATENCY
from app.services.state import AppState
from app.textutil import normalize_for_search

log = logging.getLogger(__name__)

Event = dict[str, Any]


def _chunks(text: str, size: int = 24) -> list[str]:
    """Split canned/cached text into small pieces so the UI renders progressively."""
    words = text.split(" ")
    return [" ".join(words[i : i + size]) + (" " if i + size < len(words) else "") for i in range(0, len(words), size)]


def _cache_key(question: str, result: RetrievalResult) -> str:
    payload = {
        "q": normalize_for_search(question),
        "f": result.parsed.filters.to_dict(),
        "ids": [s.record.id for s in result.sources],
        "lang": result.parsed.language,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class ChatService:
    def __init__(self, state: AppState) -> None:
        self.state = state

    async def stream(
        self,
        message: str,
        history: list[dict[str, str]],
        previous_source_ids: list[str],
        request_id: str,
    ) -> AsyncIterator[Event]:
        started = time.perf_counter()
        state = self.state
        if not state.warmup_complete:  # just started (e.g. a free instance waking up)
            await state.wait_for_warmup(state.settings.warmup_wait_s)
        if not state.ready:
            busy = state.ingestion.running
            text = (
                "The knowledge base is being built from DarGlobal and Wasalt right now - please try again in a few minutes."
                if busy
                else "The knowledge base is not available yet. Please try again shortly."
            )
            yield {"event": "meta", "data": {"request_id": request_id, "sources": [], "intent": "unavailable"}}
            yield {"event": "token", "data": {"t": text}}
            yield {"event": "done", "data": {"mode": "unavailable", "latency_ms": 0}}
            CHAT_REQUESTS.labels("unavailable", "n/a").inc()
            return

        retriever = state.retriever
        assert retriever is not None
        user_history = [t["content"] for t in history if t["role"] == "user"]
        t0 = time.perf_counter()
        result = await asyncio.to_thread(retriever.retrieve, message, user_history, previous_source_ids)
        RETRIEVAL_LATENCY.observe(time.perf_counter() - t0)
        parsed = result.parsed

        yield {
            "event": "meta",
            "data": {
                "request_id": request_id,
                "intent": parsed.intent,
                "language": parsed.language,
                "filters": parsed.filters.to_dict(),
                "filters_text": parsed.filters.describe(),
                "total_matches": result.total_matches,
                "relaxed": result.relaxed,
                "sources": source_cards(result.sources),
                "retrieval_ms": result.timings_ms.get("total"),
            },
        }

        mode, model, notice = "llm", None, None
        ttft_ms: int | None = None
        usage: dict[str, int] = {}
        prompt_chars = 0
        collected: list[str] = []

        canned = canned_answer(result)
        cache_key = _cache_key(message, result) if not history else None
        cached = state.cache.get(cache_key) if cache_key else None

        if canned is not None:
            mode = "canned"
            for piece in _chunks(canned):
                collected.append(piece)
                yield {"event": "token", "data": {"t": piece}}
        elif cached is not None:
            mode, model = "cache", cached[1]
            for piece in _chunks(cached[0]):
                collected.append(piece)
                yield {"event": "token", "data": {"t": piece}}
        else:
            messages = build_messages(
                message,
                history,
                result,
                snapshot_date=state.snapshot_date(),
                max_context_chars=state.settings.max_context_chars,
                history_messages=state.settings.prompt_history_messages,
            )
            prompt_chars = sum(len(m["content"]) for m in messages)
            stream_result = StreamResult()
            t_llm = time.perf_counter()
            try:
                async for delta in state.llm.stream_chat(messages, stream_result):
                    if ttft_ms is None:
                        ttft = time.perf_counter() - t_llm
                        ttft_ms = round(ttft * 1000)
                        LLM_TTFT.observe(ttft)
                    collected.append(delta)
                    yield {"event": "token", "data": {"t": delta}}
                model = stream_result.model
                LLM_REQUESTS.labels(model or "unknown", "success").inc()
                raw_usage = stream_result.usage or {}
                usage = {
                    k: int(raw_usage[k]) for k in ("prompt_tokens", "completion_tokens") if isinstance(raw_usage.get(k), int | float)
                }
                for kind, value in usage.items():
                    LLM_TOKENS.labels(kind.replace("_tokens", "")).inc(value)
            except LLMError as exc:
                outcome = type(exc).__name__
                LLM_REQUESTS.labels(stream_result.model or "n/a", outcome).inc()
                reason = exc.reason
                if isinstance(exc, LLMQuotaExceeded):
                    notice = "quota"
                elif isinstance(exc, LLMNotConfigured):
                    notice = "no_key"
                else:
                    notice = "llm_error"
                if collected:
                    tail = "\n\n_(The answer was interrupted because the model stopped responding. Sources are listed below.)_"
                    collected.append(tail)
                    yield {"event": "token", "data": {"t": tail}}
                    mode = "partial"
                else:
                    mode = "retrieval-only"
                    text = fallback_answer(result, reason)
                    for piece in _chunks(text):
                        collected.append(piece)
                        yield {"event": "token", "data": {"t": piece}}
                log.warning("LLM unavailable; served fallback", extra={"reason": reason, "mode": mode})

        answer = "".join(collected).strip()
        if mode == "llm" and answer and cache_key:
            state.cache.set(cache_key, (answer, model))

        latency = time.perf_counter() - started
        CHAT_LATENCY.observe(latency)
        CHAT_REQUESTS.labels(mode, parsed.intent).inc()
        log.info(
            "chat turn",
            extra={
                "intent": parsed.intent,
                "mode": mode,
                "model": model,
                "sources": len(result.sources),
                "total_matches": result.total_matches,
                "latency_ms": round(latency * 1000),
                "ttft_ms": ttft_ms,
                "prompt_chars": prompt_chars,
                **usage,
                "question_chars": len(message),
                **({"question": message} if state.settings.log_prompts else {}),
            },
        )
        yield {
            "event": "done",
            "data": {
                "mode": mode,
                "model": model,
                "notice": notice,
                "latency_ms": round(latency * 1000),
                "ttft_ms": ttft_ms,
                "tokens": usage or None,
            },
        }
