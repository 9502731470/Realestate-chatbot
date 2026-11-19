"""OpenRouter client tuned for free (``:free``) models.

Reliability features:
* live discovery of the *current* free roster (it changes weekly) with an
  operator-defined preference order; never routes to a paid model;
* OpenRouter server-side fallback (``models`` array) plus client-side retry
  across further model groups with backoff;
* daily free-quota detection -> circuit breaker until 00:00 UTC so we stop
  burning requests and serve retrieval-only answers instead;
* streaming (SSE) with mid-stream error handling and ``<think>`` stripping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from app import __version__
from app.config import Settings
from app.net import tls_verify

log = logging.getLogger(__name__)

_EXCLUDE_MODEL = re.compile(r"(?:^|[/\-_.])(?:code|coder|guard|safety|embed|embedding|moderation|ocr|tts|whisper|audio)(?:[/\-_.:]|$)", re.I)
_PARAM_SIZE = re.compile(r"(\d+(?:\.\d+)?)b(?:\b|-|_|:)", re.I)
_DAILY = re.compile(r"per[-_ ]?day|daily|free-models-per-day|requests per day", re.I)


class LLMError(Exception):
    reason = "language model error"


class LLMNotConfigured(LLMError):
    reason = "no OpenRouter API key configured"


class LLMAuthError(LLMError):
    reason = "OpenRouter rejected the API key or the account has no credit"


class LLMQuotaExceeded(LLMError):
    reason = "daily free-model quota reached (resets 00:00 UTC)"


class LLMUnavailable(LLMError):
    reason = "all free models are busy or unavailable"


class _Transient(Exception):
    pass


@dataclass
class StreamResult:
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    attempts: int = 0
    tried: list[str] = field(default_factory=list)


class ThinkFilter:
    """Removes ``<think>...</think>`` blocks from a token stream (tags may span chunks)."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.buf += text
        out: list[str] = []
        while self.buf:
            if self.inside:
                end = self.buf.find(self.CLOSE)
                if end == -1:
                    self.buf = self.buf[-(len(self.CLOSE) - 1):]
                    break
                self.buf = self.buf[end + len(self.CLOSE):]
                self.inside = False
            else:
                start = self.buf.find(self.OPEN)
                if start == -1:
                    keep = 0
                    for i in range(len(self.OPEN) - 1, 0, -1):
                        if self.buf.endswith(self.OPEN[:i]):
                            keep = i
                            break
                    out.append(self.buf[: len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                out.append(self.buf[:start])
                self.buf = self.buf[start + len(self.OPEN):]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


ROUTER_PREFIX = "openrouter/"


def must_think(model_id: str, info: dict[str, Any] | None = None) -> bool:
    """True if the model can't be told to skip its hidden reasoning.

    That is the case when OpenRouter marks its reasoning ``mandatory`` (it rejects
    ``effort: "none"``), when its supported efforts don't include "none", and for
    routers (openrouter/free), which may hand the request to such a model.
    """
    if model_id.startswith(ROUTER_PREFIX):
        return True
    reasoning = (info or {}).get("reasoning")
    if not isinstance(reasoning, dict):
        return False  # not a reasoning model
    if reasoning.get("mandatory"):
        return True
    efforts = reasoning.get("supported_efforts")
    return bool(efforts) and "none" not in efforts


def plan_groups(chain: list[str], info: dict[str, dict[str, Any]], size: int) -> list[tuple[list[str], bool]]:
    """Split the chain into OpenRouter fallback groups of at most ``size`` models.

    One request carries one ``reasoning`` setting for its whole group, so models that
    must think never share a group with models that are told not to.
    Returns ``(models, must_think)`` pairs in chain order.
    """
    groups: list[tuple[list[str], bool]] = []
    for mid in chain:
        thinks = must_think(mid, info.get(mid))
        if groups and groups[-1][1] == thinks and len(groups[-1][0]) < max(1, size):
            groups[-1][0].append(mid)
        else:
            groups.append(([mid], thinks))
    return groups


def rank_free_models(models: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    """Return usable free model ids: preferred ones first (in configured order), then
    the rest; models that must think go after all others, and the router comes last."""
    candidates: dict[str, tuple[int, float]] = {}
    info: dict[str, dict[str, Any]] = {}
    for model in models:
        mid = str(model.get("id") or "")
        if not (mid.endswith(":free") or mid == "openrouter/free"):
            continue  # never risk routing to a paid model
        arch = model.get("architecture") or {}
        if "text" not in (arch.get("output_modalities") or ["text"]) or "text" not in (arch.get("input_modalities") or ["text"]):
            continue
        if _EXCLUDE_MODEL.search(mid):
            continue
        ctx = int(model.get("context_length") or 0)
        if ctx and ctx < 8000:
            continue
        sizes = [float(s) for s in _PARAM_SIZE.findall(mid)]
        size = max(sizes) if sizes else 0.0
        if sizes and size < 7:
            continue
        candidates[mid] = (ctx, size)
        info[mid] = model
    ranked = [m for m in preferred if m in candidates and m != "openrouter/free"]
    rest = sorted(
        (m for m in candidates if m not in ranked and m != "openrouter/free"),
        key=lambda m: (-(candidates[m][1] or 30.0), -candidates[m][0], m),
    )
    ranked += rest
    # Thinking models answer slower and can run out of tokens: last resort only.
    thinkers = [m for m in ranked if must_think(m, info[m])]
    ranked = [m for m in ranked if m not in thinkers] + thinkers
    if "openrouter/free" in candidates:
        ranked.append("openrouter/free")
    return ranked


def next_utc_midnight(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


class OpenRouterClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        key = settings.openrouter_api_key.get_secret_value().strip() if settings.openrouter_api_key else ""
        self._key = key
        self._http = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            timeout=httpx.Timeout(settings.llm_read_timeout_s, connect=settings.llm_connect_timeout_s),
            transport=transport,
            verify=tls_verify(settings.tls_trust),
            headers={
                "HTTP-Referer": settings.public_url or "https://huggingface.co/spaces",
                "X-Title": "DarGlobal x Wasalt Property Assistant",
                "User-Agent": f"dg-wasalt-assistant/{__version__}",
            },
        )
        self._semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))
        self._chain: list[str] = []
        self._chain_at = 0.0
        self._model_info: dict[str, dict[str, Any]] = {}  # id -> /models entry (reasoning capabilities)
        self._chain_lock = asyncio.Lock()
        self.quota_exhausted_until: datetime | None = None
        self.last_model: str | None = None
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None
        self._key_status: dict[str, Any] | None = None
        self._key_status_at = 0.0

    @property
    def configured(self) -> bool:
        return bool(self._key)

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---------------------------------------------------------- discovery
    async def refresh_models(self, force: bool = False) -> list[str]:
        ttl = self.settings.llm_model_refresh_hours * 3600
        async with self._chain_lock:
            if self._chain and not force and time.monotonic() - self._chain_at < ttl:
                return self._chain
            preferred = list(self.settings.llm_models)
            chain = preferred
            if self.settings.llm_discover_free_models:
                try:
                    resp = await self._http.get("/models", timeout=20.0)
                    resp.raise_for_status()
                    roster = [m for m in resp.json().get("data") or [] if isinstance(m, dict)]
                    ranked = rank_free_models(roster, preferred)
                    if ranked:
                        chain = ranked
                        self._model_info = {str(m.get("id")): m for m in roster if m.get("id") in ranked}
                    log.info("free model roster refreshed", extra={"available": len(ranked), "top": ranked[:5]})
                except (httpx.HTTPError, ValueError) as exc:
                    log.warning("model discovery failed; using configured list", extra={"error": repr(exc)[:200]})
            self._chain, self._chain_at = chain, time.monotonic()
            return chain

    async def key_status(self) -> dict[str, Any] | None:
        """Remaining free-model requests today (cached for 5 minutes)."""
        if not self.configured:
            return None
        if self._key_status is not None and time.monotonic() - self._key_status_at < 300:
            return self._key_status
        try:
            resp = await self._http.get("/key", headers=self._auth(), timeout=10.0)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                self._key_status = {
                    k: data.get(k)
                    for k in ("free_model_daily_requests", "is_free_tier", "limit_remaining", "usage")
                    if k in data
                }
                self._key_status_at = time.monotonic()
        except (httpx.HTTPError, ValueError):
            pass
        return self._key_status

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "model_chain": self._chain[:6],
            # Models that can't skip hidden reasoning: tried last, with extra tokens.
            "thinking_models": [m for m in self._chain if must_think(m, self._model_info.get(m))][:6],
            "last_model": self.last_model,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error": self.last_error,
            "quota_exhausted_until": self.quota_exhausted_until.isoformat() if self.quota_exhausted_until else None,
        }

    # ------------------------------------------------------------ chatting
    def _check_available(self) -> None:
        if not self.configured:
            raise LLMNotConfigured()
        if self.quota_exhausted_until:
            if datetime.now(UTC) < self.quota_exhausted_until:
                raise LLMQuotaExceeded()
            self.quota_exhausted_until = None

    async def stream_chat(self, messages: list[dict[str, str]], result: StreamResult) -> AsyncIterator[str]:
        self._check_available()
        chain = await self.refresh_models()
        groups = plan_groups(chain, self._model_info, self.settings.llm_models_per_request)
        groups = groups[: self.settings.llm_max_attempts]
        last_error = "no models available"
        async with self._semaphore:
            for attempt, (group, thinks) in enumerate(groups):
                result.attempts = attempt + 1
                result.tried.extend(group)
                result.finish_reason = None
                yielded = False
                try:
                    async for delta in self._stream_once(group, messages, result, thinks=thinks):
                        yielded = True
                        yield delta
                    if not yielded:
                        why = " (token limit reached while the model was thinking)" if result.finish_reason == "length" else ""
                        raise _Transient("empty completion" + why)
                    self.last_model = result.model
                    self.last_success_at = datetime.now(UTC)
                    self.last_error = None
                    return
                except LLMQuotaExceeded:
                    self.quota_exhausted_until = next_utc_midnight()
                    self.last_error = LLMQuotaExceeded.reason
                    raise
                except LLMAuthError as exc:
                    self.last_error = str(exc) or LLMAuthError.reason
                    raise
                except (_Transient, httpx.HTTPError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"[:300]
                    self.last_error = last_error
                    log.warning("LLM attempt failed", extra={"models": group, "attempt": attempt + 1, "error": last_error})
                    if yielded:
                        raise LLMUnavailable(f"stream interrupted: {last_error}") from exc
                    if attempt + 1 < len(groups):
                        await asyncio.sleep(min(1.5 * (attempt + 1), 4.0))
        raise LLMUnavailable(last_error)

    def _payload(self, models: list[str], messages: list[dict[str, str]], *, thinks: bool) -> dict[str, Any]:
        """Hidden reasoning shares ``max_tokens`` with the answer, so a model that thinks
        on a tight budget can return nothing. Models that can skip it are told to
        (faster, and the whole budget goes to the answer); models that must think get
        headroom for thinking *and* the answer. Reasoning text is never returned."""
        if thinks:
            reasoning: dict[str, Any] = {"exclude": True}
            max_tokens = self.settings.llm_max_tokens + self.settings.llm_thinking_headroom_tokens
        else:
            reasoning = {"effort": "none", "exclude": True}
            max_tokens = self.settings.llm_max_tokens
        return {
            "model": models[0],
            "models": models,
            "messages": messages,
            "stream": True,
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens,
            "reasoning": reasoning,
            "usage": {"include": True},  # real prompt/completion token counts in the last chunk
        }

    async def _stream_once(
        self, models: list[str], messages: list[dict[str, str]], result: StreamResult, *, thinks: bool = False
    ) -> AsyncIterator[str]:
        payload = self._payload(models, messages, thinks=thinks)
        think = ThinkFilter()
        started = time.monotonic()
        # Thinking happens before the first visible token: allow it more time.
        deadline = self.settings.llm_thinking_first_token_timeout_s if thinks else self.settings.llm_first_token_timeout_s
        answered = False
        async with self._http.stream("POST", "/chat/completions", json=payload, headers=self._auth()) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")
                self._raise_for_error(resp.status_code, body)
            async for line in resp.aiter_lines():
                # OpenRouter sends keep-alive comments while a model is queued, so the
                # HTTP read timeout never fires; enforce a time-to-first-token deadline
                # ourselves and fail over to the next model group instead of waiting.
                if not answered and deadline and time.monotonic() - started > deadline:
                    raise _Transient(f"no first token within {deadline:.0f}s")
                if not line or line.startswith(":"):
                    continue  # SSE comment / keep-alive ("OPENROUTER PROCESSING")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    err = chunk["error"]
                    self._raise_for_error(int(err.get("code") or 500), json.dumps({"error": err}))
                result.model = chunk.get("model") or result.model
                if chunk.get("usage"):
                    result.usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason") == "error":
                        raise _Transient("upstream error mid-stream")
                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        clean = think.feed(delta)
                        if clean:
                            answered = True
                            yield clean
        tail = think.flush()
        if tail:
            yield tail

    @staticmethod
    def _raise_for_error(status: int, body: str) -> None:
        message = body[:500]
        try:
            err = json.loads(body).get("error") or {}
            message = str(err.get("message") or message)
            raw_meta = err.get("metadata") or {}
            message += " " + json.dumps(raw_meta)[:300]
        except (ValueError, AttributeError):
            pass
        if status == 429 and _DAILY.search(message):
            raise LLMQuotaExceeded(message)
        if status == 403 and re.search(r"moderat|flagged", message, re.I):
            raise _Transient(f"HTTP 403 moderation: {message[:200]}")
        if status in (401, 403):
            raise LLMAuthError(f"{status}: {message[:200]}")
        if status == 402:
            raise LLMAuthError(f"402: {message[:200]}")
        raise _Transient(f"HTTP {status}: {message[:200]}")
