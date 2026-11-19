"""A polite, resilient async HTTP client for crawling.

* robots.txt compliance (RFC 9309 semantics via Protego, incl. wildcards and
  Crawl-delay) checked before every request;
* per-host rate limiting (minimum interval + concurrency cap);
* retries with exponential backoff + jitter, honouring ``Retry-After``;
* a per-host circuit breaker that stops hammering a host that keeps failing;
* anti-bot challenge detection: we *stop* rather than try to evade it;
* response size cap and transparent handling of gzipped sitemaps.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import random
import ssl
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
from protego import Protego

from app.net import describe_error

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}
# Headers that describe the wire encoding of the ORIGINAL body; they must not
# be copied onto a response whose body has already been decoded.
_ENCODING_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


class FetchError(Exception):
    def __init__(self, url: str, reason: str, status: int | None = None):
        super().__init__(f"{reason} ({status}) for {url}" if status else f"{reason} for {url}")
        self.url = url
        self.reason = reason
        self.status = status


class NotFound(FetchError):
    pass


class RobotsDisallowed(FetchError):
    pass


class HostBlocked(FetchError):
    """The host is challenging/blocking us or has failed too often: stop."""


@dataclass
class HostPolicy:
    min_interval_s: float = 1.0
    concurrency: int = 1


@dataclass
class _HostState:
    policy: HostPolicy
    semaphore: asyncio.Semaphore
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_request: float = 0.0
    consecutive_failures: int = 0
    abandoned_reason: str | None = None
    robots: Protego | None = None
    robots_loaded: bool = False
    robots_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _looks_like_challenge(resp: httpx.Response) -> bool:
    if resp.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    if resp.status_code in (403, 503) and "text/html" in resp.headers.get("content-type", ""):
        head = resp.text[:4000].lower()
        markers = ("just a moment", "cf-chl", "challenge-platform", "attention required", "captcha")
        return any(m in head for m in markers)
    return False


class PoliteClient:
    def __init__(
        self,
        *,
        user_agent: str,
        default_policy: HostPolicy | None = None,
        host_policies: dict[str, HostPolicy] | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        max_bytes: int = 6_000_000,
        failure_budget: int = 12,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Any = asyncio.sleep,
        verify: ssl.SSLContext | bool = True,
    ) -> None:
        self.user_agent = user_agent
        self.robots_token = user_agent.split("/")[0].strip()
        self.default_policy = default_policy or HostPolicy()
        self.host_policies = host_policies or {}
        self.max_retries = max_retries
        self.max_bytes = max_bytes
        self.failure_budget = failure_budget
        self._sleep = sleep
        self._hosts: dict[str, _HostState] = {}
        self.stats: Counter[str] = Counter()
        self.status_counts: Counter[str] = Counter()
        self.robots_problems: dict[str, str] = {}  # host -> why its robots.txt couldn't be read
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": user_agent,
                "Accept-Language": "en-US,en;q=0.9,ar;q=0.6",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            follow_redirects=True,
            transport=transport,
            verify=verify,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def __aenter__(self) -> PoliteClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ hosts
    def _state(self, host: str) -> _HostState:
        state = self._hosts.get(host)
        if state is None:
            policy = self.host_policies.get(host, self.default_policy)
            state = _HostState(policy=policy, semaphore=asyncio.Semaphore(max(1, policy.concurrency)))
            self._hosts[host] = state
        return state

    def host_status(self) -> dict[str, dict[str, Any]]:
        return {
            host: {
                "abandoned": st.abandoned_reason,
                "consecutive_failures": st.consecutive_failures,
                "min_interval_s": st.policy.min_interval_s,
            }
            for host, st in self._hosts.items()
        }

    # ----------------------------------------------------------------- robots
    async def _load_robots(self, scheme: str, host: str, state: _HostState) -> None:
        async with state.robots_lock:
            if state.robots_loaded:
                return
            url = f"{scheme}://{host}/robots.txt"
            try:
                resp = await self._request(url, state, check_robots=False, retries=1)
                state.robots = Protego.parse(resp.text)
                delay = state.robots.crawl_delay(self.robots_token)
                if delay:
                    state.policy = HostPolicy(
                        min_interval_s=max(state.policy.min_interval_s, float(delay)),
                        concurrency=state.policy.concurrency,
                    )
            except NotFound:
                state.robots = None  # RFC 9309: unavailable (4xx) => allow all
            except HostBlocked:
                raise  # A protection challenge is not a robots rule or network outage.
            except FetchError as exc:
                if exc.status and 400 <= exc.status < 500:
                    state.robots = None
                else:
                    # Unreachable robots.txt (5xx/network): be conservative (RFC 9309).
                    state.robots = Protego.parse("User-agent: *\nDisallow: /")
                    reason = f"HTTP {exc.status}" if exc.status else exc.reason
                    self.robots_problems[host] = reason
                    log.warning(
                        "robots.txt unreachable; treating host as disallowed "
                        "(run `python -m app.cli doctor` to check this machine's connection)",
                        extra={"host": host, "reason": reason},
                    )
            state.robots_loaded = True

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        state = self._state(parts.netloc)
        if not state.robots_loaded:
            await self._load_robots(parts.scheme or "https", parts.netloc, state)
        if state.robots is None:
            return True
        return bool(state.robots.can_fetch(url, self.robots_token))

    # --------------------------------------------------------------- requests
    async def _throttle(self, state: _HostState) -> None:
        async with state.lock:
            wait = state.last_request + state.policy.min_interval_s - time.monotonic()
            if wait > 0:
                await self._sleep(wait)
            state.last_request = time.monotonic()

    async def _request(
        self, url: str, state: _HostState, *, check_robots: bool = True, retries: int | None = None
    ) -> httpx.Response:
        if state.abandoned_reason:
            raise HostBlocked(url, f"host abandoned: {state.abandoned_reason}")
        if check_robots and not await self.allowed(url):
            self.stats["robots_blocked"] += 1
            raise RobotsDisallowed(url, "disallowed by robots.txt")

        attempts = (self.max_retries if retries is None else retries) + 1
        last_exc: FetchError | None = None
        for attempt in range(attempts):
            async with state.semaphore:
                # A queued task may have passed the first check before another
                # request blocked this host. Recheck before sending each attempt.
                if state.abandoned_reason:
                    raise HostBlocked(url, f"host abandoned: {state.abandoned_reason}")
                await self._throttle(state)
                if state.abandoned_reason:
                    raise HostBlocked(url, f"host abandoned: {state.abandoned_reason}")
                try:
                    resp = await self._fetch_capped(url)
                except httpx.HTTPError as exc:
                    self.stats["network_errors"] += 1
                    last_exc = FetchError(url, f"network error: {describe_error(exc)}")
                    resp = None
            if resp is not None:
                self.stats["requests"] += 1
                self.status_counts[str(resp.status_code)] += 1
                if resp.status_code == 200:
                    state.consecutive_failures = 0
                    return resp
                if resp.status_code in (404, 410):
                    raise NotFound(url, "not found", resp.status_code)
                if _looks_like_challenge(resp):
                    state.abandoned_reason = "anti-bot challenge served"
                    self.stats["challenges"] += 1
                    raise HostBlocked(url, "anti-bot challenge; not evading", resp.status_code)
                if resp.status_code not in RETRY_STATUSES:
                    self._register_failure(state)
                    raise FetchError(url, "unexpected status", resp.status_code)
                last_exc = FetchError(url, "retryable status", resp.status_code)
            if attempt + 1 < attempts:
                await self._sleep(self._backoff(attempt, resp))
        self._register_failure(state)
        assert last_exc is not None
        raise last_exc

    def _register_failure(self, state: _HostState) -> None:
        state.consecutive_failures += 1
        self.stats["failures"] += 1
        if state.consecutive_failures >= self.failure_budget:
            state.abandoned_reason = f"{state.consecutive_failures} consecutive failures"

    @staticmethod
    def _backoff(attempt: int, resp: httpx.Response | None) -> float:
        if resp is not None:
            retry_after = resp.headers.get("retry-after")
            if retry_after:
                try:
                    return min(60.0, max(0.5, float(retry_after)))
                except ValueError:
                    pass
        return min(30.0, (2**attempt) * 1.5 + random.uniform(0, 1.0))

    async def _fetch_capped(self, url: str) -> httpx.Response:
        async with self._client.stream("GET", url) as resp:
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > self.max_bytes:
                    raise httpx.DecodingError(f"response exceeded {self.max_bytes} bytes")
                chunks.append(chunk)
            self.stats["bytes"] += total
            # ``aiter_bytes`` yielded *decoded* (decompressed) bytes, so this copy
            # must not carry the original encoding headers: with them, reading the
            # copy would decompress a second time and fail. That bug made every
            # live (gzip-served) fetch - robots.txt first - look like a network
            # error, while the then-uncompressed offline replicas kept passing.
            headers = [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in _ENCODING_HEADERS]
            return httpx.Response(
                status_code=resp.status_code,
                headers=headers,
                content=b"".join(chunks),
                request=resp.request,
            )

    # --------------------------------------------------------------- helpers
    async def get(self, url: str) -> httpx.Response:
        host = urlsplit(url).netloc
        return await self._request(url, self._state(host))

    async def get_text(self, url: str) -> str:
        resp = await self.get(url)
        content = resp.content
        if content[:2] == b"\x1f\x8b":  # gzipped payload (e.g. sitemap.xml.gz)
            content = gzip.decompress(content)
            return content.decode("utf-8", errors="replace")
        return resp.text

    async def get_json(self, url: str) -> Any:
        resp = await self.get(url)
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise FetchError(url, "invalid JSON") from exc

    async def get_json_with_headers(self, url: str) -> tuple[Any, httpx.Headers]:
        resp = await self.get(url)
        try:
            return resp.json(), resp.headers
        except json.JSONDecodeError as exc:
            raise FetchError(url, "invalid JSON") from exc


async def gather_isolated(coros: list[Any], report: Any, label: str) -> HostBlocked | None:
    """Run tasks concurrently; one failure never aborts the others.

    Unexpected exceptions are counted on the crawl report; a ``HostBlocked``
    is returned (not raised) so the caller can stop that host but keep
    crawling other hosts/phases.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    blocked: HostBlocked | None = None
    for result in results:
        if isinstance(result, HostBlocked):
            blocked = blocked or result
        elif isinstance(result, BaseException):
            report.errors += 1
            log.warning("crawl task failed", extra={"phase": label, "error": repr(result)[:300]})
    if blocked:
        report.ok = False
        note = f"{label}: stopped — {blocked.reason}"
        if note not in report.notes:
            report.notes.append(note)
    return blocked
