"""Connectivity self-test: ``python -m app.cli doctor``.

Checks, from *this* machine, every host the app needs (both data sources,
OpenRouter, the model download) - first with a plain HTTP client, then through
the crawler's own engine, so a crawler-only failure or an anti-bot challenge
shows up here and not only halfway through ingestion. Failures become plain advice. Typical findings on office or campus networks:
sites blocked by a web filter, or HTTPS inspection with a company certificate.
"""

from __future__ import annotations

import asyncio
import ssl
import time
import urllib.request
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.net import describe_error, tls_verify, uses_system_store
from app.scraping.http import FetchError, NotFound
from app.scraping.pipeline import make_client

TARGETS: list[tuple[str, str, bool]] = [  # (label, url, is_data_source)
    ("DarGlobal", "https://darglobal.co.uk/robots.txt", True),
    ("Wasalt", "https://wasalt.sa/robots.txt", True),
    ("Wasalt sitemaps", "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz", True),
    ("Wasalt blog", "https://blog.wasalt.sa/robots.txt", True),
    ("OpenRouter", "https://openrouter.ai/api/v1/models", False),
    ("Hugging Face (model download)", "https://huggingface.co/api/models/BAAI/bge-small-en-v1.5", False),
]
CRAWLER_TARGETS: list[tuple[str, str]] = [
    ("Crawler engine: DarGlobal", "https://darglobal.co.uk/robots.txt"),
    ("Crawler engine: Wasalt", "https://wasalt.sa/robots.txt"),
]


@dataclass
class Probe:
    label: str
    url: str
    data_source: bool
    ok: bool
    status: int | None = None
    final_url: str | None = None
    ms: int = 0
    error: str | None = None
    kind: str | None = None  # dns | tls | timeout | connect | proxy | http | other
    fixed_by_other_trust: bool = False


def classify(error: str) -> str:
    e = error.lower()
    if "anti-bot challenge" in e:
        return "challenge"
    if "certificate_verify_failed" in e or "certificate verify failed" in e or "sslcertverification" in e:
        return "tls"
    if "getaddrinfo" in e or "name or service not known" in e or "nodename nor servname" in e or (
        "name resolution" in e
    ):
        return "dns"
    if "proxy" in e or "407" in e:
        return "proxy"
    if "timeout" in e:
        return "timeout"
    if "ssl" in e or "tls" in e or "handshake" in e:
        return "tls"
    if "connect" in e or "reset" in e or "refused" in e or "forcibly closed" in e or "10054" in e:
        return "connect"
    return "other"


async def _probe(
    label: str,
    url: str,
    data_source: bool,
    settings: Settings,
    verify: ssl.SSLContext | bool,
    transport: httpx.AsyncBaseTransport | None,
) -> Probe:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent},
            timeout=httpx.Timeout(15.0, connect=10.0),
            follow_redirects=True,
            verify=verify,
            transport=transport,
        ) as client, client.stream("GET", url) as resp:
            await resp.aread()
        ms = int((time.perf_counter() - started) * 1000)
        ok = resp.status_code < 400 or resp.status_code == 404
        error = kind = None
        if not ok:
            error, kind = f"HTTP {resp.status_code}", "http"
            if resp.headers.get("cf-mitigated", "").lower() == "challenge":
                error, kind = f"HTTP {resp.status_code} (anti-bot challenge)", "challenge"
        return Probe(label, url, data_source, ok, resp.status_code, str(resp.url), ms, error, kind)
    except Exception as exc:
        error = describe_error(exc)
        return Probe(
            label, url, data_source, False, None, None,
            int((time.perf_counter() - started) * 1000), error, classify(error),
        )


async def _crawler_probe(
    label: str, url: str, settings: Settings, transport: httpx.AsyncBaseTransport | None
) -> Probe:
    """The same fetch again, but through the crawler's own client, with its own
    headers and response handling - the code path ingestion actually uses. A
    plain probe can pass while this one fails (a response-handling bug, or an
    anti-bot challenge aimed at the crawler)."""
    host = httpx.URL(url).host
    started = time.perf_counter()
    try:
        async with make_client(settings, "doctor", transport) as client:
            resp = await client._request(url, client._state(host), check_robots=False, retries=0)
        return Probe(label, url, True, True, resp.status_code, url, int((time.perf_counter() - started) * 1000))
    except NotFound:  # no robots.txt means "allowed": fine
        return Probe(label, url, True, True, 404, url, int((time.perf_counter() - started) * 1000))
    except FetchError as exc:
        error = f"{type(exc).__name__}: {exc.reason}" + (f" (HTTP {exc.status})" if exc.status else "")
    except Exception as exc:
        error = describe_error(exc)
    return Probe(label, url, True, False, None, None, int((time.perf_counter() - started) * 1000), error, classify(error))


async def run_checks(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> list[Probe]:
    verify = tls_verify(settings.tls_trust)
    plain = await asyncio.gather(
        *(_probe(label, url, src, settings, verify, transport) for label, url, src in TARGETS)
    )
    engine = await asyncio.gather(
        *(_crawler_probe(label, url, settings, transport) for label, url in CRAWLER_TARGETS)
    )
    probes = [*plain, *engine]
    # A certificate failure may be fixed by the other trust store: test it.
    other_mode = "certifi" if uses_system_store(settings.tls_trust) else "system"
    other = tls_verify(other_mode)
    for probe in probes:
        if probe.kind == "tls" and other is not verify:
            retry = await _probe(probe.label, probe.url, probe.data_source, settings, other, transport)
            probe.fixed_by_other_trust = retry.ok
    return list(probes)


def advice(probes: list[Probe], settings: Settings) -> list[str]:
    tips: list[str] = []
    failed = [p for p in probes if not p.ok]
    sources_failed = [p for p in failed if p.data_source]
    if not failed:
        return ["Everything is reachable. Run `python -m app.cli ingest` to collect the data."]
    if any(p.fixed_by_other_trust for p in failed):
        mode = "certifi" if uses_system_store(settings.tls_trust) else "system"
        tips.append(
            f"Your network inspects HTTPS with its own certificate. Add `TLS_TRUST={mode}` to .env and run ingest again."
        )
    kinds = {p.kind for p in sources_failed}
    hard_blocked = [p for p in sources_failed if p.kind != "challenge"]
    if "challenge" in kinds:
        tips.append(
            "A data site answered the crawler with an anti-bot challenge. The app never tries to sneak past "
            "these: that source is skipped and reported in /api/stats, and the bot answers from the data it "
            "has. A later run, a different network, or Render's build machines may be allowed through."
        )
    if "dns" in kinds:
        tips.append("Some site names don't resolve on this network: its DNS filter blocks them.")
    if kinds & {"connect", "timeout", "tls"} and not any(p.fixed_by_other_trust for p in sources_failed):
        tips.append("Connections to some sites are blocked or cut on this network (firewall, web filter or VPN).")
    if "proxy" in kinds:
        tips.append(f"A proxy rejected the request; proxies in effect: {_proxies() or 'none'}.")
    if "http" in kinds:
        tips.append("A site answered with an error status: it may be refusing this network's address. Try later.")
    if hard_blocked and all(p.ok for p in probes if not p.data_source):
        tips.append(
            "Your internet works (OpenRouter and Hugging Face are reachable), but the data sites are blocked on this "
            "network, which is common on office and campus networks. Options: (1) connect to a phone hotspot or home "
            "Wi-Fi, then run `python -m app.cli ingest`; (2) skip local scraping and deploy on Render: its build "
            "servers do the scraping, so your network doesn't matter."
        )
    elif any(not p.ok for p in probes if not p.data_source):
        tips.append("OpenRouter or Hugging Face is unreachable too: check the internet connection or proxy settings.")
    return tips


def _proxies() -> str:
    """HTTP(S) proxies Python will use: environment variables and, on Windows, Internet Settings."""
    found = urllib.request.getproxies()
    return ", ".join(
        f"{k}={v.split('@')[-1]}"  # never print credentials
        for k, v in found.items()
        if k in ("http", "https", "all")
    )


def report(probes: list[Probe], settings: Settings) -> str:
    trust = "system certificate store" if uses_system_store(settings.tls_trust) else "certifi bundle"
    lines = [f"Connectivity check (TLS_TRUST={settings.tls_trust}: {trust}; proxy: {_proxies() or 'none'})", ""]
    for p in probes:
        if p.ok:
            lines.append(f"  OK    {p.label:<30} HTTP {p.status}  {p.ms} ms")
        else:
            extra = "  (works with the other certificate store)" if p.fixed_by_other_trust else ""
            lines.append(f"  FAIL  {p.label:<30} {p.kind}: {p.error}{extra}")
    lines += ["", *[f"- {tip}" for tip in advice(probes, settings)]]
    return "\n".join(lines)
