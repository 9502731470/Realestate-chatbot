"""Connection problems are explained, not hidden: TLS trust, error reasons, doctor advice."""

import asyncio
import ssl
import sys

import httpx
import pytest

import app.diagnostics as diagnostics
from app.diagnostics import Probe, advice, classify, report, run_checks
from app.net import describe_error, tls_verify, uses_system_store
from app.scraping.pipeline import crawl_source
from tests.conftest import make_settings

GETADDRINFO = "[Errno 11001] getaddrinfo failed"
CERT_FAIL = "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate"


def test_tls_trust_uses_the_system_store_on_windows_and_macos(monkeypatch):
    for platform, system in (("win32", True), ("darwin", True), ("linux", False)):
        monkeypatch.setattr(sys, "platform", platform)
        assert uses_system_store("auto") is system
        verify = tls_verify("auto")
        assert isinstance(verify, ssl.SSLContext) if system else verify is True
    assert tls_verify("certifi") is True
    assert isinstance(tls_verify("system"), ssl.SSLContext)  # truststore context; verification stays on


def test_error_descriptions_keep_the_real_reason():
    try:
        try:
            raise OSError(GETADDRINFO)
        except OSError as inner:
            raise httpx.ConnectError("All connection attempts failed") from inner
    except httpx.ConnectError as exc:
        text = describe_error(exc)
    assert text.startswith("ConnectError: All connection attempts failed") and GETADDRINFO in text
    assert classify(text) == "dns" and classify(f"ConnectError: {CERT_FAIL}") == "tls"
    assert classify("ConnectTimeout: timed out") == "timeout" and classify("ProxyError: 407") == "proxy"


def test_unreachable_robots_txt_is_reported_with_its_reason(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(GETADDRINFO, request=request)

    settings = make_settings(tmp_path, crawl_max_retries=0)
    records, rep = asyncio.run(crawl_source(settings, "darglobal", httpx.MockTransport(handler)))
    assert records == []
    note = next(n for n in rep.notes if n.startswith("robots.txt of darglobal.co.uk unreachable"))
    assert f"network error: ConnectError: {GETADDRINFO}" in note


def _site_blocking_network(request: httpx.Request) -> httpx.Response:
    host = request.url.host
    if host == "darglobal.co.uk":
        raise httpx.ConnectError(GETADDRINFO, request=request)
    if host == "wasalt.sa":
        raise httpx.ConnectError(CERT_FAIL, request=request)
    if host == "blog.wasalt.sa":
        raise httpx.ConnectTimeout("timed out", request=request)
    return httpx.Response(200, text="ok")


def test_doctor_explains_a_network_that_blocks_the_data_sites(tmp_path):
    settings = make_settings(tmp_path)
    probes = asyncio.run(run_checks(settings, httpx.MockTransport(_site_blocking_network)))
    by_label = {p.label: p for p in probes}
    assert by_label["DarGlobal"].kind == "dns" and by_label["Wasalt"].kind == "tls"
    assert by_label["Wasalt blog"].kind == "timeout" and by_label["OpenRouter"].ok
    text = report(probes, settings)
    assert "FAIL  DarGlobal" in text and "OK    OpenRouter" in text
    assert "phone hotspot" in text and "deploy on Render" in text and "DNS filter" in text


def test_doctor_suggests_the_other_certificate_store_when_it_fixes_tls(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, tls_trust="certifi")
    system = object()
    monkeypatch.setattr(diagnostics, "tls_verify", lambda mode: system if mode == "system" else True)

    async def fake_probe(label, url, data_source, settings, verify, transport):
        if verify is system or not data_source:
            return Probe(label, url, data_source, True, 200)
        return Probe(label, url, data_source, False, error=f"ConnectError: {CERT_FAIL}", kind="tls")

    async def fake_crawler_probe(label, url, settings, transport):
        return Probe(label, url, True, True, 200)

    monkeypatch.setattr(diagnostics, "_probe", fake_probe)
    monkeypatch.setattr(diagnostics, "_crawler_probe", fake_crawler_probe)
    probes = asyncio.run(run_checks(settings))
    assert all(p.fixed_by_other_trust for p in probes if p.data_source and not p.ok)
    assert any("TLS_TRUST=system" in tip for tip in advice(probes, settings))


def test_doctor_all_clear(tmp_path):
    settings = make_settings(tmp_path)
    probes = asyncio.run(run_checks(settings, httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))))
    assert all(p.ok for p in probes)
    assert advice(probes, settings) == ["Everything is reachable. Run `python -m app.cli ingest` to collect the data."]


@pytest.mark.parametrize("status", [403, 500])
def test_doctor_reports_http_errors(tmp_path, status):
    settings = make_settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status if request.url.host == "wasalt.sa" else 200, text="no")

    probes = asyncio.run(run_checks(settings, httpx.MockTransport(handler)))
    wasalt = next(p for p in probes if p.label == "Wasalt")
    assert not wasalt.ok and wasalt.kind == "http" and wasalt.error == f"HTTP {status}"


def test_doctor_recognises_an_anti_bot_challenge(tmp_path):
    settings = make_settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "darglobal.co.uk":
            return httpx.Response(
                403,
                headers={"content-type": "text/html", "cf-mitigated": "challenge"},
                text="<html><title>Just a moment...</title></html>",
            )
        return httpx.Response(200, text="ok")

    probes = asyncio.run(run_checks(settings, httpx.MockTransport(handler)))
    engine = next(p for p in probes if p.label == "Crawler engine: DarGlobal")
    assert not engine.ok and engine.kind == "challenge" and "anti-bot challenge" in engine.error
    assert next(p for p in probes if p.label == "Crawler engine: Wasalt").ok
    tips = advice(probes, settings)
    assert any("anti-bot challenge" in t and "/api/stats" in t for t in tips)
    assert not any("hotspot" in t for t in tips)  # a challenge is not a blocked network


def test_doctor_crawler_probe_catches_crawler_only_failures(tmp_path):
    """A response-handling bug in the crawler must show in the doctor even when
    plain HTTP works (the plain probes all passed while ingest failed, once)."""
    import gzip as gz

    from tests.fixtures.sites import CompressedStream

    def handler(request: httpx.Request) -> httpx.Response:
        raw = gz.compress(b"User-agent: *\nAllow: /\n")
        return httpx.Response(
            200,
            headers=[("content-type", "text/plain"), ("content-encoding", "gzip"), ("content-length", str(len(raw)))],
            stream=CompressedStream(raw),
        )

    probes = asyncio.run(run_checks(make_settings(tmp_path), httpx.MockTransport(handler)))
    assert all(p.ok for p in probes), [p for p in probes if not p.ok]
