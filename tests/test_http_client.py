import asyncio
import gzip

import httpx
import pytest

from app.scraping.http import FetchError, HostBlocked, HostPolicy, NotFound, PoliteClient, RobotsDisallowed


def client_for(handler, **kw) -> tuple[PoliteClient, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = PoliteClient(
        user_agent="TestBot/1.0",
        default_policy=HostPolicy(min_interval_s=kw.pop("interval", 0), concurrency=1),
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        **kw,
    )
    return client, sleeps


ROBOTS = "User-agent: *\nDisallow: /search\nCrawl-delay: 2\n"


async def test_robots_disallow_and_crawl_delay():
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS)
        return httpx.Response(200, text="ok")

    client, sleeps = client_for(handler)
    async with client:
        with pytest.raises(RobotsDisallowed):
            await client.get("https://example.com/search?q=x")
        assert await client.get_text("https://example.com/a") == "ok"
        assert await client.get_text("https://example.com/b") == "ok"
    assert "/search" not in seen  # never requested
    assert any(s >= 1.9 for s in sleeps)  # Crawl-delay: 2 honoured between requests
    assert client.stats["robots_blocked"] == 1


async def test_retry_after_on_429_then_success():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(200, text="fine")

    client, sleeps = client_for(handler, max_retries=2)
    async with client:
        assert await client.get_text("https://example.com/x") == "fine"
    assert 3.0 in sleeps


async def test_challenge_page_stops_the_host():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(403, headers={"content-type": "text/html", "cf-mitigated": "challenge"}, text="Just a moment...")

    client, _ = client_for(handler)
    async with client:
        with pytest.raises(HostBlocked):
            await client.get("https://example.com/1")
        with pytest.raises(HostBlocked):  # abandoned immediately, no further hammering
            await client.get("https://example.com/2")
    assert client.stats["challenges"] == 1


@pytest.mark.parametrize("challenge,expected_requests", [(True, 1), (False, 3)])
async def test_queued_requests_stop_when_host_is_abandoned(challenge, expected_requests):
    requested = []

    async def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        requested.append(req.url.path)
        await asyncio.sleep(0)
        headers = {"cf-mitigated": "challenge"} if challenge else {}
        return httpx.Response(403, headers=headers, text="Forbidden")

    client, _ = client_for(handler, failure_budget=3, max_retries=0)
    async with client:
        await client.allowed("https://example.com/")
        results = await asyncio.gather(
            *(client.get(f"https://example.com/{i}") for i in range(120)),
            return_exceptions=True,
        )
    assert len(requested) == expected_requests
    assert all(isinstance(result, FetchError) for result in results)
    assert sum(isinstance(result, HostBlocked) for result in results) >= 117


async def test_not_found_and_gzip_and_size_cap():
    xml = b"<urlset><url><loc>https://example.com/a</loc></url></urlset>"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        if req.url.path == "/missing":
            return httpx.Response(404)
        if req.url.path == "/big":
            return httpx.Response(200, content=b"x" * 5000)
        return httpx.Response(200, content=gzip.compress(xml), headers={"content-type": "application/x-gzip"})

    client, _ = client_for(handler, max_bytes=1000, max_retries=0)
    async with client:
        with pytest.raises(NotFound):
            await client.get("https://example.com/missing")
        assert "<loc>https://example.com/a</loc>" in await client.get_text("https://example.com/sitemap.xml.gz")
        with pytest.raises(FetchError):  # oversized body is rejected
            await client.get("https://example.com/big")


async def test_unreachable_robots_is_conservative():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return httpx.Response(503)
        return httpx.Response(200, text="should not be fetched")

    client, _ = client_for(handler, max_retries=0)
    async with client:
        with pytest.raises(RobotsDisallowed):
            await client.get("https://example.com/page")


async def test_compressed_responses_are_decoded_exactly_once():
    """Regression: real servers gzip responses; the rebuilt (size-capped) response
    copy must not claim an encoding its body no longer has."""
    from tests.fixtures.sites import CompressedStream

    body = ("<html><body>" + "فيلا للبيع في جدة with sea view " * 120 + "</body></html>").encode()
    raw = gzip.compress(body)

    def handler(req: httpx.Request) -> httpx.Response:
        assert "gzip" in req.headers["accept-encoding"]
        return httpx.Response(
            200,
            headers=[
                ("content-type", "text/html; charset=utf-8"),
                ("content-encoding", "gzip"),
                ("content-length", str(len(raw))),
            ],
            stream=CompressedStream(raw),
        )

    client, _ = client_for(handler)
    resp = await client.get("https://example.com/page")
    assert resp.content == body and "فيلا" in resp.text
    assert "content-encoding" not in resp.headers  # the copy holds decoded bytes
    assert int(resp.headers["content-length"]) == len(body)
    assert client.stats["bytes"] == 2 * len(body)  # robots.txt and the page: same handler, both decoded


async def test_mock_sites_compress_like_real_servers():
    """The replicas must stay realistic, or bugs like the one above hide again."""
    from tests.fixtures.sites import MockSites

    transport = MockSites().transport
    async with httpx.AsyncClient(transport=transport, headers={"accept-encoding": "gzip, deflate"}) as raw_client:
        robots = await raw_client.get("https://wasalt.sa/robots.txt")
        sitemap = await raw_client.get("https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz")
    assert robots.headers.get("content-encoding") == "gzip" and "user-agent" in robots.text.lower()
    # the .gz payload is served as-is (not compressed twice), like a CDN
    assert "content-encoding" not in sitemap.headers and sitemap.content[:2] == b"\x1f\x8b"
