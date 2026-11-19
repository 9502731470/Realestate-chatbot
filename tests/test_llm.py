import json

import httpx
import pytest

from app.config import Settings
from app.llm.openrouter import (
    LLMAuthError,
    LLMQuotaExceeded,
    LLMUnavailable,
    OpenRouterClient,
    StreamResult,
    ThinkFilter,
    rank_free_models,
)

MODELS = [
    {"id": "google/gemma-4-31b-it:free", "context_length": 262144},
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "context_length": 1000000},
    {"id": "cohere/north-mini-code:free", "context_length": 256000},
    {"id": "liquid/lfm-3-1.7b:free", "context_length": 262144},
    {"id": "acme/unknown-model:free", "context_length": 131072},
    {"id": "openai/gpt-5:paid", "context_length": 400000},
    {"id": "image/maker:free", "context_length": 32000, "architecture": {"output_modalities": ["image"]}},
    {"id": "openrouter/free", "context_length": 200000},
]


def test_rank_free_models():
    ranked = rank_free_models(MODELS, ["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free", "gone/model:free"])
    assert ranked[:2] == ["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free"]
    assert "acme/unknown-model:free" in ranked
    assert ranked[-1] == "openrouter/free"
    for excluded in ("cohere/north-mini-code:free", "liquid/lfm-3-1.7b:free", "openai/gpt-5:paid", "image/maker:free", "gone/model:free"):
        assert excluded not in ranked


@pytest.mark.parametrize("chunks", [["Hello <think>secret</think> world"], ["Hel", "lo <thi", "nk>sec", "ret</th", "ink> world"]])
def test_think_filter(chunks):
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert out == "Hello  world"


def sse(*events: dict) -> bytes:
    body = ": OPENROUTER PROCESSING\n\n"
    for ev in events:
        body += f"data: {json.dumps(ev)}\n\n"
    return (body + "data: [DONE]\n\n").encode()


def settings(**kw) -> Settings:
    return Settings(openrouter_api_key="sk-test", llm_discover_free_models=False,
                    llm_models=["a/one:free", "b/two:free", "c/three:free", "d/four:free"], llm_models_per_request=2, **kw)


async def collect(client: OpenRouterClient) -> tuple[str, StreamResult]:
    result = StreamResult()
    text = ""
    async for delta in client.stream_chat([{"role": "user", "content": "hi"}], result):
        text += delta
    return text, result


async def test_streams_and_reports_model():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, content=sse(
            {"model": "a/one:free", "choices": [{"delta": {"content": "Hello "}}]},
            {"model": "a/one:free", "choices": [{"delta": {"content": "there"}, "finish_reason": "stop"}]},
        ), headers={"content-type": "text/event-stream"})

    client = OpenRouterClient(settings(), transport=httpx.MockTransport(handler))
    text, result = await collect(client)
    assert text == "Hello there" and result.model == "a/one:free"
    assert seen[0]["models"] == ["a/one:free", "b/two:free"] and seen[0]["stream"] is True
    await client.aclose()


async def test_falls_back_to_next_model_group_on_errors():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body["models"])
        if body["models"][0] == "a/one:free":
            return httpx.Response(503, json={"error": {"code": 503, "message": "No endpoints available"}})
        return httpx.Response(200, content=sse({"model": "c/three:free", "choices": [{"delta": {"content": "ok"}}]}))

    client = OpenRouterClient(settings(), transport=httpx.MockTransport(handler))
    text, _result = await collect(client)
    assert text == "ok" and calls == [["a/one:free", "b/two:free"], ["c/three:free", "d/four:free"]]
    await client.aclose()


async def test_daily_quota_opens_circuit_breaker():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"}})

    client = OpenRouterClient(settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMQuotaExceeded):
        await collect(client)
    with pytest.raises(LLMQuotaExceeded):  # short-circuits without another HTTP call
        await collect(client)
    assert calls["n"] == 1 and client.quota_exhausted_until is not None
    await client.aclose()


async def test_auth_error_and_exhausted_models():
    client = OpenRouterClient(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "No auth credentials"}})))
    with pytest.raises(LLMAuthError):
        await collect(client)
    await client.aclose()

    client = OpenRouterClient(settings(llm_max_attempts=2), transport=httpx.MockTransport(lambda r: httpx.Response(502, text="bad gateway")))
    with pytest.raises(LLMUnavailable):
        await collect(client)
    await client.aclose()


async def test_mid_stream_error_after_content_is_reported():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"model": "a/one:free", "choices": [{"delta": {"content": "Partial"}}]},
            {"choices": [{"delta": {}, "finish_reason": "error"}]},
        ))

    client = OpenRouterClient(settings(), transport=httpx.MockTransport(handler))
    result = StreamResult()
    got = []
    with pytest.raises(LLMUnavailable):
        async for d in client.stream_chat([{"role": "user", "content": "x"}], result):
            got.append(d)
    assert got == ["Partial"]
    await client.aclose()


async def test_model_discovery_uses_live_free_roster():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": MODELS})
        raise AssertionError("unexpected")

    s = Settings(openrouter_api_key="k", llm_models=["google/gemma-4-31b-it:free", "retired/model:free"])
    client = OpenRouterClient(s, transport=httpx.MockTransport(handler))
    chain = await client.refresh_models(force=True)
    assert chain[0] == "google/gemma-4-31b-it:free" and "retired/model:free" not in chain
    await client.aclose()
