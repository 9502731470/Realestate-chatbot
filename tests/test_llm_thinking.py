"""Empty answers from "thinking" models.

On OpenRouter a model's hidden reasoning counts against ``max_tokens``. With a
700-token budget and ``reasoning: {"exclude": true}`` (which hides the thinking but
does not stop it), models that think by default (Nemotron Ultra at "high" effort,
Ling) or must think (Ernie) used the whole budget and streamed nothing: the
chatbot fell back to a plain list ("all free models are busy or unavailable").
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.llm.openrouter import (
    LLMUnavailable,
    OpenRouterClient,
    StreamResult,
    must_think,
    plan_groups,
    rank_free_models,
)
from tests.test_llm import sse

GEMMA_26 = "google/gemma-4-26b-a4b-it:free"
GEMMA_31 = "google/gemma-4-31b-it:free"
LIGHTNING = "nvidia/nemotron-3.5-lightning:free"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b:free"
LING = "inclusionai/ling-3.2-flash:free"
ERNIE = "baidu/ernie-5.5-flash-preview:free"
ROUTER = "openrouter/free"

# Reasoning metadata as served by OpenRouter's /models on 20 Nov 2026.
LIVE_ROSTER = [
    {"id": GEMMA_26, "context_length": 262144, "reasoning": {"mandatory": False, "default_enabled": False}},
    {"id": GEMMA_31, "context_length": 262144, "reasoning": {"mandatory": False, "default_enabled": False}},
    {"id": LIGHTNING, "context_length": 1000000, "reasoning": {"mandatory": False}},
    {"id": ULTRA, "context_length": 1000000, "reasoning": {
        "mandatory": False, "default_enabled": True, "supports_max_tokens": True,
        "supported_efforts": ["high", "medium", "none"], "default_effort": "high"}},
    {"id": LING, "context_length": 262144, "reasoning": {"mandatory": False, "default_enabled": True}},
    {"id": ERNIE, "context_length": 1024000, "reasoning": {"mandatory": True, "default_enabled": True, "supported_efforts": ["high"]}},
    {"id": ROUTER, "context_length": 200000},
]
INFO = {m["id"]: m for m in LIVE_ROSTER}


@pytest.mark.parametrize(
    "model,expected",
    [
        (GEMMA_31, False),  # thinking optional and off by default
        (ULTRA, False),  # thinks by default, but accepts effort "none"
        (LING, False),
        (ERNIE, True),  # mandatory: rejects effort "none"
        ("x/effortful:free", True),  # optional, yet "none" isn't an offered effort
        (ROUTER, True),  # may route to a model that must think
        ("a/plain:free", False),  # not a reasoning model
    ],
)
def test_which_models_must_think(model, expected):
    info = INFO.get(model) or (
        {"reasoning": {"mandatory": False, "supported_efforts": ["low", "high"]}} if model == "x/effortful:free" else None
    )
    assert must_think(model, info) is expected


def test_models_that_must_think_and_the_router_are_tried_last():
    # Even a preferred thinker goes after every model that can skip thinking.
    ranked = rank_free_models(LIVE_ROSTER, [ERNIE, GEMMA_26, GEMMA_31, LIGHTNING, ULTRA, ROUTER])
    assert ranked == [GEMMA_26, GEMMA_31, LIGHTNING, ULTRA, LING, ERNIE, ROUTER]


def test_thinkers_never_share_a_request_with_other_models():
    chain = rank_free_models(LIVE_ROSTER, [GEMMA_26, GEMMA_31, LIGHTNING, ULTRA])
    assert plan_groups(chain, INFO, 3) == [
        ([GEMMA_26, GEMMA_31, LIGHTNING], False),
        ([ULTRA, LING], False),
        ([ERNIE, ROUTER], True),
    ]
    # Alternating classes are split even when a group has room left.
    assert plan_groups(["a:free", ERNIE, "b:free"], INFO, 3) == [(["a:free"], False), ([ERNIE], True), (["b:free"], False)]


def _settings(models: list[str], **kw) -> Settings:
    return Settings(openrouter_api_key="sk-test", llm_models=models, llm_models_per_request=3, **kw)


def thinking_server(roster: list[dict], payloads: list[dict], *, unavailable: set[str] = frozenset()):
    """Replays OpenRouter: a thinking model spends a small budget on hidden reasoning
    and streams no answer, unless told to skip thinking or given room for both."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": roster})
        body = json.loads(req.content)
        payloads.append(body)
        if body["models"][0] in unavailable:
            return httpx.Response(503, json={"error": {"code": 503, "message": "No endpoints available"}})
        model = body["models"][0]
        reasoning = INFO.get(model, {}).get("reasoning") or {}
        mandatory = bool(reasoning.get("mandatory")) or model == ROUTER  # the router may pick such a model
        thinks = mandatory or bool(reasoning.get("default_enabled"))
        if body.get("reasoning", {}).get("effort") == "none":
            if mandatory:  # as documented: models with mandatory reasoning reject effort "none"
                return httpx.Response(400, json={"error": {"code": 400, "message": "Reasoning is mandatory for this model"}})
            thinks = False
        if thinks and body["max_tokens"] < 1500:
            return httpx.Response(200, content=sse({"model": model, "choices": [{"delta": {}, "finish_reason": "length"}]}))
        return httpx.Response(200, content=sse(
            {"model": model, "choices": [{"delta": {"content": "Here are 3 options [1]."}, "finish_reason": "stop"}]}))

    return httpx.MockTransport(handler)


async def _ask(client: OpenRouterClient) -> tuple[str, StreamResult]:
    result = StreamResult()
    text = ""
    async for delta in client.stream_chat([{"role": "user", "content": "Flats in Riyadh?"}], result):
        text += delta
    return text, result


@pytest.mark.parametrize("model", [ULTRA, LING, ERNIE])
async def test_thinking_models_now_return_an_answer(model):
    # Previously: reasoning {"exclude": true} with 700 tokens -> empty -> "models unavailable".
    payloads: list[dict] = []
    client = OpenRouterClient(_settings([model], llm_max_attempts=1), transport=thinking_server([INFO[model]], payloads))
    text, result = await _ask(client)
    assert text == "Here are 3 options [1]." and result.model == model
    await client.aclose()


async def test_request_settings_per_group():
    payloads: list[dict] = []
    client = OpenRouterClient(
        _settings([GEMMA_26, GEMMA_31, LIGHTNING, ULTRA], llm_max_attempts=3),
        transport=thinking_server(LIVE_ROSTER, payloads, unavailable={GEMMA_26, ULTRA}),  # first two groups busy
    )
    text, result = await _ask(client)
    assert text and result.attempts == 3
    first, second, third = payloads
    assert first["models"] == [GEMMA_26, GEMMA_31, LIGHTNING]
    assert first["reasoning"] == {"effort": "none", "exclude": True} and first["max_tokens"] == 700
    assert second["models"] == [ULTRA, LING] and second["reasoning"] == {"effort": "none", "exclude": True}
    assert third["models"] == [ERNIE, ROUTER]
    assert third["reasoning"] == {"exclude": True}  # never "none": Ernie would reject it
    assert third["max_tokens"] == 700 + 3300  # room to think *and* answer
    assert client.status()["thinking_models"] == [ERNIE, ROUTER]
    await client.aclose()


async def test_empty_answer_reports_that_thinking_used_the_budget():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": LIVE_ROSTER})
        return httpx.Response(200, content=sse({"model": ERNIE, "choices": [{"delta": {}, "finish_reason": "length"}]}))

    client = OpenRouterClient(_settings([ERNIE], llm_max_attempts=1), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMUnavailable, match="token limit reached while the model was thinking"):
        await _ask(client)
    await client.aclose()
