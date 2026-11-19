"""Prometheus metrics (scrape ``/metrics``)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP latency", ["route"], registry=REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
CHAT_REQUESTS = Counter("chat_requests_total", "Chat turns", ["mode", "intent"], registry=REGISTRY)
CHAT_LATENCY = Histogram(
    "chat_duration_seconds", "End-to-end chat latency", registry=REGISTRY,
    buckets=(0.5, 1, 2, 4, 8, 15, 30, 60, 120),
)
LLM_REQUESTS = Counter("llm_requests_total", "LLM completions", ["model", "outcome"], registry=REGISTRY)
LLM_TTFT = Histogram(
    "llm_time_to_first_token_seconds", "Time until the first answer token arrives", registry=REGISTRY,
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30),
)
LLM_TOKENS = Counter("llm_tokens_total", "Tokens reported by OpenRouter", ["kind"], registry=REGISTRY)
RETRIEVAL_LATENCY = Histogram(
    "retrieval_duration_seconds", "Retrieval latency", registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
INDEX_RECORDS = Gauge("index_records", "Records in the active index", ["source", "kind"], registry=REGISTRY)
INGESTIONS = Counter("ingestions_total", "Ingestion runs", ["outcome"], registry=REGISTRY)
RATE_LIMITED = Counter("rate_limited_total", "Requests rejected by the rate limiter", registry=REGISTRY)
