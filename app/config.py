"""Typed, environment-driven configuration (12-factor).

Every setting can be overridden with an environment variable of the same name
(case-insensitive), e.g. ``OPENROUTER_API_KEY`` or ``WASALT_MAX_CATEGORY_PAGES``.
List settings accept either JSON (``["a","b"]``) or comma-separated values.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Ordered preference for free OpenRouter models. The live free roster changes
# frequently, so at runtime this list is intersected with what OpenRouter's
# /models endpoint reports as free *right now*; unknown free models are ranked
# after these and ``openrouter/free`` (OpenRouter's free router) is the last resort.
DEFAULT_LLM_MODELS = [
    # Latency-first: a mixture-of-experts model with ~4B active parameters
    # streams fastest; larger models follow as quality/availability fallbacks.
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "deepseek/deepseek-v4-flash:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-235b-a22b:free",
    "openrouter/free",
]

DEFAULT_BLOG_EXCLUDE = [
    "Food",
    "Hotels",
    "Activities",
    "Places to visit",
    "Malls",
    "Shopping Malls in Saudi Arabia",
    "Social Life",
    "Life",
    "lifestyle",
]

ListStr = Annotated[list[str], NoDecode]


def _split_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [part.strip() for part in text.split(",") if part.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ---- Service -------------------------------------------------------
    app_name: str = "DarGlobal × Wasalt Property Assistant"
    environment: str = "production"
    log_level: str = "INFO"
    log_prompts: bool = False  # never log raw user text unless explicitly enabled
    host: str = "0.0.0.0"
    port: int = 7860
    public_url: str | None = None  # used for OpenRouter attribution headers

    data_dir: Path = Path("data")
    model_cache_dir: Path = Path("models")

    # ---- LLM (OpenRouter) -----------------------------------------------
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_models: ListStr = Field(default_factory=lambda: list(DEFAULT_LLM_MODELS))
    llm_discover_free_models: bool = True
    llm_models_per_request: int = 3
    llm_max_attempts: int = 3
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700  # answers are concise by design (see ANSWER STYLE)
    llm_connect_timeout_s: float = 10.0
    llm_read_timeout_s: float = 45.0
    llm_first_token_timeout_s: float = 30.0  # fail over if a model hasn't started answering
    # On OpenRouter, a model's hidden "thinking" counts against max_tokens, so a model
    # that thinks can spend the whole answer budget and return an empty answer. Models
    # that can skip thinking are told to (reasoning effort "none"); models that must
    # think, and the openrouter/free router, get this many extra tokens and more time.
    llm_thinking_headroom_tokens: int = 3300
    llm_thinking_first_token_timeout_s: float = 60.0
    llm_max_concurrency: int = 4
    llm_model_refresh_hours: float = 6.0

    # ---- Retrieval -----------------------------------------------------
    # Compact English retrieval model: fits next to ChromaDB on a 512 MB instance.
    # Multilingual (EN+AR in one space, needs ~1 GB): sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    enable_dense: bool = True
    embedding_local_only: bool = False
    embedding_threads: int | None = None
    embedding_batch_size: int = Field(16, ge=1, le=256)  # small batches keep peak RAM low
    # Where chunk embeddings are stored and searched: "chroma" (ChromaDB vector
    # database, embedded, with metadata filtering) or "numpy" (exact in-memory
    # matrix, lightest). Applies to new builds; switching triggers a background re-embed.
    vector_backend: Literal["chroma", "numpy"] = "chroma"
    # Re-embed in the background when the active build's vectors don't match the
    # model/backend. Turn off on tiny CPUs (0.1 vCPU), where it takes hours: rebuild the image instead.
    reembed_if_outdated: bool = True
    # A question that arrives while the service is still starting waits up to this
    # long for semantic search to be ready, instead of being answered with keywords only.
    warmup_wait_s: float = Field(25.0, ge=0, le=120)
    chroma_ef_search: int = Field(100, ge=10, le=2000)  # HNSW search breadth (recall vs speed)
    top_k_listings: int = 8
    top_k_passages: int = 6
    min_passage_similarity: float | None = None  # dense-only passages below this are dropped (None: model default)
    prompt_history_messages: int = 4  # earlier turns sent to the LLM (follow-ups are resolved by the parser)
    max_context_chars: int = 8000  # ~2k tokens of sources

    # ---- Ingestion / crawling --------------------------------------------
    user_agent: str = (
        "DGWasaltAssistantBot/1.0 (+https://huggingface.co/spaces; "
        "polite RAG crawler; honours robots.txt)"
    )
    crawl_min_interval_s: float = 1.0
    crawl_concurrency_per_host: int = 1
    crawl_timeout_s: float = 30.0
    # Certificates for outbound HTTPS: auto = Windows/macOS system store (works behind
    # networks that inspect HTTPS), certifi elsewhere. Verification is never disabled.
    tls_trust: Literal["auto", "system", "certifi"] = "auto"
    crawl_max_retries: int = 3
    crawl_max_bytes: int = 6_000_000
    crawl_failure_budget: int = 12  # consecutive failures before a host is abandoned

    sources: ListStr = Field(default_factory=lambda: ["darglobal", "wasalt"])
    dg_max_articles: int = 120
    # Top-level DarGlobal pages fetched besides the /projects cards. The sitemap is
    # alphabetical with ~70 top-level paths, and project pages sit near its end, so
    # a cap of 40 silently dropped Pagani, Missoni, W Residences, The Astera...
    dg_max_pages: int = 100
    wasalt_max_category_pages: int = 120
    wasalt_max_detail_pages: int = 40
    wasalt_max_project_pages: int = 30
    wasalt_max_blog_posts: int = 600
    wasalt_blog_exclude_categories: ListStr = Field(
        default_factory=lambda: list(DEFAULT_BLOG_EXCLUDE)
    )

    ingest_on_startup: str = "if-missing"  # always | if-missing | if-stale | never
    refresh_interval_hours: float = 24.0
    stale_after_hours: float = 36.0

    # ---- API protection -------------------------------------------------
    rate_limit_per_minute: int = 12
    rate_limit_per_day: int = 200
    max_message_chars: int = 1500
    max_history_messages: int = 8
    admin_token: SecretStr | None = None
    cors_allow_origins: ListStr = Field(default_factory=list)
    cache_ttl_s: int = 6 * 3600
    cache_max_entries: int = 512

    @field_validator(
        "llm_models", "sources", "wasalt_blog_exclude_categories", "cors_allow_origins",
        mode="before",
    )
    @classmethod
    def _parse_lists(cls, value: Any) -> Any:
        return _split_list(value)

    @field_validator("ingest_on_startup")
    @classmethod
    def _check_ingest_mode(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"always", "if-missing", "if-stale", "never"}
        if value not in allowed:
            raise ValueError(f"ingest_on_startup must be one of {sorted(allowed)}")
        return value

    # ---- Derived paths --------------------------------------------------
    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshot"

    @property
    def index_root(self) -> Path:
        return self.data_dir / "index"

    @property
    def llm_configured(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
