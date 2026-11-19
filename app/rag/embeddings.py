"""Dense embeddings via fastembed (ONNX, CPU-only).

Default model: ``BAAI/bge-small-en-v1.5``, a compact English *retrieval*
model (33M parameters, 67 MB quantized ONNX, 384 dimensions):

* the collected content is English (DarGlobal, Wasalt's English site and its
  English blog), and a retrieval-trained model ranks passages for questions far
  better than a paraphrase model;
* it fits next to ChromaDB on a 512 MB instance (e.g. Render's free plan), so
  semantic search stays on in the free deployment.

Arabic questions are still understood: the bilingual parser extracts their
filters, keyword search is Arabic-normalised, and for an English-only model the
retriever embeds an English rendering of what the parser understood.

For Arabic-heavy use on hosts with >= 1 GB RAM, set
``EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
(English and Arabic in one vector space, 220 MB).

If the model cannot be loaded, retrieval degrades to keyword + SQL search
instead of failing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class ModelProfile:
    """How to use an embedding model well."""

    multilingual: bool  # can it embed Arabic text meaningfully?
    query_prefix: str = ""  # instruction some retrieval models expect on queries
    passage_prefix: str = ""
    # Relevance floor for passages found *only* by the vector search. Each model
    # has its own cosine scale (BGE scores unrelated text ~0.3-0.5).
    min_similarity: float = 0.3


PROFILES: dict[str, ModelProfile] = {
    DEFAULT_EMBEDDING_MODEL: ModelProfile(
        multilingual=False,
        query_prefix="Represent this sentence for searching relevant passages: ",
        min_similarity=0.5,
    ),
    MULTILINGUAL_EMBEDDING_MODEL: ModelProfile(multilingual=True, min_similarity=0.25),
}
HASHING_PROFILE = ModelProfile(multilingual=True, min_similarity=0.25)


def profile_for(model_name: str) -> ModelProfile:
    if model_name.startswith("hashing"):
        return HASHING_PROFILE
    return PROFILES.get(model_name, ModelProfile(multilingual="multilingual" in model_name.lower()))


class Embedder(Protocol):
    name: str
    dim: int
    profile: ModelProfile

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed passages (documents)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query."""
        ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FastEmbedder:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        *,
        threads: int | None = None,
        local_only: bool = False,
    ) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows; just noise
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # no "unauthenticated requests" warning
        from fastembed import TextEmbedding  # imported lazily: heavy dependency

        kwargs: dict[str, object] = {"model_name": model_name, "cache_dir": str(cache_dir)}
        if threads:
            kwargs["threads"] = threads
        if local_only:
            kwargs["local_files_only"] = True
        self._model = TextEmbedding(**kwargs)
        self.name = model_name
        self.profile = profile_for(model_name)
        self.dim = int(self._encode(["dimension probe"], 1).shape[1])

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 384)), dtype=np.float32)
        return _normalise(np.vstack(list(self._model.embed(texts, batch_size=batch_size))))

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        prefix = self.profile.passage_prefix
        return self._encode([prefix + t for t in texts] if prefix else texts, batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.profile.query_prefix + text], 1)[0]


class HashingEmbedder:
    """Deterministic, dependency-free embedder (character n-gram hashing).

    Used by the test-suite and available as an explicit opt-in fallback.
    """

    profile = HASHING_PROFILE

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        from app.textutil import normalize_for_search

        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            norm = f" {normalize_for_search(text)} "
            for n in (3, 4):
                for i in range(len(norm) - n + 1):
                    gram = norm[i : i + n]
                    h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "little")
                    mat[row, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return _normalise(mat)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def load_embedder(
    model_name: str, cache_dir: Path, *, enabled: bool = True, threads: int | None = None, local_only: bool = False
) -> Embedder | None:
    if not enabled:
        return None
    if model_name.startswith("hashing"):
        return HashingEmbedder()
    try:
        embedder = FastEmbedder(model_name, cache_dir, threads=threads, local_only=local_only)
        log.info("embedding model loaded", extra={"model": model_name, "dim": embedder.dim})
        return embedder
    except Exception as exc:  # pragma: no cover - depends on network/model files
        log.warning(
            "embedding model unavailable; dense retrieval disabled (BM25 only)",
            extra={"model": model_name, "error": repr(exc)[:300]},
        )
        return None
