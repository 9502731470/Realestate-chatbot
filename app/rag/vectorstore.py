"""Vector stores for chunk embeddings.

Two interchangeable backends behind one small interface (``VECTOR_BACKEND``):

* ``chroma`` (default) - **ChromaDB**, an open-source vector database running
  embedded in the app (no server, no account, no cost). It persists every
  chunk embedding together with its record's metadata (kind, source, purpose,
  type, price, bedrooms, area, city, district, furnishing...) and answers
  nearest-neighbour queries with an HNSW index (cosine distance).
* ``numpy`` - exact (brute-force) cosine search over an in-memory NumPy
  matrix: zero extra dependencies, 100% recall, the lightest option.

Filtered search ("3-bed rentals in Riyadh, most relevant first") is planned
per query. SQLite has already resolved the structured filters exactly (it is
the source of truth for counts and filter relaxation), so the vector store
receives the set of qualifying chunks as a boolean mask:

* broad filter (>= half the corpus) - unfiltered HNSW search with
  oversampling, then drop non-qualifying chunks (~1-2 ms);
* narrower filter - search restricted inside Chroma to the qualifying chunk
  ids (exact brute force for small sets, ~3 ms);
* candidate set unknown - Chroma's own metadata ``where`` filter
  (``search_where``; correct but ~10x slower in embedded mode, because
  Chroma evaluates it in SQLite).

Measured on ~8k chunks: unfiltered query ~1 ms, id-restricted ~3 ms, metadata
``where`` 13-48 ms. Each index *build* has its own store (``chroma/`` or
``vectors.npy``), so vectors are versioned and switched atomically together
with the text index.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.rag.embeddings import Embedder

log = logging.getLogger(__name__)

NUMPY_FILE = "vectors.npy"
CHROMA_DIR = "chroma"
CHROMA_COLLECTION = "chunks"
BACKENDS = ("chroma", "numpy")
BROAD_FILTER_FRACTION = 0.5  # at or above this share of the corpus: ANN + post-filter


class VectorStore(Protocol):
    backend: str

    def __len__(self) -> int: ...

    @property
    def dim(self) -> int: ...

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k ``(row, cosine_similarity)``, best first, among rows where ``mask``
        is True (all rows if None). Row i is chunk rowid i+1."""
        ...

    def close(self) -> None: ...


# ----------------------------------------------------------------------- NumPy
class NumpyVectorStore:
    backend = "numpy"

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)  # rows are L2-normalised

    @classmethod
    def build(cls, embedder: Embedder, texts: list[str], out_dir: Path, batch_size: int = 16) -> NumpyVectorStore:
        matrix = embedder.embed(texts, batch_size=batch_size)
        np.save(out_dir / NUMPY_FILE, matrix.astype(np.float32))
        return cls(matrix)

    @classmethod
    def load(cls, build_dir: Path, expected_rows: int) -> NumpyVectorStore | None:
        path = build_dir / NUMPY_FILE
        if not path.exists():
            return None
        matrix = np.load(path)
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
            return None
        return cls(matrix)

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def similarities(self, query: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(query, dtype=np.float32)

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        sims = self.similarities(query)
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
            k = min(k, int(mask.sum()))
        k = min(k, len(sims))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------- Chroma
def _chroma_client(path: Path) -> Any:
    # Never phone home; must be set before chromadb is imported.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    import chromadb  # heavy import: only when the Chroma backend is actually used
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))


class ChromaVectorStore:
    """Chunk embeddings in an embedded, persistent ChromaDB collection.

    Ids are chunk rowids (as strings), so hits join straight back to SQLite;
    each chunk's metadata holds its record's filterable fields.
    """

    backend = "chroma"

    def __init__(self, client: Any, collection: Any, dim: int | None = None) -> None:
        self._client = client
        self._collection = collection
        self._lock = threading.Lock()
        self._count = int(collection.count())
        self._dim = dim

    @classmethod
    def build(
        cls,
        embedder: Embedder,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        out_dir: Path,
        *,
        ef_search: int = 100,
        batch_size: int = 16,
    ) -> ChromaVectorStore:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")
        vectors = np.asarray(embedder.embed(texts, batch_size=batch_size), dtype=np.float32)
        client = _chroma_client(out_dir / CHROMA_DIR)
        try:
            collection = client.create_collection(
                CHROMA_COLLECTION,
                embedding_function=None,  # we supply our own (multilingual) embeddings
                configuration={"hnsw": {"space": "cosine", "ef_search": ef_search, "ef_construction": 200}},
                metadata={"embedding_model": embedder.name},
            )
            step = max(1, min(2000, int(client.get_max_batch_size())))
            for start in range(0, len(texts), step):
                end = min(start + step, len(texts))
                collection.add(
                    ids=[str(i + 1) for i in range(start, end)],  # chunk rowid
                    embeddings=vectors[start:end],
                    metadatas=metadatas[start:end],
                )
        except Exception:
            client.close()
            raise
        return cls(client, collection, dim=int(vectors.shape[1]))

    @classmethod
    def load(cls, build_dir: Path, expected_rows: int, dim: int | None = None) -> ChromaVectorStore | None:
        path = build_dir / CHROMA_DIR
        if not path.exists():
            return None
        try:
            client = _chroma_client(path)
        except Exception:
            log.exception("could not open the Chroma vector store", extra={"path": str(path)})
            return None
        try:
            store = cls(client, client.get_collection(CHROMA_COLLECTION, embedding_function=None), dim=dim)
        except Exception:
            log.exception("Chroma collection missing or unreadable", extra={"path": str(path)})
            client.close()
            return None
        if len(store) != expected_rows:  # also guarantees every chunk id exists
            log.warning("Chroma collection size mismatch", extra={"expected": expected_rows, "found": len(store)})
            store.close()
            return None
        return store

    def __len__(self) -> int:
        return self._count

    @property
    def dim(self) -> int:
        if self._dim is None:
            sample = self._collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            self._dim = len(embeddings[0]) if embeddings is not None and len(embeddings) else 0
        return self._dim

    def _query(
        self,
        query: np.ndarray,
        n: int,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[int, float]]:
        if self._count == 0 or n <= 0:
            return []
        with self._lock:
            res = self._collection.query(
                query_embeddings=np.asarray(query, dtype=np.float32).reshape(1, -1),
                n_results=min(n, self._count),
                ids=ids,
                where=where,
                include=["distances"],
            )
        # cosine space: distance = 1 - cosine similarity
        return [(int(cid) - 1, 1.0 - float(dist)) for cid, dist in zip(res["ids"][0], res["distances"][0], strict=True)]

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        if mask is None:
            return self._query(query, k)
        allowed = np.flatnonzero(mask)
        if len(allowed) == 0:
            return []
        k = min(k, len(allowed))
        if len(allowed) >= self._count:
            return self._query(query, k)
        fraction = len(allowed) / self._count
        if fraction >= BROAD_FILTER_FRACTION:
            # Broad filter: oversampled HNSW search, then drop non-qualifying chunks.
            n = min(self._count, math.ceil(k / fraction * 1.5) + 10)
            hits = [(row, sim) for row, sim in self._query(query, n) if mask[row]]
            if len(hits) >= k or n >= self._count:
                return hits[:k]
        # Search restricted to the qualifying chunks, inside Chroma.
        return self._query(query, k, ids=[str(row + 1) for row in allowed])

    def search_where(self, query: np.ndarray, k: int, where: dict[str, Any] | None) -> list[tuple[int, float]]:
        """Filtered search evaluated by Chroma's metadata index (``where``)."""
        return self._query(query, k, where=where)

    def close(self) -> None:
        try:
            self._client.close()  # reference-counted: the last client releases the files
        except Exception:  # pragma: no cover - best effort
            log.debug("Chroma client close failed", exc_info=True)
