"""Persistent hybrid search index (one immutable *build* per data refresh).

``data/index/<build_id>/``
* ``store.sqlite`` - normalised records (structured columns for SQL filters),
  chunk text, and an FTS5 table for BM25 keyword search (EN + AR normalised,
  Arabic light-stemmed);
* ``chroma/`` (default) or ``vectors.npy`` - the vector store holding the chunk
  embeddings (ChromaDB, or an exact NumPy matrix); see ``vectorstore.py``;
* ``meta.json``   - build metadata (format version, counts, embedding model).

Builds are activated atomically by rewriting ``data/index/CURRENT``
(blue/green): a refresh never disturbs the index that is serving traffic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.models import Record, comparable_usd, to_usd, utcnow
from app.rag.chunking import chunk_record
from app.rag.embeddings import Embedder
from app.rag.query import QueryFilters, Vocabulary, default_currency_for
from app.rag.vectorstore import BACKENDS, ChromaVectorStore, NumpyVectorStore, VectorStore
from app.textutil import normalize_for_index, normalize_for_search, place_key, search_terms
from app.vocab import detect_brands, find_places

log = logging.getLogger(__name__)

INDEX_FORMAT = 3  # bump when the schema changes; old builds are rebuilt from the snapshot
RECORD_SCAN_LIMIT = 5000  # max SQL candidates re-ranked semantically by search_records

SCHEMA = """
CREATE TABLE records (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  kind TEXT NOT NULL,
  url TEXT NOT NULL,
  title TEXT NOT NULL,
  purpose TEXT,
  property_type TEXT,
  ptypes TEXT,
  category TEXT,
  price REAL,
  price_max REAL,
  price_cmp_usd REAL,
  price_max_cmp_usd REAL,
  currency TEXT,
  bedrooms INTEGER,
  bathrooms INTEGER,
  area_sqm REAL,
  area_max_sqm REAL,
  city TEXT,
  city_key TEXT,
  district TEXT,
  district_key TEXT,
  country TEXT,
  furnishing_key TEXT,
  sort_date TEXT,
  chunk_start INTEGER NOT NULL,
  chunk_end INTEGER NOT NULL,
  data TEXT NOT NULL
);
CREATE INDEX ix_records_filter ON records(kind, source, city_key, purpose);
CREATE INDEX ix_records_price ON records(price_cmp_usd);
CREATE TABLE chunks (
  rowid INTEGER PRIMARY KEY,
  record_id TEXT NOT NULL,
  ord INTEGER NOT NULL,
  kind TEXT NOT NULL,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  text TEXT NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(title, body, tokenize = 'unicode61 remove_diacritics 2');
"""

SORT_SQL = {
    "price_asc": "price_cmp_usd IS NULL, price_cmp_usd ASC",
    "price_desc": "price_cmp_usd IS NULL, price_cmp_usd DESC",
    "area_desc": "area_sqm IS NULL, area_sqm DESC",
    "newest": "sort_date IS NULL, sort_date DESC",
}

_GENERIC_DISTRICT_WORDS = {
    "north", "south", "east", "west", "center", "centre", "central", "industrial", "airport",
    "downtown", "city", "new", "old", "park", "garden", "gardens", "beach", "corniche", "hills",
    "valley", "village", "marina", "palm", "bay", "heights", "district", "area", "town", "port",
    "station", "university", "hospital", "market", "school", "king", "road", "street", "the",
}
_GENERIC_PROJECT_ALIASES = {
    "villas", "villa", "residences", "residence", "tower", "towers", "mansions", "apartments",
    "project", "hotel", "resort", "golf", "the", "living", "homes", "interiors", "oman", "dubai",
    "jeddah", "riyadh", "doha", "qatar", "saudi arabia", "uae", "spain", "london",
}


class IndexFormatError(RuntimeError):
    """The on-disk build was written by an older version of the code."""


def furnishing_key(value: str | None) -> str | None:
    if not value:
        return None
    v = normalize_for_search(value).replace(" ", "")
    if "semi" in v:
        return "semi-furnished"
    if v.startswith("un") or "not" in v:
        return "unfurnished"
    if "furnish" in v:
        return "furnished"
    return None


def city_key(city: str | None) -> str | None:
    if not city:
        return None
    cities, _ = find_places(city)
    return place_key(cities[0] if cities else city)


def project_aliases(title: str) -> set[str]:
    aliases = {normalize_for_search(title)}
    norm = normalize_for_search(title)
    if norm.startswith("the "):
        aliases.add(norm[4:])
    for part in re.split(r",|\bby\b|\binteriors\b|–|-|\(|\)", title, flags=re.I):
        p = normalize_for_search(part)
        if p.startswith("the "):
            p = p[4:]
        aliases.add(p)
    for brand in detect_brands(title):
        aliases.add(normalize_for_search(brand))
    return {
        a for a in aliases
        if a and a not in _GENERIC_PROJECT_ALIASES and (len(a) >= 4 or any(ch.isdigit() for ch in a))
    }


def chunk_metadata(rec: Record) -> dict[str, Any]:
    """Record-level fields stored with every chunk in the vector database.

    They make the database self-describing (inspectable with any Chroma tool)
    and let Chroma apply the filters itself when the SQL candidate set is too
    large to pass as ids. Values use the same normalised forms as the SQL
    columns (``city_key``, ``price_cmp_usd``...), and ``vector_where`` mirrors
    ``SearchIndex._where`` clause by clause; a test asserts both select exactly
    the same records.
    """
    md: dict[str, Any] = {"record_id": rec.id, "kind": rec.kind, "source": rec.source}
    if rec.purpose:
        md["purpose"] = rec.purpose
    if rec.category:
        md["category"] = rec.category
    for ptype in rec.property_types or ([rec.property_type] if rec.property_type else []):
        md[f"t_{ptype}"] = True
    if rec.bedrooms is not None:
        md["bedrooms"] = int(rec.bedrooms)
    price = comparable_usd(rec.price, rec.currency, rec.purpose, rec.rent_period)
    price_hi = comparable_usd(rec.price_max, rec.currency, rec.purpose, rec.rent_period)
    if price is not None:
        md["price_usd"] = float(price)
    if price_hi is not None or price is not None:
        md["price_hi_usd"] = float(price_hi if price_hi is not None else price)
    if rec.area_sqm is not None:
        md["area"] = float(rec.area_sqm)
    area_hi = rec.area_max_sqm if rec.area_max_sqm is not None else rec.area_sqm
    if area_hi is not None:
        md["area_hi"] = float(area_hi)
    for key, value in (
        ("city_key", city_key(rec.city)),
        ("district_key", place_key(rec.district) or None),
        ("country", rec.country),
        ("furnishing_key", furnishing_key(rec.furnishing)),
    ):
        if value:
            md[key] = value
    return md


def vector_where(f: QueryFilters, kinds: list[str]) -> dict[str, Any]:
    """The same filters as the SQL ``_where`` clause, as a Chroma metadata filter."""
    clauses: list[dict[str, Any]] = [{"kind": {"$in": list(kinds)}}]
    if f.sources:
        clauses.append({"source": {"$in": list(f.sources)}})
    if f.project_ids:
        clauses.append({"record_id": {"$in": list(f.project_ids)}})
    if f.purpose:
        clauses.append({"purpose": {"$eq": f.purpose}})
    if f.property_types:
        types = [{f"t_{t}": {"$eq": True}} for t in f.property_types]
        clauses.append(types[0] if len(types) == 1 else {"$or": types})
    if f.category:
        clauses.append({"category": {"$eq": f.category}})
    if f.bedrooms is not None:
        clauses.append({"bedrooms": {"$eq": int(f.bedrooms)}})
    if f.min_bedrooms is not None:
        clauses.append({"bedrooms": {"$gte": int(f.min_bedrooms)}})
    if f.max_bedrooms is not None:
        clauses.append({"bedrooms": {"$lte": int(f.max_bedrooms)}})
    currency = default_currency_for(f)
    if f.max_price is not None:
        usd = to_usd(f.max_price, currency)  # unknown currency -> matches nothing, as in SQL
        clauses.append({"price_usd": {"$lte": float(usd) if usd is not None else -1.0}})
    if f.min_price is not None:
        usd = to_usd(f.min_price, currency)
        clauses.append({"price_hi_usd": {"$gte": float(usd) if usd is not None else 1e308}})
    if f.min_area is not None:
        clauses.append({"area_hi": {"$gte": float(f.min_area)}})
    if f.max_area is not None:
        clauses.append({"area": {"$lte": float(f.max_area)}})
    if f.cities:
        keys = [k for k in (city_key(c) for c in f.cities) if k]
        clauses.append({"city_key": {"$in": keys}})
    if f.districts:
        clauses.append({"district_key": {"$in": [place_key(d) for d in f.districts]}})
    if f.countries:
        clauses.append({"country": {"$in": list(f.countries)}})
    if f.furnishing:
        clauses.append({"furnishing_key": {"$eq": f.furnishing}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# --------------------------------------------------------------------- building
def build_index(
    records: list[Record],
    out_dir: Path,
    embedder: Embedder | None,
    *,
    backend: str = "chroma",
    ef_search: int = 100,
    batch_size: int = 16,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown vector backend {backend!r}; expected one of {BACKENDS}")
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "store.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    chunk_rows: list[tuple[Any, ...]] = []
    fts_rows: list[tuple[Any, ...]] = []
    record_rows: list[tuple[Any, ...]] = []
    chunk_texts: list[str] = []
    chunk_metas: list[dict[str, Any]] = []
    rowid = 0
    for rec in records:
        chunks = chunk_record(rec)
        start = rowid + 1
        for ch in chunks:
            rowid += 1
            chunk_rows.append((rowid, rec.id, ch.ord, rec.kind, rec.source, ch.title, ch.text))
            fts_rows.append((rowid, normalize_for_index(ch.title), normalize_for_index(ch.text)))
            chunk_texts.append(ch.text)
        rec_meta = chunk_metadata(rec)
        chunk_metas.extend(rec_meta for _ in chunks)
        ptypes = " " + " ".join(rec.property_types or ([rec.property_type] if rec.property_type else [])) + " "
        sort_date = (rec.updated_at or rec.published_at or rec.scraped_at).isoformat()
        record_rows.append(
            (
                rec.id, rec.source, rec.kind, rec.url, rec.title, rec.purpose, rec.property_type, ptypes,
                rec.category, rec.price, rec.price_max,
                comparable_usd(rec.price, rec.currency, rec.purpose, rec.rent_period),
                comparable_usd(rec.price_max, rec.currency, rec.purpose, rec.rent_period),
                rec.currency, rec.bedrooms, rec.bathrooms, rec.area_sqm, rec.area_max_sqm,
                rec.city, city_key(rec.city), rec.district, place_key(rec.district) or None,
                rec.country, furnishing_key(rec.furnishing), sort_date, start, rowid, rec.model_dump_json(),
            )
        )
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", chunk_rows)
    conn.executemany("INSERT INTO chunks_fts(rowid, title, body) VALUES (?,?,?)", fts_rows)
    conn.executemany(f"INSERT INTO records VALUES ({','.join('?' * 28)})", record_rows)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    conn.commit()
    conn.close()

    dim = None
    stored_backend = None
    if embedder is not None and chunk_texts:
        t0 = time.perf_counter()
        store: VectorStore
        if backend == "chroma":
            store = ChromaVectorStore.build(
                embedder, chunk_texts, chunk_metas, out_dir, ef_search=ef_search, batch_size=batch_size
            )
        else:
            store = NumpyVectorStore.build(embedder, chunk_texts, out_dir, batch_size=batch_size)
        dim, stored_backend = store.dim, store.backend
        store.close()
        log.info(
            "embedded chunks into vector store",
            extra={"chunks": len(chunk_texts), "backend": backend, "seconds": round(time.perf_counter() - t0, 1)},
        )

    by_source_kind = Counter(f"{r.source}:{r.kind}" for r in records)
    meta = {
        "format": INDEX_FORMAT,
        "built_at": utcnow().isoformat(),
        "records": len(records),
        "chunks": rowid,
        "embedding_model": embedder.name if embedder is not None else None,
        "embedding_dim": dim,
        "vector_backend": stored_backend,
        "counts": dict(sorted(by_source_kind.items())),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def new_build_dir(index_root: Path) -> Path:
    """Unique, lexicographically time-ordered build directory."""
    for _ in range(5):
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%f")
        path = index_root / f"build-{stamp}-{os.getpid()}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            time.sleep(0.001)
    raise RuntimeError("could not allocate a unique index build directory")


def activate(index_root: Path, build_dir: Path, keep: int = 2) -> None:
    tmp = index_root / "CURRENT.tmp"
    tmp.write_text(build_dir.name)
    os.replace(tmp, index_root / "CURRENT")
    builds = sorted((p for p in index_root.glob("build-*") if p.is_dir()), key=lambda p: p.name)
    for old in builds[:-keep]:
        if old.name != build_dir.name:
            shutil.rmtree(old, ignore_errors=True)


def current_build_dir(index_root: Path) -> Path | None:
    pointer = index_root / "CURRENT"
    if not pointer.exists():
        return None
    path = index_root / pointer.read_text().strip()
    return path if (path / "store.sqlite").exists() else None


# -------------------------------------------------------------------- searching
@dataclass
class ChunkHit:
    rowid: int
    record_id: str
    text: str
    score: float
    similarity: float | None = None
    keyword_match: bool = False


@dataclass
class RecordSearch:
    records: list[Record]
    total: int


class SearchIndex:
    def __init__(self, build_dir: Path, embedder: Embedder | None) -> None:
        self.dir = build_dir
        self.meta: dict[str, Any] = json.loads((build_dir / "meta.json").read_text())
        if self.meta.get("format") != INDEX_FORMAT:
            raise IndexFormatError(f"index format {self.meta.get('format')} != {INDEX_FORMAT}")
        self._conn = sqlite3.connect(f"file:{build_dir / 'store.sqlite'}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()

        rows = self._conn.execute("SELECT rowid, record_id, kind, source FROM chunks ORDER BY rowid").fetchall()
        self.chunk_record = [r[1] for r in rows]
        self.chunk_kind = np.array([r[2] for r in rows]) if rows else np.array([], dtype=str)
        self.chunk_source = np.array([r[3] for r in rows]) if rows else np.array([], dtype=str)

        self.vectors: VectorStore | None = None
        self.embedder: Embedder | None = None
        self.record_chunks: dict[str, tuple[int, int]] = {}
        for rid, start, end in self._conn.execute("SELECT id, chunk_start, chunk_end FROM records"):
            self.record_chunks[rid] = (start, end)
        if embedder is not None:
            self.attach_embedder(embedder)

        self.records: dict[str, Record] = {}
        for rid, data in self._conn.execute("SELECT id, data FROM records"):
            self.records[rid] = Record.model_validate_json(data)
        self.vocab = self._build_vocab()

    def attach_embedder(self, embedder: Embedder) -> bool:
        """Open this build's vector store for ``embedder`` and enable semantic search.

        Also used to upgrade an index that is already serving keyword search
        (start-up warm-up) in place, so the records aren't loaded a second time:
        that matters on a 512 MB instance.
        """
        if self.vectors is not None:
            return True
        if self.meta.get("embedding_model") != embedder.name:
            return False
        rows = len(self.chunk_record)
        vectors: VectorStore | None = None
        if self.meta.get("vector_backend") == "chroma":
            vectors = ChromaVectorStore.load(self.dir, rows, dim=self.meta.get("embedding_dim"))
        elif self.meta.get("vector_backend") == "numpy":
            vectors = NumpyVectorStore.load(self.dir, expected_rows=rows)
        if vectors is None:
            return False
        self.embedder = embedder  # set before vectors: searches check `vectors` to go dense
        self.vectors = vectors
        return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        if self.vectors is not None:
            self.vectors.close()

    @property
    def dense_enabled(self) -> bool:
        return self.vectors is not None

    @property
    def multilingual_vectors(self) -> bool:
        """Can the loaded embedding model read Arabic? (False for English-only models.)"""
        return self.embedder is None or self.embedder.profile.multilingual

    # ------------------------------------------------------------ vocabulary
    def _build_vocab(self) -> Vocabulary:
        districts: dict[str, tuple[str, str | None]] = {}
        projects: list[tuple[str, str, str]] = []
        cities: set[str] = set()
        for rec in self.records.values():
            if rec.city:
                cities.add(rec.city)
            if rec.kind in ("listing", "project") and rec.district:
                key = place_key(rec.district)
                if len(key) >= 4 and key not in _GENERIC_DISTRICT_WORDS and not any(ch.isdigit() for ch in key):
                    districts.setdefault(key, (rec.district, rec.city))
            if rec.kind == "project":
                for alias in project_aliases(rec.title):
                    projects.append((alias, rec.id, rec.title))
        projects.sort(key=lambda p: len(p[0]), reverse=True)
        return Vocabulary(districts=districts, projects=projects, cities=cities)

    # ----------------------------------------------------------------- query
    def embed_query(self, text: str) -> np.ndarray | None:
        if self.embedder is None or not text.strip():
            return None
        try:
            return self.embedder.embed_query(text)
        except Exception:  # pragma: no cover - defensive
            log.exception("query embedding failed")
            return None

    def _execute(self, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _chunk_mask(
        self, kinds: list[str] | None, sources: list[str] | None, record_ids: list[str] | None
    ) -> np.ndarray | None:
        """Boolean mask of the chunks a filtered vector search may return."""
        if not (kinds or sources or record_ids):
            return None
        mask = np.ones(len(self.chunk_record), dtype=bool)
        if kinds:
            mask &= np.isin(self.chunk_kind, kinds)
        if sources:
            mask &= np.isin(self.chunk_source, sources)
        if record_ids:
            allowed = np.zeros(len(self.chunk_record), dtype=bool)
            for rid in record_ids:
                if rid in self.record_chunks:
                    start, end = self.record_chunks[rid]
                    allowed[start - 1 : end] = True
            mask &= allowed
        return mask

    def search_chunks(
        self,
        query: str,
        qvec: np.ndarray | None,
        *,
        k: int = 8,
        kinds: list[str] | None = None,
        sources: list[str] | None = None,
        record_ids: list[str] | None = None,
        per_record: int = 2,
        candidates: int = 60,
        min_similarity: float = 0.0,
    ) -> list[ChunkHit]:
        """Hybrid search: BM25 (FTS5) + dense cosine, fused with Reciprocal Rank Fusion.

        ``min_similarity`` is a relevance floor for chunks found *only* by the
        dense retriever: RRF always returns *something*, so without a floor an
        unrelated passage can fill a slot (and spend prompt tokens).
        """
        scores: dict[int, float] = {}
        keyword_rows: set[int] = set()
        sims: dict[int, float] = {}

        terms = search_terms(query)
        if terms:
            match = " OR ".join(f'"{t}"*' if len(t) >= 4 and t.isascii() else f'"{t}"' for t in terms)
            sql = (
                "SELECT chunks_fts.rowid FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ?"
            )
            params: list[Any] = [match]
            if kinds:
                sql += f" AND chunks.kind IN ({','.join('?' * len(kinds))})"
                params += kinds
            if sources:
                sql += f" AND chunks.source IN ({','.join('?' * len(sources))})"
                params += sources
            if record_ids:
                sql += f" AND chunks.record_id IN ({','.join('?' * len(record_ids))})"
                params += record_ids
            sql += " ORDER BY bm25(chunks_fts, 3.0, 1.0) LIMIT ?"
            params.append(candidates)
            try:
                for rank, (rowid,) in enumerate(self._execute(sql, params)):
                    keyword_rows.add(rowid)
                    scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (60 + rank)
            except sqlite3.OperationalError:
                log.warning("FTS query failed", extra={"match": match[:200]})

        if qvec is not None and self.vectors is not None and len(self.chunk_record):
            try:
                dense = self.vectors.search(qvec, candidates, self._chunk_mask(kinds, sources, record_ids))
            except Exception:
                log.exception("vector search failed; using keyword results only")
                dense = []
            for rank, (row, sim) in enumerate(dense):
                rowid = row + 1
                sims[rowid] = sim
                scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (60 + rank)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        hits: list[ChunkHit] = []
        per: Counter[str] = Counter()
        for rowid, score in ranked:
            if rowid not in keyword_rows and sims.get(rowid, -1.0) < min_similarity:
                continue  # dense-only and below the relevance floor
            rid = self.chunk_record[rowid - 1]
            if per[rid] >= per_record:
                continue
            per[rid] += 1
            text = self._execute("SELECT text FROM chunks WHERE rowid = ?", (rowid,))[0][0]
            hits.append(ChunkHit(rowid, rid, text, score, sims.get(rowid), rowid in keyword_rows))
            if len(hits) >= k:
                break
        return hits

    def _where(self, f: QueryFilters, kinds: list[str]) -> tuple[str, list[Any]]:
        clauses = [f"kind IN ({','.join('?' * len(kinds))})"]
        params: list[Any] = list(kinds)
        if f.sources:
            clauses.append(f"source IN ({','.join('?' * len(f.sources))})")
            params += f.sources
        if f.project_ids:
            clauses.append(f"id IN ({','.join('?' * len(f.project_ids))})")
            params += f.project_ids
        if f.purpose:
            clauses.append("purpose = ?")
            params.append(f.purpose)
        if f.property_types:
            clauses.append("(" + " OR ".join("ptypes LIKE ?" for _ in f.property_types) + ")")
            params += [f"% {t} %" for t in f.property_types]
        if f.category:
            clauses.append("category = ?")
            params.append(f.category)
        if f.bedrooms is not None:
            clauses.append("bedrooms = ?")
            params.append(f.bedrooms)
        if f.min_bedrooms is not None:
            clauses.append("bedrooms >= ?")
            params.append(f.min_bedrooms)
        if f.max_bedrooms is not None:
            clauses.append("bedrooms <= ?")
            params.append(f.max_bedrooms)
        # Budgets are compared on a common basis: USD, and rents annualised
        # (the parser already turns "4,000 per month" into 48,000 per year).
        currency = default_currency_for(f)
        if f.max_price is not None:
            clauses.append("price_cmp_usd IS NOT NULL AND price_cmp_usd <= ?")
            params.append(to_usd(f.max_price, currency))
        if f.min_price is not None:
            clauses.append("COALESCE(price_max_cmp_usd, price_cmp_usd) >= ?")
            params.append(to_usd(f.min_price, currency))
        if f.min_area is not None:
            clauses.append("COALESCE(area_max_sqm, area_sqm) >= ?")
            params.append(f.min_area)
        if f.max_area is not None:
            clauses.append("area_sqm <= ?")
            params.append(f.max_area)
        if f.cities:
            keys = [k for k in (city_key(c) for c in f.cities) if k]
            clauses.append(f"city_key IN ({','.join('?' * len(keys))})")
            params += keys
        if f.districts:
            keys = [place_key(d) for d in f.districts]
            clauses.append(f"district_key IN ({','.join('?' * len(keys))})")
            params += keys
        if f.countries:
            clauses.append(f"country IN ({','.join('?' * len(f.countries))})")
            params += f.countries
        if f.furnishing:
            clauses.append("furnishing_key = ?")
            params.append(f.furnishing)
        return " AND ".join(clauses), params

    def search_records(
        self,
        f: QueryFilters,
        *,
        kinds: list[str],
        qvec: np.ndarray | None = None,
        limit: int = 8,
    ) -> RecordSearch:
        where, params = self._where(f, kinds)
        total = int(self._execute(f"SELECT COUNT(*) FROM records WHERE {where}", params)[0][0])
        if total == 0:
            return RecordSearch(records=[], total=0)
        if f.sort in SORT_SQL:
            rows = self._execute(f"SELECT id FROM records WHERE {where} ORDER BY {SORT_SQL[f.sort]} LIMIT ?", [*params, limit])
            return RecordSearch(records=[self.records[r[0]] for r in rows], total=total)

        rows = self._execute(
            f"SELECT id, chunk_start, chunk_end FROM records WHERE {where} ORDER BY sort_date DESC LIMIT ?",
            [*params, RECORD_SCAN_LIMIT],
        )
        if qvec is not None and isinstance(self.vectors, ChromaVectorStore) and len(rows) > limit:
            ids = self._chroma_rank(f, kinds, qvec, rows, total, limit)
            if ids is not None:
                return RecordSearch(records=[self.records[i] for i in ids], total=total)
        if qvec is not None and isinstance(self.vectors, NumpyVectorStore) and len(rows) > limit:
            sims = self.vectors.similarities(qvec)
            scored = []
            for position, (rid, start, end) in enumerate(rows):
                best = float(np.max(sims[start - 1 : end]))
                recency = 0.02 * (1 - position / len(rows))  # mild tie-breaker toward fresher records
                scored.append((best + recency, rid))
            scored.sort(reverse=True)
            ids = [rid for _, rid in scored[:limit]]
        else:
            ids = [r[0] for r in rows[:limit]]
        return RecordSearch(records=[self.records[i] for i in ids], total=total)

    def _chroma_rank(
        self,
        f: QueryFilters,
        kinds: list[str],
        qvec: np.ndarray,
        rows: list[tuple[Any, ...]],
        total: int,
        limit: int,
    ) -> list[str] | None:
        """Semantic ranking of the records matching ``f``, inside Chroma.

        SQLite has already resolved the filters (exact count, relaxation), so
        Chroma searches only the chunks of those records. If the SQL candidate
        list was truncated, Chroma applies the same filters itself (``where``).
        The list is topped up in recency order if fewer than ``limit`` records
        come back, and gets the same mild recency tie-breaker as the NumPy path.
        """
        assert isinstance(self.vectors, ChromaVectorStore)
        complete = len(rows) == total
        try:
            if complete:
                mask = np.zeros(len(self.chunk_record), dtype=bool)
                for _rid, start, end in rows:
                    mask[start - 1 : end] = True
                hits = self.vectors.search(qvec, limit * 4, mask)
            else:
                hits = self.vectors.search_where(qvec, limit * 4, vector_where(f, kinds))
        except Exception:
            log.exception("vector record search failed; falling back to recency order")
            return None
        position = {r[0]: i for i, r in enumerate(rows)}
        best: dict[str, float] = {}
        for row, sim in hits:
            rid = self.chunk_record[row]
            if sim > best.get(rid, -2.0):
                best[rid] = sim
        scored = sorted(
            ((sim + 0.02 * (1 - position.get(rid, len(rows)) / len(rows)), rid) for rid, sim in best.items()),
            reverse=True,
        )
        ids = [rid for _, rid in scored]
        seen = set(ids)
        for rid, *_ in rows:
            if len(ids) >= limit:
                break
            if rid not in seen:
                ids.append(rid)
                seen.add(rid)
        return ids[:limit]

    def stats(self) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        cities: Counter[str] = Counter()
        for rec in self.records.values():
            counts[f"{rec.source}:{rec.kind}"] += 1
            if rec.kind == "listing" and rec.city:
                cities[rec.city] += 1
        return {
            "built_at": self.meta.get("built_at"),
            "records": len(self.records),
            "chunks": len(self.chunk_record),
            "counts": dict(sorted(counts.items())),
            "top_listing_cities": dict(cities.most_common(10)),
            "dense_retrieval": self.dense_enabled,
            "vector_store": {
                "backend": self.vectors.backend,
                "vectors": len(self.vectors),
                "dim": self.vectors.dim,
                "loaded": True,
            }
            if self.vectors is not None
            else {
                "backend": self.meta.get("vector_backend"),
                "vectors": self.meta.get("chunks") if self.meta.get("vector_backend") else 0,
                "dim": self.meta.get("embedding_dim"),
                "loaded": False,
            },
            "embedding_model": self.meta.get("embedding_model") if self.dense_enabled else None,
            "multilingual_vectors": self.multilingual_vectors if self.dense_enabled else None,
        }
