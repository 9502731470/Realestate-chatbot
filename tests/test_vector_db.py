"""ChromaDB vector store: persistence, agreement with the exact NumPy store,
filter equivalence with SQL, graceful degradation and backend switching."""

import asyncio
import json
import os
import random
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import app.rag.index as index_module
from app.config import Settings
from app.main import _warm_up
from app.models import Record
from app.rag.embeddings import HashingEmbedder
from app.rag.index import SearchIndex, build_index, chunk_metadata, current_build_dir, vector_where
from app.rag.query import QueryFilters
from app.rag.vectorstore import CHROMA_DIR, NUMPY_FILE, ChromaVectorStore, NumpyVectorStore
from app.scraping.pipeline import ingest
from app.services.state import AppState
from tests.conftest import make_settings

ROOT = Path(__file__).resolve().parents[1]
CITIES = [
    ("Riyadh", "Saudi Arabia", "SAR"),
    ("Jeddah", "Saudi Arabia", "SAR"),
    ("Dubai", "United Arab Emirates", "AED"),
    ("Muscat", "Oman", "OMR"),
    ("London", "United Kingdom", "GBP"),
]
DISTRICTS = ["Al Malqa", "Al Narjis", "Hittin", "Al Rawdah", "Dubai Marina", "Al Mouj", "Mayfair"]
TYPES = ["apartment", "villa", "townhouse", "penthouse", "office", "land"]
FURNISHING = [None, "Furnished", "Unfurnished", "Semi-furnished"]
WORDS = [
    "sea", "view", "pool", "gym", "garden", "metro", "school", "mall",
    "quiet", "family", "luxury", "modern", "smart", "parking", "balcony",
]


def synthetic_records(n: int = 240, seed: int = 7) -> list[Record]:
    """Diverse records covering every filterable field (incl. missing values)."""
    rng = random.Random(seed)
    base = datetime(2026, 11, 1, tzinfo=UTC)
    records = []
    for i in range(n):
        city, country, currency = rng.choice(CITIES)
        purpose = rng.choice(["sale", "rent"])
        types = rng.sample(TYPES, rng.choice([1, 1, 2]))
        price = rng.choice([None, round(rng.uniform(20_000, 5_000_000), -3)])
        area = rng.choice([None, float(rng.randint(40, 1500))])
        source, kind = rng.choice([("wasalt", "listing"), ("wasalt", "listing"), ("darglobal", "project")])
        records.append(
            Record(
                id=f"{source}:{kind}:{i}",
                source=source,
                kind=kind,
                url=f"https://example.test/{i}",
                title=f"{types[0].title()} in {city} #{i}",
                text=" ".join(rng.sample(WORDS, 6)) + f" {types[0]} in {city}",
                purpose=purpose,
                property_type=types[0],
                property_types=types,
                category="commercial" if types[0] == "office" else "residential",
                price=price,
                price_max=round(price * rng.uniform(1.1, 2.0)) if price and rng.random() < 0.3 else None,
                currency=currency,
                rent_period=rng.choice(["year", "month"]) if purpose == "rent" else None,
                bedrooms=rng.choice([None, 1, 2, 3, 4, 5]),
                area_sqm=area,
                area_max_sqm=area * 1.5 if area and rng.random() < 0.2 else None,
                city=city,
                district=rng.choice([*DISTRICTS, None]),
                country=country,
                furnishing=rng.choice(FURNISHING),
                updated_at=base - timedelta(days=i),
            )
        )
    return records


def random_filters(rng: random.Random, records: list[Record]) -> QueryFilters:
    f = QueryFilters()
    if rng.random() < 0.5:
        f.purpose = rng.choice(["sale", "rent"])
    if rng.random() < 0.4:
        f.property_types = rng.sample(TYPES, rng.choice([1, 2]))
    if rng.random() < 0.2:
        f.category = rng.choice(["residential", "commercial"])
    r = rng.random()
    if r < 0.2:
        f.bedrooms = rng.randint(1, 5)
    elif r < 0.35:
        f.min_bedrooms = rng.randint(1, 4)
    elif r < 0.45:
        f.max_bedrooms = rng.randint(2, 5)
    if rng.random() < 0.35:
        f.max_price = float(rng.choice([100_000, 500_000, 2_000_000]))
        f.currency = rng.choice([None, "SAR", "USD", "AED"])
    if rng.random() < 0.25:
        f.min_price = float(rng.choice([50_000, 300_000, 1_000_000]))
    if rng.random() < 0.2:
        f.min_area = float(rng.choice([100, 300]))
    if rng.random() < 0.2:
        f.max_area = float(rng.choice([200, 800]))
    if rng.random() < 0.4:
        f.cities = [c[0] for c in rng.sample(CITIES, rng.choice([1, 2]))]
    if rng.random() < 0.15:
        f.districts = rng.sample(DISTRICTS, 1)
    if rng.random() < 0.15:
        f.countries = [rng.choice(CITIES)[1]]
    if rng.random() < 0.2:
        f.furnishing = rng.choice(["furnished", "unfurnished", "semi-furnished"])
    if rng.random() < 0.2:
        f.sources = [rng.choice(["wasalt", "darglobal"])]
    if rng.random() < 0.1:
        f.project_ids = [rng.choice(records).id for _ in range(3)]
    return f


def _build(tmp_path: Path, records: list[Record], backend: str) -> SearchIndex:
    out = tmp_path / backend
    meta = build_index(records, out, HashingEmbedder(), backend=backend)
    assert meta["vector_backend"] == backend
    return SearchIndex(out, HashingEmbedder())


def _sql_ids(index: SearchIndex, f: QueryFilters, kinds: list[str]) -> set[str]:
    where, params = index._where(f, kinds)
    return {row[0] for row in index._execute(f"SELECT id FROM records WHERE {where}", params)}


def _chroma_ids(index: SearchIndex, f: QueryFilters, kinds: list[str]) -> set[str]:
    assert isinstance(index.vectors, ChromaVectorStore)
    got = index.vectors._collection.get(where=vector_where(f, kinds), include=["metadatas"])
    return {meta["record_id"] for meta in got["metadatas"]}


@pytest.fixture(scope="module")
def pair(tmp_path_factory) -> tuple[SearchIndex, SearchIndex, list[Record]]:
    tmp = tmp_path_factory.mktemp("vector-db")
    records = synthetic_records()
    return _build(tmp, records, "chroma"), _build(tmp, records, "numpy"), records


def test_chroma_store_is_persisted_per_build(tmp_path: Path):
    records = synthetic_records(60)
    out = tmp_path / "build"
    meta = build_index(records, out, HashingEmbedder(), backend="chroma")
    assert (out / CHROMA_DIR).is_dir() and not (out / NUMPY_FILE).exists()

    first = SearchIndex(out, HashingEmbedder())
    assert isinstance(first.vectors, ChromaVectorStore) and len(first.vectors) == meta["chunks"]
    second = SearchIndex(out, HashingEmbedder())  # same build loaded twice (hot swap): shared, ref-counted
    first.close()
    qvec = HashingEmbedder().embed(["villa with pool"])[0]
    assert second.vectors.search(qvec, 3)  # still usable after the other client closed
    second.close()

    reopened = SearchIndex(out, HashingEmbedder())  # data survives closing every client
    assert reopened.stats()["vector_store"] == {"backend": "chroma", "vectors": meta["chunks"], "dim": 256, "loaded": True}
    got = reopened.vectors._collection.get(ids=["1"], include=["metadatas", "embeddings"])
    assert got["metadatas"][0] == chunk_metadata(records[0])  # chunk 1 belongs to the first record
    assert len(got["embeddings"][0]) == 256
    reopened.close()


def test_chunk_metadata_mirrors_the_sql_columns():
    rec = synthetic_records(1)[0].model_copy(
        update={"price": 4000.0, "price_max": None, "currency": "SAR", "purpose": "rent", "rent_period": "month",
                "area_sqm": None, "area_max_sqm": 250.0, "bedrooms": 0, "furnishing": "Semi furnished"}
    )
    md = chunk_metadata(rec)
    assert md["price_usd"] == md["price_hi_usd"] == pytest.approx(4000 * 12 / 3.75)  # annualised, in USD
    assert "area" not in md and md["area_hi"] == 250.0  # COALESCE(area_max_sqm, area_sqm)
    assert md["bedrooms"] == 0 and md["furnishing_key"] == "semi-furnished"
    assert all(v is not None for v in md.values())  # Chroma rejects None values


def test_vector_where_selects_exactly_the_sql_rows(pair):
    chroma, _numpy, records = pair
    rng = random.Random(11)
    checked = 0
    for _ in range(300):
        f = random_filters(rng, records)
        kinds = rng.choice([["listing"], ["project"], ["listing", "project"]])
        expected = _sql_ids(chroma, f, kinds)
        assert _chroma_ids(chroma, f, kinds) == expected, f
        checked += bool(expected)
    assert checked > 100  # most combinations select something


def test_chroma_agrees_with_exact_numpy_search(pair):
    chroma, numpy_index, _ = pair
    emb = HashingEmbedder()
    recalls = []
    for query in ["sea view villa in Jeddah", "family apartment near metro", "office", "luxury penthouse pool",
                  "شقة للايجار في الرياض"]:
        qvec = emb.embed([query])[0]
        exact = [row for row, _ in numpy_index.vectors.search(qvec, 10)]
        approx = chroma.vectors.search(qvec, 10)
        recalls.append(len(set(exact) & {row for row, _ in approx}) / 10)
        for row, sim in approx:  # similarities are true cosine similarities
            assert sim == pytest.approx(float(numpy_index.vectors.matrix[row] @ qvec), abs=1e-4)
        mask = chroma._chunk_mask(["project"], ["darglobal"], None)
        rows = [row for row, _ in chroma.vectors.search(qvec, 8, mask)]
        assert rows and all(chroma.chunk_kind[r] == "project" and chroma.chunk_source[r] == "darglobal" for r in rows)
        assert rows == [row for row, _ in numpy_index.vectors.search(qvec, 8, mask)]  # small set: exact
    assert np.mean(recalls) >= 0.9


@pytest.mark.parametrize("share", [0.0, 0.002, 0.05, 0.3, 0.7, 0.95, 1.0])
def test_filtered_search_plans_stay_inside_the_filter(pair, share):
    """Narrow filters search inside Chroma by id (exact); broad ones use ANN + post-filter."""
    chroma, numpy_index, _ = pair
    rng = np.random.default_rng(int(share * 1000))
    n = len(chroma.chunk_record)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=round(n * share), replace=False)] = True
    qvec = HashingEmbedder().embed(["modern apartment with balcony and gym"])[0]
    got = chroma.vectors.search(qvec, 10, mask)
    exact = numpy_index.vectors.search(qvec, 10, mask)
    assert len(got) == len(exact) == min(10, int(mask.sum()))
    assert all(mask[row] for row, _ in got)
    assert [s for _, s in got] == sorted((s for _, s in got), reverse=True)
    overlap = len({r for r, _ in got} & {r for r, _ in exact})
    assert overlap >= (len(exact) if share < 0.5 else len(exact) - 2)  # exact below the broad threshold


def test_filtered_record_ranking_runs_in_the_vector_db(pair):
    chroma, numpy_index, _ = pair
    qvec = HashingEmbedder().embed(["quiet family apartment with gym and parking"])[0]
    for f in (QueryFilters(purpose="rent", cities=["Riyadh"]), QueryFilters(property_types=["villa"]), QueryFilters()):
        allowed = _sql_ids(chroma, f, ["listing"])
        got = chroma.search_records(f, kinds=["listing"], qvec=qvec, limit=5)
        ref = numpy_index.search_records(f, kinds=["listing"], qvec=qvec, limit=5)
        assert got.total == ref.total == len(allowed)
        assert len(got.records) == min(5, len(allowed)) and {r.id for r in got.records} <= allowed
        assert len({r.id for r in got.records} & {r.id for r in ref.records}) >= len(ref.records) - 1


def test_truncated_candidates_use_chroma_metadata_filtering(pair, monkeypatch):
    chroma, _numpy, _ = pair
    monkeypatch.setattr(index_module, "RECORD_SCAN_LIMIT", 10)  # SQL candidate list gets truncated
    calls = []
    original = chroma.vectors.search_where
    monkeypatch.setattr(chroma.vectors, "search_where", lambda *a: calls.append(a[2]) or original(*a))
    f = QueryFilters(purpose="sale", property_types=["apartment", "villa"])
    qvec = HashingEmbedder().embed(["sea view"])[0]
    res = chroma.search_records(f, kinds=["listing"], qvec=qvec, limit=5)
    assert calls == [vector_where(f, ["listing"])]
    assert res.total > 10 and len(res.records) == 5 and {r.id for r in res.records} <= _sql_ids(chroma, f, ["listing"])


def test_vector_db_failure_degrades_to_keyword_search(pair, monkeypatch):
    chroma, _numpy, _ = pair

    def down(*_args, **_kwargs):
        raise RuntimeError("vector database unavailable")

    monkeypatch.setattr(chroma.vectors, "search", down)
    monkeypatch.setattr(chroma.vectors, "search_where", down)
    qvec = HashingEmbedder().embed(["villa in Jeddah"])[0]
    hits = chroma.search_chunks("villa Jeddah", qvec, k=5)
    assert hits and all(h.keyword_match for h in hits)
    res = chroma.search_records(QueryFilters(cities=["Riyadh"]), kinds=["listing"], qvec=qvec, limit=3)
    assert len(res.records) == 3  # recency order instead of an error


def test_filter_translation_shapes():
    assert vector_where(QueryFilters(), ["listing"]) == {"kind": {"$in": ["listing"]}}
    where = vector_where(QueryFilters(property_types=["villa", "townhouse"], bedrooms=3), ["listing"])
    assert {"$or": [{"t_villa": {"$eq": True}}, {"t_townhouse": {"$eq": True}}]} in where["$and"]


def test_switching_backend_reembeds_from_the_snapshot(ingested: Settings, tmp_path: Path):
    state = AppState(make_settings(tmp_path, vector_backend="numpy"))  # same data dir as `ingested`
    state.embedder = HashingEmbedder()
    assert state.load_index() and state.index.vectors.backend == "chroma"  # the active build keeps serving
    assert state.vectors_outdated()
    assert asyncio.run(state.ingestion.reindex("backend changed"))
    assert state.ingestion.status["state"] == "succeeded"
    assert isinstance(state.index.vectors, NumpyVectorStore) and state.index.meta["vector_backend"] == "numpy"
    assert not state.vectors_outdated()


def test_warm_up_adds_the_vector_store_to_a_keyword_only_build(tmp_path: Path, sites):
    settings = make_settings(tmp_path)
    asyncio.run(ingest(settings, None, transport=sites.transport, min_records=5))  # built without embeddings
    state = AppState(settings)
    asyncio.run(_warm_up(state))
    assert state.index.dense_enabled and state.index.meta["vector_backend"] == "chroma"
    assert not state.vectors_outdated()


def test_keyword_only_mode_never_loads_the_vector_db(ingested: Settings):
    """ENABLE_DENSE=false (512 MB hosts): ChromaDB is not even imported."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from app.rag.index import SearchIndex, current_build_dir\n"
        f"index = SearchIndex(current_build_dir(Path({str(ingested.index_root)!r})), None)\n"
        "assert index.vectors is None and index.search_chunks('villa', None, k=3)\n"
        "assert index.stats()['vector_store']['backend'] == 'chroma'\n"
        "assert 'chromadb' not in sys.modules, 'chromadb was imported'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert current_build_dir(ingested.index_root) is not None


def test_vsearch_cli_queries_the_vector_db(ingested: Settings):
    env = dict(
        os.environ,
        DATA_DIR=str(ingested.data_dir),
        MODEL_CACHE_DIR=str(ingested.model_cache_dir),
        EMBEDDING_MODEL="hashing",
        LOG_LEVEL="WARNING",
        OPENROUTER_API_KEY="",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "app.cli", "vsearch", "villa in Jeddah", "-k", "3"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [line for line in proc.stdout.splitlines() if line and not line.startswith('{"ts"')]
    assert json.loads(lines[0])["backend"] == "chroma" and json.loads(lines[0])["loaded"]
    assert len(lines) == 4 and all(float(line.split()[0]) <= 1.0 for line in lines[1:])
