from app.config import Settings
from app.models import Record
from app.rag.chunking import chunk_record
from app.rag.embeddings import HashingEmbedder
from app.rag.index import SearchIndex, current_build_dir, project_aliases
from app.rag.prompt import build_messages, fallback_answer
from app.rag.query import QueryFilters
from app.rag.retriever import Retriever


def _index(settings: Settings) -> SearchIndex:
    return SearchIndex(current_build_dir(settings.index_root), HashingEmbedder())


def test_chunking_adds_contextual_headers():
    rec = Record(id="r", source="darglobal", kind="project", url="u", title="DG1", city="Dubai", country="United Arab Emirates",
                 text="# DG1\n\nFacts here\n\n## About\n" + "word " * 400 + "\n\n## FAQs\nQ: one?\nA: yes")
    chunks = chunk_record(rec, max_words=150, overlap_words=30)
    assert len(chunks) >= 4
    assert all(c.text.startswith("DG1 — Dubai, United Arab Emirates [DarGlobal project]") for c in chunks)
    assert any("FAQs: Q: one?" in c.text for c in chunks)


def test_project_aliases():
    aliases = project_aliases("Da Vinci Tower, Interiors by Pagani")
    assert {"da vinci tower", "pagani"} <= aliases
    assert "tower" not in aliases
    assert "dg1" in project_aliases("DG1")


def test_structured_search_filters(ingested: Settings):
    idx = _index(ingested)
    res = idx.search_records(QueryFilters(purpose="rent", property_types=["apartment"], cities=["Riyadh"], max_price=100_000),
                             kinds=["listing"], limit=10)
    assert res.total == 3
    assert all(r.purpose == "rent" and r.city == "Riyadh" and r.price <= 100_000 for r in res.records)
    cheapest = idx.search_records(QueryFilters(purpose="rent", cities=["Riyadh"], sort="price_asc"), kinds=["listing"], limit=1)
    assert cheapest.records[0].price == 60_000
    dg = idx.search_records(QueryFilters(sources=["darglobal"], countries=["United Arab Emirates"]), kinds=["project"], limit=5)
    assert [r.title for r in dg.records] == ["DG1"]


def test_vocabulary_from_data(ingested: Settings):
    idx = _index(ingested)
    assert "narjis" in idx.vocab.districts
    assert any(alias == "trump tower jeddah" for alias, _, _ in idx.vocab.projects)


def test_retriever_relaxes_least_important_filters_first(ingested: Settings):
    ret = Retriever(_index(ingested))
    r = ret.retrieve("apartments in An-Narjis with 5 bedrooms")
    assert r.relaxed[:1] == ["exact bedroom count"] and "district" not in r.relaxed
    assert all(s.record.district == "An-Narjis" for s in r.sources if s.reason == "structured match")


def test_retriever_named_project_and_general(ingested: Settings):
    ret = Retriever(_index(ingested))
    r = ret.retrieve("Is Trump Tower Jeddah developed by the Trump Organization?")
    assert r.sources[0].record.id == "darglobal:project:trump-tower-jeddah"
    assert any("Trump Organization" in p for p in r.sources[0].passages)
    g = ret.retrieve("How do I buy a house in Saudi Arabia?")
    assert g.parsed.intent == "general"
    assert any(s.record.kind == "article" for s in g.sources)


def test_retriever_follow_up_with_ordinal(ingested: Settings):
    ret = Retriever(_index(ingested))
    first = ret.retrieve("villas for sale in Jeddah")
    ids = [s.record.id for s in first.sources]
    follow = ret.retrieve("tell me more about the second one", ["villas for sale in Jeddah"], ids)
    assert follow.sources[0].record.id == ids[1]


def test_prompt_is_grounded_and_fallback_is_useful(ingested: Settings):
    ret = Retriever(_index(ingested))
    r = ret.retrieve("3 bedroom apartments for rent in Riyadh under 100k")
    msgs = build_messages("q", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], r,
                          snapshot_date="2026-11-18", max_context_chars=6000)
    assert msgs[0]["role"] == "system" and "Use only the SOURCES" in msgs[0]["content"]
    assert "[1] Wasalt.sa listing |" in msgs[-1]["content"] and "SEARCH NOTES" in msgs[-1]["content"]
    assert "ANSWER STYLE" in msgs[-1]["content"] and "Reply in: English" in msgs[-1]["content"]
    fb = fallback_answer(r, "quota")
    assert "[1]" in fb and "Source: Wasalt.sa" in fb
