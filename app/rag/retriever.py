"""Retrieval orchestration: structured search + hybrid passages + citations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from app.models import Record
from app.rag.index import ChunkHit, SearchIndex, city_key
from app.rag.query import ParsedQuery, QueryFilters, QueryParser
from app.textutil import is_arabic

log = logging.getLogger(__name__)

MAX_SOURCES = 12


@dataclass
class Source:
    n: int
    record: Record
    passages: list[str] = field(default_factory=list)
    reason: str = "match"

    def card(self) -> dict[str, Any]:
        rec = self.record
        return {
            "n": self.n,
            "id": rec.id,
            "title": rec.title,
            "url": rec.url,
            "source": rec.source,
            "source_label": rec.source_label,
            "kind": rec.kind,
            "price": rec.price_label(),
            "purpose": rec.purpose,
            "property_type": rec.property_type,
            "bedrooms": rec.bedrooms,
            "bathrooms": rec.bathrooms,
            "area_sqm": rec.area_sqm,
            "location": rec.location_label() or None,
            "status": rec.status,
            "completion": rec.completion,
            "updated": (rec.updated_at or rec.published_at).date().isoformat()
            if (rec.updated_at or rec.published_at)
            else None,
            "snippet": (self.passages[0] if self.passages else rec.summary or "")[:280],
        }


@dataclass
class RetrievalResult:
    parsed: ParsedQuery
    sources: list[Source]
    total_matches: int | None = None
    relaxed: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.sources


Relaxer = Callable[[QueryFilters], QueryFilters | None]

# Least important constraints are relaxed first; city and purpose are never relaxed
# (silently changing them would mislead the user). Location beats bedroom count.
RELAXATION_STEPS: list[tuple[str, Relaxer]] = [
    (
        "exact bedroom count",
        lambda f: replace(f, bedrooms=None, min_bedrooms=f.bedrooms) if f.bedrooms is not None else None,
    ),
    ("furnishing", lambda f: replace(f, furnishing=None) if f.furnishing else None),
    (
        "price range (±25%)",
        lambda f: replace(
            f,
            max_price=f.max_price * 1.25 if f.max_price else None,
            min_price=f.min_price * 0.75 if f.min_price else None,
        )
        if (f.max_price or f.min_price)
        else None,
    ),
    ("area", lambda f: replace(f, min_area=None, max_area=None) if (f.min_area or f.max_area) else None),
    (
        "bedrooms",
        lambda f: replace(f, min_bedrooms=None, max_bedrooms=None) if (f.min_bedrooms or f.max_bedrooms) else None,
    ),
    ("district", lambda f: replace(f, districts=[]) if f.districts else None),
    ("property type", lambda f: replace(f, property_types=[], category=None) if f.property_types else None),
]


class Retriever:
    def __init__(
        self,
        index: SearchIndex,
        *,
        top_k_listings: int = 8,
        top_k_passages: int = 6,
        min_passage_similarity: float = 0.25,
    ) -> None:
        self.index = index
        self.parser = QueryParser(index.vocab)
        self.top_k_listings = top_k_listings
        self.top_k_passages = top_k_passages
        self.min_sim = min_passage_similarity

    def dense_query_text(self, parsed: ParsedQuery) -> str | None:
        """The text to embed for the vector search.

        An English-only embedding model can't read Arabic. For an Arabic question
        it gets an English rendering of what the bilingual parser understood
        ("3 bedroom apartment for rent in Al Malqa, Riyadh"), or no vector search
        when nothing was understood (keyword search and SQL filters still run).
        """
        text = parsed.search_text or parsed.text
        if self.index.multilingual_vectors or not is_arabic(text):
            return text
        f = parsed.filters
        words: list[str] = []
        if f.bedrooms is not None:
            words.append(f"{f.bedrooms} bedroom")
        elif f.min_bedrooms is not None:
            words.append(f"{f.min_bedrooms}+ bedroom")
        words += [t.replace("_", " ").replace("-", " ") for t in f.property_types]
        if f.furnishing:
            words.append(f.furnishing)
        if f.category:
            words.append(f.category)
        if f.purpose:
            words.append("for rent" if f.purpose == "rent" else "for sale")
        places = [p for p in (*f.districts, *f.cities, *f.countries) if p and not is_arabic(p)]
        if places:
            words.append("in " + ", ".join(places))
        words += [self.index.records[rid].title for rid in f.project_ids if rid in self.index.records]
        english = " ".join(w for w in words if w and not is_arabic(w)).strip()
        return english or None

    def _search_with_relaxation(
        self, f: QueryFilters, kinds: list[str], qvec: Any, limit: int
    ) -> tuple[list[Record], int, list[str]]:
        relaxed: list[str] = []
        res = self.index.search_records(f, kinds=kinds, qvec=qvec, limit=limit)
        for label, step in RELAXATION_STEPS:
            if res.total > 0:
                break
            new_f = step(f)
            if new_f is None:
                continue
            f = new_f
            relaxed.append(label)
            res = self.index.search_records(f, kinds=kinds, qvec=qvec, limit=limit)
        return res.records, res.total, relaxed

    def retrieve(
        self,
        message: str,
        history_user_messages: list[str] | None = None,
        previous_source_ids: list[str] | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        parsed = self.parser.parse(message, history_user_messages or [])
        result = RetrievalResult(parsed=parsed, sources=[])
        if parsed.intent in ("greeting", "thanks", "help"):
            return result

        dense_text = self.dense_query_text(parsed)
        qvec = self.index.embed_query(dense_text) if dense_text else None
        t_embed = time.perf_counter()
        ordered: dict[str, Source] = {}

        def add(rec: Record | None, reason: str, passage: str | None = None) -> None:
            if rec is None:
                return
            src = ordered.get(rec.id)
            if src is None:
                if len(ordered) >= MAX_SOURCES:
                    return
                src = Source(n=len(ordered) + 1, record=rec, reason=reason)
                ordered[rec.id] = src
            if passage and passage not in src.passages and len(src.passages) < 3:
                src.passages.append(passage)

        f = parsed.filters
        # 1) Explicit references to earlier answers ("the second one").
        if parsed.references_previous and previous_source_ids:
            ids = previous_source_ids
            if parsed.ordinal and 0 < parsed.ordinal <= len(ids):
                ids = [ids[parsed.ordinal - 1]]
            for rid in ids[:4]:
                add(self.index.records.get(rid), "previous answer")

        # 2) Named projects.
        for rid in f.project_ids[:4]:
            rec = self.index.records.get(rid)
            if rec:
                hits = self.index.search_chunks(
                    parsed.search_text, qvec, k=3, record_ids=[rid], per_record=3, min_similarity=self.min_sim
                )
                if not hits:
                    add(rec, "named project", rec.text[:1200])
                for hit in hits:
                    add(rec, "named project", hit.text)

        # 3) Structured search for listings / projects.
        if parsed.intent in ("property_search", "stats") and (f.is_structured() or f.kinds or f.sources):
            limit = min(f.limit or self.top_k_listings, 15)
            kinds = f.kinds or (["project"] if f.sources == ["darglobal"] else ["listing", "project"])
            records, total, relaxed = self._search_with_relaxation(
                replace(f, project_ids=[]) if parsed.intent == "property_search" else f, kinds, qvec, limit
            )
            result.total_matches, result.relaxed = total, relaxed
            for rec in records:
                add(rec, "structured match")
            if parsed.intent == "stats":
                stats = self.index.search_records(replace(f, sources=["wasalt"], districts=[], bedrooms=None,
                                                          min_bedrooms=None, max_bedrooms=None, min_price=None,
                                                          max_price=None, min_area=None, max_area=None,
                                                          furnishing=None, project_ids=[]), kinds=["stat"], limit=3)
                for rec in stats.records:
                    add(rec, "market statistic")
        t_struct = time.perf_counter()

        # 4) Hybrid passages (articles, FAQs, guides, project details).
        passage_kinds = None
        k = self.top_k_passages
        if parsed.intent == "property_search":
            passage_kinds = ["project", "article", "page", "faq", "guide"]
            k = max(3, self.top_k_passages // 2)
        elif parsed.intent == "project_detail":
            passage_kinds = ["project", "article", "page", "faq"]
            k = max(4, self.top_k_passages // 2)
        elif parsed.intent == "general":
            passage_kinds = ["article", "page", "faq", "guide", "project", "stat"]
        sources = f.sources or None
        hits: list[ChunkHit] = self.index.search_chunks(
            parsed.search_text, qvec, k=k + 3, kinds=passage_kinds, sources=sources, min_similarity=self.min_sim
        )
        allowed_cities = {city_key(c) for c in f.cities} if f.cities else None
        added = 0
        for hit in hits:
            rec = self.index.records.get(hit.record_id)
            if rec is None:
                continue
            # A search for Riyadh rentals must not pull in a Jeddah project just
            # because "apartments" matched: drop passages that contradict the location.
            if parsed.intent in ("property_search", "stats"):
                if allowed_cities and rec.city and city_key(rec.city) not in allowed_cities:
                    continue
                if f.countries and rec.country and rec.country not in f.countries:
                    continue
            add(rec, "passage", hit.text)
            added += 1
            if added >= k:
                break

        result.sources = list(ordered.values())
        t_end = time.perf_counter()
        result.timings_ms = {
            "embed": round((t_embed - t0) * 1000, 1),
            "structured": round((t_struct - t_embed) * 1000, 1),
            "passages": round((t_end - t_struct) * 1000, 1),
            "total": round((t_end - t0) * 1000, 1),
        }
        return result
