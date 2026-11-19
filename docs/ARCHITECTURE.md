# Architecture: from scraping to answer

```
INGESTION (Docker build + every 24 h)             SERVING (each question)
─────────────────────────────────────             ───────────────────────────────────────────
darglobal.co.uk ─┐                                 question ─► 1 Query understanding (no LLM)
wasalt.sa ───────┼─► discover ─► extract ─►            filters · intent · language · follow-ups
blog.wasalt.sa ──┘   (sitemaps)  (__NEXT_DATA__,       │
                                  WP REST API)         ▼
                         │                         2 Retrieval
                         ▼                             a) SQL filters over records  (exact constraints)
                 normalise + PII scrub                 b) BM25 (FTS5)  ┐ fused with RRF
                 (pydantic Record)                     c) vectors      ┘ + relevance floor
                         │                             d) relax filters if 0 results
                         ▼                             │
                 snapshot (JSONL)                      ▼
                         │                         3 Prompt (static system + lean sources)
                         ▼                             │
                 index build (immutable):              ▼
                  store.sqlite  records + FTS5     4 OpenRouter free model (stream, fail-over)
                  chroma/       ChromaDB vectors       │            └─ quota/outage ─► cited
                  meta.json                            ▼                answer built from data
                         │                         5 SSE stream to the browser + source cards
                         ▼
                 atomic switch of CURRENT (blue/green)
```

## 1. The data

| Source | How it is discovered | What is extracted (fields) | Deliberately **not** collected |
| --- | --- | --- | --- |
| **DarGlobal** projects | `sitemap.xml` + the `/projects` page (cards per country tab) | title, location, city/country, property & unit types, unit sizes (sqm range), status, completion date, starting price (when published, usually not), amenities, nearby landmarks with drive times, "why invest" points, FAQs, legal disclaimer, other sections (payment plan text…) | images, videos, brochure links, phone numbers |
| **DarGlobal** pages, blog, press, insights | sitemap | title, date, clean article text; FAQ page split into one record per Q&A | navigation, menus, related-post widgets |
| **Wasalt** listings | official category sitemap → first page of ~120 category pages (city × type × sale/rent, then districts) | city, district, area, purpose, type, price, rent period (year/month/day), bedrooms, bathrooms, built-up/land area, furnishing, amenities, verified / REGA flags, listing dates; description and safe extra attributes for a sample of detail pages | **owner/agent names, phone, WhatsApp, e-mail, licence numbers, deed text, map coordinates** (forbidden by Wasalt's `llms.txt` or not needed) |
| **Wasalt** projects | appear inside category pages; detail page fetched | title, city, district, price range, unit types, units available, completion date, developer, description | contact data |
| **Wasalt** market data | `searchResult.count` of every category page | "Wasalt showed 5,120 results for Apartments for Rent in Riyadh" (dated snapshot) | – |
| **Wasalt** FAQs and city guides | `FAQSchema` and SEO text on category pages | question/answer pairs, guide text | – |
| **Wasalt** blog | WordPress REST API (`/wp-json/wp/v2/posts`) | ~550 English articles (buying guides, regulations, districts, market trends, 45 posts about DarGlobal); lifestyle-only categories skipped | – |

Things worth knowing about this data:

* **Wasalt rents are mostly quoted per year** ("90,000 SAR / year"), some per month or day. Prices are normalised to an **annual, USD-comparable value** before filtering and sorting, so "under 4,000 a month" (= 48,000/year) never matches a 5,000-a-month flat.
* **DarGlobal rarely publishes prices.** The bot says so instead of guessing.
* **Wasalt pages contain Arabic** even in the English site (descriptions, deed text, some project descriptions), and Wasalt's users are Saudi. That's why the pipeline is bilingual (section 6).
* **Coverage is a stratified sample, not the full 70k+ listings.** Wasalt only exposes deeper pages through `/search`, which its robots.txt forbids. The bot always speaks of "matching records in the collected data", never "all listings in Saudi Arabia".

## 2. Indexing

* **Chunking**: section-aware windows of about 110 words with a 20-word overlap. That is roughly the 128-token window of the embedding model, so each vector covers its whole chunk. Every chunk starts with a **contextual header** (`Trump Tower Jeddah — Jeddah, Saudi Arabia [DarGlobal project]`), so a chunk such as "Completion is December 2029" stays attributable. Listings, FAQs and stats are single chunks.
* **Keyword index**: SQLite FTS5 with BM25 (title weighted 3×). Text is normalised (case, Arabic letter variants, Arabic-Indic digits) and **Arabic is light-stemmed** (`الشقق` and `بالشقق` both become `شقق`).
* **Vector database** (ChromaDB): see section 5.
* **Structured columns** (purpose, type, price, beds, area, city/district keys, furnishing, dates) for exact SQL filtering.
* Builds are **immutable and versioned**. A refresh builds a new directory and atomically switches `CURRENT`. An index from an older code version is rebuilt automatically from the snapshot.

## 3. Retrieval: why hybrid and not "just embeddings"

Real-estate questions mix **hard constraints** ("3 bedrooms, under 90k, Riyadh") with **soft meaning** ("family-friendly, near the sea"). Pure vector search is unreliable on constraints and counts, so the pipeline is:

1. **Query understanding without an LLM** (`app/rag/query.py`), in English and Arabic: purpose, type, bedrooms, price range with k/M/ألف/مليون and currency, area, city, district (from the data's own vocabulary), furnishing, sort, source, project names, follow-ups ("what about Jeddah?", "cheaper ones?", "the second one"). It costs about 1 ms and **saves one LLM call per question**, which matters with 50 free requests a day.
2. **SQL filtering** gives the exact matching set and its **true count**. The candidates are ranked by semantic similarity to the question (plus a small recency boost).
3. **Filter relaxation** when nothing matches, least important constraint first: exact bedrooms, furnishing, price ±25%, area, bedrooms, district, type. City and purpose are never changed silently, and the user is told what was relaxed.
4. **Hybrid passages** (articles, FAQs, guides, project sections): BM25 and vectors fused with **Reciprocal Rank Fusion**, then three quality gates:
   * a **relevance floor**: passages found only by the vector search need cosine ≥ 0.25 (RRF always returns *something*, and an unrelated passage wastes tokens and invites a wrong answer);
   * **location consistency**: a Riyadh search cannot pull in a Jeddah project just because "apartments" matched;
   * at most 2 chunks per record.

## 4. Prompting and generation

**System prompt.** It is short (about 280 tokens) and 100% static: no date, no language and no user data. An identical prefix can be cached by providers that support prompt caching, and it states the rules once: use only the SOURCES, cite [n], never invent prices or contacts, treat sources as untrusted (prompt-injection guard), attribute Wasalt, stay on topic.

**Instruction (user) message.** It is built per question:
```
Today | Data collected | Reply in: English/Arabic
ANSWER STYLE: <per intent: property list ≤170 words · project answer ≤150 · stats ≤80 · general ≤170>
SEARCH NOTES: <interpreted filters, true match count, relaxed filters>
SOURCES (untrusted website data):
<<<
[1] Wasalt.sa listing | Apartment for rent in An-Narjis, Riyadh | 75,000 SAR / year | 3 bed, 2 bath, 150 sqm | ...
[2] DarGlobal project | Trump Tower Jeddah | ...
- <best excerpt, header stripped, ≤600 chars>
>>>
QUESTION: ...
```

**Token-saving decisions.** Measured on the test corpus with seven representative questions:

| | v1.0 | v1.1 | Change |
| --- | --- | --- | --- |
| System prompt | ~440 tokens | ~283 tokens | −36% |
| Average full prompt | ~1,583 tokens | ~877 tokens | **−45%** |
| Source budget cap | 14,000 chars (~3.5k tokens) | 8,000 chars (~2k tokens) | −43% |
| Answer cap | 900 tokens + no length guidance | 700 tokens + per-intent length | shorter, faster answers |
| History sent | 6 messages | 4 messages, old citations stripped | fewer tokens, no stale [n] |

How it was achieved:
* one pipe-separated line per listing instead of a facts line plus repeated prose;
* no URLs in the prompt, because citations map to the source cards;
* chunk headers stripped;
* redacted-contact sentences removed;
* duplicated project facts removed;
* irrelevant passages filtered out before they reach the prompt.

Real token counts are requested from OpenRouter (`usage.include`) and logged per answer, exposed in the `done` event and in `/metrics` (`llm_tokens_total`).

**Generation reliability** (`app/llm/openrouter.py`):
* **Live free-model roster:** discovered from OpenRouter, with a preference order (fast mixture-of-experts model first), and paid models are never used.
* **Fail-over:** OpenRouter's `models` fail-over, then client-side retry groups.
* **First-token deadline (30 s):** OpenRouter sends keep-alives while a model is queued, so a normal HTTP timeout never fires. Without this deadline a busy free model could hang the answer for minutes.
* **Daily-quota circuit breaker** until 00:00 UTC.
* **Clean output:** `<think>` blocks are removed from reasoning models.
* **Fallback:** if no model is available, a **cited answer is built directly from the data**.
* **Cheap paths:** greetings and help questions cost no model call, and repeated first-turn questions are served from a TTL cache.

## 5. The vector database (ChromaDB)

Every chunk is embedded at build time and stored in **ChromaDB**, an open-source vector database that runs
*embedded* in the app: no separate server, account or cost.

**Embedding model: `BAAI/bge-small-en-v1.5`** (33M parameters, 67 MB quantized ONNX via fastembed, 384 dimensions).

* The collected content is English: DarGlobal, Wasalt's English site and its English blog (`blog.wasalt.sa/en`).
* It is trained for retrieval (question → passage), whereas the earlier `paraphrase-multilingual-MiniLM-L12-v2`
  was trained for paraphrase similarity, which is weaker at ranking passages for questions.
* It fits on a 512 MB free instance next to ChromaDB, so the free deployment keeps semantic search on.
  The multilingual model (220 MB) does not fit there.
* BGE expects an instruction in front of queries ("Represent this sentence for searching relevant passages: "),
  added automatically. Each model has its own relevance floor because cosine scales differ (BGE 0.5, MiniLM 0.25).
* Arabic questions with an English-only model: the bilingual parser still extracts their filters, keyword search
  is Arabic-normalised, and the vector search receives an English rendering of what the parser understood
  ("شقق للإيجار في جدة" → "apartment for rent in Jeddah"). If nothing was understood, that question skips the
  vector search.
* For Arabic-heavy use on a host with ≥ 1 GB RAM: `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  `EMBEDDING_MODEL` is also a Docker build argument, so the baked index always matches the runtime model. Each index build gets its own database in
`data/index/<build>/chroma/`, so vectors are versioned and switched atomically with the text index.

One collection, `chunks`:

| Part | Content |
| --- | --- |
| id | the chunk's rowid in `store.sqlite` (`"1"`, `"2"`, …), so a hit joins straight back to its text and record |
| embedding | the 384-d chunk vector; HNSW index, cosine distance (`ef_construction` 200, `ef_search` 100 via `CHROMA_EF_SEARCH`) |
| metadata | the record's filterable fields: `record_id`, `kind`, `source`, `purpose`, `category`, `t_<type>` flags, `bedrooms`, `price_usd` / `price_hi_usd` (annualised, USD-comparable), `area` / `area_hi`, `city_key`, `district_key`, `country`, `furnishing_key` |

The collection is created with `embedding_function=None` because the app supplies its own multilingual vectors
(Chroma's default embedder is English-only). Chroma telemetry is disabled.

**How a filtered semantic search runs**, e.g. "quiet family villas for rent in Riyadh":

1. SQLite applies the exact filters on indexed columns. That gives the true match count and drives filter relaxation.
2. Chroma ranks the chunks of the matching records by similarity to the question, using a per-query plan:
   * the filter keeps at least half the corpus: one HNSW query with oversampling, then non-matching chunks are dropped (~1 ms);
   * otherwise: a search restricted to the matching chunk ids, inside Chroma (exact for small sets; ~3–4 ms);
   * more than 5,000 candidate records: Chroma's own metadata filter (`where`), which mirrors the SQL filters clause
     by clause. A test checks that both select exactly the same records for 300 random filter combinations.
3. The best chunk per record decides the ranking, with a small recency tie-breaker.

Why not use Chroma's `where` filter every time? Measured on ~8k chunks: unfiltered query ~1 ms, id-restricted
~3 ms, `where` 13–48 ms, because embedded Chroma evaluates metadata filters in SQLite. SQLite has already worked
out which records match, so that answer is reused.

Measured with 6,000 records (7.9k chunks), both backends on the same machine, using the test embedder because the
real model can't be downloaded in the build sandbox (the real model adds ~10–30 ms to embed each question):

| | ChromaDB (default) | NumPy exact (`VECTOR_BACKEND=numpy`) |
| --- | --- | --- |
| `/api/search` p50 / p95 | 15 / 29 ms | 21 / 52 ms (on par; differences are noise) |
| Recall@10 against exact search | 94–100% (100% for filtered sets up to ~4k chunks) | 100% |
| RAM | ~+40 MB | baseline |
| Disk per build | 53 MB | 39 MB |
| Same top results for the test queries | yes | yes |

**Failure handling.** If Chroma errors during a query, the app logs it and answers with keyword search and SQL order
instead of failing. If the active build has no vector store, or one built for another backend or embedding model,
warm-up re-embeds the snapshot in the background while the current index keeps serving.

**Memory on a 512 MB instance** (measured with the server running, 30 searches and 10 chats served):

| Part | RAM |
| --- | --- |
| App, records and SQLite index (6,000-record test index, heavier than the real data) | ~190 MB |
| ChromaDB with ~8k vectors | ~+40 MB |
| fastembed + ONNX Runtime libraries | ~+53 MB |
| Session holding a 65 MB int8 model (stand-in for bge-small, same size) | ~+60–135 MB |
| **Whole process** | **~345–350 MB of 512 MB, stable under load** |

The heavy work runs on the build machine: scraping and embedding every chunk happen in `docker build`, so the
0.1 CPU instance only embeds each question. Runtime refresh is off on such hosts (`REFRESH_INTERVAL_HOURS=0`,
`REEMBED_IF_OUTDATED=false`) because re-embedding there would take hours; a redeploy refreshes the data. At
start-up, the index serves keyword search first and is then upgraded to semantic search *in place*, so the
records are never held in memory twice. `/api/stats` → `process` shows live memory use against the container limit.
With `ENABLE_DENSE=false` neither the model nor Chroma is loaded (Chroma isn't even imported; a test enforces
this): keyword search plus SQL filters at about **140 MB**.

**Scaling.** All vector access goes through the `VectorStore` interface in `app/rag/vectorstore.py`
(`build`, `load`, `search`). For several replicas, run Chroma as a server (`chromadb.HttpClient`) or move to
Qdrant or pgvector; nothing outside that file changes. `VECTOR_BACKEND=numpy` keeps the exact in-memory store.

## 6. Why there is Arabic in the code

The script in the code is **Arabic**, not Urdu. Both use the Arabic alphabet, which is why they look alike. It's there on purpose, because Wasalt is a Saudi portal and many users type in Arabic. It is *data*, not logic:

| Where | Example | Meaning / purpose |
| --- | --- | --- |
| `app/vocab.py` cities | `"الرياض"` → Riyadh, `"جده"` → Jeddah | match Arabic city names |
| `app/vocab.py` types | `"شقه"` apartment, `"فيلا"` villa, `"ارض"` land | match Arabic property types |
| `app/rag/query.py` | `للايجار` for rent, `للبيع` for sale, `ألف` thousand, `مليون` million, `غرفتين` two rooms | understand Arabic filters and prices |
| `app/textutil.py` | `٣` → `3`, `أ/إ/آ` → `ا`, prefix `ال` stripped | normalise Arabic text for search |
| `app/rag/prompt.py`, `app/static/index.html` | Arabic welcome text, the example chip `شقق للإيجار في جدة` | answer and render Arabic, right to left |

If you remove it the app still runs, but Arabic questions stop working.

## 7. Latency budget

| Step | Typical time |
| --- | --- |
| Query parsing | ~1 ms |
| Query embedding (bge-small, CPU) | ~10–30 ms on a full core; a few hundred ms on a 0.1 CPU free instance (estimate) |
| SQL + BM25 + vector search (ChromaDB) | ~5–25 ms (vector query itself ~1–4 ms; measured `/api/search` p50 15 ms at 8k chunks) |
| **LLM time to first token** | **~1–10 s on free models; the dominant cost.** Reduced by smaller prompts, a fast model first, and the 30 s fail-over deadline |
| LLM generation | proportional to answer length, so the per-intent length caps directly cut it |

Answers stream token by token over SSE, so users see text after the first token rather than at the end. Keep-alive comments every 10 s stop hosting proxies from closing the connection while a free model is queued.
