# Changelog

## 1.3.3 — Fixes from testing the live deployment

- **City names mistaken for districts.** "Riyadh" also matched a district called Al-Riyadh in Jeddah, so
  "apartments for rent in Riyadh" (a homepage example) returned one Jeddah flat instead of the Riyadh listings;
  "Khobar" matched a district of Madinah, "Jubail" and "Dhahran" districts inside those cities. A district that
  shares a city's name now only counts when its own city is named too ("Al-Riyadh district, Jeddah").
- **Wasalt city spellings.** Wasalt writes Aldammam, Alttayif, Bariduh, Hayil, Tbwk, Earear, Eanizah, Alzahran…,
  so searches for Dammam, Taif, Buraydah, Hail, Tabuk, Arar, Unaizah or Dhahran found nothing. These spellings are
  now aliases (plus Abqaiq, Hafar Al Batin, Al Khafji, Rabigh and other towns found in the data) and listings are
  stored under the standard name. Wasalt project URLs keep Wasalt's own spelling.
- **Villas stored as townhouses.** Wasalt files villas under the slug `villa-townhouse`; the scraper read the slug
  first, so all villas became townhouses and villa searches returned nothing. The listing's label now decides.
- **Missing DarGlobal projects.** When `/projects` ships no project data, the crawler took the first 40 paths of
  the alphabetical sitemap, which cut off Pagani, Missoni, W Residences, The Astera and Marea. It now reads the
  project links in the page's HTML first, and fetches up to 100 top-level pages (`DG_MAX_PAGES`).
- **Duplicate and false DarGlobal projects.** Project records are keyed on the page's canonical URL, so a
  redirecting alias no longer duplicates a project, and same-name projects in the same place are merged. Pages
  with no location (e.g. "Buy Property with Crypto") are kept as pages, not projects.
- 47 new tests (227 in total). Each bug above has tests that fail on the previous code.

## 1.3.2 — Fix: compressed responses crashed the crawler

- Real servers send pages gzip-compressed. The crawler decompressed them while reading, then handed the copy on
  still labelled as compressed, so reading it decompressed a second time and failed — and every live fetch,
  `robots.txt` first, looked like a network error ("robots.txt unreachable", 0 records). The offline replicas
  were uncompressed, so 159 tests missed it. Fixed by dropping the encoding headers from the already-decoded copy.
- The mock sites now gzip their responses like the live sites, so the whole scraper suite exercises response
  decoding from now on (on the unfixed code, the suite fails).
- `python -m app.cli doctor` now also fetches through the crawler's own engine, not just plain HTTP, so a
  crawler-only failure like this can't hide again — and it recognises anti-bot challenges, explaining that the
  app skips that source (never evades) and answers from the data it has.
- 4 new tests (163 in total).

## 1.3.1 — Clear diagnostics for blocked networks

- `python -m app.cli doctor`: tests each data site, OpenRouter and the model download from this machine and turns
  failures (DNS filter, firewall, HTTPS inspection, proxy, HTTP errors) into concrete advice.
- Crawler errors now carry the real reason (e.g. `ConnectError: [Errno 11001] getaddrinfo failed`) in the logs and
  in the crawl report (`/api/stats`), instead of only "robots.txt unreachable".
- `TLS_TRUST` (`auto` default): on Windows and macOS, HTTPS is verified against the system certificate store, like
  browsers and pip, so networks that inspect HTTPS with a company certificate work. Verification is never disabled.
- Quieter first model download on Windows (no symlink or unauthenticated-request warnings).
- 8 new tests (159 in total).

## 1.3.0 — Semantic search on the free 512 MB plan

- Default embedding model is now `BAAI/bge-small-en-v1.5` (67 MB, retrieval-trained) instead of the 220 MB
  multilingual paraphrase model, which stays available via `EMBEDDING_MODEL` for hosts with ≥ 1 GB.
  Model profiles add BGE's query instruction and a relevance floor per model.
- The free Render plan runs embeddings + ChromaDB: whole process ~350 MB of 512 MB
  (measured with a same-size stand-in for the model). `render.yaml`
  Blueprint presets everything; `EMBEDDING_MODEL`, `ENABLE_DENSE` and `VECTOR_BACKEND` are Docker build
  arguments, so the baked index always matches the runtime model.
- Arabic questions with an English-only model: the vector search receives an English rendering of the parsed
  question ("شقق للإيجار في جدة" → "apartment for rent in Jeddah").
- Start-up: semantic search is attached to the already-loaded index in place (no second copy of the records);
  the first question after a cold start waits up to `WARMUP_WAIT_S` (25 s) for it instead of using keywords only.
- `REEMBED_IF_OUTDATED` lets tiny CPUs skip runtime re-embedding; `EMBEDDING_BATCH_SIZE` (16) keeps peak RAM low.
- `/api/stats` → `process` reports memory use and the container limit.
- Optional weekly Render rebuild from CI (`RENDER_DEPLOY_HOOK_URL` secret).
- 15 new tests (151 in total).

## 1.2.0 — ChromaDB vector database

- Chunk embeddings are stored in **ChromaDB** (embedded, persistent, HNSW cosine index), with each record's
  filterable fields as metadata. One database per index build, in `data/index/<build>/chroma/`.
- Filtered semantic search is planned per query: SQLite resolves the filters, and Chroma searches inside that set
  (id-restricted search, or oversampled HNSW + post-filter for broad filters). Chroma's own `where` filter is used
  when the candidate set is too large to enumerate. Typical vector query: 1–4 ms.
- New settings: `VECTOR_BACKEND` (`chroma` default, `numpy` for the exact in-memory store) and `CHROMA_EF_SEARCH`.
- Self-healing: a vector store that is missing or was built for another backend/model is re-embedded from the
  snapshot in the background; a vector-database error at query time falls back to keyword search.
- `python -m app.cli vsearch "text"` queries the vector database directly; `/api/stats` → `index.vector_store`
  shows the backend, vector count, dimension and whether it is loaded.
- Index format 3: older builds are rebuilt from the snapshot automatically.
- 19 new tests (136 in total), including SQL/Chroma filter equivalence over 300 random filters and agreement with
  exact search; CI checks that the container serves semantic search from Chroma.

## 1.1.0 — Leaner prompts, faster first token

- About 45% fewer prompt tokens, latency-first free model order, 30 s first-token fail-over, SSE keep-alives,
  Arabic light stemming, relevance floor for passages, annualised USD price comparison.

## 1.0.0

- Polite crawlers for DarGlobal and Wasalt, hybrid retrieval (SQL + BM25 + embeddings), OpenRouter free models,
  Docker image, offline test suite.
