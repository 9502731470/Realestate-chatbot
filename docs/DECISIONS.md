# Design decisions: paths considered

Each layer lists the realistic options, what was chosen, and why.

## 1. Getting the data

| Option | Verdict |
| --- | --- |
| CSS/XPath scraping of rendered HTML | Rejected: brittle to redesigns; both sites are Next.js and render client-side widgets |
| Headless browser (Playwright) for every page | Rejected as default: 10–50× heavier, slower and more aggressive on the target; kept as a possible future fallback |
| **Parse `__NEXT_DATA__` server payloads** | **Chosen**: it is exactly the data the page renders, structured (Strapi/listing JSON), stable across visual redesigns, no JS execution |
| Call the sites' private/internal APIs (e.g. Wasalt `/search` backend) | Rejected: `/search` is disallowed by robots.txt; internal APIs are not a published interface |
| Official public APIs | Used where one exists: the Wasalt blog's WordPress REST API |
| Third-party scraping services / proxy rotation / TLS impersonation | Rejected: would mean evading the sites' bot controls; the crawler stops on a challenge instead |
| One-off manual export | Rejected: not reproducible; the pipeline re-runs at build time and on a schedule |

Wasalt coverage strategy: listing pagination is only exposed through `/search`
(disallowed), so the crawler samples the first page (32 listings) of a stratified
set of category pages: country level, then core types × sale/rent across the top
cities (interleaved so every city is covered), then further cities, extra types,
bedroom-specific rentals and districts.

## 2. Storing and retrieving knowledge

| Option | Verdict |
| --- | --- |
| Pure vector RAG (embed everything, top-k) | Rejected alone: weak at "3-bed under 80k in Riyadh" style constraints and at counts |
| Text-to-SQL by the LLM | Rejected: costs extra LLM calls (scarce on the free tier) and is error-prone with small free models |
| **Deterministic bilingual parser → SQL filters + hybrid (BM25 + dense) passages, RRF** | **Chosen**: precise constraints, semantic recall, Arabic support, one LLM call per turn |
| Managed vector DB (Pinecone, Weaviate) | Rejected: external dependency, account and network hop for a corpus of only ~10k chunks |
| NumPy exact matrix (v1.0–v1.1 default) | Kept as `VECTOR_BACKEND=numpy`: 100% recall and the lightest option, but no metadata, no database tooling and no path to a shared server |
| Qdrant / pgvector | The scale-out path for many replicas or millions of chunks; an extra service today |
| **SQLite (structured + FTS5) + ChromaDB (embedded, persistent) in one immutable build directory** | **Chosen (v1.2)**: a real vector database at zero cost and zero ops. Embeddings are stored with their metadata, HNSW cosine search runs in ~1–4 ms, and the store is versioned with the text index for atomic blue/green activation. SQLite resolves structured filters and Chroma searches inside that set: Chroma's own `where` filter measured 13–48 ms vs ~3 ms for an id-restricted search. Costs ~40 MB RAM and ~200 MB of image (44 extra Python packages). `app/rag/vectorstore.py` is the swap point |

### Embedding model

The constraint is the free, no-card host (Render: 512 MB RAM, 0.1 CPU), and the requirement is that semantic search
and the vector database stay **on** in the deployed demo.

| Option | Verdict |
| --- | --- |
| Turn semantic search off on 512 MB (v1.1–v1.2 advice) | Rejected: the live demo would run on keyword search only |
| `paraphrase-multilingual-MiniLM-L12-v2` (220 MB, EN+AR) | Kept as an option for hosts with ≥ 1 GB; with ChromaDB it doesn't fit in 512 MB with a safe margin, and as a paraphrase model it is weaker at question → passage retrieval |
| Remote embedding API (Gemini, Cloudflare, Jina…) | Rejected for now: another account and key, a secret needed at build time, quotas, and a network hop per question |
| Separate embedding micro-service on a second free instance | Rejected: two services with independent cold starts; too fragile for a demo |
| **`BAAI/bge-small-en-v1.5` (67 MB quantized, retrieval-trained)** | **Chosen (v1.3)**: the corpus is English, it ranks passages well, and the whole process measured ~350 MB of 512 MB (with a same-size stand-in for the model). Arabic is covered by the bilingual parser, Arabic-normalised BM25 and an English rendering of parsed Arabic questions for the vector search |
| Larger models (multilingual-e5-large, bge-m3, ~2.2 GB) | Rejected: image size and CPU latency on free hosting |

Corpus embedding runs at image build time (Render's build machines), never on the 0.1 CPU instance;
`EMBEDDING_MODEL` is a build argument, so the baked index and the runtime model can't diverge.

## 3. Language model

| Option | Verdict |
| --- | --- |
| Hard-code one `:free` model | Rejected: the free roster changes weekly; one model is a single point of failure |
| **Live roster discovery + preference list + OpenRouter `models` fallback + client-side retry groups** | **Chosen** |
| `openrouter/free` router only | Used as last resort: convenient but less predictable behaviour |
| Paid fallbacks | Rejected by brief ("free model"); paid IDs are explicitly filtered out to avoid surprise billing |
| Agentic tool-calling loop | Rejected: several calls per turn would exhaust 50 requests/day; tool-calling quality varies across free models |

## 4. Serving

FastAPI + Uvicorn with Server-Sent Events streaming (simple, proxy-friendly,
works through Hugging Face's reverse proxy). Vanilla JS front end with no build
step and no CDNs, so a strict Content-Security-Policy is possible.

## 5. Hosting the Docker image (as of Nov 2026)

| Option | Verdict |
| --- | --- |
| **Render (free web service)** | **Chosen for $0**: builds the Dockerfile from GitHub, HTTPS URL, no card; 512 MB / 0.1 CPU, sleeps after 15 min (about 1 min to wake), so run with `ENABLE_DENSE=false` (measured ~140 MB) |
| Google Cloud Run | Best free option *with* a card: 1 GiB+ keeps semantic search on, scale-to-zero, SSE supported |
| Hugging Face Spaces (Docker SDK) | Docker Spaces now require a paid PRO plan (about $9/month); 16 GB RAM; the repo is already configured for it |
| Koyeb / Railway / Northflank | Viable alternatives with caveats (card hold, trial credits, limited sandbox) |
| Fly.io / AWS App Runner | No longer available for new free users |
| Kubernetes | The scale-out target (separate ingestion CronJob, Redis, pgvector); overkill for the demo |

## 6. Failure modes designed for

| Failure | Behaviour |
| --- | --- |
| Source blocks or is down | Stop that host, keep previous snapshot for that source, flag stale |
| Partial crawl/parsing bug | Per-task isolation; a crash in one adapter never stops the other |
| Tiny/empty crawl | Refuses to activate (keeps the current index) |
| Embedding model missing | Keyword-only search; the server still starts instantly |
| No API key / daily quota / all free models busy | Cited, deterministic answer from the data; quota circuit breaker until 00:00 UTC |
| Stream breaks mid-answer | Partial answer kept, user told, sources still shown |
| Abuse / floods | Per-client minute + day limits, LLM concurrency cap, input limits |
| Prompt injection inside scraped text | Context is delimited and declared untrusted; model instructed to ignore it |
