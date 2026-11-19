# Deploy and get a working URL

> **Nov 2026 update:** Hugging Face no longer lets free accounts create Docker Spaces (a PRO plan is
> required). The recommended free route is now **GitHub + Render**: no credit card, an
> `https://<name>.onrender.com` URL, and semantic search with ChromaDB included (`render.yaml` sets it all up).

## 0. Before anything: keep your key secret

* Copy `.env.example` to `.env` and put `OPENROUTER_API_KEY=sk-or-v1-...` in it. That file is for **local runs only**.
* `.env` is listed in `.gitignore`, so Git will not upload it. Run `git status` before every commit and make sure `.env` is **not** listed.
* On the hosting platform, paste the key into its **environment variables / secrets** screen, never into code.
* If a key ever leaks, revoke it at <https://openrouter.ai/keys> and create a new one.

## Option A: Render (free, no card) [recommended, semantic search on]

Render's free web service has 512 MB RAM and 0.1 CPU, sleeps after 15 minutes without visitors and takes about
1 minute to wake. The app runs **full semantic search** there: the compact embedding model
`BAAI/bge-small-en-v1.5` (67 MB) plus the **ChromaDB** vector database. Measured: about **350 MB** of the 512 MB,
stable under load (with a same-size stand-in for the model; `/api/stats` → `process` shows the real figure). The heavy work (scraping and embedding every chunk) happens on Render's build machines while
the Docker image is built, so the small instance only embeds each question.

**1. Put the code on GitHub** (VS Code terminal, inside the project folder):
```powershell
git init
git add .
git status            # .env must NOT appear
git commit -m "Real-estate assistant (FastAPI + Docker + ChromaDB)"
```
On github.com create an empty repository (no README), e.g. `realestate-assistant`, then:
```powershell
git branch -M main
git remote add origin https://github.com/<you>/realestate-assistant.git
git push -u origin main
```

**2. Create the service from the Blueprint** (everything is preset in `render.yaml`):
* At <https://render.com> choose **Sign up with GitHub**, then **New + → Blueprint**, and pick the repository.
* Render reads `render.yaml` and asks for **`OPENROUTER_API_KEY`**: paste your key. Click **Apply**.

The Blueprint sets: Docker, Free plan, health check `/api/health`, `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`,
`ENABLE_DENSE=true`, `VECTOR_BACKEND=chroma`, `EMBEDDING_THREADS=1`, `REFRESH_INTERVAL_HOURS=0`,
`REEMBED_IF_OUTDATED=false` and a generated `ADMIN_TOKEN`. Render passes these to the Docker build as build
arguments, so the index baked into the image always matches the model used at run time.

<details><summary>Prefer to click through the form instead of the Blueprint?</summary>

**New → Web Service** → pick the repository → Language **Docker** → Instance type **Free** → health check path
`/api/health` → add the environment variables listed above (and `OPENROUTER_API_KEY`) → **Create Web Service**.
Do not set `ENABLE_DENSE=false`.
</details>

**3. Watch the build log** (about 10–20 minutes): installing packages, downloading the embedding model,
`crawl finished` for `darglobal` and `wasalt`, `embedded chunks into vector store` (backend `chroma`),
`ingestion complete`. When the log says your service is live:

**4. Verify**
* `https://<name>.onrender.com/api/health` shows `{"status":"ok"}`.
* `https://<name>.onrender.com/api/stats` shows:
  * non-zero `wasalt:listing` and `darglobal:project` counts and a crawl report per source;
  * `index.dense_retrieval: true`, `index.embedding_model: "BAAI/bge-small-en-v1.5"` and
    `index.vector_store: {"backend": "chroma", "vectors": <number of chunks>, "dim": 384, "loaded": true}`;
  * `process.rss_mb` (memory in use) well under `process.limit_mb` (512).
* `https://<name>.onrender.com/` opens the chat. Try *"3-bedroom apartments for rent in Riyadh under 90,000 SAR"*,
  *"Which DarGlobal projects are in Dubai?"* and a meaning-based one such as *"a quiet family home near good schools"*.

**Waking up:** after 15 idle minutes the first visit takes about a minute, and the first question may take up to
~25 s more while the embedding model loads (it waits instead of answering with keywords only). Open the URL
yourself shortly before reviewers test it.

**Fresh data:** use **Manual Deploy → Deploy latest commit**. The build re-scrapes and re-embeds on Render's
build machines. Embedding on the 0.1 CPU instance itself would take hours, which is why runtime refresh is off.
Optional: add the service's Deploy Hook URL as the GitHub secret `RENDER_DEPLOY_HOOK_URL`, and CI rebuilds it every
Monday.

## Option B: Google Cloud Run (free tier, needs a card) [semantic search on]

In <https://console.cloud.google.com>, create a project, link billing (set a ₹100 budget alert), and open **Cloud Shell**:
```bash
git clone https://github.com/<you>/realestate-assistant.git && cd realestate-assistant
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy property-assistant --source . --region us-central1 --allow-unauthenticated \
  --memory 1Gi --cpu 1 --timeout 3600 --set-env-vars OPENROUTER_API_KEY=sk-or-v1-YOURKEY
```
The command prints a `https://property-assistant-….run.app` URL. Cloud Run sets `PORT`, and the app follows it.

Check `https://<service>.run.app/api/stats`: `vector_store` should read
`{"backend": "chroma", "vectors": <number of chunks>, "dim": 384, "loaded": true}`, which means the ChromaDB vector
database is serving semantic search.

## Option C: Hugging Face Spaces (PRO plan, about $9/month)

Create a Space with **SDK: Docker** and add the secret `OPENROUTER_API_KEY` under Settings, then upload with
`huggingface-cli upload <you>/<space> . . --repo-type=space`. The `README.md` header already configures it
(port 7860). With 16 GB RAM, semantic search stays on.

## Run locally (optional)

```powershell
docker build -t property-assistant .
docker run -p 7860:7860 --env-file .env property-assistant      # then open http://localhost:7860
# or:  docker compose up --build
```
Without Docker: `py -3.13 -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, `pip install -r requirements-dev.txt`,
`python -m app.cli ingest`, `python -m app`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Answers begin "The language model is unavailable…" | Key missing or wrong (`/api/stats` → `llm.last_error`), or the free daily limit (50 requests) is used up. It resets at 05:30 IST, and a one-time $10 OpenRouter credit raises it to 1,000/day. |
| Render: crash / exit 137 | Out of memory. Check `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` (the multilingual model needs ~1 GB) and `EMBEDDING_THREADS=1`. As a last resort, `ENABLE_DENSE=false` turns semantic search off (~140 MB). |
| `/api/stats` shows `dense_retrieval: false` on Render | The model or vector store didn't load: check the build log for the `warmup` step, and that `EMBEDDING_MODEL` was set **before** the build (after changing it, use **Manual Deploy → Clear build cache & deploy**). |
| First page load takes about a minute | Render free was asleep. Open the URL yourself just before reviewers test it; the first question can take ~25 s more while semantic search warms up. |
| `/api/stats` shows `"stale_sources": ["wasalt"]` or a crawl note `anti-bot challenge` | The site refused the build server. The app serves the other source; redeploy later. It never tries to evade blocks. |
| Log: `anti-bot challenge served` for a site (even its robots.txt) | That site's protection is refusing automated clients from your address. The app never evades: it skips the source, indexes the other one, and reports it under `/api/stats` → `snapshot.crawl_reports`. Try later, from another network, or let Render's build machines try — the bot works with the data it has. |
| Local `ingest`: `robots.txt unreachable` for every site, 0 records | This machine's network blocks the sites (common on office and campus networks). Run `python -m app.cli doctor`: it tests each site and says what to do (switch `TLS_TRUST`, use a phone hotspot or home Wi-Fi). This doesn't affect Render, whose build servers do the scraping. |
| Build log: `ingestion produced only 0 records` | Both sites were unreachable during the build. The app scrapes at startup instead (`/api/ready` returns 503 until done). |
