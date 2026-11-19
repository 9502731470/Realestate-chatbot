# DarGlobal x Wasalt Property Assistant
#   builder  - resolves the locked dependency set into an isolated virtualenv
#   runtime  - slim, non-root (uid 1000, as Hugging Face Spaces expects), with the
#              embedding model and a fresh data snapshot baked into the image
#
# Build:  docker build -t dg-wasalt-assistant .
# Run:    docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... dg-wasalt-assistant

# ------------------------------------------------------------------ builder
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.lock /tmp/requirements.lock
RUN pip install --require-virtualenv -r /tmp/requirements.lock

# ------------------------------------------------------------------ runtime
FROM python:3.13-slim AS runtime

ARG INGEST_AT_BUILD=1
# Semantic-search settings are build args too, so the index baked into the image
# matches the model used at run time. Render passes a service's environment
# variables to the build as build args, so setting them there is enough.
#   EMBEDDING_MODEL: BAAI/bge-small-en-v1.5 (default; fits 512 MB with ChromaDB) or
#   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (EN+AR, needs ~1 GB)
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG ENABLE_DENSE=true
ARG VECTOR_BACKEND=chroma
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/user \
    DATA_DIR=/home/user/app/data \
    MODEL_CACHE_DIR=/home/user/app/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False \
    EMBEDDING_LOCAL_ONLY=true \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    ENABLE_DENSE=${ENABLE_DENSE} \
    VECTOR_BACKEND=${VECTOR_BACKEND} \
    PORT=7860 \
    INGEST_ON_STARTUP=if-missing \
    REFRESH_INTERVAL_HOURS=24

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin user \
 && mkdir -p /home/user/app/data /home/user/app/models \
 && chown -R user:user /home/user/app
COPY --from=builder /opt/venv /opt/venv

WORKDIR /home/user/app
COPY --chown=user:user app ./app
COPY --chown=user:user pyproject.toml README.md ./
USER user

# Precompile bytecode: faster cold starts on small instances (e.g. 0.1 vCPU free tiers).
RUN python -m compileall -q app

# Bake the configured embedding model into the image (no download at runtime).
RUN python -m app.cli warmup

# Bake a fresh data snapshot + search index (SQLite/FTS5 + ChromaDB vector store)
# into the image. A release must include projects and listings from both sources.
# A blocked source fails this build; retry later without bypassing the challenge.
# INGEST_AT_BUILD=0 is for development images with a separately supplied index.
RUN if [ "$INGEST_AT_BUILD" = "1" ]; then python -m app.cli ingest --require-all-sources; fi

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '7860'), timeout=4)"]

CMD ["python", "-m", "app"]
