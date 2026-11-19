"""Operational CLI.

    python -m app.cli ingest [--sources darglobal,wasalt] [--require-all-sources | --allow-empty]
    python -m app.cli reindex          # rebuild the index from the current snapshot
    python -m app.cli warmup           # download/cache the embedding model
    python -m app.cli stats            # print index statistics
    python -m app.cli vsearch "text"   # semantic search straight against the vector database
    python -m app.cli doctor           # can this machine reach the sites, OpenRouter and the model?
    python -m app.cli verify --url URL # verify coverage, semantic search and two cited AI replies
    python -m app.cli ask "question"   # one-shot question (uses OpenRouter if configured)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import get_settings
from app.logging_setup import configure_logging


def _embedder(allow_download: bool = False):
    from app.rag.embeddings import load_embedder

    s = get_settings()
    return load_embedder(
        s.embedding_model, s.model_cache_dir, enabled=s.enable_dense, threads=s.embedding_threads,
        local_only=s.embedding_local_only and not allow_download,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    from app.scraping.pipeline import ingest

    settings = get_settings()
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    try:
        summary = asyncio.run(ingest(settings, _embedder(), sources=sources, require_all_sources=args.require_all_sources))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": repr(exc)}), file=sys.stderr)
        return 0 if args.allow_empty else 1
    print(json.dumps({"ok": True, **summary}, indent=2, default=str))
    return 0


def cmd_reindex(_: argparse.Namespace) -> int:
    from app.scraping.pipeline import rebuild_from_snapshot

    print(json.dumps(rebuild_from_snapshot(get_settings(), _embedder()), indent=2, default=str))
    return 0


def cmd_warmup(_: argparse.Namespace) -> int:
    embedder = _embedder(allow_download=True)
    print(json.dumps({"embedding_model": getattr(embedder, "name", None), "dim": getattr(embedder, "dim", None)}))
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    from app.rag.index import SearchIndex, current_build_dir

    build = current_build_dir(get_settings().index_root)
    if not build:
        print("no index", file=sys.stderr)
        return 1
    print(json.dumps(SearchIndex(build, None).stats(), indent=2))
    return 0


def cmd_vsearch(args: argparse.Namespace) -> int:
    """Query the vector database directly (no filters, no LLM): shows what is stored."""
    from app.rag.index import SearchIndex, current_build_dir

    build = current_build_dir(get_settings().index_root)
    if not build:
        print("no index; run `python -m app.cli ingest` first", file=sys.stderr)
        return 1
    embedder = _embedder()
    if embedder is None:
        print("semantic search is off (ENABLE_DENSE=false) or the embedding model is unavailable", file=sys.stderr)
        return 1
    index = SearchIndex(build, embedder)
    try:
        if index.vectors is None:
            print("this build has no vector store for the current model; run `python -m app.cli reindex`", file=sys.stderr)
            return 1
        qvec = index.embed_query(args.query)
        print(json.dumps(index.stats()["vector_store"]))
        for row, sim in index.vectors.search(qvec, args.k):
            rec = index.records[index.chunk_record[row]]
            print(f"{sim:.3f}  {rec.title}  [{rec.source_label} {rec.kind}]  {rec.url}")
    finally:
        index.close()
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    from app.diagnostics import report, run_checks

    settings = get_settings()
    probes = asyncio.run(run_checks(settings))
    print(report(probes, settings))
    return 0 if all(p.ok for p in probes) else 1


def cmd_verify(args: argparse.Namespace) -> int:
    import httpx

    from app.deployment import verify_deployment
    from app.net import tls_verify

    async def run():
        async with httpx.AsyncClient(
            base_url=args.url.rstrip("/") + "/", timeout=180,
            verify=tls_verify(get_settings().tls_trust),
        ) as client:
            return await verify_deployment(client, check_chat=not args.skip_chat)

    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_ask(args: argparse.Namespace) -> int:
    from app.services.chat import ChatService
    from app.services.state import AppState

    async def run() -> None:
        state = AppState(get_settings())
        state.load_embedder()
        if not state.load_index():
            print("no index; run `python -m app.cli ingest` first", file=sys.stderr)
            return
        async for ev in ChatService(state).stream(args.question, [], [], "cli"):
            if ev["event"] == "token":
                print(ev["data"]["t"], end="", flush=True)
            elif ev["event"] == "meta":
                print(f"[intent={ev['data']['intent']} filters={ev['data']['filters']} sources={len(ev['data']['sources'])}]\n")
            elif ev["event"] == "done":
                print(f"\n\n[{ev['data']}]")
        await state.llm.aclose()

    asyncio.run(run())
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="scrape sources and build + activate a new index")
    p.add_argument("--sources", help="comma-separated subset, e.g. darglobal")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--allow-empty", action="store_true", help="exit 0 even if ingestion fails (development only)")
    mode.add_argument("--require-all-sources", action="store_true", help="require fresh DarGlobal projects and Wasalt listings before activation")
    p.set_defaults(func=cmd_ingest)
    sub.add_parser("reindex").set_defaults(func=cmd_reindex)
    sub.add_parser("warmup").set_defaults(func=cmd_warmup)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    verify = sub.add_parser("verify", help="verify a deployed app; makes two chat requests by default")
    verify.add_argument("--url", required=True, help="base URL, e.g. https://your-app.onrender.com")
    verify.add_argument("--skip-chat", action="store_true", help="check retrieval only (does not verify OpenRouter)")
    verify.set_defaults(func=cmd_verify)
    sub.add_parser("doctor", help="check this machine can reach the data sites, OpenRouter and the model").set_defaults(
        func=cmd_doctor
    )
    v = sub.add_parser("vsearch", help="semantic search straight against the vector database")
    v.add_argument("query")
    v.add_argument("-k", type=int, default=5)
    v.set_defaults(func=cmd_vsearch)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.set_defaults(func=cmd_ask)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
