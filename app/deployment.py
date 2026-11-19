"""Check a running deployment, including grounded replies from both sources."""

from typing import Any
from urllib.parse import urlparse

import httpx

from app.coverage import REQUIRED_KINDS, source_coverage


async def verify_deployment(client: httpx.AsyncClient, *, check_chat: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    result: dict[str, Any] = {"ok": False, "errors": errors, "chat_checked": False}
    try:
        ready = await client.get("api/ready")
        ready.raise_for_status()
        if not ready.json().get("ready"):
            errors.append("service is not ready")
        response = await client.get("api/stats")
        response.raise_for_status()
        stats = response.json()
        index = stats.get("index") or {}
        stale = (stats.get("snapshot") or {}).get("stale_sources", [])
        coverage = source_coverage(index.get("counts", {}), stale)
        result["coverage"] = coverage
        if not coverage["complete"]:
            errors.append("missing property data: " + ", ".join(coverage["missing_sources"]))
        if stale:
            errors.append("source refresh failed: " + ", ".join(stale))
        vectors = index.get("vector_store") or {}
        if not index.get("dense_retrieval") or not vectors.get("loaded") or not vectors.get("vectors"):
            errors.append("semantic retrieval is not loaded")
        if check_chat and not (stats.get("llm") or {}).get("configured"):
            errors.append("OpenRouter is not configured")
        if errors:
            return result
        if check_chat:
            result["chat_checked"] = True
            result["replies"] = []
            for source, question in (
                ("darglobal", "Show DarGlobal projects in Dubai"),
                ("wasalt", "Show properties from Wasalt"),
            ):
                response = await client.post("api/chat", json={"message": question, "stream": False})
                response.raise_for_status()
                reply = response.json()
                model = reply.get("model") or ""
                if reply.get("mode") not in {"llm", "cache"} or not (model.endswith(":free") or model == "openrouter/free"):
                    errors.append(f"{source}: no successful free-model reply")
                host = "darglobal.co.uk" if source == "darglobal" else "wasalt.sa"
                citations = [
                    card for card in reply.get("sources", [])
                    if card.get("source") == source and card.get("kind") == REQUIRED_KINDS[source]
                    and urlparse(card.get("url", "")).scheme == "https"
                    and urlparse(card.get("url", "")).hostname in {host, f"www.{host}"}
                    and f"[{card.get('n')}]" in reply.get("answer", "")
                ]
                if not citations:
                    errors.append(f"{source}: reply lacks a citation to that source's property data")
                result["replies"].append({"source": source, "model": model, "citations": len(citations)})
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        errors.append(f"deployment check failed: {type(exc).__name__}")
    result["ok"] = not errors
    return result
