"""Coverage of the two property sources required for a complete release."""

from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_KINDS = {"darglobal": "project", "wasalt": "listing"}


def source_coverage(counts: Mapping[str, int], stale_sources: Sequence[str] = ()) -> dict[str, Any]:
    sources = {}
    missing = []
    for source, kind in REQUIRED_KINDS.items():
        records = sum(count for key, count in counts.items() if key.startswith(f"{source}:"))
        properties = counts.get(f"{source}:{kind}", 0)
        if not properties:
            missing.append(source)
        status = "available"
        if not records:
            status = "unavailable"
        elif not properties:
            status = "missing_property_data"
        elif source in stale_sources:
            status = "stale"
        sources[source] = {
            "status": status, "records": records, "property_records": properties, "required_kind": kind,
        }
    return {"complete": not missing, "missing_sources": missing, "sources": sources}
