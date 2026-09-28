#!/usr/bin/env python3
"""Capture read-only Atlas Vector Search index metadata and collection footprint.

Atlas database commands expose index definitions and readiness, but not dedicated
Search-node index bytes. The snapshot deliberately records that boundary instead
of treating collStats.totalIndexSize as Vector Search index storage.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_dotenv() -> None:
    path = ROOT / ".env"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def vector_fields(definition: dict) -> list[dict]:
    return [field for field in definition.get("fields", []) if field.get("type") == "vector"]


def main() -> None:
    load_dotenv()
    if not os.getenv("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI in .env or the shell.")
    database = os.getenv("BENCHMARK_DB", "atlas_voyage_v4_benchmark")
    collection_name = os.getenv("BENCHMARK_COLLECTION", "chunks")
    collection = MongoClient(os.environ["MONGODB_URI"], appname="atlas-voyage-v4-benchmark-index-snapshot")[database][collection_name]
    collection_stats = collection.database.command("collstats", collection_name)
    count = collection_stats.get("count", 0)
    indexes = []
    for item in collection.list_search_indexes():
        definition = item.get("latestDefinition", {})
        fields = vector_fields(definition)
        indexes.append(
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "status": item.get("status"),
                "queryable": item.get("queryable"),
                "status_detail": item.get("statusDetail"),
                "definition_version": item.get("latestDefinitionVersion"),
                "vector_fields": [
                    {
                        "path": field.get("path"),
                        "dimensions": field.get("numDimensions"),
                        "similarity": field.get("similarity"),
                        "quantization": field.get("quantization", "none"),
                        "nominal_float32_payload_bytes": count * int(field.get("numDimensions", 0)) * 4,
                    }
                    for field in fields
                ],
            }
        )
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "database": database,
        "collection": collection_name,
        "collection_stats": {key: collection_stats.get(key) for key in ("count", "size", "storageSize", "avgObjSize", "totalIndexSize")},
        "search_indexes": indexes,
        "search_node_index_bytes": None,
        "search_node_index_bytes_note": "Not returned by MongoDB collStats or listSearchIndexes. Capture this from Atlas Search-node metrics or the Atlas Administration API with separate monitoring credentials.",
        "nominal_float32_payload_note": "A dimensions × 4 × document-count comparison aid only. It is not BSON collection bytes, Atlas Search-index bytes, or a scalar-quantization storage estimate.",
    }
    RESULTS.mkdir(exist_ok=True)
    destination = RESULTS / f"index-snapshot-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    destination.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")
    print(json.dumps({"collection_documents": count, "search_indexes": len(indexes), "ready_indexes": sum(item["status"] == "READY" for item in indexes)}, indent=2))


if __name__ == "__main__":
    main()
