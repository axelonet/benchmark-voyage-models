#!/usr/bin/env python3
"""Verify Atlas Vector Search ACL pre-filtering returns zero cross-group documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"
RESULTS = ROOT / "results"
INDEX_NAME = "vs_voyage_4_1024_acl"
VECTOR_FIELD = "embedding_voyage_4_1024"
QUERY_KEY = "voyage_4_1024"
GROUPS = ("group:a", "group:b")


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def synthetic_group(parent_doc_id: str) -> str:
    digest = hashlib.sha256(parent_doc_id.encode("utf-8")).hexdigest()
    return GROUPS[int(digest, 16) % len(GROUPS)]


def query_as(collection: Any, principal: str, query_vector: list[float]) -> list[dict[str, Any]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": VECTOR_FIELD,
                "queryVector": query_vector,
                "numCandidates": 400,
                "limit": 50,
                "filter": {"effective_principal_ids": principal},
            }
        },
        {"$project": {"_id": 0, "chunk_id": 1, "parent_doc_id": 1}},
    ]
    return list(collection.aggregate(pipeline))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-queries", type=int, help="lower the query count for a diagnostic run")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI in .env or the shell.")
    queries_path = PREPARED / "queries.json"
    if not queries_path.exists():
        raise SystemExit("Run the prepare and embedding steps first.")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if args.max_queries:
        queries = queries[: args.max_queries]
    database = os.getenv("BENCHMARK_DB", "atlas_voyage_v4_benchmark")
    collection_name = os.getenv("BENCHMARK_COLLECTION", "chunks")
    collection = MongoClient(os.environ["MONGODB_URI"], appname="atlas-voyage-v4-benchmark")[database][collection_name]
    violations = []
    for query in queries:
        vector = query["query_vectors"][QUERY_KEY]
        for principal, other in ((GROUPS[0], GROUPS[1]), (GROUPS[1], GROUPS[0])):
            rows = query_as(collection, principal, vector)
            leaked_ids = sorted({row["parent_doc_id"] for row in rows if synthetic_group(row["parent_doc_id"]) == other})
            if leaked_ids:
                violations.append({"query_id": query["query_id"], "queried_as": principal, "leaked_parent_doc_ids": leaked_ids})
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "query_count": len(queries),
        "status": "pass" if not violations else "fail",
        "violations": violations,
    }
    destination = RESULTS / f"permission-isolation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
