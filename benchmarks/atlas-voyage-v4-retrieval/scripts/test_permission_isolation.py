#!/usr/bin/env python3
"""Verify Atlas Vector Search ACL pre-filtering returns no document the asker may not see."""

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
HRPOC_PREPARED = ROOT / "data" / "prepared-hrpoc"
RESULTS = ROOT / "results"
INDEX_NAME = "vs_voyage_4_1024_acl"
HRPOC_INDEX_NAME = "vs_voyage_4_1024"  # hrpoc declares the audience filter fields on this index; it has no _acl twin
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


def hrpoc_visibility() -> dict[str, tuple[bool, set[str]]]:
    """Document visibility as recorded by prepare_hrpoc.py from the source rule trees, read from the prepared file rather than Atlas."""
    visibility: dict[str, tuple[bool, set[str]]] = {}
    with (HRPOC_PREPARED / "chunks.ndjson").open(encoding="utf-8") as handle:
        for line in handle:
            chunk = json.loads(line)
            visibility[chunk["parent_doc_id"]] = (chunk["open"], set(chunk["audience_ids"]))
    return visibility


def hrpoc_search(collection: Any, vector: list[float], acl: dict[str, Any] | None) -> list[str]:
    stage: dict[str, Any] = {"index": HRPOC_INDEX_NAME, "path": VECTOR_FIELD, "queryVector": vector, "numCandidates": 400, "limit": 50}
    if acl is not None:
        stage["filter"] = acl
    return [row["parent_doc_id"] for row in collection.aggregate([{"$vectorSearch": stage}, {"$project": {"_id": 0, "parent_doc_id": 1}}])]


def run_hrpoc(max_queries: int | None) -> None:
    queries_path = HRPOC_PREPARED / "queries.json"
    if not queries_path.exists():
        raise SystemExit("Run scripts/prepare_hrpoc.py and the embedding step first.")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if max_queries:
        queries = queries[:max_queries]
    visibility = hrpoc_visibility()
    database = os.getenv("BENCHMARK_DB", "atlas_voyage_v4_benchmark")
    collection = MongoClient(os.environ["MONGODB_URI"], appname="atlas-voyage-v4-benchmark")[database][os.getenv("HRPOC_COLLECTION", "hrpoc_chunks")]
    violations, filter_excluded_documents = [], 0
    for query in queries:
        vector = query["query_vectors"][QUERY_KEY]
        asker = set(query["asker_audience_ids"])
        def visible(parent_doc_id: str, audiences: set[str]) -> bool:
            is_open, document_audiences = visibility[parent_doc_id]
            return is_open or bool(document_audiences & audiences)
        filtered = hrpoc_search(collection, vector, {"$or": [{"open": True}, {"audience_ids": {"$in": query["asker_audience_ids"]}}]})
        control = hrpoc_search(collection, vector, {"open": True})
        unfiltered = hrpoc_search(collection, vector, None)
        if any(not visible(doc, asker) for doc in unfiltered):
            filter_excluded_documents += 1
        for label, returned, audiences in (("asker", filtered, asker), ("no-audience control", control, set())):
            leaked = sorted({doc for doc in returned if not visible(doc, audiences)})
            if leaked:
                violations.append({"query_id": query["query_id"], "queried_as": label, "leaked_parent_doc_ids": leaked})
    status = "fail" if violations else ("pass" if filter_excluded_documents else "inconclusive")
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "hrpoc",
        "query_count": len(queries),
        "queries_where_filter_excluded_documents": filter_excluded_documents,
        "status": status,
        "violations": violations,
    }
    destination = RESULTS / f"hrpoc-permission-isolation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "pass":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scifact", "hrpoc"), default="scifact", help="scifact checks the synthetic groups; hrpoc checks the real audience filter")
    parser.add_argument("--max-queries", type=int, help="lower the query count for a diagnostic run")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI in .env or the shell.")
    if args.dataset == "hrpoc":
        return run_hrpoc(args.max_queries)
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
