#!/usr/bin/env python3
"""Backfill synthetic tenant_id and effective_principal_ids fields onto every chunk."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from pymongo import MongoClient, UpdateOne


ROOT = Path(__file__).resolve().parents[1]
TENANT_ID = "scifact-poc"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI in .env or the shell.")
    database = os.getenv("BENCHMARK_DB", "atlas_voyage_v4_benchmark")
    collection_name = os.getenv("BENCHMARK_COLLECTION", "chunks")
    collection = MongoClient(os.environ["MONGODB_URI"], appname="atlas-voyage-v4-benchmark")[database][collection_name]
    operations = [
        UpdateOne(
            {"_id": chunk["_id"]},
            {"$set": {"tenant_id": TENANT_ID, "effective_principal_ids": [TENANT_ID, synthetic_group(chunk["parent_doc_id"])]}},
        )
        for chunk in collection.find({}, {"_id": 1, "parent_doc_id": 1})
    ]
    if not operations:
        raise SystemExit("No chunks found. Run the import step first.")
    result = collection.bulk_write(operations)
    print(f"Backfilled ACL fields on {result.modified_count} of {len(operations)} chunks.")


if __name__ == "__main__":
    main()
