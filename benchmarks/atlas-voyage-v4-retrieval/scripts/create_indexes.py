#!/usr/bin/env python3
"""Create only the search indexes used by this dedicated benchmark database."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel


ROOT = Path(__file__).resolve().parents[1]
INDEXES = [
    ("vs_voyage_4_large_1024", "embedding_voyage_4_large_1024", 1024, None),
    ("vs_voyage_4_1024", "embedding_voyage_4_1024", 1024, None),
    ("vs_voyage_4_lite_1024", "embedding_voyage_4_lite_1024", 1024, None),
    ("vs_voyage_context_4_1024", "embedding_voyage_context_4_1024", 1024, None),
    ("vs_voyage_4_512", "embedding_voyage_4_512", 512, None),
    ("vs_voyage_4_256", "embedding_voyage_4_256", 256, None),
    ("vs_voyage_4_1024_scalar", "embedding_voyage_4_1024", 1024, "scalar"),
]
# ACL filter fields backfilled onto every chunk by scripts/backfill_acl_fields.py.
ACL_FILTER_FIELDS = ("tenant_id", "effective_principal_ids")
# Additive: a second vector index on the same field, carrying ACL filter fields, so the
# 7 indexes above never need an in-place edit for the hybrid/permission-isolation work.
ACL_VECTOR_INDEX = ("vs_voyage_4_1024_acl", "embedding_voyage_4_1024", 1024, None)
# BM25 text index used by the native $rankFusion hybrid profile.
HYBRID_TEXT_INDEX = "search_text_bm25"
# hrpoc: every query is audience pre-filtered, so each vector index declares the filter fields itself
# (no separate _acl index) and the BM25 index carries the same fields. open is boolean, audience_ids a token array.
HRPOC_FILTER_FIELDS = {"open": "boolean", "audience_ids": "token"}
DATASETS = {
    "scifact": {"collection_env": "BENCHMARK_COLLECTION", "collection": "chunks"},
    "hrpoc": {"collection_env": "HRPOC_COLLECTION", "collection": "hrpoc_chunks"},
}


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


def definition(path: str, dimensions: int, quantization: str | None, filters: tuple[str, ...] = ()) -> dict:
    field = {"type": "vector", "path": path, "numDimensions": dimensions, "similarity": "cosine"}
    if quantization:
        field["quantization"] = quantization
    return {"fields": [field] + [{"type": "filter", "path": name} for name in filters]}


def text_definition(filters: dict[str, str]) -> dict:
    mappings = {"dynamic": False, "fields": {"text": {"type": "string"}}}
    for name, field_type in filters.items():
        mappings["fields"][name] = {"type": field_type}
    return {"mappings": mappings}


def index_specs(dataset: str = "scifact") -> list[tuple[str, str, dict]]:
    if dataset == "hrpoc":
        specs = [(name, "vectorSearch", definition(field, dimensions, quantization, filters=tuple(HRPOC_FILTER_FIELDS))) for name, field, dimensions, quantization in INDEXES]
        specs.append((HYBRID_TEXT_INDEX, "search", text_definition(HRPOC_FILTER_FIELDS)))
        return specs
    specs = [(name, "vectorSearch", definition(field, dimensions, quantization)) for name, field, dimensions, quantization in INDEXES]
    acl_name, acl_field, acl_dimensions, acl_quantization = ACL_VECTOR_INDEX
    specs.append((acl_name, "vectorSearch", definition(acl_field, acl_dimensions, acl_quantization, filters=ACL_FILTER_FIELDS)))
    specs.append((HYBRID_TEXT_INDEX, "search", text_definition({name: "token" for name in ACL_FILTER_FIELDS})))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="scifact", help="which dataset's collection and index definitions to create")
    parser.add_argument("--wait", action="store_true", help="wait up to ten minutes for every requested index to become queryable")
    parser.add_argument("--replace", action="store_true", help="replace only the named benchmark indexes for the selected dataset")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI in .env or the shell.")
    database = os.getenv("BENCHMARK_DB", "atlas_voyage_v4_benchmark")
    collection_name = os.getenv(DATASETS[args.dataset]["collection_env"], DATASETS[args.dataset]["collection"])
    collection = MongoClient(os.environ["MONGODB_URI"], appname="atlas-voyage-v4-benchmark")[database][collection_name]
    existing = {item["name"]: item for item in collection.list_search_indexes()}
    specs = index_specs(args.dataset)
    for name, kind, index_definition in specs:
        if name in existing:
            if not args.replace:
                print(f"Keeping existing index {name}; use --replace to recreate it.")
                continue
            print(f"Dropping benchmark index {name}")
            collection.drop_search_index(name)
        print(f"Creating {name}")
        collection.create_search_index(SearchIndexModel(definition=index_definition, name=name, type=kind))
    if not args.wait:
        return
    expected = {entry[0] for entry in specs}
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        states = {item["name"]: item.get("status", "UNKNOWN") for item in collection.list_search_indexes() if item["name"] in expected}
        print(states)
        if len(states) == len(expected) and all(state.upper() == "READY" for state in states.values()):
            print("All benchmark indexes are READY.")
            return
        time.sleep(10)
    raise SystemExit("Timed out waiting for the named benchmark indexes to become READY.")


if __name__ == "__main__":
    main()
