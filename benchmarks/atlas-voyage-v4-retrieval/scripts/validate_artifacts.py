#!/usr/bin/env python3
"""Validate prepared SciFact documents, labels, and (when present) Voyage vectors."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"
SPECS = {
    "embedding_voyage_4_large_1024": 1024,
    "embedding_voyage_4_1024": 1024,
    "embedding_voyage_4_lite_1024": 1024,
    "embedding_voyage_4_512": 512,
    "embedding_voyage_4_256": 256,
    "embedding_voyage_context_4_1024": 1024,
}


def main() -> None:
    chunks_path, queries_path, manifest_path = PREPARED / "chunks.ndjson", PREPARED / "queries.json", PREPARED / "manifest.json"
    if not all(path.exists() for path in (chunks_path, queries_path, manifest_path)):
        raise SystemExit("Run scripts/prepare_scifact.py first.")
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    parent_ids = {chunk.get("parent_doc_id") for chunk in chunks}
    if not chunks:
        errors.append("No chunks found.")
    if len({chunk.get("chunk_id") for chunk in chunks}) != len(chunks):
        errors.append("Chunk IDs are not unique.")
    for query in queries:
        if not query.get("query_text"):
            errors.append(f"{query.get('query_id')}: missing query text")
        if not set(query.get("relevance", {})).intersection(parent_ids):
            errors.append(f"{query.get('query_id')}: no labelled positive document is present in the corpus")
    if manifest.get("embeddings", {}).get("status") == "generated":
        for chunk in chunks:
            for field, dimension in SPECS.items():
                vector = chunk.get(field)
                if not isinstance(vector, list) or len(vector) != dimension:
                    errors.append(f"{chunk.get('chunk_id')}: {field} does not have {dimension} dimensions")
                    break
        query_keys = {field.removeprefix("embedding_") for field in SPECS}
        for query in queries:
            vectors = query.get("query_vectors", {})
            for key in query_keys:
                expected = SPECS[f"embedding_{key}"]
                vector = vectors.get(key)
                if not isinstance(vector, list) or len(vector) != expected:
                    errors.append(f"{query.get('query_id')}: {key} query vector does not have {expected} dimensions")
                    break
    else:
        errors.append("Embeddings have not been generated.")
    if errors:
        print("Validation failed:")
        print("\n".join(f"- {error}" for error in errors[:25]))
        raise SystemExit(1)
    print(json.dumps({"status": "valid", "chunks": len(chunks), "queries": len(queries), "profile": manifest.get("profile")}, indent=2))


if __name__ == "__main__":
    main()
