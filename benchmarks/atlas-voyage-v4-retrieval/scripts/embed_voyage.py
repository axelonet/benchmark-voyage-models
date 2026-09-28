#!/usr/bin/env python3
"""Generate real Voyage v4 document and query embeddings for the prepared SciFact set."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import voyageai


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"
SPECS = [
    {"field": "embedding_voyage_4_large_1024", "query_key": "voyage_4_large_1024", "model": "voyage-4-large", "dimensions": 1024, "contextualized": False},
    {"field": "embedding_voyage_4_1024", "query_key": "voyage_4_1024", "model": "voyage-4", "dimensions": 1024, "contextualized": False},
    {"field": "embedding_voyage_4_lite_1024", "query_key": "voyage_4_lite_1024", "model": "voyage-4-lite", "dimensions": 1024, "contextualized": False},
    {"field": "embedding_voyage_4_512", "query_key": "voyage_4_512", "model": "voyage-4", "dimensions": 512, "contextualized": False},
    {"field": "embedding_voyage_4_256", "query_key": "voyage_4_256", "model": "voyage-4", "dimensions": 256, "contextualized": False},
    {"field": "embedding_voyage_context_4_1024", "query_key": "voyage_context_4_1024", "model": "voyage-context-4", "dimensions": 1024, "contextualized": True},
]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def batches(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def total_tokens(response: Any) -> int:
    value = getattr(response, "total_tokens", None)
    if value is None and isinstance(response, dict):
        value = response.get("total_tokens", response.get("usage", {}).get("total_tokens", 0))
    return int(value or 0)


def response_embeddings(response: Any) -> list[list[float]]:
    return list(getattr(response, "embeddings", response["embeddings"] if isinstance(response, dict) else []))


def context_embeddings(response: Any) -> list[list[list[float]]]:
    results = getattr(response, "results", response["results"] if isinstance(response, dict) else [])
    extracted = []
    for result in results:
        extracted.append(list(getattr(result, "embeddings", result["embeddings"] if isinstance(result, dict) else [])))
    return extracted


def call_with_retry(operation: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return operation()
        except Exception as error:  # API errors are surfaced after a short bounded retry.
            last_error = error
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Voyage request failed after retries: {last_error}")


def embed_standard(client: Any, chunks: list[dict[str, Any]], queries: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, int]:
    usage = {"document_tokens": 0, "query_tokens": 0, "document_requests": 0, "query_requests": 0}
    for batch in batches(chunks, 96):
        response = call_with_retry(lambda: client.embed([item["text"] for item in batch], model=spec["model"], input_type="document", output_dimension=spec["dimensions"], truncation=False))
        values = response_embeddings(response)
        if len(values) != len(batch):
            raise RuntimeError(f"{spec['model']} returned {len(values)} document vectors for {len(batch)} chunks")
        for item, vector in zip(batch, values):
            item[spec["field"]] = vector
        usage["document_tokens"] += total_tokens(response)
        usage["document_requests"] += 1
    for batch in batches(queries, 96):
        response = call_with_retry(lambda: client.embed([item["query_text"] for item in batch], model=spec["model"], input_type="query", output_dimension=spec["dimensions"], truncation=False))
        values = response_embeddings(response)
        if len(values) != len(batch):
            raise RuntimeError(f"{spec['model']} returned {len(values)} query vectors for {len(batch)} queries")
        for item, vector in zip(batch, values):
            item.setdefault("query_vectors", {})[spec["query_key"]] = vector
        usage["query_tokens"] += total_tokens(response)
        usage["query_requests"] += 1
    return usage


def embed_contextualized(client: Any, chunks: list[dict[str, Any]], queries: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, int]:
    usage = {"document_tokens": 0, "query_tokens": 0, "document_requests": 0, "query_requests": 0}
    by_document: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk["parent_doc_id"], []).append(chunk)
    groups = [sorted(group, key=lambda row: row["chunk_position"]) for _, group in sorted(by_document.items())]
    for batch in batches(groups, 32):
        inputs = [[chunk["text"] for chunk in group] for group in batch]
        response = call_with_retry(lambda: client.contextualized_embed(inputs=inputs, model=spec["model"], input_type="document", output_dimension=spec["dimensions"], enable_auto_chunking=False))
        values = context_embeddings(response)
        if len(values) != len(batch):
            raise RuntimeError(f"{spec['model']} returned {len(values)} document groups for {len(batch)} groups")
        for group, vectors in zip(batch, values):
            if len(vectors) != len(group):
                raise RuntimeError(f"{spec['model']} returned {len(vectors)} vectors for a group of {len(group)} chunks")
            for chunk, vector in zip(group, vectors):
                chunk[spec["field"]] = vector
        usage["document_tokens"] += total_tokens(response)
        usage["document_requests"] += 1
    for batch in batches(queries, 96):
        response = call_with_retry(lambda: client.contextualized_embed(inputs=[item["query_text"] for item in batch], model=spec["model"], input_type="query", output_dimension=spec["dimensions"], enable_auto_chunking=False))
        values = context_embeddings(response)
        if len(values) != len(batch):
            raise RuntimeError(f"{spec['model']} returned {len(values)} query groups for {len(batch)} queries")
        for item, vectors in zip(batch, values):
            if len(vectors) != 1:
                raise RuntimeError(f"{spec['model']} returned {len(vectors)} vectors for a query")
            item.setdefault("query_vectors", {})[spec["query_key"]] = vectors[0]
        usage["query_tokens"] += total_tokens(response)
        usage["query_requests"] += 1
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace embeddings in the prepared artifacts and make new billable API calls")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("VOYAGE_API_KEY"):
        raise SystemExit("Set VOYAGE_API_KEY in .env or the shell. The key is not printed.")
    chunks_path, queries_path, manifest_path = PREPARED / "chunks.ndjson", PREPARED / "queries.json", PREPARED / "manifest.json"
    if not all(path.exists() for path in (chunks_path, queries_path, manifest_path)):
        raise SystemExit("Run scripts/prepare_scifact.py first.")
    chunks, queries = read_jsonl(chunks_path), json.loads(queries_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("embeddings", {}).get("status") == "generated" and not args.force:
        raise SystemExit("Embeddings already exist. Use --force only when you intend to make a new set of API calls.")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    usage: dict[str, dict[str, int]] = {}
    for spec in SPECS:
        print(f"Embedding {spec['field']}")
        usage[spec["field"]] = embed_contextualized(client, chunks, queries, spec) if spec["contextualized"] else embed_standard(client, chunks, queries, spec)
    write_jsonl(chunks_path, chunks)
    queries_path.write_text(json.dumps(queries, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest["embeddings"] = {"status": "generated", "generated_at": datetime.now(timezone.utc).isoformat(), "specs": SPECS, "voyage_usage": usage}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chunks": len(chunks), "queries": len(queries), "usage": usage}, indent=2))


if __name__ == "__main__":
    main()
