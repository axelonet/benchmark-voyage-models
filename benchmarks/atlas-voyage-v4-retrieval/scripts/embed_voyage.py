#!/usr/bin/env python3
"""Generate real Voyage v4 document and query embeddings for a prepared dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import voyageai


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    # hrpoc sections are not truncated locally; Voyage clips the few inputs above the model context limit
    # instead of rejecting a whole batch after earlier batches were already billed.
    "scifact": {"prepared": ROOT / "data" / "prepared", "truncation": False, "prepare": "scripts/prepare_scifact.py"},
    "hrpoc": {"prepared": ROOT / "data" / "prepared-hrpoc", "truncation": True, "prepare": "scripts/prepare_hrpoc.py"},
}
MAX_BATCH_ITEMS = 96
MAX_BATCH_CHARS = 160_000
# A contextualized group is one document; very long documents are split into consecutive groups so no single group exceeds the model context.
MAX_GROUP_CHARS = 60_000
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


def batches(rows: list[Any], size: int, weight: Any = None) -> Iterable[list[Any]]:
    """Fixed-size batches; with a weight function, also close a batch before it exceeds MAX_BATCH_CHARS."""
    batch: list[Any] = []
    batch_weight = 0
    for row in rows:
        row_weight = weight(row) if weight else 0
        if batch and (len(batch) >= size or batch_weight + row_weight > MAX_BATCH_CHARS):
            yield batch
            batch, batch_weight = [], 0
        batch.append(row)
        batch_weight += row_weight
    if batch:
        yield batch


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


def embed_standard(client: Any, chunks: list[dict[str, Any]], queries: list[dict[str, Any]], spec: dict[str, Any], truncation: bool) -> dict[str, int]:
    usage = {"document_tokens": 0, "query_tokens": 0, "document_requests": 0, "query_requests": 0}
    for batch in batches(chunks, MAX_BATCH_ITEMS, lambda item: len(item["text"])):
        response = call_with_retry(lambda: client.embed([item["text"] for item in batch], model=spec["model"], input_type="document", output_dimension=spec["dimensions"], truncation=truncation))
        values = response_embeddings(response)
        if len(values) != len(batch):
            raise RuntimeError(f"{spec['model']} returned {len(values)} document vectors for {len(batch)} chunks")
        for item, vector in zip(batch, values):
            item[spec["field"]] = vector
        usage["document_tokens"] += total_tokens(response)
        usage["document_requests"] += 1
    for batch in batches(queries, 96):
        response = call_with_retry(lambda: client.embed([item["query_text"] for item in batch], model=spec["model"], input_type="query", output_dimension=spec["dimensions"], truncation=truncation))
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
    groups: list[list[dict[str, Any]]] = []
    for _, document in sorted(by_document.items()):
        current: list[dict[str, Any]] = []
        current_chars = 0
        for chunk in sorted(document, key=lambda row: row["chunk_position"]):
            if current and current_chars + len(chunk["text"]) > MAX_GROUP_CHARS:
                groups.append(current)
                current, current_chars = [], 0
            current.append(chunk)
            current_chars += len(chunk["text"])
        groups.append(current)
    for batch in batches(groups, 32, lambda group: sum(len(chunk["text"]) for chunk in group)):
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
        response = call_with_retry(lambda: client.contextualized_embed(inputs=[[item["query_text"]] for item in batch], model=spec["model"], input_type="query", output_dimension=spec["dimensions"], enable_auto_chunking=False))
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


def checkpoint_paths(directory: Path, spec: dict[str, Any]) -> tuple[Path, Path, Path]:
    return directory / f"{spec['field']}.vectors.ndjson", directory / f"{spec['field']}.queries.json", directory / f"{spec['field']}.usage.json"


def save_checkpoint(directory: Path, spec: dict[str, Any], chunks: list[dict[str, Any]], queries: list[dict[str, Any]], usage: dict[str, int]) -> None:
    """Persist one finished representation so a later failure does not discard already-billed work."""
    vectors_path, queries_path, usage_path = checkpoint_paths(directory, spec)
    vectors_tmp = vectors_path.with_suffix(".tmp")
    with vectors_tmp.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({"chunk_id": chunk["chunk_id"], "v": chunk.pop(spec["field"])}, separators=(",", ":")) + "\n")
    vectors_tmp.replace(vectors_path)
    queries_path.write_text(json.dumps({item["query_id"]: item["query_vectors"][spec["query_key"]] for item in queries}, separators=(",", ":")), encoding="utf-8")
    usage_path.write_text(json.dumps(usage), encoding="utf-8")


def checkpoint_complete(directory: Path, spec: dict[str, Any], chunk_count: int) -> bool:
    vectors_path, queries_path, usage_path = checkpoint_paths(directory, spec)
    if not (vectors_path.exists() and queries_path.exists() and usage_path.exists()):
        return False
    with vectors_path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) == chunk_count


def merge_checkpoints(directory: Path, chunks_path: Path, chunks: list[dict[str, Any]]) -> None:
    """Stream the per-representation files back into chunks.ndjson one chunk at a time to bound memory."""
    handles = {spec["field"]: checkpoint_paths(directory, spec)[0].open(encoding="utf-8") for spec in SPECS}
    temporary = chunks_path.with_suffix(".merged")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for chunk in chunks:
                for field, handle in handles.items():
                    record = json.loads(handle.readline())
                    if record["chunk_id"] != chunk["chunk_id"]:
                        raise RuntimeError(f"{field} checkpoint is out of order at chunk {chunk['chunk_id']}")
                    chunk[field] = record["v"]
                output.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
                for field in handles:
                    del chunk[field]
    finally:
        for handle in handles.values():
            handle.close()
    temporary.replace(chunks_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="scifact", help="which prepared dataset to embed")
    parser.add_argument("--force", action="store_true", help="discard saved progress and existing embeddings, then make new billable API calls for every representation")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("VOYAGE_API_KEY"):
        raise SystemExit("Set VOYAGE_API_KEY in .env or the shell. The key is not printed.")
    dataset = DATASETS[args.dataset]
    prepared = dataset["prepared"]
    chunks_path, queries_path, manifest_path = prepared / "chunks.ndjson", prepared / "queries.json", prepared / "manifest.json"
    if not all(path.exists() for path in (chunks_path, queries_path, manifest_path)):
        raise SystemExit(f"Run {dataset['prepare']} first.")
    chunks, queries = read_jsonl(chunks_path), json.loads(queries_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("embeddings", {}).get("status") == "generated" and not args.force:
        raise SystemExit("Embeddings already exist. Use --force only when you intend to make a new set of API calls.")
    if args.force and manifest.get("embeddings", {}).get("status") == "generated":
        for chunk in chunks:
            for spec in SPECS:
                chunk.pop(spec["field"], None)
        for item in queries:
            item.pop("query_vectors", None)
    directory = prepared / "embedding-progress"
    if args.force and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(exist_ok=True)
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    usage: dict[str, dict[str, int]] = {}
    for spec in SPECS:
        if checkpoint_complete(directory, spec, len(chunks)):
            print(f"Reusing saved {spec['field']}; no API calls")
            usage[spec["field"]] = json.loads(checkpoint_paths(directory, spec)[2].read_text(encoding="utf-8"))
            saved = json.loads(checkpoint_paths(directory, spec)[1].read_text(encoding="utf-8"))
            for item in queries:
                item.setdefault("query_vectors", {})[spec["query_key"]] = saved[item["query_id"]]
            continue
        print(f"Embedding {spec['field']}")
        usage[spec["field"]] = embed_contextualized(client, chunks, queries, spec) if spec["contextualized"] else embed_standard(client, chunks, queries, spec, dataset["truncation"])
        save_checkpoint(directory, spec, chunks, queries, usage[spec["field"]])
    merge_checkpoints(directory, chunks_path, chunks)
    queries_path.write_text(json.dumps(queries, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest["embeddings"] = {"status": "generated", "generated_at": datetime.now(timezone.utc).isoformat(), "specs": SPECS, "truncation": dataset["truncation"], "contextualized_group_max_chars": MAX_GROUP_CHARS, "voyage_usage": usage}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(directory)
    print(json.dumps({"chunks": len(chunks), "queries": len(queries), "usage": usage}, indent=2))


if __name__ == "__main__":
    main()
