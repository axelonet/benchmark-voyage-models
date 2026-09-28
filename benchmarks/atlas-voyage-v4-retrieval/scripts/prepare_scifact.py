#!/usr/bin/env python3
"""Download and prepare a bounded, labelled BEIR SciFact retrieval corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_DIR = DATA / "source"
PREPARED = DATA / "prepared"
ARCHIVE = SOURCE_DIR / "scifact.zip"
SOURCE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
PROFILES = {"starter": {"documents": 1000, "queries": 80}, "full": {"documents": 5183, "queries": 300}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict]:
    with archive.open(name) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chunks_for_document(title: str, body: str, size: int = 160, overlap: int = 30) -> list[str]:
    """A deterministic word-window chunker; the title is retained on every chunk."""
    words = body.split()
    if not words:
        words = [title]
    step = size - overlap
    passages = [" ".join(words[offset : offset + size]) for offset in range(0, len(words), step)]
    return [f"{title}\n\n{passage}".strip() for passage in passages if passage.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="starter")
    parser.add_argument("--force", action="store_true", help="replace only data/prepared for this benchmark")
    args = parser.parse_args()

    if PREPARED.exists() and any(PREPARED.iterdir()):
        if not args.force:
            raise SystemExit(f"{PREPARED} already has artifacts. Re-run with --force after confirming the target.")
        shutil.rmtree(PREPARED)
    PREPARED.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, ARCHIVE)

    with zipfile.ZipFile(ARCHIVE) as archive:
        corpus = {str(item["_id"]): item for item in jsonl_from_zip(archive, "scifact/corpus.jsonl")}
        source_queries = {str(item["_id"]): item["text"] for item in jsonl_from_zip(archive, "scifact/queries.jsonl")}
        with archive.open("scifact/qrels/test.tsv") as handle:
            qrels_lines = [line.decode("utf-8").strip().split("\t") for line in handle if line.strip()]

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for query_id, corpus_id, score in qrels_lines[1:]:
        relevance = int(score)
        if relevance > 0:
            qrels[query_id][corpus_id] = relevance
    eligible = sorted(query_id for query_id, rels in qrels.items() if query_id in source_queries and any(doc_id in corpus for doc_id in rels))
    desired = min(PROFILES[args.profile]["queries"], len(eligible))
    selected_query_ids = sorted(random.Random(20260921).sample(eligible, desired))
    needed_docs = {doc_id for query_id in selected_query_ids for doc_id in qrels[query_id] if doc_id in corpus}
    target_docs = max(PROFILES[args.profile]["documents"], len(needed_docs))
    remaining = sorted(set(corpus).difference(needed_docs))
    rng = random.Random(20260921)
    selected_doc_ids = sorted(needed_docs | set(rng.sample(remaining, target_docs - len(needed_docs))))

    chunks: list[dict] = []
    for doc_id in selected_doc_ids:
        source = corpus[doc_id]
        title = str(source.get("title", "")).strip()
        body = str(source.get("text", "")).strip()
        for position, text in enumerate(chunks_for_document(title, body)):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}:{position}",
                    "parent_doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "chunk_position": position,
                    "source": "BEIR SciFact",
                }
            )

    queries = [
        {
            "query_id": query_id,
            "query_text": source_queries[query_id],
            "relevance": {doc_id: score for doc_id, score in qrels[query_id].items() if doc_id in selected_doc_ids},
        }
        for query_id in selected_query_ids
    ]
    write_jsonl(PREPARED / "chunks.ndjson", chunks)
    (PREPARED / "queries.json").write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "profile": args.profile,
        "source": {"url": SOURCE_URL, "archive_sha256": sha256(ARCHIVE), "dataset": "BEIR SciFact", "qrels_split": "test"},
        "selection": {"seed": 20260921, "source_documents": len(selected_doc_ids), "queries": len(queries), "chunks": len(chunks)},
        "chunking": {"method": "word-window", "chunk_size_words": 160, "chunk_overlap_words": 30, "title_in_each_chunk": True},
        "embeddings": {"status": "not-generated"},
    }
    (PREPARED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["selection"], indent=2))


if __name__ == "__main__":
    main()
