#!/usr/bin/env python3
"""Prepare the client HR-policy POC package as section and table-piece chunks, labelled queries, and resolved audiences."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source" / "hrpoc"
PREPARED = ROOT / "data" / "prepared-hrpoc"
IMAGE_PLACEHOLDER = "Image description unavailable"


def sha256_16(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_manifest() -> dict[str, Any]:
    """MANIFEST.json predates table_sections.json, so that file is required but checksummed only for the record."""
    if not (SOURCE / "table_sections.json").exists():
        raise SystemExit(f"Missing {SOURCE / 'table_sections.json'}, the table-hash to section_id mapping from the client.")
    manifest = json.loads((SOURCE / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = SOURCE / name
        if not path.exists():
            raise SystemExit(f"Missing {path}. Copy the client package into {SOURCE}.")
        if path.stat().st_size != expected["bytes"] or sha256_16(path) != expected["sha256_16"]:
            raise SystemExit(f"{name} does not match MANIFEST.json (size or sha256_16).")
    return manifest


def evaluate(node: dict[str, Any], params: dict[str, Any]) -> bool:
    """A rule is {operand, operator: in, value}; a node ANDs/ORs its children. A missing attribute never matches."""
    if "children" in node:
        results = [evaluate(child, params) for child in node["children"]]
        return all(results) if node.get("combinator") == "AND" else any(results)
    if node.get("operator") != "in":
        raise SystemExit(f"Unsupported audience operator: {node.get('operator')}")
    actual = params.get(node["operand"])
    if actual is None:
        return False
    actual_values = actual if isinstance(actual, list) else [actual]
    return any(item in node["value"] for item in actual_values)


def hierarchy_paths(nodes: list[dict[str, Any]], trail: list[str], paths: dict[str, str]) -> None:
    for node in nodes:
        here = trail + [str(node.get("title") or "").strip()]
        paths[node.get("section_id") or node.get("id")] = " > ".join(part for part in here if part)
        hierarchy_paths(node.get("children", []), here, paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace only data/prepared-hrpoc (never data/prepared)")
    parser.add_argument("--drop-empty", action="store_true", help="remove sections whose content is empty; default keeps every section")
    args = parser.parse_args()

    manifest_in = verify_manifest()
    if PREPARED.exists() and any(PREPARED.iterdir()):
        if not args.force:
            raise SystemExit(f"{PREPARED} already has artifacts. Re-run with --force after confirming the target.")
        shutil.rmtree(PREPARED)
    PREPARED.mkdir(parents=True, exist_ok=True)

    audiences = json.loads((SOURCE / "bot_audiences.json").read_text(encoding="utf-8"))
    trees = {item["audience_id"]: json.loads(item["filters"]) for item in audiences if item["status"] == "active"}
    queries_in = read_jsonl(SOURCE / "queries.jsonl")
    gold = {item["query_id"]: item for item in read_jsonl(SOURCE / "gold.jsonl")}
    if [item["query_id"] for item in queries_in] != list(gold):
        raise SystemExit("queries.jsonl and gold.jsonl query_ids differ.")

    asker_cache: dict[str, list[str]] = {}
    queries: list[dict[str, Any]] = []
    for item in queries_in:
        params = item["user_params"]
        key = json.dumps(params, sort_keys=True)
        if key not in asker_cache:
            asker_cache[key] = sorted(audience_id for audience_id, tree in trees.items() if evaluate(tree, params))
        label = gold[item["query_id"]]
        queries.append(
            {
                "query_id": item["query_id"],
                "query_text": item["query"],
                "language": item["language"],
                "user_params": params,
                "asker_audience_ids": asker_cache[key],
                "answerability": label["answerability"],
                "query_type": label["query_type"],
                "expected_answer": label["expected_answer"],
                "gold_policy_ids": label["gold_policy_ids"],
                "gold_section_ids": label["gold_section_ids"],
                "relevance": {policy_id: 1 for policy_id in label["gold_policy_ids"]},
            }
        )
    gold_sections = {section_id for item in queries for section_id in item["gold_section_ids"]}

    table_sentences = {table_hash: entry[0] for table_hash, entry in json.loads((SOURCE / "table_hash.json").read_text(encoding="utf-8")).items()}
    table_sections = json.loads((SOURCE / "table_sections.json").read_text(encoding="utf-8"))
    if set(table_sections) != set(table_sentences):
        raise SystemExit("table_sections.json and table_hash.json do not list the same table hashes.")
    sections_of_table: dict[str, list[str]] = {}
    for table_hash, section_ids in table_sections.items():
        for section_id in section_ids:
            sections_of_table.setdefault(section_id, []).append(table_hash)

    chunks: list[dict[str, Any]] = []
    stats = {"documents": 0, "sections_seen": 0, "empty_content": 0, "dropped_empty": 0, "image_placeholder": 0, "open_documents": 0, "restricted_documents": 0, "table_pieces": 0, "table_pieces_skipped_no_sentences": 0}
    visibility_mismatch = 0
    doc_lang: dict[str, str] = {}
    with (SOURCE / "policies_v2.ndjson").open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            stats["documents"] += 1
            doc_lang[doc["policy_id"]] = doc.get("lang")
            tree = doc.get("audiences") or None
            audience_id = ((doc.get("audience") or {}).get("audienceFilter") or {}).get("audienceId")
            is_open = tree is None
            stats["open_documents" if is_open else "restricted_documents"] += 1
            if not is_open:
                for key, asker in asker_cache.items():
                    direct = evaluate(tree, json.loads(key))
                    if direct != (audience_id in asker):
                        visibility_mismatch += 1
            paths: dict[str, str] = {}
            hierarchy_paths(doc.get("policy_search_sections_hierarchy", []), [], paths)
            next_table_position = len(doc.get("policy_search_sections", []))
            for position, section in enumerate(doc.get("policy_search_sections", [])):
                stats["sections_seen"] += 1
                if section.get("audiences") or (section.get("audience") or {}).get("applicableToAll") is not True:
                    raise SystemExit(f"Section {section.get('section_id')} has its own audience, which overrides the document's. prepare_hrpoc.py only supports document-level audiences.")
                content = (section.get("content") or "").strip()
                section_id = section["section_id"]
                if not content:
                    stats["empty_content"] += 1
                    if args.drop_empty:
                        if section_id in gold_sections:
                            raise SystemExit(f"Gold section {section_id} has empty content; re-run without --drop-empty.")
                        stats["dropped_empty"] += 1
                        continue
                if IMAGE_PLACEHOLDER in content:
                    stats["image_placeholder"] += 1
                title = (section.get("title") or "").strip()
                title_path = paths.get(section_id) or " > ".join(part for part in (doc.get("name"), title) if part)
                text = "\n\n".join(part for part in (title_path, content) if part)
                section_chunk = {
                    "chunk_id": section_id,
                    "section_id": section_id,
                    "parent_doc_id": doc["policy_id"],
                    "doc_name": doc.get("name"),
                    "lang": doc.get("lang"),
                    "title": title,
                    "title_path": title_path,
                    "text": text,
                    "chunk_position": position,
                    "kind": "section",
                    "keywords": section.get("keywords") or [],
                    "open": is_open,
                    "audience_ids": [] if is_open or not audience_id else [audience_id],
                    "source": "HR policy POC",
                }
                chunks.append(section_chunk)
                # Production indexes table-to-text sentences as extra pieces of the owning section; a table that appears
                # in both the .pdf and .docx copy yields one piece per owning section. Pieces keep the section's audience.
                for table_hash in sections_of_table.get(section_id, []):
                    sentences = table_sentences[table_hash]
                    if not sentences:
                        stats["table_pieces_skipped_no_sentences"] += 1
                        continue
                    stats["table_pieces"] += 1
                    next_table_position += 1
                    chunks.append(
                        {
                            **{key: value for key, value in section_chunk.items() if key not in ("chunk_id", "text", "chunk_position", "kind", "keywords")},
                            "chunk_id": f"{section_id}#table-{table_hash[:12]}",
                            "text": "\n\n".join(part for part in (title_path, " ".join(sentences)) if part),
                            "chunk_position": next_table_position,
                            "kind": "table",
                            "table_hash": table_hash,
                            "keywords": [],
                        }
                    )
    if visibility_mismatch:
        raise SystemExit(f"{visibility_mismatch} document/asker pairs disagree between the document rule tree and bot_audiences evaluation.")
    section_ids = {chunk["section_id"] for chunk in chunks if chunk["kind"] == "section"}
    unknown_table_sections = {section_id for ids in table_sections.values() for section_id in ids} - section_ids
    if unknown_table_sections:
        raise SystemExit(f"{len(unknown_table_sections)} section ids in table_sections.json are not prepared sections (dropped or absent).")
    missing_gold = gold_sections - section_ids
    if missing_gold:
        raise SystemExit(f"{len(missing_gold)} gold_section_ids are not in the prepared chunks.")
    for item in queries:
        item["gold_doc_langs"] = sorted({doc_lang[policy_id] for policy_id in item["gold_policy_ids"] if policy_id in doc_lang})
        item["cross_lingual"] = bool(item["gold_doc_langs"]) and item["language"] not in item["gold_doc_langs"]

    write_jsonl(PREPARED / "chunks.ndjson", chunks)
    (PREPARED / "queries.json").write_text(json.dumps(queries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "source": {"dataset": "HR policy POC (client package)", "files": manifest_in, "table_sections.json": {"sha256_16": sha256_16(SOURCE / "table_sections.json"), "tables": len(table_sections)}},
        "selection": {**stats, "sections": len(section_ids), "queries": len(queries), "chunks": len(chunks), "distinct_asker_profiles": len(asker_cache)},
        "text_construction": {"fields": "title_path + content", "truncated_locally": False, "html_content_used": False, "table_sentences": "indexed as one extra piece per (table, owning section) via table_sections.json; tables with no sentences are skipped"},
        "empty_sections": "dropped" if args.drop_empty else "kept",
        "embeddings": {"status": "not-generated"},
    }
    (PREPARED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["selection"], indent=2))


if __name__ == "__main__":
    main()
