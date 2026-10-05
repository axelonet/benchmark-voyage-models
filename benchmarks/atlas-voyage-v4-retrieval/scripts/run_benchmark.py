#!/usr/bin/env python3
"""Run serial, labelled Atlas Vector Search measurements for the Voyage v4 matrix on a prepared dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = {
    "scifact": {"name": "scifact", "prepared": ROOT / "data" / "prepared", "collection_env": "BENCHMARK_COLLECTION", "collection": "chunks", "prefix": "", "result_limit": 10, "index_overrides": {}, "prepare": "the prepare and embedding steps"},
    # hrpoc keeps the 50-section result list the client asked for; its audience pre-filter is per query,
    # and vs_voyage_4_1024 already declares the filter fields so no duplicate _acl index is created.
    "hrpoc": {"name": "hrpoc", "prepared": ROOT / "data" / "prepared-hrpoc", "collection_env": "HRPOC_COLLECTION", "collection": "hrpoc_chunks", "prefix": "hrpoc-", "result_limit": 50, "index_overrides": {"vs_voyage_4_1024_acl": "vs_voyage_4_1024"}, "prepare": "scripts/prepare_hrpoc.py and the embedding step"},
}
VARIANTS = json.loads((ROOT / "config" / "variants.json").read_text(encoding="utf-8"))


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


def percentile(values: list[float], point: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * point
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def index_name(variant: dict[str, Any], dataset: dict[str, Any]) -> str:
    return dataset["index_overrides"].get(variant["index"], variant["index"])


def vector_filter(variant: dict[str, Any], query: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any] | None:
    if dataset["name"] == "hrpoc":
        return {"$or": [{"open": True}, {"audience_ids": {"$in": query["asker_audience_ids"]}}]}
    if variant["family"] == "hybrid":
        return {"effective_principal_ids": variant["acl_tenant_id"]}
    return None


def text_filter(variant: dict[str, Any], query: dict[str, Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    if dataset["name"] == "hrpoc":
        should: list[dict[str, Any]] = [{"equals": {"path": "open", "value": True}}]
        if query["asker_audience_ids"]:
            should.append({"in": {"path": "audience_ids", "value": query["asker_audience_ids"]}})
        return [{"compound": {"should": should, "minimumShouldMatch": 1}}]
    return [{"equals": {"path": "effective_principal_ids", "value": variant["acl_tenant_id"]}}]


def retrieval_pipeline(variant: dict[str, Any], query: dict[str, Any], exact: bool = False, dataset: dict[str, Any] = DATASETS["scifact"]) -> list[dict[str, Any]]:
    vector = {
        "index": index_name(variant, dataset),
        "path": variant["vector_field"],
        "queryVector": query["query_vectors"][variant["query_key"]],
        "limit": variant["fetch_k"],
    }
    if exact:
        vector["exact"] = True
    else:
        vector["numCandidates"] = variant["num_candidates"]
    if (acl := vector_filter(variant, query, dataset)) is not None:
        vector["filter"] = acl
    return [
        {"$vectorSearch": vector},
        {"$project": {"_id": 0, "chunk_id": 1, "section_id": 1, "parent_doc_id": 1, "text": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]


def native_rerank_pipeline(variant: dict[str, Any], query: dict[str, Any], dataset: dict[str, Any] = DATASETS["scifact"]) -> list[dict[str, Any]]:
    """Rerank Atlas Vector Search candidates in one aggregation request.

    The $project stage makes the Vector Search score an ordinary field before
    $rerank replaces the score metadata with its own score.
    """
    vector = {
        "index": index_name(variant, dataset),
        "path": variant["vector_field"],
        "queryVector": query["query_vectors"][variant["query_key"]],
        "numCandidates": variant["num_candidates"],
        "limit": variant["fetch_k"],
    }
    if (acl := vector_filter(variant, query, dataset)) is not None:
        vector["filter"] = acl
    return [
        {"$vectorSearch": vector},
        {"$project": {"_id": 0, "chunk_id": 1, "section_id": 1, "parent_doc_id": 1, "text": 1, "vector_search_score": {"$meta": "vectorSearchScore"}}},
        {
            "$rerank": {
                "model": os.getenv("ATLAS_RERANK_MODEL", "rerank-2.5-lite"),
                "query": {"text": query["query_text"]},
                "path": "text",
                "numDocsToRerank": variant.get("rerank_k", variant["fetch_k"]),
            }
        },
        {"$set": {"rerank_score": {"$meta": "score"}}},
        {"$limit": dataset["result_limit"]},
        {"$project": {"chunk_id": 1, "section_id": 1, "parent_doc_id": 1, "text": 1, "vector_search_score": 1, "rerank_score": 1}},
    ]


def hybrid_pipeline(variant: dict[str, Any], query: dict[str, Any], dataset: dict[str, Any] = DATASETS["scifact"]) -> list[dict[str, Any]]:
    """Native $rankFusion combining one $vectorSearch pipeline and one BM25 $search pipeline.

    The ACL filter here uses acl_tenant_id, a value every chunk carries regardless of its
    synthetic permission group, so this pipeline's accuracy numbers are unaffected by the
    filter. Group-level isolation is verified separately by
    scripts/test_permission_isolation.py, which filters by a specific group instead.
    For the hrpoc dataset the filter is the asker's real audience (open documents or a matching audience_ids).
    """
    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": index_name(variant, dataset),
                                    "path": variant["vector_field"],
                                    "queryVector": query["query_vectors"][variant["query_key"]],
                                    "numCandidates": variant["num_candidates"],
                                    "limit": variant["fetch_k"],
                                    "filter": vector_filter(variant, query, dataset),
                                }
                            }
                        ],
                        "text": [
                            {
                                "$search": {
                                    "index": variant["text_index"],
                                    "compound": {
                                        "must": [{"text": {"query": query["query_text"], "path": "text"}}],
                                        "filter": text_filter(variant, query, dataset),
                                    },
                                }
                            },
                            {"$limit": variant["fetch_k"]},
                        ],
                    }
                },
                "combination": {"weights": variant["hybrid_weights"]},
            }
        },
        {"$limit": dataset["result_limit"]},
        {"$project": {"_id": 0, "chunk_id": 1, "section_id": 1, "parent_doc_id": 1, "text": 1, "score": {"$meta": "score"}}},
    ]


def invoke(collection: Any, pipeline: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    records = list(collection.aggregate(pipeline))
    return records, round((time.perf_counter() - started) * 1000, 3)


def dedupe_documents(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    output, seen = [], set()
    for row in rows:
        if row["parent_doc_id"] not in seen:
            seen.add(row["parent_doc_id"])
            output.append(row)
        if len(output) == limit:
            break
    return output


def metrics(relevance: dict[str, int], rows: list[dict[str, Any]], limit: int = 10) -> dict[str, float | bool]:
    returned = [row["parent_doc_id"] for row in dedupe_documents(rows, limit)]
    relevant = {doc_id: grade for doc_id, grade in relevance.items() if grade > 0}
    relevant_returned = [doc_id for doc_id in returned if doc_id in relevant]
    reciprocal_rank = next((1 / position for position, doc_id in enumerate(returned, start=1) if doc_id in relevant), 0.0)
    dcg = sum(((2 ** relevant.get(doc_id, 0) - 1) / math.log2(position + 1)) for position, doc_id in enumerate(returned, start=1))
    ideal_grades = sorted(relevant.values(), reverse=True)[:limit]
    ideal_dcg = sum(((2**grade - 1) / math.log2(position + 1)) for position, grade in enumerate(ideal_grades, start=1))
    return {
        "hit_at_10": bool(relevant_returned),
        "hit_at_5": bool(set(returned[:5]) & set(relevant)),
        "hit_at_1": bool(set(returned[:1]) & set(relevant)),
        "recall_at_10": len(set(relevant_returned)) / max(len(relevant), 1),
        "precision_at_10": len(set(relevant_returned)) / limit,
        "mrr_at_10": reciprocal_rank,
        "ndcg_at_10": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def dedupe_sections(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """hrpoc indexes a section and its table-to-text pieces as separate rows; a result list holds each section once, at its best rank."""
    output, seen = [], set()
    for row in rows:
        if row["section_id"] not in seen:
            seen.add(row["section_id"])
            output.append(row)
        if len(output) == limit:
            break
    return output


def section_metrics(query: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, bool]:
    """The client's scoring: a returned section counts if its document, or the section itself, is gold, within the first N results."""
    gold_documents, gold_sections = set(query["gold_policy_ids"]), set(query["gold_section_ids"])
    documents, sections = [row["parent_doc_id"] for row in rows], [row["section_id"] for row in rows]
    scored = {f"doc_hit_at_{n}": bool(gold_documents & set(documents[:n])) for n in (1, 5, 10)}
    if gold_sections:
        scored.update({f"passage_hit_at_{n}": bool(gold_sections & set(sections[:n])) for n in (5, 10)})
    return scored


def rerank(query_text: str, rows: list[dict[str, Any]], limit: int = 10) -> tuple[list[dict[str, Any]], float]:
    if not os.getenv("VOYAGE_API_KEY"):
        raise RuntimeError("Set VOYAGE_API_KEY to run reranking.")
    from voyageai import Client

    started = time.perf_counter()
    response = Client(api_key=os.environ["VOYAGE_API_KEY"]).rerank(query_text, [row["text"] for row in rows], model=os.getenv("VOYAGE_RERANK_MODEL", "rerank-3-lite"), top_k=min(limit, len(rows)))
    ranked = []
    for item in response.results:
        position = item.index if hasattr(item, "index") else item["index"]
        ranked.append(rows[position])
    return ranked, round((time.perf_counter() - started) * 1000, 3)


def select_variants(profile: str) -> tuple[list[dict[str, Any]], str]:
    if profile == "smoke":
        return [variant for variant in VARIANTS if variant["name"] in {"voyage-4-1024", "voyage-context-4-1024", "voyage-4-1024-scalar"}], "standard"
    rerank_only_families = {"rerank_depth", "dimension:matryoshka+rerank", "quantization+rerank", "hybrid"}
    if profile == "standard":
        return [variant for variant in VARIANTS if variant["family"] not in rerank_only_families], "standard"
    rerank_variant_names = {
        "voyage-4-1024-rerank50", "voyage-4-1024-rerank10", "voyage-4-1024-rerank20",
        "voyage-4-1024-fk10", "voyage-4-1024-fk20",
        "voyage-context-4-1024",
        "voyage-4-512-rerank", "voyage-4-1024-scalar-rerank",
    }
    if profile == "rerank":
        return [variant for variant in VARIANTS if variant["name"] in rerank_variant_names], "client-rerank"
    if profile == "native-rerank":
        return [variant for variant in VARIANTS if variant["name"] in rerank_variant_names], "native-rerank"
    if profile == "hybrid":
        return [variant for variant in VARIANTS if variant["family"] == "hybrid"], "hybrid"
    raise ValueError(profile)


def top_score(rows: list[dict[str, Any]], mode: str) -> float | None:
    """Best Vector Search similarity among the retrieved sections, so reranked and plain rows share one scale. Hybrid rows only carry a $rankFusion score."""
    if not rows:
        return None
    if mode == "hybrid":
        return rows[0].get("score")
    scores = [row.get("vector_search_score", row.get("score")) for row in rows]
    return max((score for score in scores if score is not None), default=None)


def mean_of(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record["metrics"][key] for record in records if key in record["metrics"]]
    return round(statistics.mean(values), 4) if values else None


def hrpoc_summary(scored: list[dict[str, Any]], unanswerable: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    keys = ("doc_hit_at_1", "doc_hit_at_5", "doc_hit_at_10", "passage_hit_at_5", "passage_hit_at_10")
    def group(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {"queries": len(records), **{key: mean_of(records, key) for key in keys}}
    scores = [record["top_score"] for record in unanswerable if record.get("top_score") is not None]
    return {
        "by_language": {language: group([record for record in scored if record["language"] == language]) for language in sorted({record["language"] for record in scored})},
        "by_language_match": {
            "same_language": group([record for record in scored if not record["cross_lingual"]]),
            "cross_lingual": group([record for record in scored if record["cross_lingual"]]),
        },
        "unanswerable": {
            "queries": len(unanswerable),
            "top_score_basis": "$rankFusion score" if mode == "hybrid" else "Vector Search similarity",
            "top_score_p50": round(percentile(scores, 0.5), 4) if scores else None,
            "top_score_p95": round(percentile(scores, 0.95), 4) if scores else None,
            "top_score_max": round(max(scores), 4) if scores else None,
        },
    }


def run_variant(collection: Any, variant: dict[str, Any], queries: list[dict[str, Any]], mode: str, dataset: dict[str, Any] = DATASETS["scifact"]) -> dict[str, Any]:
    records = []
    hrpoc = dataset["name"] == "hrpoc"
    for query in queries:
        try:
            if mode == "native-rerank":
                final_rows, pipeline_ms = invoke(collection, native_rerank_pipeline(variant, query, dataset))
                rows, retrieval_ms, rerank_ms, total_ms = final_rows, None, None, pipeline_ms
            elif mode == "hybrid":
                final_rows, pipeline_ms = invoke(collection, hybrid_pipeline(variant, query, dataset))
                rows, retrieval_ms, rerank_ms, total_ms = final_rows, None, None, pipeline_ms
            else:
                rows, retrieval_ms = invoke(collection, retrieval_pipeline(variant, query, dataset=dataset))
                final_rows, rerank_ms = rows, None
                if mode == "client-rerank":
                    final_rows, rerank_ms = rerank(query["query_text"], rows[: variant.get("rerank_k", len(rows))], dataset["result_limit"])
                total_ms = round(retrieval_ms + (rerank_ms or 0), 3)
            if hrpoc:
                final_rows = dedupe_sections(final_rows, dataset["result_limit"])
            unanswerable = hrpoc and query["answerability"] == "none"
            result = None if unanswerable else metrics(query["relevance"], final_rows)
            if result is not None and hrpoc:
                result.update(section_metrics(query, final_rows))
            enn_overlap = None
            if variant["enn_reference"] and mode == "standard" and not unanswerable:
                enn_rows, _ = invoke(collection, retrieval_pipeline(variant, query, exact=True, dataset=dataset))
                ann_ids = {row["parent_doc_id"] for row in dedupe_documents(rows)}
                enn_ids = {row["parent_doc_id"] for row in dedupe_documents(enn_rows)}
                enn_overlap = len(ann_ids.intersection(enn_ids)) / max(len(enn_ids), 1)
            if hrpoc:
                retrieved = [
                    {"section_id": row["section_id"], "chunk_id": row["chunk_id"], "parent_doc_id": row["parent_doc_id"], "score": row.get("rerank_score") if row.get("rerank_score") is not None else row.get("score", row.get("vector_search_score"))}
                    for row in final_rows
                ]
            else:
                retrieved = [
                    {
                        "parent_doc_id": row["parent_doc_id"],
                        "chunk_id": row["chunk_id"],
                        "score": row.get("score"),
                        "vector_search_score": row.get("vector_search_score"),
                        "rerank_score": row.get("rerank_score"),
                    }
                    for row in dedupe_documents(final_rows)
                ]
            record = {
                "query_id": query["query_id"],
                "retrieved": retrieved,
                "metrics": result,
                "ann_enn_overlap_at_10": enn_overlap,
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "pipeline_ms": pipeline_ms if mode in ("native-rerank", "hybrid") else None,
                "total_ms": total_ms,
            }
            if hrpoc:
                record.update({"language": query["language"], "cross_lingual": query["cross_lingual"], "answerability": query["answerability"], "top_score": top_score(rows, mode)})
            records.append(record)
        except Exception as error:
            records.append({"query_id": query["query_id"], "error": str(error)})
    complete = [record for record in records if "error" not in record]
    if not complete:
        return {"variant": variant, "status": "failed", "records": records}
    scored = [record for record in complete if record["metrics"] is not None]
    if not scored:
        return {"variant": variant, "status": "failed", "records": records}
    result: dict[str, Any] = {
        "variant": variant,
        "status": "complete" if len(complete) == len(records) else "partial",
        "metrics": {
            "queries": len(scored),
            "recall_at_10": round(statistics.mean(record["metrics"]["recall_at_10"] for record in scored), 4),
            "mrr_at_10": round(statistics.mean(record["metrics"]["mrr_at_10"] for record in scored), 4),
            "ndcg_at_10": round(statistics.mean(record["metrics"]["ndcg_at_10"] for record in scored), 4),
            "hit_at_10": round(statistics.mean(record["metrics"]["hit_at_10"] for record in scored), 4),
            "hit_at_5": round(statistics.mean(record["metrics"]["hit_at_5"] for record in scored), 4),
            "hit_at_1": round(statistics.mean(record["metrics"]["hit_at_1"] for record in scored), 4),
            "precision_at_10": round(statistics.mean(record["metrics"]["precision_at_10"] for record in scored), 4),
            "ann_enn_overlap_at_10": round(statistics.mean(record["ann_enn_overlap_at_10"] for record in scored if record["ann_enn_overlap_at_10"] is not None), 4) if any(record["ann_enn_overlap_at_10"] is not None for record in scored) else None,
            "retrieval_p50_ms": round(percentile([record["retrieval_ms"] for record in complete if record["retrieval_ms"] is not None], 0.5), 3) if mode not in ("native-rerank", "hybrid") else None,
            "retrieval_p95_ms": round(percentile([record["retrieval_ms"] for record in complete if record["retrieval_ms"] is not None], 0.95), 3) if mode not in ("native-rerank", "hybrid") else None,
            "rerank_p50_ms": round(percentile([record["rerank_ms"] for record in complete if record["rerank_ms"] is not None], 0.5), 3) if mode == "client-rerank" else None,
            "native_pipeline_p50_ms": round(percentile([record["pipeline_ms"] for record in complete if record["pipeline_ms"] is not None], 0.5), 3) if mode == "native-rerank" else None,
            "native_pipeline_p95_ms": round(percentile([record["pipeline_ms"] for record in complete if record["pipeline_ms"] is not None], 0.95), 3) if mode == "native-rerank" else None,
            "hybrid_pipeline_p50_ms": round(percentile([record["pipeline_ms"] for record in complete if record["pipeline_ms"] is not None], 0.5), 3) if mode == "hybrid" else None,
            "hybrid_pipeline_p95_ms": round(percentile([record["pipeline_ms"] for record in complete if record["pipeline_ms"] is not None], 0.95), 3) if mode == "hybrid" else None,
            "total_p50_ms": round(percentile([record["total_ms"] for record in complete], 0.5), 3),
            "total_p95_ms": round(percentile([record["total_ms"] for record in complete], 0.95), 3),
        },
        "timing_scope": {
            "standard": "one client timer around $vectorSearch aggregation",
            "client-rerank": "separate client timers around $vectorSearch aggregation and Voyage rerank API call",
            "native-rerank": "one client timer around $vectorSearch + $rerank aggregation; the two server-side components are not separable from this client observation",
            "hybrid": "one client timer around the $rankFusion aggregation combining a $vectorSearch pipeline and a $search (BM25) pipeline; the two component times are not separable from this client observation",
        }[mode],
        "records": records,
    }
    if hrpoc:
        for key in ("doc_hit_at_1", "doc_hit_at_5", "doc_hit_at_10", "passage_hit_at_5", "passage_hit_at_10"):
            result["metrics"][key] = mean_of(scored, key)
        result["metrics"]["passage_queries"] = sum(1 for record in scored if "passage_hit_at_5" in record["metrics"])
        result["hrpoc"] = hrpoc_summary(scored, [record for record in complete if record["metrics"] is None], mode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="scifact", help="which prepared dataset and collection to run against")
    parser.add_argument("--profile", choices=("smoke", "standard", "rerank", "native-rerank", "hybrid"), default="smoke")
    parser.add_argument("--max-queries", type=int, help="lower the profile query count for a diagnostic run")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("MONGODB_URI"):
        raise SystemExit("Set MONGODB_URI in .env or the shell.")
    dataset = DATASETS[args.dataset]
    manifest_path, queries_path = dataset["prepared"] / "manifest.json", dataset["prepared"] / "queries.json"
    if not manifest_path.exists() or not queries_path.exists():
        raise SystemExit(f"Run {dataset['prepare']} first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("embeddings", {}).get("status") != "generated":
        raise SystemExit(f"Embeddings are not generated. Run scripts/embed_voyage.py --dataset {args.dataset} first.")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    variants, mode = select_variants(args.profile)
    if args.max_queries:
        queries = queries[: args.max_queries]
    if args.profile == "smoke":
        queries = queries[:10]
    database = os.getenv("BENCHMARK_DB", "atlas_voyage_v4_benchmark")
    collection_name = os.getenv(dataset["collection_env"], dataset["collection"])
    collection = MongoClient(os.environ["MONGODB_URI"], appname="atlas-voyage-v4-benchmark")[database][collection_name]
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "dataset_name": args.dataset,
        "collection": collection_name,
        "concurrency": 1,
        "dataset": manifest,
        "query_count": len(queries),
        "variants": [],
    }
    if variants and queries:
        warm_up_pipeline = retrieval_pipeline(variants[0], queries[0], dataset=dataset)
        try:
            invoke(collection, warm_up_pipeline)
            print("Warm-up query completed (not included in recorded metrics).")
        except Exception as error:
            print(f"Warm-up query failed, continuing without it: {error}")
    for variant in variants:
        print(f"Running {variant['name']} against {len(queries)} labelled queries")
        output["variants"].append(run_variant(collection, variant, queries, mode, dataset))
    RESULTS.mkdir(exist_ok=True)
    destination = RESULTS / f"{dataset['prefix']}{args.profile}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if dataset["name"] == "hrpoc":
        # Per-query records hold 50 sections each; keep them compact, and write a records-free summary the dashboard reads.
        destination.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
        summary = {**output, "variants": [{key: value for key, value in item.items() if key != "records"} for item in output["variants"]]}
        summary_destination = RESULTS / f"hrpoc-summary-{args.profile}-{destination.stem.rsplit('-', 1)[1]}.json"
        summary_destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {summary_destination}")
    else:
        destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
