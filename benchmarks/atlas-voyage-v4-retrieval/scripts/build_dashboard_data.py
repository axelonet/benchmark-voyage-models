#!/usr/bin/env python3
"""Publish the most recent local benchmark result in a compact browser-friendly form."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WEB = ROOT / "web"
DESTINATION = WEB / "results.json"
SCRIPT_DESTINATION = WEB / "results.js"


def find_latest(profile: str, skip_previous: bool) -> tuple[Path, Path | None] | None:
    matches = sorted(RESULTS.glob(f"{profile}-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        return None
    if skip_previous:
        return matches[0], None
    latest = json.loads(matches[0].read_text(encoding="utf-8"))
    previous = next(
        (
            candidate
            for candidate in matches[1:]
            if json.loads(candidate.read_text(encoding="utf-8")).get("query_count") == latest.get("query_count")
        ),
        None,
    )
    return matches[0], previous


def variant_row(run: dict[str, Any], item: dict[str, Any], prior: dict[str, Any], previous_run_at: str | None) -> dict[str, Any]:
    variant = item.get("variant", {})
    row = {
        "run_profile": run.get("profile"),
        "run_at": run.get("run_at"),
        "name": variant.get("name"),
        "model": variant.get("model"),
        "family": variant.get("family"),
        "dimensions": variant.get("dimensions"),
        "quantization": variant.get("quantization"),
        "num_candidates": variant.get("num_candidates"),
        "fetch_k": variant.get("fetch_k"),
        "num_docs_to_rerank": variant.get("rerank_k"),
        "hybrid_weights": variant.get("hybrid_weights"),
        "status": item.get("status"),
        "timing_scope": item.get("timing_scope"),
        "metrics": item.get("metrics", {}),
        "previous_metrics": prior.get("metrics"),
        "previous_run_at": previous_run_at if prior else None,
    }
    if item.get("hrpoc"):
        row["hrpoc"] = item["hrpoc"]
    return row


def hrpoc_section() -> dict[str, Any] | None:
    """The client HR-policy dataset, published as its own section; never merged into the SciFact rows.

    Reads the records-free hrpoc-summary-*.json files; the full per-query result files can be hundreds of MB.
    """
    runs = []
    for profile in ("standard", "rerank", "native-rerank", "hybrid"):
        found = find_latest(f"hrpoc-summary-{profile}", skip_previous=True)
        if found:
            runs.append(json.loads(found[0].read_text(encoding="utf-8")))
    if not runs:
        return None
    section: dict[str, Any] = {
        "run_at": max(run.get("run_at", "") for run in runs),
        "query_count": runs[0].get("query_count"),
        "dataset": runs[0].get("dataset", {}).get("selection", {}),
        "variants": [variant_row(run, item, {}, None) for run in runs for item in run.get("variants", [])],
    }
    snapshots = sorted(RESULTS.glob("hrpoc-index-snapshot-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if snapshots:
        snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
        section["index_snapshot"] = {
            "captured_at": snapshot.get("captured_at"),
            "collection_stats": snapshot.get("collection_stats"),
            "search_indexes": [
                {"name": item.get("name"), "status": item.get("status"), "queryable": item.get("queryable"), "vector_fields": item.get("vector_fields", [])}
                for item in snapshot.get("search_indexes", [])
            ],
            "search_node_index_bytes_note": snapshot.get("search_node_index_bytes_note"),
            "nominal_float32_payload_note": snapshot.get("nominal_float32_payload_note"),
        }
    isolation_runs = sorted(RESULTS.glob("hrpoc-permission-isolation-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if isolation_runs:
        isolation = json.loads(isolation_runs[0].read_text(encoding="utf-8"))
        section["permission_isolation"] = {key: isolation.get(key) for key in ("run_at", "query_count", "status", "queries_where_filter_excluded_documents", "violations")}
    return section


def main() -> None:
    selected: list[tuple[Path, Path | None]] = []
    for profile in ("standard", "rerank", "native-rerank"):
        found = find_latest(profile, skip_previous=profile == "native-rerank")
        if found:
            selected.append(found)
    hybrid_found = find_latest("hybrid", skip_previous=False)
    hrpoc = hrpoc_section()
    if not selected and not hybrid_found and not hrpoc:
        payload = {"status": "not-run", "message": "No benchmark run has been published yet.", "variants": []}
    else:
        runs = [(json.loads(path.read_text(encoding="utf-8")), previous) for path, previous in selected]
        all_runs = runs + ([(json.loads(hybrid_found[0].read_text(encoding="utf-8")), hybrid_found[1])] if hybrid_found else [])
        payload = {
            "status": "ready",
            "source_files": [path.name for path, _ in selected] + ([hybrid_found[0].name] if hybrid_found else []),
            "run_at": max((run.get("run_at", "") for run, _ in all_runs), default=hrpoc["run_at"] if hrpoc else ""),
            "query_count": all_runs[0][0].get("query_count") if all_runs else None,
            "dataset": all_runs[0][0].get("dataset", {}).get("selection", {}) if all_runs else {},
            "variants": [],
            "hybrid_variants": [],
        }
        if hrpoc:
            payload["hrpoc"] = hrpoc
        for run, previous_path in runs:
            previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path else {}
            previous_by_name = {item.get("variant", {}).get("name"): item for item in previous.get("variants", [])}
            for item in run.get("variants", []):
                prior = previous_by_name.get(item.get("variant", {}).get("name"), {})
                payload["variants"].append(variant_row(run, item, prior, previous.get("run_at")))
        if hybrid_found:
            hybrid_run, hybrid_previous_path = json.loads(hybrid_found[0].read_text(encoding="utf-8")), hybrid_found[1]
            hybrid_previous = json.loads(hybrid_previous_path.read_text(encoding="utf-8")) if hybrid_previous_path else {}
            hybrid_previous_by_name = {item.get("variant", {}).get("name"): item for item in hybrid_previous.get("variants", [])}
            for item in hybrid_run.get("variants", []):
                prior = hybrid_previous_by_name.get(item.get("variant", {}).get("name"), {})
                payload["hybrid_variants"].append(variant_row(hybrid_run, item, prior, hybrid_previous.get("run_at")))
        snapshots = sorted(RESULTS.glob("index-snapshot-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if snapshots:
            snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
            payload["index_snapshot"] = {
                "captured_at": snapshot.get("captured_at"),
                "collection_stats": snapshot.get("collection_stats"),
                "search_indexes": [
                    {"name": item.get("name"), "status": item.get("status"), "queryable": item.get("queryable"), "vector_fields": item.get("vector_fields", [])}
                    for item in snapshot.get("search_indexes", [])
                ],
                "search_node_index_bytes_note": snapshot.get("search_node_index_bytes_note"),
                "nominal_float32_payload_note": snapshot.get("nominal_float32_payload_note"),
            }
        admin_snapshots = sorted(RESULTS.glob("atlas-admin-search-metrics-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if admin_snapshots:
            admin = json.loads(admin_snapshots[0].read_text(encoding="utf-8"))
            payload["atlas_admin_metrics"] = {
                "captured_at": admin.get("captured_at"),
                "metric_data_point_count": admin.get("metric_data_point_count", {}),
                "interpretation": admin.get("interpretation"),
            }
        ui_snapshots = sorted(RESULTS.glob("atlas-ui-index-size-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if ui_snapshots:
            ui = json.loads(ui_snapshots[0].read_text(encoding="utf-8"))
            payload["atlas_ui_index_size"] = {
                "captured_on": ui.get("captured_on"),
                "source": ui.get("source"),
                "documents_indexed": ui.get("documents_indexed"),
                "indexes": ui.get("indexes", {}),
                "notes": ui.get("notes", []),
            }
        isolation_runs = sorted(RESULTS.glob("permission-isolation-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if isolation_runs:
            isolation = json.loads(isolation_runs[0].read_text(encoding="utf-8"))
            payload["permission_isolation"] = {
                "run_at": isolation.get("run_at"),
                "query_count": isolation.get("query_count"),
                "status": isolation.get("status"),
                "violations": isolation.get("violations", []),
            }
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    SCRIPT_DESTINATION.write_text("window.BENCHMARK_RESULTS = " + json.dumps(payload) + ";\n", encoding="utf-8")
    archive_destination = WEB / f"results-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    archive_destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DESTINATION}")
    print(f"Wrote {SCRIPT_DESTINATION}")
    print(f"Archived {archive_destination}")


if __name__ == "__main__":
    main()
