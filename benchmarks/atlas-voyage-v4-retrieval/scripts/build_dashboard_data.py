#!/usr/bin/env python3
"""Publish the most recent local benchmark result in a compact browser-friendly form."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DESTINATION = ROOT / "web" / "results.json"
SCRIPT_DESTINATION = ROOT / "web" / "results.js"


def main() -> None:
    selected: list[tuple[Path, Path | None]] = []
    for profile in ("standard", "rerank", "native-rerank"):
        matches = sorted(RESULTS.glob(f"{profile}-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if matches:
            latest = json.loads(matches[0].read_text(encoding="utf-8"))
            previous = next(
                (
                    candidate
                    for candidate in matches[1:]
                    if json.loads(candidate.read_text(encoding="utf-8")).get("query_count") == latest.get("query_count")
                ),
                None,
            ) if profile != "native-rerank" else None
            selected.append((matches[0], previous))
    if not selected:
        payload = {"status": "not-run", "message": "No benchmark run has been published yet.", "variants": []}
    else:
        runs = [(json.loads(path.read_text(encoding="utf-8")), previous) for path, previous in selected]
        primary = runs[0]
        payload = {
            "status": "ready",
            "source_files": [path.name for path, _ in selected],
            "run_at": max(run.get("run_at", "") for run, _ in runs),
            "query_count": primary[0].get("query_count"),
            "dataset": primary[0].get("dataset", {}).get("selection", {}),
            "variants": [],
        }
        for run, previous_path in runs:
            previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path else {}
            previous_by_name = {item.get("variant", {}).get("name"): item for item in previous.get("variants", [])}
            for item in run.get("variants", []):
                name = item.get("variant", {}).get("name")
                prior = previous_by_name.get(name, {})
                payload["variants"].append({
                    "run_profile": run.get("profile"),
                    "run_at": run.get("run_at"),
                    "name": name,
                    "family": item.get("variant", {}).get("family"),
                    "dimensions": item.get("variant", {}).get("dimensions"),
                    "quantization": item.get("variant", {}).get("quantization"),
                    "num_candidates": item.get("variant", {}).get("num_candidates"),
                    "status": item.get("status"),
                    "timing_scope": item.get("timing_scope"),
                    "metrics": item.get("metrics", {}),
                    "previous_metrics": prior.get("metrics"),
                    "previous_run_at": previous.get("run_at") if prior else None,
                })
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
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    SCRIPT_DESTINATION.write_text("window.BENCHMARK_RESULTS = " + json.dumps(payload) + ";\n", encoding="utf-8")
    print(f"Wrote {DESTINATION}")
    print(f"Wrote {SCRIPT_DESTINATION}")


if __name__ == "__main__":
    main()
