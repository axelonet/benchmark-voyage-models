#!/usr/bin/env python3
"""Persist Atlas Admin API Search-index storage telemetry for one completed run.

Requires an authenticated Atlas CLI session with Project Read Only or higher.
The script stores raw API responses, including an empty data-point series when
the newly created Search deployment has not emitted metrics yet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_dotenv() -> None:
    path = ROOT / ".env"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def atlas_json(arguments: list[str]) -> dict[str, Any]:
    command = ["atlas", "api", *arguments, "--output", "json"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def latest_index_names() -> list[str]:
    snapshots = sorted(RESULTS.glob("index-snapshot-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not snapshots:
        raise SystemExit("Run scripts/collect_index_snapshot.py before collecting Admin API metrics.")
    snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
    return [item["name"] for item in snapshot.get("search_indexes", [])]


def data_point_count(metric_response: dict[str, Any]) -> int:
    return sum(len(series.get("dataPoints", [])) for series in metric_response.get("indexStatsMeasurements", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--cluster-name", required=True)
    parser.add_argument("--process-id", required=True)
    parser.add_argument("--period", default="P1D")
    parser.add_argument("--granularity", default="PT5M")
    args = parser.parse_args()
    load_dotenv()
    database = os.getenv("BENCHMARK_DB", "atlas_voyage_v4_benchmark")
    collection = os.getenv("BENCHMARK_COLLECTION", "chunks")
    common = ["--groupId", args.project_id, "--processId", args.process_id, "--period", args.period, "--granularity", args.granularity]
    deployment = atlas_json(["atlasSearch", "getAtlasSearchDeployment", "--groupId", args.project_id, "--clusterName", args.cluster_name])
    search_node_metrics = atlas_json(["monitoringAndLogs", "getMeasurements", *common, "--metrics", "FTS_DISK_USAGE"])
    index_metrics = {}
    for index_name in latest_index_names():
        index_metrics[index_name] = atlas_json(["monitoringAndLogs", "getIndexMetrics", *common, "--databaseName", database, "--collectionName", collection, "--indexName", index_name, "--metrics", "INDEX_SIZE_ON_DISK"])
    data_points = {name: data_point_count(response) for name, response in index_metrics.items()}
    artifact = {
        "artifact_type": "atlas-admin-api-search-metrics",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_context": {"project_id": args.project_id, "cluster_name": args.cluster_name, "process_id": args.process_id, "database": database, "collection": collection, "period": args.period, "granularity": args.granularity},
        "search_deployment": deployment,
        "search_node_disk_usage": search_node_metrics,
        "per_index_size_on_disk": index_metrics,
        "metric_data_point_count": data_points,
        "interpretation": "A zero data-point count means Atlas Admin API metrics have not yet been emitted for this new Search deployment/window. It is not a zero-byte index size.",
    }
    RESULTS.mkdir(exist_ok=True)
    destination = RESULTS / f"atlas-admin-search-metrics-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")
    print(json.dumps({"indexes": len(index_metrics), "index_metric_data_points": data_points, "search_node_disk_data_points": sum(len(series.get("dataPoints", [])) for series in search_node_metrics.get("hardwareMeasurements", []))}, indent=2))


if __name__ == "__main__":
    main()
