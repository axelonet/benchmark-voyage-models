const fmt = (value, digits = 3) => value === null || value === undefined ? "—" : Number(value).toFixed(digits);
const delta = (current, previous, digits = 3) => current === null || current === undefined || previous === null || previous === undefined ? "—" : `${current - previous >= 0 ? "+" : ""}${(current - previous).toFixed(digits)}`;
const observedP50 = (metrics) => metrics.native_pipeline_p50_ms ?? (metrics.rerank_p50_ms !== null && metrics.rerank_p50_ms !== undefined ? metrics.total_p50_ms : metrics.retrieval_p50_ms);

async function render() {
  const summary = document.querySelector("#run-summary");
  const body = document.querySelector("#results-body");
  try {
    let data = window.BENCHMARK_RESULTS;
    if (!data) {
      const response = await fetch("results.json", { cache: "no-store" });
      data = await response.json();
    }
    if (data.status !== "ready") {
      summary.textContent = "No run published yet. Prepare the corpus, generate Voyage embeddings, import it into the dedicated Atlas database, and run the standard profile.";
      return;
    }
    summary.textContent = `${data.query_count} labelled queries · ${data.dataset.chunks} chunks · latest artifact ${new Date(data.run_at).toLocaleString()}`;
    for (const variant of data.variants) {
      const m = variant.metrics || {};
      const row = document.createElement("tr");
      row.innerHTML = `<td title="${variant.timing_scope || ""}">${variant.run_profile || "—"}</td><td><code>${variant.name || "—"}</code></td><td>${variant.family || "—"}</td><td>${variant.dimensions || "—"}</td><td>${variant.quantization || "—"}</td><td>${variant.num_candidates || "—"}</td><td>${fmt(m.recall_at_10)}</td><td>${fmt(m.mrr_at_10)}</td><td>${fmt(m.ndcg_at_10)}</td><td>${fmt(m.retrieval_p50_ms, 1)} / ${fmt(m.retrieval_p95_ms, 1)}</td><td>${fmt(m.native_pipeline_p50_ms, 1)} / ${fmt(m.native_pipeline_p95_ms, 1)}</td><td>${fmt(m.rerank_p50_ms, 1)}</td><td>${fmt(m.total_p50_ms, 1)}</td><td>${fmt(m.ann_enn_overlap_at_10)}</td>`;
      body.append(row);
    }
    const comparisonBody = document.querySelector("#comparison-body");
    const comparisons = data.variants.filter((variant) => variant.previous_metrics);
    if (!comparisons.length) {
      comparisonBody.innerHTML = '<tr><td colspan="7">No same-profile predecessor is available for the published rows.</td></tr>';
    } else {
      for (const variant of comparisons) {
        const current = variant.metrics || {};
        const previous = variant.previous_metrics || {};
        const row = document.createElement("tr");
        row.innerHTML = `<td>${variant.run_profile || "—"}</td><td><code>${variant.name || "—"}</code></td><td>${variant.previous_run_at ? new Date(variant.previous_run_at).toLocaleString() : "—"}</td><td>${delta(current.recall_at_10, previous.recall_at_10)}</td><td>${delta(current.mrr_at_10, previous.mrr_at_10)}</td><td>${delta(current.ndcg_at_10, previous.ndcg_at_10)}</td><td>${delta(observedP50(current), observedP50(previous), 1)}</td>`;
        comparisonBody.append(row);
      }
    }
    const snapshot = data.index_snapshot;
    const indexSummary = document.querySelector("#index-summary");
    const indexBody = document.querySelector("#index-body");
    const indexNote = document.querySelector("#index-note");
    if (!snapshot) {
      indexSummary.textContent = "No index snapshot has been published yet.";
    } else {
      const stats = snapshot.collection_stats || {};
      const uiSizes = data.atlas_ui_index_size;
      const uiMessage = uiSizes ? ` Atlas UI size snapshot: ${uiSizes.documents_indexed} indexed documents, captured ${uiSizes.captured_on}.` : "";
      indexSummary.textContent = `${stats.count || "—"} documents · ${(stats.size / 1024 / 1024).toFixed(1)} MiB logical collection data · ${(stats.storageSize / 1024 / 1024).toFixed(1)} MiB collection storage · captured ${new Date(snapshot.captured_at).toLocaleString()}.${uiMessage}`;
      for (const index of snapshot.search_indexes || []) {
        for (const field of index.vector_fields || []) {
          const ui = uiSizes?.indexes?.[index.name] || {};
          const row = document.createElement("tr");
          row.innerHTML = `<td><code>${index.name}</code></td><td>${index.status}${index.queryable ? " · queryable" : ""}</td><td><code>${field.path}</code></td><td>${field.dimensions}</td><td>${field.quantization}</td><td>${ui.total_size_mb ?? "—"} MB</td><td>${ui.recommended_memory_mb ?? "—"} MB</td><td>${(field.nominal_float32_payload_bytes / 1024 / 1024).toFixed(1)} MiB</td>`;
          indexBody.append(row);
        }
      }
      const admin = data.atlas_admin_metrics;
      const adminMessage = admin ? ` Atlas Admin API captured ${new Date(admin.captured_at).toLocaleString()}: ${Object.values(admin.metric_data_point_count || {}).reduce((total, count) => total + count, 0)} per-index size datapoints emitted. ${admin.interpretation}` : "";
      indexNote.textContent = `${snapshot.nominal_float32_payload_note} ${snapshot.search_node_index_bytes_note}${adminMessage}`;
    }
  } catch (error) {
    summary.textContent = `Could not load local results: ${error.message}`;
  }
}
render();
