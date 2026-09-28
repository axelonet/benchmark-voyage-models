# Atlas + Voyage v4 retrieval benchmark

This is a bounded retrieval-quality and query-latency benchmark. It uses the public [BEIR SciFact](https://github.com/beir-cellar/beir) corpus and held-out relevance labels; it does not use synthetic vectors, production data, an Elasticsearch baseline, ACL behaviour, hybrid search, or client-load testing.

The benchmark answers narrowly useful questions:

- Which Voyage v4 embedding family gives the best retrieval quality for this labelled corpus?
- What quality/latency trade-off comes from 1024, 512, and 256 dimensions?
- Does `voyage-context-4` improve chunk retrieval when chunks are encoded in their source-document context?
- What is the observed impact of scalar quantization and `numCandidates` on Atlas Vector Search?
- How do a Voyage client reranker and the native Atlas <code>$rerank</code> stage compare on the same shortlist?

## Cost and safety bounds

The default **starter** dataset has 1,000 real source documents and 80 labelled test queries. It includes every positive document for those queries plus deterministic distractors. Source documents are chunked into short word-window passages, so the imported chunk count can be modestly higher than 1,000.

The default matrix has ten retrieval rows and runs serially. It records 80 query results per row. Embedding calls are batched, their API-reported token counts are written to the manifest, and no reranking request is made unless explicitly requested. The **full** dataset is 5,183 documents / 300 test queries and must be selected explicitly.

Use a dedicated Atlas database. `mongoimport --drop` below deletes only the configured benchmark collection.

## Measured environment

The completed starter run used an Atlas **M30 Gen 2** cluster on **AWS**, with **two dedicated S30 Search Nodes**. This configuration is user-provided and recorded in [`config/environment.json`](config/environment.json). It is part of the run context, not a universal sizing recommendation.

## What is compared

| Focus | Rows | Fixed conditions |
| --- | --- | --- |
| Model family | `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-context-4` at 1024d | cosine, ANN, `numCandidates=400`, fetch 50 chunks, evaluate top 10 documents |
| Matryoshka dimensions | `voyage-4` at 1024d, 512d, 256d | same retrieval conditions |
| Atlas scalar quantization | `voyage-4` 1024d float vs scalar-quantized index | same vectors and retrieval conditions |
| Candidate curve | `voyage-4` 1024d at `numCandidates=100/400/1000` | same index, fetch, and evaluation |
| Client reranking | selected `voyage-4` 1024d candidates through Voyage `rerank-3-lite` | separately timed from retrieval |
| Native pipeline reranking | selected candidates through Atlas `$vectorSearch → $rerank` using `rerank-2.5-lite` | one aggregation timer; component times cannot be separated |

`voyage-context-4` is contextualized over chunks belonging to the same SciFact source document. It is the current Voyage contextualized-chunk model. The other rows embed the same chunk text with the general embedding API. This is an implementation comparison, not a claim that one corpus predicts every workload.

## Setup

```bash
cd benchmarks/atlas-voyage-v4-retrieval
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put the Atlas URI and Voyage key in `.env`. Confirm `BENCHMARK_DB` is a new, dedicated database before importing.
Quoted values are supported, which is useful when a URI contains shell-special characters.

## Use the published starter snapshot

The repository includes a compressed, three-part snapshot of the completed starter corpus: 2,047 SciFact chunks, 80 labelled queries, and six generated Voyage v4 representations. Restore it before importing to avoid downloading the corpus or making new embedding API calls:

```bash
# Reassembles the three parts, verifies SHA-256, and restores data/prepared/.
python3 scripts/restore_starter_snapshot.py

# Confirms the corpus, labels, and vector dimensions before importing.
.venv/bin/python scripts/validate_artifacts.py
```

The snapshot is fixed to the manifest's 2026-09-21 starter selection and model specifications. It does not include Atlas indexes, Atlas data, query results, credentials, or any reranking output. A Voyage API key is unnecessary unless you run client reranking or generate a fresh embedding set.

## Regenerate the starter data instead

Use this path only when you need a fresh corpus preparation or embeddings. It downloads the source data and the embedding command makes billable Voyage API calls.

```bash
# Downloads and prepares 1,000 source documents and 80 labelled test queries.
.venv/bin/python scripts/prepare_scifact.py --profile starter

# Calls Voyage once per selected representation and records usage tokens.
.venv/bin/python scripts/embed_voyage.py
```

## Load, index, and run

```bash

# Validate the prepared corpus and embeddings before touching Atlas.
.venv/bin/python scripts/validate_artifacts.py

# Imports only the dedicated collection named in .env.
set -a; . ./.env; set +a
mongoimport --uri "$MONGODB_URI" --db "$BENCHMARK_DB" --collection "$BENCHMARK_COLLECTION" \
  --file data/prepared/chunks.ndjson --type json --drop

# Create seven search indexes and wait for readiness.
.venv/bin/python scripts/create_indexes.py --wait

# Ten retrieval rows × 80 held-out queries, serially.
.venv/bin/python scripts/run_benchmark.py --profile standard

# Optional: two reranking rows. This creates additional Voyage API calls.
.venv/bin/python scripts/run_benchmark.py --profile rerank

# Two matching Atlas-native rerank rows. Requires Native Reranking enabled in
# Atlas Project Settings and MongoDB 8.3+; this benchmark recorded 9.0.2.
.venv/bin/python scripts/run_benchmark.py --profile native-rerank

.venv/bin/python scripts/build_dashboard_data.py
python3 -m http.server --directory web 8000
# Then open http://localhost:8000 in a browser.
```

The early smoke option uses ten labelled queries. The full corpus requires an explicit rebuild:

```bash
.venv/bin/python scripts/prepare_scifact.py --profile full
```

Re-run embedding, validation, import, index creation, and the benchmark after changing profiles. Never combine artifacts across profiles.

## Evidence captured

Each run retains query-level rankings and these aggregate values:

- Recall@10, MRR@10, nDCG@10, and Hit@10 against the SciFact relevance labels
- Atlas client-observed retrieval P50/P95, separately timed Voyage client-rerank P50, and native `$vectorSearch + $rerank` pipeline P50/P95 when selected
- ANN-to-ENN overlap for the candidate-curve rows
- run configuration: model, dimensions, index, scalar quantization, fetch size, `numCandidates`, query count, timestamps, source checksum, and embedding API token usage

The timing is client-observed one-user query latency from the benchmark process. The native rerank timer covers the entire aggregation request and does not split Vector Search from reranking. It is not a capacity, concurrency, or end-to-end application-latency claim.

## Vector Search index evidence

After index creation and each benchmark run, capture a read-only index snapshot:

```bash
.venv/bin/python scripts/collect_index_snapshot.py
```

The snapshot records each Vector Search index definition, dimensions, quantization regime, readiness, collection document count, and collection logical/physical storage. It also calculates a nominal `dimensions × 4 × document count` comparison aid for each index.

Do **not** use `collStats.totalIndexSize` as Atlas Vector Search index storage; it covers ordinary MongoDB indexes such as `_id`. Dedicated Search-node index bytes must be collected separately from Atlas Search-node metrics or the Atlas Administration API with monitoring credentials. Record the metric name, observation window, and index configuration together with that value.

With an authenticated Atlas CLI session, store the Admin API evidence for this completed run:

```bash
.venv/bin/python scripts/collect_atlas_admin_metrics.py \
  --project-id <Atlas-project-id> \
  --cluster-name <cluster-name> \
  --process-id <search-metrics-process-id>
```

The artifact contains `INDEX_SIZE_ON_DISK` for each Vector Search index and `FTS_DISK_USAGE` for the Search deployment when Atlas has emitted data points. An empty series means the new deployment has not emitted telemetry yet; it must remain `unknown`, not be recorded as zero.

For this completed starter run, the Atlas UI index-size screenshot is preserved as `results/atlas-ui-index-size-20260921.json`. It records the displayed Total Size and Recommended Memory for every index at 2,047 indexed documents.

## Sources and version assumptions

- Dataset: [BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip), including `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`.
- Voyage models and dimensions: [Voyage text embeddings](https://docs.voyageai.com/docs/embeddings) and [contextualized chunk embeddings](https://docs.voyageai.com/docs/contextualized-chunk-embeddings).
- Atlas Vector Search configuration and query support must be validated on the target cluster. The index creation script preserves server errors rather than substituting assumptions.
