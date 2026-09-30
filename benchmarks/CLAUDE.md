# CLAUDE.md

Guidance for any AI coding agent working in this repository.

## 1. Project overview

This repo hosts benchmark suites that measure MongoDB Atlas Vector Search performance against various embedding models. Today it contains exactly one benchmark:

- `benchmarks/atlas-voyage-v4-retrieval/` — retrieval quality and query latency of Atlas Vector Search with Voyage v4 embeddings, against the BEIR SciFact corpus.

There is no root-level README by design. This file lives at `benchmarks/CLAUDE.md` and governs everything under `benchmarks/` — i.e. every benchmark in this repo. For what a specific benchmark measures, how to set it up, and how to run it end-to-end, read that benchmark's own README (e.g. [`atlas-voyage-v4-retrieval/README.md`](atlas-voyage-v4-retrieval/README.md)). This file does not repeat that content — it covers conventions and policy that apply across all benchmarks and to any agent editing code here.

If a second benchmark is added under `benchmarks/`, it should follow the same split: a benchmark-local README for setup/run/methodology, and this file for repo-wide conventions. Don't assume that split for a new benchmark without confirming it first.

## 2. Repository layout

```
.
├── .gitignore
├── LICENSE
└── benchmarks/
    ├── CLAUDE.md
    └── atlas-voyage-v4-retrieval/
        ├── .env             (local only, not committed)
        ├── .env.example     (committed template)
        ├── README.md
        ├── requirements.txt
        ├── config/          (environment.json, variants.json)
        ├── data/
        │   ├── source/      (gitignored — downloaded corpus)
        │   ├── prepared/    (gitignored — generated chunks/embeddings)
        │   └── snapshots/   (checked in — starter embedding tar parts)
        ├── results/         (gitignored except .gitkeep — generated run artifacts)
        ├── scripts/         (9 standalone Python scripts)
        └── web/             (static results dashboard)
```

There is no shared Python package. Every script under `scripts/` is standalone — each one re-implements its own small helpers (env loading, hashing, JSONL I/O) rather than importing from a sibling script or a shared module. See §4 before "fixing" this.

## 3. Coding conventions (house style)

Follow the existing style exactly when adding or editing scripts:

- `#!/usr/bin/env python3` shebang, followed by a single-line module docstring stating the script's purpose. That docstring is reused as `argparse(description=__doc__)`.
- `from __future__ import annotations` as the first import.
- Import order: stdlib first, then a blank line, then third-party (`pymongo`, `voyageai`). No local/package imports — scripts stay flat and standalone.
- `ROOT = Path(__file__).resolve().parents[1]` for locating the benchmark directory; derive other path constants (`PREPARED`, `RESULTS`, etc.) from `ROOT`.
- `argparse` flags follow existing naming: `--profile`, `--force`, `--wait`, `--replace`. Any flag that repeats billable API work or is destructive (drops/replaces data) must require an explicit opt-in flag, with a `help=` string that says so.
- Any script that talks to MongoDB or Voyage guards required environment variables first: `if not os.getenv("X"): raise SystemExit("Set X in .env or the shell.")`, before doing any other work.
- Full type hints on function signatures (`list[dict[str, Any]]`, `-> None`, etc.). Plain dicts are the data structure of choice — no dataclasses or pydantic models.
- snake_case for functions and variables, UPPER_CASE for module-level constants.
- Minimal comments — only where something is genuinely non-obvious (a workaround, a subtle invariant). Don't add comments that restate the code.
- Docstrings and any documentation prose favor precise, falsifiable claims ("this is not X, it is Y", explicit about what's excluded) over marketing language. Match this tone in any new documentation.

## 4. Explicit anti-patterns — don't do these

- **Don't add a sixth copy of `load_dotenv()`.** It's already duplicated near-identically across five scripts (`collect_atlas_admin_metrics.py`, `collect_index_snapshot.py`, `create_indexes.py`, `embed_voyage.py`, `run_benchmark.py`), and the copies have already drifted (some guard for a missing `.env` file, some don't). If you touch env-loading logic, fix every copy consistently in the same change and call out the duplication to the human reviewer — don't unilaterally introduce a shared module as a side effect of an unrelated change.
- **Don't introduce a shared `lib/`/common package casually.** The flat, standalone-script layout is an intentional (if debatable) existing pattern. Changing it is an architecture decision that needs sign-off, not a drive-by refactor bundled into another change.
- **Don't add a test framework speculatively.** There is none today — see §7 for what "verified" means in this repo.
- **Don't add CI, linter, or formatter config unless asked.** None of that exists currently by choice/omission; don't assume it should.

## 5. Safety and cost-bound rules

These apply to every benchmark in this repo, not just the current one:

- **Database blast radius**: `MONGODB_URI` / the configured benchmark database must always point at a dedicated, disposable Atlas database — never an application or production database. Any `--drop`-equivalent operation must only ever target the single configured benchmark collection. Verify this invariant before running or modifying import/index code.
- **Credentials**: never print, log, or echo API keys or Mongo credentials — including in error messages, debug output, or committed artifacts.
- **Billable API calls**: embedding generation and reranking cost real money. Scripts that do this are gated behind explicit flags (`--force`, specific `--profile` choices) — do not remove or bypass these guards, and do not casually re-run them while iterating or debugging.
- **Generated vs. committed artifacts**: directories like `data/prepared/` and files like `results/*.json` are gitignored, generated outputs. Never hand-edit them directly — regenerate via the documented scripts. Checked-in binary artifacts (e.g. `data/snapshots/*.tar.gz.part-*`) are fixed, versioned inputs — don't regenerate or replace them casually.

## 6. The cross-file sync sharp edge

There is **no single source of truth** for the retrieval-variant schema in `benchmarks/atlas-voyage-v4-retrieval/`. The following must be kept in lockstep by hand whenever a model, dimension, or index variant is added, renamed, or removed:

- `config/variants.json`
- `scripts/create_indexes.py`'s hardcoded `INDEXES` list
- `scripts/embed_voyage.py`'s hardcoded `SPECS` list
- `scripts/validate_artifacts.py`'s hardcoded `SPECS` dict — a separate `field_name -> dimension` mapping used only to validate chunk/query vector shapes; it must gain the same new entry whenever `embed_voyage.py`'s `SPECS` gains one, even though the two are never read from the same place.
- `scripts/run_benchmark.py`'s `select_variants()` — its hardcoded `rerank_variant_names` set must list the exact same variant names present in `config/variants.json` for any variant meant to run under the `rerank`/`native-rerank` profiles. Unlike the schema files above, this one isn't about model/dimension/index shape — it's a name-based selection list, so it also breaks on a plain **rename** of an existing variant, not just on adding or removing one. Its sibling `rerank_only_families` set (used to exclude rerank-only families from the `standard` profile) has the same failure mode in reverse: a new family meant only for reranking that isn't added there will leak into `standard` as a duplicate.

Before considering any change to the retrieval pipeline complete, explicitly diff these five for field-name, count, dimension, and name consistency (e.g. embedding field names like `embedding_voyage_4_1024` must match exactly across both `SPECS` lists and `INDEXES`; variant names referenced in `rerank_variant_names` must exist verbatim in `config/variants.json`). Do not assume changing one file is sufficient.

This has caused real bugs before, and they "looked right" until traced through carefully:
- `embed_voyage.py`'s contextualized query-embedding call was missing required list nesting (each query needs to be its own single-item group, not a flat list) — an easy thing to miss by analogy with the document path a few lines above.
- `create_indexes.py`'s `--wait` polling loop doesn't detect a `FAILED` index status — it only checks for `READY`, so a bad index definition spins for the full timeout instead of surfacing the real cause.
- Renaming a variant in `config/variants.json` (e.g. `voyage-4-1024-rerank30` → `voyage-4-1024-rerank20` after retuning its `rerank_k`) silently breaks its inclusion in `select_variants()`'s `rerank_variant_names` set unless the name is updated there too — the variant just stops appearing in `rerank`/`native-rerank` runs, with no error.
- Adding a new embedding representation (e.g. a new Matryoshka dimension the model actually supports) to `embed_voyage.py`'s `SPECS` doesn't automatically validate anything about it — `validate_artifacts.py` has its own independent `SPECS` dict that has to be updated separately, or a genuinely missing/malformed vector for the new field would pass validation silently. Confirm the dimension is actually supported by the model before adding it anywhere — Voyage does not support every dimension (384 was tried and reverted this session for exactly this reason).

## 7. Verification / what "done" means

There is no test suite or CI in this repo. Define "done" for a change as follows:

1. After any change touching data preparation, embeddings, or config schemas, run `.venv/bin/python scripts/validate_artifacts.py`.
2. For any change to index/embedding schema, manually re-check the §6 three-file sync.
3. Prefer `restore_starter_snapshot.py` + `validate_artifacts.py` as the cheap, no-API-call path to sanity-check most changes without spending money.
4. Never consider a change complete solely because it "ran without throwing." Check for silently wrong shapes or states — e.g. an index stuck in `PENDING` or `FAILED` that a naive wait-loop wouldn't surface, or a response shape that satisfies a count check but not a content check.
5. For documentation-only changes, verify that internal links resolve and that any command or flag mentioned actually matches what's currently in the scripts.

## 8. Git conventions

- Imperative, capitalized subject lines (e.g. "Add reusable starter embedding snapshot"), no conventional-commits prefix (`feat:`, `fix:`) — this matches the existing commit history.
- Keep commits scoped and atomic.
- There is no CONTRIBUTING.md or PR template. Don't add one unprompted.

## 9. Open questions — flag these rather than resolving them silently

- Whether a new benchmark under `benchmarks/` should follow the exact same README/CLAUDE.md split used here.
- Any request to modify `LICENSE` or authorship/copyright information — treat as a human decision.
