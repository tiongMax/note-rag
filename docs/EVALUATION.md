# RAG evaluation

`note-rag-eval` evaluates retrieval, answer behavior, grounding proxies,
latency, and optional RAGAS judge metrics. It records every question-level
trace so aggregates are reproducible and failures can be inspected.

## Metrics

Deterministic metrics are always enabled:

- retrieval: hit rate, precision, evidence-group recall, complete-evidence
  rate, MRR, and graded NDCG at 1, 3, 5, and 10;
- answers: normalized exact match and token F1;
- grounding safeguards: citation presence and citation-ID validity;
- behavior: refusal accuracy and false-refusal rate;
- operations: retrieval, generation, total latency, context tokens, output
  tokens, and estimated cost when the provider exposes it.

`--ragas` additionally evaluates faithfulness, answer relevancy, reference-aware
context precision, and answer correctness. RAGAS is isolated behind
`RagasEvaluator`; it does not define the deterministic relevance labels.

Citation validity is intentionally computed from citation tokens in the raw
answer. This detects nonexistent citations that the application may omit when
it resolves citation metadata.

## Install

Core deterministic evaluation has no additional dependencies:

```powershell
pip install -e .
```

Install the optional RAGAS adapter:

```powershell
pip install -e ".[evaluation]"
```

## Run against the API

Start Note RAG with an indexed, frozen benchmark corpus, then run:

```powershell
note-rag-eval `
  --dataset benchmarks/test.jsonl `
  --experiment recursive-cross-encoder
```

Use the same benchmark, corpus, and held-out split for every experiment. Useful
retrieval controls are exposed directly:

```powershell
note-rag-eval `
  --dataset benchmarks/test.jsonl `
  --experiment hybrid-rerank-070 `
  --mode hybrid `
  --candidate-k 20 `
  --max-chunks 8 `
  --vector-weight 0.7 `
  --rerank `
  --rerank-weight 0.7
```

If API authentication is enabled, set `API_AUTH_TOKEN` or pass `--api-token`.
Set `NOTE_RAG_EVAL_BASE_URL` when the service is not at
`http://127.0.0.1:8001`.

## Enable RAGAS

Set `GEMINI_API_KEY` and run:

```powershell
note-rag-eval `
  --dataset benchmarks/test.jsonl `
  --experiment hybrid-ragas `
  --ragas `
  --judge-model gemini-3.5-flash `
  --judge-embedding-model gemini-embedding-001
```

The judge model, embedding model, benchmark hash, Git commit, retrieval
configuration, and timestamp are written to `config.json`. Judge calls use a
per-experiment disk cache.

Pass `--corpus-manifest path/to/manifest.json` to record a hash of the frozen
corpus manifest. For application cost estimates, pass both
`--input-cost-per-million` and `--output-cost-per-million`. Input tokens are an
approximation based on question and context tokens; provider-side system prompt
or protocol overhead is not exposed by the current API.

## Recalculate metrics offline

`per_question.jsonl` from an earlier run can be used as trace input. This
recalculates deterministic metrics or changes judge metrics without querying
the RAG application again:

```powershell
note-rag-eval `
  --dataset benchmarks/test.jsonl `
  --experiment offline-recheck `
  --trace-input reports/recursive-cross-encoder/per_question.jsonl
```

A no-network smoke example is included:

```powershell
note-rag-eval `
  --dataset benchmarks/example.jsonl `
  --experiment example-offline `
  --trace-input benchmarks/example-traces.jsonl
```

## Outputs

Each experiment creates:

```text
reports/<experiment>/
├── config.json
├── metrics.json
├── per_question.jsonl
├── results.csv
├── failures.jsonl
└── .ragas-cache/       # only with --ragas
```

`metrics.json` contains macro means plus p50/p95 latency. `results.csv` is
suitable for spreadsheet comparisons. `failures.jsonl` selects execution
errors, incomplete Recall@5, invalid citations, incorrect refusals, or RAGAS
faithfulness/correctness below 0.8.

For a résumé claim, compare two configurations on the exact same held-out test
set and report the question count. Do not use selected development examples as
the before/after result.
