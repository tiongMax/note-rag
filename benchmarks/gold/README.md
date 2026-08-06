# Gold retrieval dataset

This directory contains the first human-authored retrieval benchmark for the
local operating-systems lecture corpus.

## Files

- `annotations.jsonl` contains questions, reference answers, PDF page numbers,
  and text markers that a reviewer can edit.
- `dataset.jsonl` is generated from the annotations. Its source character
  offsets match `note_rag.ingest.PdfParser` output.
- `manifest.json` pins source-file, parsed-text, and generated-dataset hashes.

The source PDFs live in the repository's ignored `documents/` directory. They
are not copied into the benchmark.

## Build and validate

From the repository root:

```powershell
.venv\Scripts\python.exe scripts\build_gold_dataset.py
.venv\Scripts\python.exe scripts\validate_gold_dataset.py
```

Rebuild after changing an annotation. Validation should fail if a source PDF,
parser result, passage offset, page number, or generated dataset changes
without rebuilding the manifest.

## Labelling rules

1. Questions must be answerable from the labelled passage alone.
2. Answers should paraphrase the source rather than depend on exact wording.
3. Passage labels use original-document coordinates, never generated chunk IDs.
4. Passages should include the smallest coherent section that supports the
   complete answer.
5. Add a second passage only when the question genuinely requires both.
6. Keep retrieval questions factual; save subjective generation scoring for a
   separate dataset.

## Using labels with chunking strategies

A retrieved chunk overlaps a gold passage by:

```text
overlap = max(0, min(chunk.char_end, passage.char_end)
                 - max(chunk.char_start, passage.char_start))
passage_coverage = overlap / (passage.char_end - passage.char_start)
```

Keep the continuous coverage score for NDCG and passage-coverage metrics. A
reasonable initial binary relevance threshold is `passage_coverage >= 0.5`.
Boundary completeness is true when one retrieved chunk fully contains a gold
passage.

## Run the retrieval benchmark

The runner calls the HTTP search API, restricts every query to the six
manifested filenames, checks that those documents are ready and indexed, and
writes detailed JSONL plus JSON and CSV summaries.

Start one API against the fixed-window index, then run:

```powershell
.venv\Scripts\python.exe scripts\run_retrieval_benchmark.py `
  --label fixed `
  --base-url http://127.0.0.1:8001
```

Start the isolated recursive benchmark API and database, then run:

```powershell
docker compose --profile benchmark up -d --build app-recursive

.venv\Scripts\python.exe scripts\run_retrieval_benchmark.py `
  --label recursive `
  --base-url http://127.0.0.1:8002
```

Results default to `tmp/benchmarks/`. Set `API_AUTH_TOKEN` when the API uses
authentication; the token is read from the environment and is never written
to result artifacts.

The default run makes one request for each of the 48 questions. Use
`--repetitions 3` for a larger latency sample only after raising the API rate
limit or setting an appropriate `--delay-ms`.

## Retrieval and reranking matrix

Run every system against the same API, corpus, embeddings, dataset hash, and
`top_k`. BM25 is deliberately benchmark-only: it builds one in-process Okapi
BM25 index from the API's stored chunks. A gain here justifies evaluating a
production BM25 backend; it does not make the in-process index production-ready.

```powershell
$python = ".venv\Scripts\python.exe"
$runner = "scripts\run_retrieval_benchmark.py"

& $python $runner --label dense --system vector
& $python $runner --label postgres-fts --system keyword
& $python $runner --label current-hybrid --system current_hybrid
& $python $runner --label bm25 --system bm25
& $python $runner --label bm25-dense --system bm25_dense

& $python $runner --label current-hybrid-lexical `
  --system current_hybrid --reranker lexical --candidate-k 50
& $python $runner --label bm25-dense-lexical `
  --system bm25_dense --reranker lexical --candidate-k 50
```

Install the optional local cross-encoder once:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[reranking]"
```

Then run both cross-encoded hybrids:

```powershell
& $python $runner --label current-hybrid-cross-encoder `
  --system current_hybrid --reranker cross_encoder --candidate-k 50
& $python $runner --label bm25-dense-cross-encoder `
  --system bm25_dense --reranker cross_encoder --candidate-k 50
```

The runner records total, retrieval-only, and reranking-only mean/p50/p95
latencies, throughput, BM25 build time, Recall@5/10/20, MRR, NDCG@10,
Precision@5, and mean irrelevant top-five results. Use at least three
repetitions for final latency reporting and otherwise idle hardware.

The application can also use the local cross-encoder for normal context
building:

```dotenv
RERANKER_BACKEND=cross_encoder
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
CROSS_ENCODER_DEVICE=cpu
CROSS_ENCODER_BATCH_SIZE=16
```

Compare the controlled summaries:

```powershell
.venv\Scripts\python.exe scripts\compare_retrieval_benchmarks.py `
  tmp\benchmarks\fixed.summary.json `
  tmp\benchmarks\recursive.summary.json `
  --output-prefix tmp\benchmarks\fixed-vs-recursive
```

The comparison refuses to run when dataset hash, embedding model, retrieval
mode, `top_k`, vector weight, relevance threshold, or repetition count differ.

## Metric interpretation

- Recall at K is the macro-average proportion of labelled passages for which
  one result covers at least 50% of the passage.
- Precision at 5 is the number of top-five chunks crossing that threshold,
  divided by five.
- Irrelevant at 5 is `5 - relevant top-five chunks`; its relative reduction
  versus dense retrieval is the résumé bullet's `Y%`.
- MRR uses the rank of the first chunk crossing the threshold.
- NDCG at 10 uses character-coverage proportion as graded relevance and
  measures the ordering of the returned top-ten set.
- The relative NDCG@10 change versus dense retrieval is the résumé bullet's
  `X%`; the comparison script calculates it directly.
- Passage coverage at K uses the union of all retrieved character ranges, so
  adjacent chunks can jointly cover one passage.
- Boundary completeness at K requires one chunk to contain the entire labelled
  passage.
- Embedding budget tokens sum the application's deterministic token counts for
  every indexed chunk, including overlap. They are a reproducible cost proxy,
  not provider-reported billed tokens.
