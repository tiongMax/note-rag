# Point 1: production retrieval experiment

This report verifies the first résumé claim against the committed 48-query gold
set (`869335a0ed4cb222743f5c7666c61772be8b3f651815c6b04a75bb605cf697f0`).
It compares fixed versus structure-aware recursive chunking and dense-only
versus request-path PostgreSQL BM25+dense retrieval in a controlled 2x2 design.

## Result

| Cell | Chunking | Retrieval | MRR | Recall@5 | NDCG@10 |
| --- | --- | --- | ---: | ---: | ---: |
| A | fixed | dense | 0.6539 | 0.8750 | 0.7605 |
| B | recursive | dense | 0.8142 | 0.9583 | 0.8778 |
| C | fixed | PostgreSQL BM25+dense RRF | 0.7787 | 0.9688 | 0.8569 |
| D | recursive | PostgreSQL BM25+dense RRF | 0.8406 | 0.9792 | 0.8967 |

The combined A-to-D effect is **+28.6% MRR**, **+11.9% Recall@5**, and
**+17.9% NDCG@10**. Paired 10,000-sample query bootstrap intervals are
11.5% to 50.4% for MRR, 0.0% to 27.0% for Recall@5, and 7.2% to 29.5% for
NDCG@10. All four cells produced stable rankings across three repetitions.

The original résumé value of +22.3% MRR is not reproduced by this production
request-path implementation. Use +28.6% for this experiment, or retain 22.3%
only with the older run and exact methodology that produced it. Recall@5
reproduces the stated +11.9%.

## Controls

- 48 human-labeled questions, six indexed lecture PDFs, Gemini embedding model
  `gemini-embedding-2`, top-k 20, vector weight 0.7, BM25 `k1=1.5`, `b=0.75`,
  RRF `k=60`, and passage relevance threshold 0.5.
- Each cell ran through the FastAPI request path for three repetitions. The
  runner checked `/health` and refused mislabeled chunking or lexical settings.
- Query embedding cache was warm and retrieval result cache was disabled. This
  held vectors constant and avoided external quota/latency while still running
  vector search, persisted PostgreSQL BM25 scoring, and RRF for each request.
- The migration backfilled lexical statistics for 144/144 existing fixed-index
  chunks and 148/148 existing recursive-index chunks before the experiment.

The machine-readable report is in `point1-retrieval-report.json`; raw
per-query artifacts remain local under `tmp/point1-final` because they include
large retrieved-text payloads.
