# Point 2: automated retrieval and answer evaluation

This report verifies the automated evaluation system on the complete
48-question operating-systems benchmark. Retrieval metrics come from the
controlled Point 1 production cell; answer generation and deterministic replay
were rerun from clean commit `ed10d0f82475da50689ef839a6e18a14f2c261e5`.

## Result

| Evaluation | Metric | Result |
| --- | --- | ---: |
| Retrieval | Recall@5 | 0.9792 |
| Retrieval | MRR | 0.8406 |
| Retrieval | NDCG@10 | 0.8967 |
| Deterministic proxy | Reference-token F1 | 0.4395 |
| Deterministic proxy | Answer support by context | 0.8379 |
| Deterministic proxy | Reference coverage by context | 0.7853 |
| Deterministic proxy | Citation presence | 0.9792 |
| Deterministic proxy | Valid-citation rate | 0.9688 |

The deterministic scores are inexpensive lexical/citation regression proxies.
They are not semantic judges and must not be called RAGAS metrics.

## Controls and provenance

- 48/48 version-controlled questions and 49 labelled passages across six PDFs.
- Collection made exactly one `/api/v1/chat` request per question and captured
  the context package used by that request; no independent retrieval request
  was substituted for generation context.
- The runner verified all six indexed source hashes, ready/indexed state,
  recursive chunking, PostgreSQL BM25, the lexical reranker, and
  `gemini-3.5-flash-lite` before collection.
- Corpus version remained `2`; all 48 answers used one model and have distinct
  SHA-256 hashes for their exact prompt inputs.
- Candidate-k 20, maximum eight chunks, 1,200 application-token context budget,
  vector weight 0.7, and rerank weight 0.7.
- Dataset normalized-text SHA-256 is
  `c4d2adc7bf7a80313e651c24f1b24aa2b9bb4218903ad7363b726b1cee03190d`;
  the checked-out file-byte SHA-256 is
  `869335a0ed4cb222743f5c7666c61772be8b3f651815c6b04a75bb605cf697f0`.
- Mean chat latency was 1,452.8 ms, p50 was 1,347.3 ms, and p95 was
  2,151.6 ms. These are generation-path timings, not evaluator overhead.

## RAGAS status

The code supports explicit `--evaluator ragas-google` judging for
faithfulness, answer relevancy, answer correctness, context precision, and
context recall. A genuine run was not executed because it requires separate
authorization to send the saved questions, answers, references, and retrieved
lecture excerpts to Google. Therefore this report does not verify a RAGAS
result or a résumé claim that RAGAS judging was completed.

Until that authorized 48-case judge run exists, use this truthful wording:

> Built an automated evaluation pipeline over 48 version-controlled queries,
> reporting Recall@5/10/20, MRR, NDCG@10, exact-context answer samples, and
> deterministic lexical/citation quality proxies with hash-bound replay.

Raw per-query payloads remain ignored under `tmp/point2-committed-final`.
Their hashes are pinned in `point2-evaluation-report.json`.
