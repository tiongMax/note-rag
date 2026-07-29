# Evaluation

The eval file must be a non-empty JSON array. Every object requires:

```json
[
  {
    "question": "What is the stated objective?",
    "answer": "The reference answer used for generation evaluation.",
    "relevant_chunk_ids": ["3d31d0c5-..."],
    "category": "single-chunk"
  }
]
```

`category` must be `single-chunk`, `multi-chunk`, or `out-of-corpus`. Chunk IDs
may be the `chunk_id` metadata field or stable PGVector document IDs.

## Retrieval benchmark

Run all three retrieval configurations:

```powershell
note-rag `
  --eval-set evaluations/datasets/l2_process_abstraction.json `
  --eval-output evaluations/results/retrieval_eval.csv
```

The benchmark reports precision@5, recall@5, F1@5, hit rate@5, mean reciprocal
rank (MRR), mean average precision@5 (MAP@5), and normalized discounted
cumulative gain@5 (nDCG@5).

Metrics are macro-averaged over in-corpus questions and sliced into
`single-chunk` and `multi-chunk` results. `out-of-corpus` examples are excluded
because retrieval precision and recall are undefined without relevant
documents. The end-to-end benchmark measures their refusal correctness.

The command writes:

- `retrieval_eval.csv`: aggregate metrics by retriever and dataset slice;
- `retrieval_eval_details.csv`: expected and returned IDs, ranks, and metrics.

## End-to-end benchmark

Add `--eval-generation` to evaluate the strategy selected by `--strategy`:

```powershell
note-rag `
  --eval-set evaluations/datasets/l2_process_abstraction.json `
  --eval-output evaluations/results/retrieval_eval.csv `
  --strategy hybrid+rerank `
  --eval-generation
```

This generates answers at temperature 0 and uses a structured Gemini judge for
answer correctness, faithfulness, answer relevance, context relevance, and
out-of-corpus refusal correctness.

It writes `retrieval_eval_generation_details.csv` and
`retrieval_eval_generation_summary.csv`. This mode makes two Gemini calls per
example, so it is opt-in. Calibrate LLM-judge scores against a small
human-reviewed sample before using them as a release gate.
