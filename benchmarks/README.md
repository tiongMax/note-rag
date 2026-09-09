# Building a benchmark

The authoritative benchmark is JSONL containing human-written questions,
reference answers, and evidence quotes. Evidence is stored as stable source
text rather than only as chunk IDs, because chunk IDs change when chunking is
modified.

## Recommended first benchmark

Build 100–150 cases and keep a held-out test split:

| Category | Count | Purpose |
|---|---:|---|
| Single-document factual | 40 | Basic retrieval |
| Paraphrased or indirect | 20 | Semantic retrieval |
| Multi-document or multi-hop | 20 | Evidence coverage |
| Unanswerable | 20 | Refusal behavior |
| Ambiguous or filter-sensitive | 10 | Metadata filtering |

Use 70–100 cases for development, 30–50 held out for final comparisons, and
an optional 10-case smoke set for CI.

## Labelling rules

1. Write questions from the frozen evaluation corpus, ideally based on real
   user needs.
2. Find the answer manually rather than using the retriever being tested.
3. Copy the smallest exact source quote that supports each required fact.
4. Give independently required facts distinct `group` values. Retrieval recall
   is measured over these groups, so overlapping chunks do not inflate recall.
5. Grade relevance as `3` for direct evidence, `2` for necessary supporting
   evidence, and `1` for useful background.
6. Add explicit `chunk_ids` only when needed. Exact evidence quotes remain the
   authoritative labels.
7. Have a second annotator review at least 20–30% of the test cases.

An answerable row requires `reference_answer` and at least one evidence span.
An unanswerable row must have no evidence and should set
`expected_behavior` to `refuse`. See `example.jsonl` for both forms.

For a multi-hop question, use multiple evidence groups:

```json
{
  "id": "multi-hop-001",
  "question": "Which release introduced feature X, and who approved it?",
  "answerable": true,
  "reference_answer": "Release 2.0 introduced X, and Alex approved it.",
  "evidence": [
    {
      "document": "releases.md",
      "quote": "Feature X was introduced in release 2.0.",
      "group": "release",
      "relevance": 3
    },
    {
      "document": "approvals.md",
      "quote": "Alex approved the feature X launch.",
      "group": "approver",
      "relevance": 3
    }
  ],
  "tags": ["multi-hop"]
}
```
