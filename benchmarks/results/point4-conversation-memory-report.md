# Point 4: budgeted conversation memory

This report verifies the conversation-memory implementation at clean commit
`4b00be644ddd24cd47381b6b633f23b82b8a56c3`.

## Result

| Policy | Fact-retention accuracy | Mean history tokens |
| --- | ---: | ---: |
| Full raw conversation baseline | 100.00% | 222.08 |
| Previous recent-only policy | 0.00% | not used for the reduction claim |
| Production memory policy | 100.00% | 129.94 |

The production policy reduced mean assembled conversation-history tokens by
**41.49%** versus full raw history and matched the baseline fact-retention
accuracy exactly: **0.0 percentage-point difference** across 50 cases.

This is deterministic subject-and-exact-fact retention in the assembled prompt,
not model-generated answer accuracy. The original resume phrase “multi-turn
accuracy” is too broad unless a separate model-answer evaluation is run.

## What is implemented

- Every successfully validated user/assistant pair is appended atomically under
  a PostgreSQL conversation row lock, preventing interleaved pair positions.
- Each completed turn receives a deterministic extractive summary capped at 64
  application tokens. Prior citation numbers are removed from summaries.
- The configured embedding provider embeds summaries; invalid or unavailable
  embeddings degrade to persisted summaries plus recent history without being
  labeled as semantic recall.
- Four newest complete turns remain verbatim. Up to two older compatible
  memories are selected by query-embedding cosine similarity at a 0.25 minimum.
- Selected memory and recent history share the 2,000-token history cap and the
  global prompt budget with fixed prompt text, retrieved context, and output
  reserve.
- Bounded conversation cues augment retrieval for follow-ups while always
  preserving the complete current question.
- Chat responses and SSE metadata expose selected message/memory counts, token
  usage, embedding model, and contextualized-query token count.
- Alembic migration `20260813_0009` persists summaries, embedding provenance,
  and vectors with all-or-none embedding-field constraints.

## Experiment contract

The 50 synthetic cases each contain one explicitly stated target fact followed
by 15 unrelated completed turns and a paraphrased follow-up. A case is correct
only when the assembled history retains both the case subject and exact expected
answer. The previous recent-only result executes the exact pre-Point-4
20-message/2,000-token selection loop.

The production cell used a locally cached 384-dimensional
`sentence-transformers/all-MiniLM-L6-v2` snapshot
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; no external calls or user-data
transfer occurred.

The corpus was inspected while configuring Point 4, is not held out, and has no
independent adjudication. It is a targeted regression experiment, not a general
multi-turn conversational-quality benchmark. Long-history token savings also do
not imply the same reduction for short conversations.

## Reproduction and verification

```powershell
$modelPath = "<local all-MiniLM-L6-v2 snapshot>"
.venv\Scripts\python.exe scripts\run_conversation_memory_benchmark.py `
  --label point4-memory-final `
  --output-dir tmp\point4-memory-final `
  --model-path $modelPath
```

Verification gates passed: 248 local tests plus 6 database-gated skips, all 6
live PostgreSQL integration tests, five consecutive worker-concurrency tests,
an empty-database migration to head, Point-4 downgrade/upgrade, Ruff, Pyright
with zero errors, frontend build, and Compose validation.

Raw artifacts remain ignored under `tmp/point4-memory-final`; their hashes are
pinned in `point4-conversation-memory-report.json`.

## Resume wording

Use this evidence-bound version:

> Fixed lost long-history facts and prompt overflow with persisted extractive
> turn summaries, four verbatim recent turns, embedding-based older-memory
> recall, and token-budgeted prompt assembly; cut mean history tokens 41.5%
> while matching full-history fact retention on a 50-case synthetic regression
> set.

Do not claim “multi-turn answer accuracy within 2 percentage points” from this
experiment. That requires a separately labeled generated-answer benchmark.
