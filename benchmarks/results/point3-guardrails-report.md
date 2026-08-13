# Point 3: RAG guardrails

This report verifies the deterministic guardrail path at clean implementation
commit `e5ff3a4740ab5ea53bb98bf21ed8418d0854602d`.

## Result

| Check | Result |
| --- | ---: |
| Synthetic unsafe regression cases detected | 18/18 |
| Synthetic safe regression cases allowed | 17/18 |
| Overall accuracy | 97.22% |
| Precision | 94.74% |
| Recall | 100.00% |
| Normal-fixture median / p95 / p99 | 0.58 / 1.00 / 1.20 ms |
| Max-fixture median / p95 / p99 | 22.95 / 35.93 / 45.60 ms |

The only false positive was `guard-output-safe-003`, a safe non-disclosure
paraphrase rejected by the deliberately conservative lexical groundedness
threshold. The threshold was not weakened to optimize this inspected corpus.

## What is enforced

- NFKC/format-control normalization and direct input prompt-injection checks.
- Indirect-injection checks across retrieved text, filename, media type, and
  metadata before canonical context re-rendering and citation renumbering.
- Input, complete prompt, accumulated output, request-body, generic API-rate,
  and separate chat-rate budgets. Authentication precedes expensive chat quota.
- Output citation validation, prompt/secret leakage checks, and explicitly
  lexical support and relevance filters with fixed safe fallbacks.
- Full SSE buffering and validation before any generated text or source
  metadata is released.
- Successful user/assistant pair persistence only after validated generation,
  preventing failed or filtered generations from leaving orphan user turns.
- Strict production configuration parsing and a fail-closed production gate
  when guardrails are disabled without an explicit break-glass setting.

## Latency scope

The `<100 ms median overhead` statement is supported for the local,
in-process deterministic guardrail path only. The benchmark executes input,
context, and output stages together and excludes HTTP, database, retrieval,
model/provider, network, and persistence latency.

Each fixture used 1,000 warm-ups and 10,000 measured iterations with
`time.perf_counter_ns`. The maximum fixture contains 512 question tokens,
1,200 context-body tokens across eight chunks, and 1,024 answer tokens. Its
22.95 ms median is the conservative number to cite.

## Evaluation boundaries

The 36 cases are a versioned, manually authored synthetic regression corpus:
12 input, 12 context, and 12 output cases, balanced between safe and unsafe,
including 14 hard negatives. The corpus was inspected while implementing Point
3, is not held out, and has no independent adjudication. Its deterministic
keyword, regular-expression, citation, and token-overlap outcomes are
non-semantic proxies—not a production safety certification.

Remaining production limitations include a per-process in-memory rate limiter,
reverse-proxy dependence for streamed/chunked request-size enforcement,
response/metric rather than durable per-request guardrail audit traces, and no
semantic entailment judge. These controls are defense in depth, not a complete
security boundary.

## Reproduction and verification

Controlled benchmark:

```powershell
.venv\Scripts\python.exe scripts\run_guardrail_benchmark.py `
  --label point3-guardrails-final `
  --output-dir tmp\point3-guardrails-final `
  --iterations 10000 `
  --warmup-iterations 1000
```

Verification gates passed: 233 full-suite tests plus 6 skipped, 6 PostgreSQL
integration tests, Ruff, Pyright with zero errors, the frontend production
build, and Docker Compose configuration validation.

Raw per-case and latency artifacts remain ignored under
`tmp/point3-guardrails-final`. Their SHA-256 hashes and the clean implementation
commit are pinned in `point3-guardrails-report.json`.

## Resume wording

Use this evidence-bound version:

> Hardened the RAG pipeline with prompt-injection checks, token and separate
> chat-rate budgets, buffered SSE validation, and output citation plus lexical
> support/relevance filters; detected 18/18 unsafe synthetic regression cases
> and measured 22.95 ms median local guardrail overhead on a max-size fixture.
