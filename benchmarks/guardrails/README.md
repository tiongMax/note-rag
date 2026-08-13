# Guardrail evaluation cases

This directory contains version `1.0.1` of a synthetic guardrail regression
benchmark. It covers the three evaluation stages exactly: `input`, `context`,
and `output`. The cases are committed fixtures, not generated from production
traffic, and contain no real credentials, private prompts, or personal data.

This corpus was inspected while implementing Point 3 and is therefore not an
untouched holdout. It is useful for transparent, versioned regression checks,
not independent generalization claims. A separate blind code audit did not
inspect these cases, but its findings are not dataset performance evidence.

Treat this set as evaluation data. Do not copy case wording into detection
rules or tune thresholds against individual cases. Add newly authored cases in
a later dataset version, keep existing IDs stable, and never reuse a retired
ID for different content.

## Files

- `cases.jsonl` has one complete case per line.
- `manifest.json` pins the case-file hash, schema and ruleset versions, counts,
  and provenance.

## Record schema

Every JSONL record has all of these fields:

| Field | Type | Rule |
| --- | --- | --- |
| `schema_version` | string | Record schema version; currently `1.0`. |
| `ruleset_version` | string | Expected-label policy version; currently `1.0`. |
| `id` | string | Unique, immutable stable ID. |
| `stage` | string | Exactly one of `input`, `context`, or `output`. |
| `label` | string | `unsafe` or `safe`; must agree with `expected_unsafe`. |
| `category` | string | Stable snake-case behavior class used for slicing, never for constructing the fixture. |
| `question` | string | Complete user input fixture. |
| `contexts` | array of strings | Complete ordered retrieved-context fixture. |
| `context_roles` | array of strings | Parallel to `contexts`; each item is `supporting`, `distractor`, or `adversarial`. |
| `answer` | string | Complete generated-output fixture. |
| `expected_unsafe` | boolean | Expected binary decision at the target `stage`. |
| `hard_negative` | boolean | `true` only for a benign case intentionally resembling an unsafe pattern. |
| `rationale` | string | Short annotation justification; not evaluator input. |
| `tags` | array of strings | Descriptive slices; not evaluator input. |

`contexts` and `context_roles` must be non-empty and have the same length.
`hard_negative: true` requires `expected_unsafe: false`. The runner should pass
the actual fixture field to the stage implementation and compare its result to
`expected_unsafe`; it must not derive test text or expected behavior from
`label`, `category`, `rationale`, or `tags`.

Safe factual output fixtures follow the production answer contract by citing
supporting context IDs. Safe no-answer fixtures use the canonical abstention
language accepted by the production output guardrail. Dataset `1.0.1` added
that contract alignment without changing unsafe or context-stage wording.

## Stage semantics

- For `input`, `question` is the target. `contexts` and `answer` are benign
  defaults so every record retains one common shape. Unsafe cases cover direct
  prompt injection, jailbreaks, instruction redirection, and prompt or secret
  exfiltration. Safe cases include ordinary controls and quoted or educational
  hard negatives.
- For `context`, the ordered `contexts` list is the target. Unsafe cases embed
  indirect instructions in retrieved text. Safe hard negatives contain quoted
  attacks, defensive policy, research prose, domain-specific imperatives, or
  inert code literals that must not be treated as executable model commands.
- For `output`, `answer` is the target, evaluated against `question` and
  `contexts`. The parallel roles make supporting evidence and distractors
  explicit. Unsafe cases cover secret or prompt leakage, contradictions,
  unsupported claims, fabricated attribution, and irrelevant answers. Safe
  cases cover grounded answers, non-disclosure, calibrated abstention,
  harmless placeholders, and distractor resistance.

Here, `unsafe` means the stage guardrail is expected to reject or flag the
fixture. At the output stage this umbrella includes evidence-integrity and
relevance failures as well as conventional security leakage.

## Interpreting results

Deterministic keyword, regular-expression, token-overlap, and other lexical
checks are **non-semantic proxies**. They can provide reproducible regression
signals, but they do not establish semantic prompt-injection resistance,
confidentiality, groundedness, answer relevance, or production safety. Report
such results explicitly as deterministic or lexical proxy metrics, never as a
semantic safety evaluation. Hard negatives are included specifically to expose
false positives from surface-form matching.

Report confusion-matrix counts and rates overall and separately by stage and
label. Small category slices are diagnostic only; this compact synthetic set
does not support broad statistical claims.

## Running the evaluator and latency benchmark

From the repository root, run the production guardrail service against all
regression cases and benchmark both the `normal` and `max` complete local paths:

```powershell
python scripts/run_guardrail_benchmark.py --label guardrails
```

The defaults perform 1,000 unmeasured warmup calls and 10,000 measured calls
per fixture using `time.perf_counter_ns`. Use `--fixture normal` or
`--fixture max` to select one fixture. The runner writes per-case JSONL and a
summary JSON under `tmp/benchmarks`; the summary contains per-stage confusion
matrices, false-positive and false-negative IDs, median/p95/p99 latency, and
machine, Python, corpus, manifest, and result hashes.

## Integrity and updates

From the repository root, verify the committed cases hash in PowerShell with:

```powershell
(Get-FileHash benchmarks/guardrails/cases.jsonl -Algorithm SHA256).Hash.ToLower()
```

Any case change requires a dataset-version decision and an updated SHA-256 and
counts in `manifest.json`. Schema meaning changes require a new
`schema_version`; annotation-policy changes require a new `ruleset_version`.
