"""Compare raw, recent-only, and production conversation-memory assembly."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from note_rag.chat.memory import (  # noqa: E402
    ExtractiveTurnSummarizer,
    MemoryEmbeddingProvider,
    MemoryItem,
    assemble_conversation_history,
)
from note_rag.chunking import RegexTokenCounter  # noqa: E402
from note_rag.evaluation import (  # noqa: E402
    load_jsonl,
    sha256_file,
    validate_run_label,
)

DEFAULT_CASES = REPOSITORY_ROOT / "benchmarks" / "conversation_memory" / "cases.jsonl"
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "benchmarks" / "conversation_memory" / "manifest.json"
)
FILLER_TURNS = (
    ("The design review moved to Friday.", "Friday is noted for the review."),
    ("The lunch headcount is now fourteen.", "I have noted fourteen attendees."),
    ("The staging environment is available.", "Staging availability is noted."),
    ("The onboarding document needs proofreading.", "The proofreading task is noted."),
    ("The weekly sync remains thirty minutes.", "The meeting duration is noted."),
    ("The accessibility audit finished yesterday.", "The completed audit is noted."),
    ("The office delivery arrives after noon.", "The delivery timing is noted."),
    (
        "The quarterly planning template was approved.",
        "The approved template is noted.",
    ),
    ("The demo recording needs updated captions.", "The caption update is noted."),
    ("The travel request is waiting for finance.", "The finance dependency is noted."),
    (
        "The mobile layout passed visual review.",
        "The successful visual review is noted.",
    ),
    ("The support handbook has a new appendix.", "The handbook appendix is noted."),
    ("The training survey closes next week.", "The survey deadline is noted."),
    ("The hardware inventory was reconciled.", "The reconciled inventory is noted."),
    (
        "The team retrospective has five topics.",
        "The retrospective topic count is noted.",
    ),
)


@dataclass(frozen=True, slots=True)
class Case:
    identifier: str
    subject: str
    fact: str
    query: str
    expected_answer: str


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_path: Path) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_path = model_path.resolve()
        self._model = SentenceTransformer(
            str(self.model_path),
            local_files_only=True,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("sentence transformer did not expose a dimension")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        values = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in values]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


def load_cases(cases_path: Path, manifest_path: Path) -> list[Case]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("conversation-memory manifest schema is unsupported")
    if manifest.get("held_out") is not False:
        raise ValueError("conversation-memory manifest must disclose held_out=false")
    if manifest.get("split") != "regression":
        raise ValueError("conversation-memory manifest must use regression split")
    if manifest.get("cases_sha256") != sha256_file(cases_path):
        raise ValueError("conversation-memory cases hash does not match manifest")
    rows = load_jsonl(cases_path)
    if manifest.get("case_count") != len(rows):
        raise ValueError("conversation-memory case count does not match manifest")
    cases: list[Case] = []
    ids: set[str] = set()
    required = {
        "schema_version",
        "id",
        "subject",
        "fact",
        "query",
        "expected_answer",
    }
    for number, row in enumerate(rows, start=1):
        if set(row) != required:
            raise ValueError(f"case {number} has unexpected fields")
        if row["schema_version"] != "1.0":
            raise ValueError(f"case {number} has an unsupported schema")
        valid_strings = all(
            isinstance(row[field], str) and row[field].strip()
            for field in required
        )
        if not valid_strings:
            raise ValueError(f"case {number} has an invalid string field")
        if row["id"] in ids:
            raise ValueError(f"duplicate case id: {row['id']}")
        ids.add(row["id"])
        cases.append(
            Case(
                identifier=row["id"],
                subject=row["subject"],
                fact=row["fact"],
                query=row["query"],
                expected_answer=row["expected_answer"],
            )
        )
    return cases


def build_conversation(case: Case, counter: RegexTokenCounter) -> tuple[
    list[tuple[int, str, str, int]],
    list[tuple[str, str]],
]:
    turns = [
        (
            f"For {case.subject}, remember this confirmed detail: {case.fact}",
            f"I will remember the confirmed {case.subject} detail.",
        ),
        *FILLER_TURNS,
    ]
    messages: list[tuple[int, str, str, int]] = []
    for turn_number, (user, assistant) in enumerate(turns):
        messages.extend(
            [
                (turn_number * 2, "user", user, counter.count(user)),
                (
                    turn_number * 2 + 1,
                    "assistant",
                    assistant,
                    counter.count(assistant),
                ),
            ]
        )
    return messages, turns


def evaluate(
    cases: list[Case],
    provider: MemoryEmbeddingProvider,
    *,
    history_budget: int,
    recent_turns: int,
    semantic_k: int,
    minimum_similarity: float,
    summary_max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counter = RegexTokenCounter()
    summarizer = ExtractiveTurnSummarizer(
        counter,
        max_tokens=summary_max_tokens,
    )
    prepared: list[tuple[Case, list[tuple[int, str, str, int]], list[str]]] = []
    summaries: list[str] = []
    for case in cases:
        messages, turns = build_conversation(case, counter)
        case_summaries = [summarizer.summarize(user, answer) for user, answer in turns]
        summaries.extend(case_summaries)
        prepared.append((case, messages, case_summaries))
    vectors = provider.embed(summaries)
    if len(vectors) != len(summaries):
        raise ValueError("embedding provider returned the wrong vector count")

    records: list[dict[str, Any]] = []
    vector_offset = 0
    for case, messages, case_summaries in prepared:
        memories: list[MemoryItem] = []
        for index, summary in enumerate(case_summaries):
            memories.append(
                MemoryItem(
                    start_position=index * 2,
                    end_position=index * 2 + 1,
                    summary=summary,
                    token_count=counter.count(summary),
                    embedding_model=provider.model_name,
                    embedding_dimension=provider.dimension,
                    embedding=vectors[vector_offset],
                )
            )
            vector_offset += 1
        assembly = assemble_conversation_history(
            messages,
            memories,
            question=case.query,
            token_counter=counter,
            embedding_provider=provider,
            recent_turns=recent_turns,
            semantic_k=semantic_k,
            semantic_min_similarity=minimum_similarity,
            maximum_tokens=history_budget,
            maximum_recent_messages=20,
        )
        raw_history = "\n".join(content for _, _, content, _ in messages)
        previous_history = select_previous_recent_only_history(
            messages,
            maximum_messages=20,
            maximum_tokens=history_budget,
        )
        recent_only = "\n".join(content for _, _, content, _ in previous_history)
        memory_history = "\n".join(turn.content for turn in assembly.turns)
        expected = case.expected_answer.casefold()
        subject = case.subject.casefold()
        full_correct = (
            expected in raw_history.casefold() and subject in raw_history.casefold()
        )
        recent_correct = (
            expected in recent_only.casefold() and subject in recent_only.casefold()
        )
        memory_correct = (
            expected in memory_history.casefold()
            and subject in memory_history.casefold()
        )
        full_tokens = sum(message[3] for message in messages)
        memory_tokens = counter.count(memory_history)
        records.append(
            {
                "id": case.identifier,
                "full_history_correct": full_correct,
                "recent_only_correct": recent_correct,
                "memory_correct": memory_correct,
                "full_history_tokens": full_tokens,
                "memory_history_tokens": memory_tokens,
                "token_reduction": 1 - (memory_tokens / full_tokens),
                "recent_message_count": assembly.trace.recent_message_count,
                "semantic_memory_count": assembly.trace.semantic_memory_count,
            }
        )
    count = len(records)
    return records, {
        "case_count": count,
        "full_history_fact_retention_accuracy": sum(
            row["full_history_correct"] for row in records
        ) / count,
        "recent_only_fact_retention_accuracy": sum(
            row["recent_only_correct"] for row in records
        ) / count,
        "memory_fact_retention_accuracy": sum(
            row["memory_correct"] for row in records
        ) / count,
        "memory_accuracy_delta_pp_vs_full": 100 * (
            sum(row["memory_correct"] for row in records) / count
            - sum(row["full_history_correct"] for row in records) / count
        ),
        "mean_full_history_tokens": sum(
            row["full_history_tokens"] for row in records
        ) / count,
        "mean_memory_history_tokens": sum(
            row["memory_history_tokens"] for row in records
        ) / count,
        "mean_token_reduction": sum(row["token_reduction"] for row in records)
        / count,
        "memory_failure_ids": [
            row["id"] for row in records if not row["memory_correct"]
        ],
    }


def select_previous_recent_only_history(
    messages: list[tuple[int, str, str, int]],
    *,
    maximum_messages: int,
    maximum_tokens: int,
) -> list[tuple[int, str, str, int]]:
    """Execute the pre-Point-4 history-selection algorithm exactly."""

    selected: list[tuple[int, str, str, int]] = []
    used_tokens = 0
    for message in reversed(messages[-maximum_messages:]):
        if used_tokens + message[3] > maximum_tokens:
            break
        selected.append(message)
        used_tokens += message[3]
    selected.reverse()
    return selected


def git_value(*args: str, allow_empty: bool = False) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPOSITORY_ROOT.as_posix()}",
                *args,
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip()
    return output if output or allow_empty else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--history-budget", type=int, default=2000)
    parser.add_argument("--recent-turns", type=int, default=4)
    parser.add_argument("--semantic-k", type=int, default=2)
    parser.add_argument("--minimum-similarity", type=float, default=0.25)
    parser.add_argument("--summary-max-tokens", type=int, default=64)
    args = parser.parse_args()
    validate_run_label(args.label)
    cases = load_cases(args.cases, args.manifest)
    provider = SentenceTransformerEmbeddingProvider(args.model_path)
    records, metrics = evaluate(
        cases,
        provider,
        history_budget=args.history_budget,
        recent_turns=args.recent_turns,
        semantic_k=args.semantic_k,
        minimum_similarity=args.minimum_similarity,
        summary_max_tokens=args.summary_max_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.label}.memory.results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
        newline="\n",
    )
    git_commit = git_value("rev-parse", "HEAD")
    git_status = git_value("status", "--porcelain", allow_empty=True)
    summary = {
        "schema_version": "1.0",
        "run_label": args.label,
        "created_at": datetime.now(UTC).isoformat(),
        "metric_scope": (
            "deterministic fact retention and history-token selection; not "
            "model-generated answer accuracy"
        ),
        "dataset": {
            "path": args.cases.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(args.cases),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "configuration": {
            "history_budget": args.history_budget,
            "recent_turns": args.recent_turns,
            "semantic_k": args.semantic_k,
            "minimum_similarity": args.minimum_similarity,
            "summary_max_tokens": args.summary_max_tokens,
            "embedding_model": provider.model_name,
            "embedding_dimension": provider.dimension,
            "model_snapshot": args.model_path.name,
            "previous_recent_only_max_messages": 20,
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "implementation": {
            "git_commit": git_commit,
            "git_dirty": None if git_status is None else bool(git_status),
            "memory_source_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "note_rag" / "chat" / "memory.py"
            ),
        },
        "metrics": metrics,
        "results_sha256": sha256_file(results_path),
    }
    summary_path = args.output_dir / f"{args.label}.memory.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output = {
        "results": str(results_path),
        "summary": str(summary_path),
        **metrics,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
