"""Evaluation dataset, trace, and report contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    """A stable evidence span independent of a particular chunking run."""

    model_config = ConfigDict(extra="forbid")

    document: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    group: str = Field(min_length=1)
    relevance: int = Field(default=3, ge=1, le=3)
    chunk_ids: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    """One human-labelled benchmark question."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerable: bool = True
    reference_answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    expected_behavior: Literal["answer", "refuse"] | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> "BenchmarkCase":
        behavior = self.expected_behavior or ("answer" if self.answerable else "refuse")
        self.expected_behavior = behavior
        if self.answerable != (behavior == "answer"):
            raise ValueError("expected_behavior must agree with the answerable label")
        if self.answerable and not self.reference_answer:
            raise ValueError("answerable cases require a non-empty reference_answer")
        if self.answerable and not self.evidence:
            raise ValueError("answerable cases require at least one evidence span")
        if not self.answerable and self.evidence:
            raise ValueError("unanswerable cases cannot contain evidence")
        return self


class RetrievedContext(BaseModel):
    """A context chunk returned by the system under test."""

    model_config = ConfigDict(extra="allow")

    citation_id: int = Field(ge=1)
    chunk_id: str
    document_id: str
    filename: str
    position: int = Field(ge=0)
    text: str
    retrieval_score: float
    rerank_score: float | None = None
    score: float


class EvaluationTrace(BaseModel):
    """Captured system output used for deterministic and judge evaluation."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    retrieved: list[RetrievedContext] = Field(default_factory=list)
    answer: str
    resolved_citation_ids: list[int] = Field(default_factory=list)
    model_name: str | None = None
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    context_tokens: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    error: str | None = None


class QuestionResult(BaseModel):
    """Per-question trace and scores written to JSONL."""

    case: BenchmarkCase
    trace: EvaluationTrace
    metrics: dict[str, float | int | None]
    judge: dict[str, float | str | None] = Field(default_factory=dict)


class ExperimentSnapshot(BaseModel):
    """Configuration required to reproduce an evaluation run."""

    experiment: str
    benchmark_path: str
    benchmark_sha256: str
    benchmark_cases: int
    corpus_manifest: str | None
    corpus_sha256: str | None
    base_url: str | None
    git_commit: str | None
    created_at: str
    retrieval: dict[str, Any]
    ragas: dict[str, Any]
