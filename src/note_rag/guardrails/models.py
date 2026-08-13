"""Framework-independent guardrail decisions and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from note_rag.context import ContextPackage


class GuardrailStage(StrEnum):
    """Pipeline stage at which a guardrail decision was made."""

    INPUT = "input"
    CONTEXT = "context"
    OUTPUT = "output"


class GuardrailAction(StrEnum):
    """Action the caller must take after a guardrail decision."""

    ALLOW = "allow"
    BLOCK = "block"
    FILTER = "filter"


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    """One deterministic decision with stable machine-readable reasons."""

    stage: GuardrailStage
    action: GuardrailAction
    reason_codes: tuple[str, ...]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ContextGuardResult:
    """A context package after unsafe chunks have been removed."""

    context: ContextPackage
    filtered_chunk_count: int
    decision: GuardrailDecision


@dataclass(frozen=True, slots=True)
class OutputGuardResult:
    """The safe answer and explicitly lexical quality proxies."""

    answer: str
    decision: GuardrailDecision
    lexical_groundedness_proxy: float | None
    lexical_relevance_proxy: float | None
