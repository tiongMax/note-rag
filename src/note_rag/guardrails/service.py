"""Deterministic input, retrieved-context, and output guardrails."""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace

from note_rag.chunking import RegexTokenCounter
from note_rag.context import ContextChunk, ContextPackage
from note_rag.guardrails.models import (
    ContextGuardResult,
    GuardrailAction,
    GuardrailDecision,
    GuardrailStage,
    OutputGuardResult,
)
from note_rag.guardrails.rules import (
    detect_prompt_injection,
    detect_prompt_leakage,
    detect_sensitive_value,
    extract_citation_ids,
    has_malformed_citation,
    lexical_groundedness_proxy,
    lexical_relevance_proxy,
    normalize_guardrail_text,
)

BLOCKED_INPUT_RESPONSE = "I can't help with that request."
UNSAFE_OUTPUT_RESPONSE = "I can't provide that response safely."
UNSUPPORTED_OUTPUT_RESPONSE = (
    "I can't provide a supported answer from the available sources."
)

_ABSTENTION_PATTERN = re.compile(
    r"^(?:i\s+)?(?:do\s+not|don't|cannot|can't)\s+"
    r"(?:have\s+)?(?:enough|sufficient)?\s*information"
    r"(?:\s+in\s+the\s+supplied\s+context)?"
    r"(?:\s+from\s+the\s+available\s+sources?)?"
    r"(?:\s+to\s+(?:answer|provide|give|determine|state)\b[^.!?]*)?[.!?]?$|"
    r"^insufficient\s+information(?:\s+in\s+the\s+supplied\s+context)?[.!?]?$|"
    r"^i\s+(?:cannot|can't)\s+provide\s+a\s+supported\s+answer\s+"
    r"from\s+the\s+available\s+sources[.!?]?$")


class GuardrailService:
    """Apply cheap local checks around the retrieval-generation pipeline."""

    def __init__(
        self,
        *,
        token_counter: RegexTokenCounter | None = None,
        input_max_tokens: int = 512,
        input_max_characters: int = 8192,
        output_max_tokens: int = 1024,
        output_max_characters: int = 16_384,
        groundedness_threshold: float = 0.45,
        relevance_threshold: float = 0.05,
    ) -> None:
        if input_max_tokens <= 0:
            raise ValueError("input_max_tokens must be greater than zero")
        if output_max_tokens <= 0:
            raise ValueError("output_max_tokens must be greater than zero")
        if input_max_characters <= 0:
            raise ValueError("input_max_characters must be greater than zero")
        if output_max_characters <= 0:
            raise ValueError("output_max_characters must be greater than zero")
        if not 0.0 <= groundedness_threshold <= 1.0:
            raise ValueError("groundedness_threshold must be between zero and one")
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between zero and one")
        self.token_counter = token_counter or RegexTokenCounter()
        self.input_max_tokens = input_max_tokens
        self.input_max_characters = input_max_characters
        self.output_max_tokens = output_max_tokens
        self.output_max_characters = output_max_characters
        self.groundedness_threshold = groundedness_threshold
        self.relevance_threshold = relevance_threshold

    def inspect_input(self, question: str) -> GuardrailDecision:
        """Block oversized input or high-signal prompt-injection attempts."""

        started = time.perf_counter_ns()
        reasons = list(detect_prompt_injection(question))
        if len(question) > self.input_max_characters:
            reasons.append("input_character_limit")
        elif self.token_counter.count(question) > self.input_max_tokens:
            reasons.append("input_token_limit")
        return self._decision(
            started=started,
            stage=GuardrailStage.INPUT,
            action=(GuardrailAction.BLOCK if reasons else GuardrailAction.ALLOW),
            reasons=reasons,
        )

    def sanitize_context(self, context: ContextPackage) -> ContextGuardResult:
        """Remove injected chunks and rebuild canonical contiguous citations."""

        started = time.perf_counter_ns()
        safe_chunks: list[ContextChunk] = []
        reasons: list[str] = []
        for chunk in context.chunks:
            chunk_reasons = self._context_reasons(chunk)
            if chunk_reasons:
                self._append_reason(reasons, "context_instruction")
                for reason in chunk_reasons:
                    self._append_reason(reasons, reason)
                continue
            safe_chunks.append(replace(chunk, citation_id=len(safe_chunks) + 1))

        rendered_context = self._render_context(safe_chunks)
        sanitized = replace(
            context,
            context=rendered_context,
            chunks=safe_chunks,
            token_count=self.token_counter.count(rendered_context),
        )
        filtered_count = len(context.chunks) - len(safe_chunks)
        decision = self._decision(
            started=started,
            stage=GuardrailStage.CONTEXT,
            action=(
                GuardrailAction.FILTER
                if filtered_count
                else GuardrailAction.ALLOW
            ),
            reasons=reasons,
        )
        return ContextGuardResult(
            context=sanitized,
            filtered_chunk_count=filtered_count,
            decision=decision,
        )

    def validate_output(
        self,
        *,
        question: str,
        answer: str,
        context: ContextPackage,
    ) -> OutputGuardResult:
        """Filter leaked, malformed, unsupported, or irrelevant answers."""

        started = time.perf_counter_ns()
        stripped = answer.strip()
        reasons: list[str] = []
        unsafe = False

        if not stripped:
            reasons.append("empty_output")
        if len(stripped) > self.output_max_characters:
            reasons.append("output_character_limit")
        elif self.token_counter.count(stripped) > self.output_max_tokens:
            reasons.append("output_token_limit")
        if detect_prompt_leakage(stripped):
            reasons.append("prompt_leakage")
            unsafe = True
        if detect_sensitive_value(stripped):
            reasons.append("sensitive_value")
            unsafe = True

        if self._is_abstention(stripped):
            safe_abstention = (
                UNSAFE_OUTPUT_RESPONSE
                if unsafe
                else UNSUPPORTED_OUTPUT_RESPONSE
                if reasons
                else stripped
            )
            return OutputGuardResult(
                answer=safe_abstention,
                decision=self._decision(
                    started=started,
                    stage=GuardrailStage.OUTPUT,
                    action=(
                        GuardrailAction.FILTER
                        if reasons
                        else GuardrailAction.ALLOW
                    ),
                    reasons=reasons,
                ),
                lexical_groundedness_proxy=None,
                lexical_relevance_proxy=None,
            )

        available = {chunk.citation_id: chunk for chunk in context.chunks}
        cited_ids = extract_citation_ids(stripped)
        malformed_citation = has_malformed_citation(stripped)
        invalid_ids = [
            citation_id
            for citation_id in cited_ids
            if citation_id not in available
        ]
        if invalid_ids or malformed_citation:
            reasons.append("invalid_citation")
        valid_ids = [
            citation_id for citation_id in cited_ids if citation_id in available
        ]
        if not cited_ids:
            reasons.append("missing_citation")

        cited_contexts = [available[citation_id].text for citation_id in valid_ids]
        groundedness = lexical_groundedness_proxy(stripped, cited_contexts)
        relevance = lexical_relevance_proxy(question, cited_contexts)
        if groundedness < self.groundedness_threshold:
            reasons.append("unsupported_claim")
        if relevance < self.relevance_threshold:
            reasons.append("irrelevant_answer")

        action = GuardrailAction.FILTER if reasons else GuardrailAction.ALLOW
        safe_answer = (
            UNSAFE_OUTPUT_RESPONSE
            if unsafe
            else UNSUPPORTED_OUTPUT_RESPONSE
            if reasons
            else stripped
        )
        return OutputGuardResult(
            answer=safe_answer,
            decision=self._decision(
                started=started,
                stage=GuardrailStage.OUTPUT,
                action=action,
                reasons=reasons,
            ),
            lexical_groundedness_proxy=groundedness,
            lexical_relevance_proxy=relevance,
        )

    @staticmethod
    def _context_reasons(chunk: ContextChunk) -> tuple[str, ...]:
        values = (
            chunk.text,
            chunk.filename,
            chunk.media_type,
            json.dumps(
                chunk.source_metadata,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ),
        )
        reasons: list[str] = []
        for value in values:
            for reason in detect_prompt_injection(value):
                GuardrailService._append_reason(reasons, reason)
        return tuple(reasons)

    @staticmethod
    def _render_context(chunks: list[ContextChunk]) -> str:
        rendered = []
        for chunk in chunks:
            metadata = json.dumps(
                chunk.source_metadata,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
            header = (
                f"[{chunk.citation_id}] Source: {chunk.filename}; "
                f"media_type={chunk.media_type}; position={chunk.position}; "
                f"metadata={metadata}"
            )
            rendered.append(f"{header}\n{chunk.text}")
        return "\n\n".join(rendered)

    @staticmethod
    def _is_abstention(answer: str) -> bool:
        normalized = normalize_guardrail_text(answer)
        return bool(_ABSTENTION_PATTERN.fullmatch(normalized))

    @staticmethod
    def _append_reason(reasons: list[str], reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    @staticmethod
    def _decision(
        *,
        started: int,
        stage: GuardrailStage,
        action: GuardrailAction,
        reasons: list[str],
    ) -> GuardrailDecision:
        return GuardrailDecision(
            stage=stage,
            action=action,
            reason_codes=tuple(dict.fromkeys(reasons)),
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
        )
