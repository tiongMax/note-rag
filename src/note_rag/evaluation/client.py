"""HTTP system-under-test adapter for a running Note RAG API."""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from note_rag.chunking import RegexTokenCounter
from note_rag.evaluation.models import (
    BenchmarkCase,
    EvaluationTrace,
    RetrievedContext,
)


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    mode: str = "hybrid"
    candidate_k: int = 20
    max_chunks: int = 8
    max_context_tokens: int = 1200
    vector_weight: float = 0.7
    rerank: bool = True
    rerank_weight: float = 0.7

    def as_payload(self, case: BenchmarkCase) -> dict[str, Any]:
        return {
            "query": case.question,
            "mode": self.mode,
            "candidate_k": self.candidate_k,
            "max_chunks": self.max_chunks,
            "max_context_tokens": self.max_context_tokens,
            "vector_weight": self.vector_weight,
            "rerank": self.rerank,
            "rerank_weight": self.rerank_weight,
            "filters": case.filters,
        }


@dataclass(slots=True)
class NoteRagHttpClient:
    base_url: str
    api_token: str = ""
    timeout_seconds: float = 120.0
    options: RetrievalOptions = field(default_factory=RetrievalOptions)
    token_counter: RegexTokenCounter = field(default_factory=RegexTokenCounter)
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    def run(self, case: BenchmarkCase) -> EvaluationTrace:
        """Capture context and answer responses for one independent question."""

        payload = self.options.as_payload(case)
        retrieval_started = time.perf_counter()
        context = self._post("/api/v1/retrieval/context", payload)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        generation_started = time.perf_counter()
        chat = self._post("/api/v1/chat", payload)
        chat_total_ms = (time.perf_counter() - generation_started) * 1000
        generation_ms = max(0.0, chat_total_ms - retrieval_ms)
        answer = str(chat["answer"])
        input_tokens = int(context.get("token_count", 0))
        input_tokens += self.token_counter.count(case.question)
        output_tokens = self.token_counter.count(answer)
        estimated_cost = self._estimated_cost(input_tokens, output_tokens)
        return EvaluationTrace(
            question_id=case.id,
            retrieved=[
                RetrievedContext.model_validate(chunk)
                for chunk in context.get("chunks", [])
            ],
            answer=answer,
            resolved_citation_ids=[
                int(citation["citation_id"]) for citation in chat.get("citations", [])
            ],
            model_name=chat.get("model_name"),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            context_tokens=int(context.get("token_count", 0)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
        )

    def _estimated_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float | None:
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Note RAG API returned HTTP {error.code}: {details}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"cannot reach Note RAG API at {self.base_url}: {error.reason}"
            ) from error
        if not isinstance(result, dict):
            raise RuntimeError(f"Note RAG API returned invalid JSON for {path}")
        return result
