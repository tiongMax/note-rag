"""Optional RAGAS 0.4 collections-API judge adapter."""

import importlib
from pathlib import Path
from typing import Any

from note_rag.evaluation.models import BenchmarkCase, EvaluationTrace


class RagasEvaluator:
    """Run pinned RAGAS judge metrics without leaking RAGAS into core metrics."""

    def __init__(
        self,
        *,
        api_key: str,
        judge_model: str,
        embedding_model: str,
        cache_dir: Path,
    ) -> None:
        if not api_key:
            raise ValueError("a Gemini API key is required for RAGAS evaluation")
        try:
            genai = importlib.import_module("google.genai")
            ragas_cache = importlib.import_module("ragas.cache")
            ragas_embeddings = importlib.import_module("ragas.embeddings")
            ragas_llms = importlib.import_module("ragas.llms")
            ragas_metrics = importlib.import_module("ragas.metrics.collections")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "RAGAS is not installed; run 'pip install -e .[evaluation]' first"
            ) from error

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = ragas_cache.DiskCacheBackend(cache_dir=str(cache_dir))
        client = genai.Client(api_key=api_key)
        llm = ragas_llms.llm_factory(
            judge_model,
            provider="google",
            client=client,
            cache=cache,
            temperature=0,
        )
        embeddings = ragas_embeddings.GoogleEmbeddings(
            client=client,
            model=embedding_model,
            cache=cache,
        )
        self._metrics: list[tuple[str, Any, tuple[str, ...]]] = [
            (
                "faithfulness",
                ragas_metrics.Faithfulness(llm=llm),
                ("user_input", "response", "retrieved_contexts"),
            ),
            (
                "answer_relevancy",
                ragas_metrics.AnswerRelevancy(
                    llm=llm,
                    embeddings=embeddings,
                ),
                ("user_input", "response"),
            ),
            (
                "context_precision",
                ragas_metrics.ContextPrecisionWithReference(llm=llm),
                ("user_input", "reference", "retrieved_contexts"),
            ),
            (
                "answer_correctness",
                ragas_metrics.AnswerCorrectness(
                    llm=llm,
                    embeddings=embeddings,
                ),
                ("user_input", "response", "reference"),
            ),
        ]

    def evaluate(
        self,
        case: BenchmarkCase,
        trace: EvaluationTrace,
    ) -> dict[str, float | str | None]:
        if trace.error:
            return {"ragas_skipped": "trace contains an execution error"}
        if not case.answerable:
            return {"ragas_skipped": "unanswerable case"}

        inputs: dict[str, Any] = {
            "user_input": case.question,
            "response": trace.answer,
            "reference": case.reference_answer,
            "retrieved_contexts": [chunk.text for chunk in trace.retrieved],
        }
        scores: dict[str, float | str | None] = {}
        for name, metric, required in self._metrics:
            try:
                result = metric.score(**{field: inputs[field] for field in required})
                scores[name] = float(result.value)
                reason = getattr(result, "reason", None)
                if reason:
                    scores[f"{name}_reason"] = str(reason)
            except Exception as error:
                scores[name] = None
                scores[f"{name}_error"] = f"{type(error).__name__}: {error}"
        return scores
