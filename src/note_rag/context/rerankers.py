"""Reranker contracts and a deterministic local baseline."""

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from note_rag.chunking import RegexTokenCounter


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class LexicalReranker:
    """Score query/document token overlap without an external model."""

    model_name = "lexical-overlap-v1"

    def __init__(self, token_counter: RegexTokenCounter | None = None) -> None:
        self.token_counter = token_counter or RegexTokenCounter()

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_terms = self._terms(query)
        if not query_terms:
            return [0.0] * len(documents)
        scores = []
        for document in documents:
            document_terms = self._terms(document)
            if not document_terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & document_terms)
            scores.append(
                overlap / math.sqrt(len(query_terms) * len(document_terms))
            )
        return scores

    def _terms(self, text: str) -> set[str]:
        return {
            token.text.casefold()
            for token in self.token_counter.tokenize(text)
            if any(character.isalnum() for character in token.text)
        }


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder adapter.

    The model is loaded on first use so importing and testing the application
    does not require the optional reranking dependency or download model data.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 16,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("cross-encoder model name cannot be empty")
        if batch_size <= 0:
            raise ValueError("cross-encoder batch size must be greater than zero")
        self._model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._load_model()
        raw_scores: Sequence[Any] = model.predict(
            [(query, document) for document in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = [float(score) for score in raw_scores]
        if len(scores) != len(documents):
            raise ValueError("cross-encoder returned the wrong score count")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("cross-encoder returned a non-finite score")
        if not all(0.0 <= score <= 1.0 for score in scores):
            raise ValueError(
                "cross-encoder scores must be between zero and one; "
                "configure a sigmoid activation"
            )
        return scores

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._model_factory is None:
            try:
                from sentence_transformers import CrossEncoder
                from torch.nn import Sigmoid
            except ImportError as error:
                raise RuntimeError(
                    "cross-encoder reranking requires the 'reranking' "
                    "optional dependency"
                ) from error
            if self.device:
                self._model = CrossEncoder(
                    self._model_name,
                    activation_fn=Sigmoid(),
                    device=self.device,
                )
            else:
                self._model = CrossEncoder(
                    self._model_name,
                    activation_fn=Sigmoid(),
                )
        else:
            if self.device:
                self._model = self._model_factory(
                    self._model_name,
                    device=self.device,
                )
            else:
                self._model = self._model_factory(self._model_name)
        return self._model
