"""Embedding-provider contracts and the first local implementation."""

from typing import Any, Protocol, cast


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class QueryEmbeddingProvider(EmbeddingProvider, Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class GeminiEmbeddingProvider:
    """Gemini Embedding 2 provider with one vector per input document."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        expected_dimension: int = 768,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._expected_dimension = expected_dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._expected_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(
            texts,
            instruction="Represent this document for retrieval:",
        )

    def embed_query(self, query: str) -> list[float]:
        vectors = self._embed(
            [query],
            instruction="Represent this query for retrieving relevant documents:",
        )
        return vectors[0]

    def _embed(
        self,
        texts: list[str],
        *,
        instruction: str,
    ) -> list[list[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        from google import genai
        from google.genai import types

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"{instruction}\n{text}"
                    )
                ],
            )
            for text in texts
        ]
        with genai.Client(api_key=self._api_key) as client:
            response = client.models.embed_content(
                model=self._model_name,
                contents=cast(Any, contents),
                config=types.EmbedContentConfig(
                    output_dimensionality=self._expected_dimension
                ),
            )
        if not response.embeddings:
            raise RuntimeError("Gemini returned no embeddings")
        return [
            list(embedding.values or [])
            for embedding in response.embeddings
        ]
