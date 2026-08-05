"""Chat-provider abstraction and Gemini implementation."""

from collections.abc import Iterator
from typing import Any, Protocol, cast

from note_rag.chat.models import ChatTurn


class ChatProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> str: ...

    def stream(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> Iterator[str]: ...


class GeminiChatProvider:
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> str:
        self._require_api_key()
        from google import genai

        with genai.Client(api_key=self._api_key) as client:
            response = client.models.generate_content(
                model=self._model_name,
                contents=cast(Any, self._contents(turns)),
                config=self._config(system_instruction),
            )
        return response.text or ""

    def stream(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> Iterator[str]:
        self._require_api_key()
        from google import genai

        with genai.Client(api_key=self._api_key) as client:
            responses = client.models.generate_content_stream(
                model=self._model_name,
                contents=cast(Any, self._contents(turns)),
                config=self._config(system_instruction),
            )
            for response in responses:
                if response.text:
                    yield response.text

    def _contents(self, turns: list[ChatTurn]) -> list[Any]:
        from google.genai import types

        return [
            types.Content(
                role="model" if turn.role == "assistant" else "user",
                parts=[types.Part.from_text(text=turn.content)],
            )
            for turn in turns
        ]

    def _config(self, system_instruction: str):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def _require_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
