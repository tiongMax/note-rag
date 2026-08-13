from __future__ import annotations

from note_rag.chat.memory import (
    ExtractiveTurnSummarizer,
    MemoryItem,
    assemble_conversation_history,
    contextualized_retrieval_query,
)
from note_rag.chat.models import ChatTurn
from note_rag.chunking import RegexTokenCounter


class TopicEmbeddingProvider:
    model_name = "topic-embedding-v1"
    dimension = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        return [
            float("orion" in normalized or "project" in normalized),
            float("allergy" in normalized or "peanut" in normalized),
            float("vacation" in normalized or "kyoto" in normalized),
        ]


def memory(
    start: int,
    summary: str,
    provider: TopicEmbeddingProvider,
) -> MemoryItem:
    return MemoryItem(
        start_position=start,
        end_position=start + 1,
        summary=summary,
        token_count=RegexTokenCounter().count(summary),
        embedding_model=provider.model_name,
        embedding_dimension=provider.dimension,
        embedding=provider.embed([summary])[0],
    )


def test_keeps_recent_turn_verbatim_and_recalls_relevant_older_memory() -> None:
    counter = RegexTokenCounter()
    provider = TopicEmbeddingProvider()
    messages = [
        (0, "user", "The Orion launch code is amber.", 7),
        (1, "assistant", "I will remember the Orion detail.", 7),
        (2, "user", "I am planning a vacation.", 6),
        (3, "assistant", "Kyoto is on the shortlist.", 7),
        (4, "user", "My meeting is on Tuesday.", 7),
        (5, "assistant", "Tuesday is noted.", 4),
    ]
    memories = [
        memory(0, "Earlier user: Orion launch code amber.", provider),
        memory(2, "Earlier user: Vacation shortlist Kyoto.", provider),
        memory(4, "Earlier user: Meeting Tuesday.", provider),
    ]

    assembled = assemble_conversation_history(
        messages,
        memories,
        question="What was the project launch code?",
        token_counter=counter,
        embedding_provider=provider,
        recent_turns=1,
        semantic_k=1,
        semantic_min_similarity=0.0,
        maximum_tokens=80,
        maximum_recent_messages=20,
    )

    assert assembled.turns[-2:] == [
        ChatTurn(role="user", content="My meeting is on Tuesday."),
        ChatTurn(role="assistant", content="Tuesday is noted."),
    ]
    assert "Orion launch code amber" in assembled.turns[0].content
    assert "Kyoto" not in assembled.turns[0].content
    assert assembled.trace.recent_message_count == 2
    assert assembled.trace.semantic_memory_count == 1


def test_oversized_recent_pair_is_replaced_by_bounded_memory() -> None:
    counter = RegexTokenCounter()
    provider = TopicEmbeddingProvider()
    long_content = "orion " * 100
    messages = [
        (0, "user", long_content, 100),
        (1, "assistant", long_content, 100),
    ]
    memories = [memory(0, "Earlier user: Orion code amber.", provider)]

    assembled = assemble_conversation_history(
        messages,
        memories,
        question="What is the Orion code?",
        token_counter=counter,
        embedding_provider=provider,
        recent_turns=1,
        semantic_k=1,
        semantic_min_similarity=0.0,
        maximum_tokens=40,
        maximum_recent_messages=20,
    )

    assert assembled.trace.recent_message_count == 0
    assert assembled.trace.semantic_memory_count == 1
    assert "amber" in assembled.turns[0].content


def test_no_provider_means_no_claimed_semantic_retrieval() -> None:
    assembled = assemble_conversation_history(
        [],
        [],
        question="Question",
        token_counter=RegexTokenCounter(),
        embedding_provider=None,
        recent_turns=1,
        semantic_k=3,
        semantic_min_similarity=0.25,
        maximum_tokens=20,
        maximum_recent_messages=20,
    )

    assert assembled.turns == []
    assert assembled.trace.embedding_model is None


def test_extractive_summary_is_bounded_and_keeps_both_sides() -> None:
    counter = RegexTokenCounter()
    summarizer = ExtractiveTurnSummarizer(counter, max_tokens=24)

    summary = summarizer.summarize("question " * 30, "answer " * 30)

    assert counter.count(summary) <= 24
    assert "Earlier user" in summary
    assert "assistant" in summary


def test_extractive_summary_removes_stale_citation_numbers() -> None:
    summary = ExtractiveTurnSummarizer(
        RegexTokenCounter(),
        max_tokens=32,
    ).summarize("Question", "Prior answer [1] with another source [22].")

    assert "[1]" not in summary
    assert "[22]" not in summary


def test_contextualized_query_is_bounded() -> None:
    counter = RegexTokenCounter()

    rendered = contextualized_retrieval_query(
        "What about that project exactly?",
        [ChatTurn(role="user", content="Orion " * 100)],
        token_counter=counter,
        max_tokens=30,
    )

    assert counter.count(rendered) <= 30
    assert "Conversation cues" in rendered
    assert rendered.endswith("What about that project exactly?")


def test_query_embedding_failure_degrades_to_recent_history() -> None:
    class BrokenQueryProvider(TopicEmbeddingProvider):
        def embed_query(self, query: str) -> list[float]:
            del query
            raise RuntimeError("temporarily unavailable")

    provider = BrokenQueryProvider()
    assembled = assemble_conversation_history(
        [(0, "user", "Orion code amber", 3), (1, "assistant", "Noted", 1)],
        [memory(0, "Earlier user: Orion code amber", provider)],
        question="What is the project code?",
        token_counter=RegexTokenCounter(),
        embedding_provider=provider,
        recent_turns=1,
        semantic_k=1,
        semantic_min_similarity=0.0,
        maximum_tokens=2,
        maximum_recent_messages=20,
    )

    assert assembled.turns == []
    assert assembled.trace.embedding_model is None
