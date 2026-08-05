import uuid
from collections.abc import Iterator

import pytest

from note_rag.chat import ChatService, ChatTurn
from note_rag.context import ContextChunk, ContextPackage
from note_rag.persistence import (
    ChatMessageRepository,
    ChatRole,
    ConversationRepository,
    Database,
)
from note_rag.retrieval import SearchFilters, SearchMode


class StubContextBuilder:
    def build(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        candidate_k: int = 20,
        max_chunks: int = 8,
        max_context_tokens: int = 1200,
        vector_weight: float = 0.7,
        rerank: bool = True,
        rerank_weight: float = 0.7,
        filters: SearchFilters | None = None,
    ) -> ContextPackage:
        chunk = ContextChunk(
            citation_id=1,
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="lesson.txt",
            media_type="text/plain",
            position=0,
            text="Apples grow in orchards.",
            token_count=5,
            source_metadata={"page": 2},
            retrieval_score=0.9,
            rerank_score=1.0,
            score=0.97,
        )
        return ContextPackage(
            query=query,
            mode=SearchMode.HYBRID,
            context="[1] Source: lesson.txt\nApples grow in orchards.",
            chunks=[chunk],
            token_count=14,
            token_budget=100,
            candidates_considered=1,
            duplicates_removed=0,
            truncated=False,
            reranker_model="fake-reranker",
        )


class FakeChatProvider:
    model_name = "fake-chat"

    def __init__(self) -> None:
        self.calls: list[list[ChatTurn]] = []

    def generate(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> str:
        self.calls.append(turns)
        return "Apples grow in orchards [1]. Ignore invalid [99]."

    def stream(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> Iterator[str]:
        self.calls.append(turns)
        yield "Apples grow "
        yield "in orchards [1]."


def test_persists_chat_and_valid_citations(database: Database) -> None:
    provider = FakeChatProvider()
    service = ChatService(database, StubContextBuilder(), provider)

    result = service.ask("Where do apples grow?")

    assert result.answer.startswith("Apples grow")
    assert [citation.citation_id for citation in result.citations] == [1]
    with database.session() as session:
        conversation = ConversationRepository(session).get(
            result.conversation_id
        )
        messages = ChatMessageRepository(session).list_for_conversation(
            result.conversation_id
        )
        assert conversation is not None
        assert conversation.title == "Where do apples grow?"
        assert [message.role for message in messages] == [
            ChatRole.USER,
            ChatRole.ASSISTANT,
        ]
        assert messages[1].citations[0]["filename"] == "lesson.txt"
        assert messages[1].model_name == "fake-chat"


def test_reuses_conversation_history(database: Database) -> None:
    provider = FakeChatProvider()
    service = ChatService(database, StubContextBuilder(), provider)
    first = service.ask("First question")

    service.ask("Follow-up question", conversation_id=first.conversation_id)

    turns = provider.calls[-1]
    assert [turn.role for turn in turns] == ["user", "assistant", "user"]
    assert turns[0].content == "First question"
    assert turns[1].content.startswith("Apples grow")
    assert "Follow-up question" in turns[2].content


def test_streams_events_then_persists_answer(database: Database) -> None:
    service = ChatService(
        database,
        StubContextBuilder(),
        FakeChatProvider(),
    )

    events = list(service.stream("Where do apples grow?"))

    assert [event.event for event in events] == [
        "metadata",
        "delta",
        "delta",
        "done",
    ]
    conversation_id = uuid.UUID(events[0].data["conversation_id"])
    assert events[-1].data["citations"][0]["citation_id"] == 1
    with database.session() as session:
        messages = ChatMessageRepository(session).list_for_conversation(
            conversation_id
        )
        assert messages[-1].content == "Apples grow in orchards [1]."


def test_rejects_unknown_conversation(database: Database) -> None:
    service = ChatService(
        database,
        StubContextBuilder(),
        FakeChatProvider(),
    )

    with pytest.raises(LookupError, match="conversation not found"):
        service.ask("Question", conversation_id=uuid.uuid4())
