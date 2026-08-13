import uuid

from note_rag.chat import (
    GROUNDED_SYSTEM_PROMPT,
    NO_RETRIEVED_CONTEXT,
    ChatTurn,
    build_chat_turns,
    prompt_sha256,
    rendered_generation_context,
)
from note_rag.context import ContextChunk, ContextPackage
from note_rag.retrieval import SearchMode


def test_prompt_includes_history_context_question_and_citation_rules() -> None:
    context = ContextPackage(
        query="question",
        mode=SearchMode.HYBRID,
        context="[1] Source: notes.txt\nGrounded fact.",
        chunks=[
            ContextChunk(
                citation_id=1,
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                filename="notes.txt",
                media_type="text/plain",
                position=0,
                text="Grounded fact.",
                token_count=3,
                source_metadata={},
                retrieval_score=1.0,
                rerank_score=1.0,
                score=1.0,
            )
        ],
        token_count=10,
        token_budget=100,
        candidates_considered=1,
        duplicates_removed=0,
        truncated=False,
        reranker_model="test",
    )
    history = [ChatTurn(role="user", content="Earlier question")]

    turns = build_chat_turns("Current question", context, history)

    assert turns[0] == history[0]
    assert "[1] Source: notes.txt" in turns[-1].content
    assert "Current question" in turns[-1].content
    assert "only the supplied context" in GROUNDED_SYSTEM_PROMPT
    assert "[1]" in GROUNDED_SYSTEM_PROMPT

    first_hash = prompt_sha256(GROUNDED_SYSTEM_PROMPT, turns)
    assert first_hash == prompt_sha256(GROUNDED_SYSTEM_PROMPT, list(turns))
    assert first_hash != prompt_sha256(
        GROUNDED_SYSTEM_PROMPT,
        [*turns, ChatTurn(role="user", content="changed")],
    )


def test_empty_context_uses_one_canonical_generation_fallback() -> None:
    context = ContextPackage(
        query="unknown",
        mode=SearchMode.HYBRID,
        context="",
        chunks=[],
        token_count=0,
        token_budget=100,
        candidates_considered=0,
        duplicates_removed=0,
        truncated=False,
        reranker_model="test",
    )

    turns = build_chat_turns("Unknown question", context, [])

    assert rendered_generation_context(context) == NO_RETRIEVED_CONTEXT
    assert NO_RETRIEVED_CONTEXT in turns[-1].content
