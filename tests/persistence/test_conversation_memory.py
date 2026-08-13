from __future__ import annotations

from note_rag.persistence import (
    ChatMessageRepository,
    Conversation,
    ConversationMemoryRepository,
    ConversationRepository,
    Database,
)


def test_persists_complete_pair_and_memory_atomically(database: Database) -> None:
    with database.session() as session:
        conversation = ConversationRepository(session).add(
            Conversation(title="Memory test")
        )
        user, assistant = ChatMessageRepository(session).add_pair(
            conversation,
            user_content="The launch code is amber",
            user_token_count=5,
            assistant_content="The code is amber.",
            assistant_token_count=5,
            model_name="test-model",
        )
        ConversationMemoryRepository(session).add(
            conversation,
            start_position=user.position,
            end_position=assistant.position,
            summary="Earlier user: launch code amber",
            token_count=6,
            embedding_model="test-embedding",
            embedding_dimension=2,
            embedding=[1.0, 0.0],
        )
        conversation_id = conversation.id

    with database.session() as session:
        memories = ConversationMemoryRepository(
            session
        ).list_for_conversation(conversation_id)

    assert len(memories) == 1
    assert memories[0].embedding == [1.0, 0.0]
    assert memories[0].end_position == 1
