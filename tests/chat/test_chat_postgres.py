import os

import pytest

from note_rag.persistence import (
    ChatMessageRepository,
    ChatRole,
    Conversation,
    ConversationRepository,
    Database,
)
from note_rag.persistence.settings import DatabaseSettings

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for the PostgreSQL integration test",
)
def test_live_conversation_message_round_trip() -> None:
    assert TEST_DATABASE_URL is not None
    database = Database(DatabaseSettings(url=TEST_DATABASE_URL))
    conversation_id = None
    try:
        with database.session() as session:
            conversation = ConversationRepository(session).add(
                Conversation(title="Phase seven smoke")
            )
            conversation_id = conversation.id
            ChatMessageRepository(session).add(
                conversation,
                role=ChatRole.USER,
                content="Stored question",
                token_count=2,
            )
            ChatMessageRepository(session).add(
                conversation,
                role=ChatRole.ASSISTANT,
                content="Stored answer [1]",
                token_count=4,
                citations=[{"citation_id": 1}],
                context_token_count=20,
                model_name="fake-chat",
            )

        with database.session() as session:
            messages = ChatMessageRepository(session).list_for_conversation(
                conversation_id
            )
            assert [message.position for message in messages] == [0, 1]
            assert messages[1].citations == [{"citation_id": 1}]
    finally:
        if conversation_id is not None:
            with database.session() as session:
                conversation = ConversationRepository(session).get(
                    conversation_id
                )
                if conversation is not None:
                    ConversationRepository(session).delete(conversation)
        database.dispose()
