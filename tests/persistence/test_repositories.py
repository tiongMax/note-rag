import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from note_rag.chunking import TokenChunker
from note_rag.persistence import (
    ChatMessageRepository,
    ChatRole,
    ChunkRepository,
    Conversation,
    ConversationRepository,
    Document,
    DocumentRepository,
    IngestionJob,
    IngestionJobRepository,
    IngestionJobStatus,
)


def test_stores_and_inspects_documents_and_chunks(session: Session) -> None:
    documents = DocumentRepository(session)
    chunks = ChunkRepository(session)
    document = documents.add(
        Document(filename="lesson.md", media_type="text/markdown")
    )
    generated = TokenChunker(chunk_size=3, chunk_overlap=1).chunk(
        "zero one two three four",
        source_id="lesson.md",
    )

    stored_chunks = chunks.add_from_chunks(document, generated)

    assert documents.get(document.id) is document
    assert documents.list() == [document]
    assert [chunk.text for chunk in stored_chunks] == [
        "zero one two",
        "two three four",
    ]
    assert chunks.list_for_document(document.id) == stored_chunks
    assert document.chunk_count == 2
    assert document.token_count == 5
    assert stored_chunks[0].source_metadata == {"source_id": "lesson.md"}


def test_tracks_ingestion_job_status(session: Session) -> None:
    document = DocumentRepository(session).add(
        Document(filename="notes.txt", media_type="text/plain")
    )
    jobs = IngestionJobRepository(session)
    job = jobs.add(IngestionJob(document=document))

    jobs.set_status(job, IngestionJobStatus.CHUNKING, progress=40)

    assert jobs.get(job.id) is job
    assert jobs.list_for_document(document.id) == [job]
    assert job.status is IngestionJobStatus.CHUNKING
    assert job.progress == 40


def test_rejects_invalid_job_progress(session: Session) -> None:
    document = DocumentRepository(session).add(
        Document(filename="notes.txt", media_type="text/plain")
    )
    jobs = IngestionJobRepository(session)
    job = jobs.add(IngestionJob(document=document))

    with pytest.raises(ValueError, match="between 0 and 100"):
        jobs.set_status(job, IngestionJobStatus.PARSING, progress=101)


def test_document_delete_cascades_to_chunks_and_jobs(session: Session) -> None:
    documents = DocumentRepository(session)
    document = documents.add(
        Document(filename="lesson.md", media_type="text/markdown")
    )
    chunks = ChunkRepository(session)
    chunks.add_from_chunks(
        document,
        TokenChunker(chunk_size=5, chunk_overlap=0).chunk("stored text"),
    )
    jobs = IngestionJobRepository(session)
    job = jobs.add(IngestionJob(document=document))
    document_id = document.id
    job_id = job.id

    documents.delete(document)

    assert documents.get(document_id) is None
    assert chunks.list_for_document(document_id) == []
    assert jobs.get(job_id) is None


def test_chunk_position_is_unique_per_document(session: Session) -> None:
    document = DocumentRepository(session).add(
        Document(filename="lesson.md", media_type="text/markdown")
    )
    generated = TokenChunker(chunk_size=5, chunk_overlap=0).chunk("stored text")
    repository = ChunkRepository(session)
    repository.add_from_chunks(document, generated)

    with pytest.raises(IntegrityError):
        repository.add_from_chunks(document, generated)
    session.rollback()


def test_conversation_messages_and_cascade_delete(session: Session) -> None:
    conversations = ConversationRepository(session)
    conversation = conversations.add(Conversation(title="Stored chat"))
    messages = ChatMessageRepository(session)
    user = messages.add(
        conversation,
        role=ChatRole.USER,
        content="Question",
        token_count=1,
    )
    assistant = messages.add(
        conversation,
        role=ChatRole.ASSISTANT,
        content="Answer [1]",
        token_count=3,
        citations=[{"citation_id": 1}],
        context_token_count=10,
        model_name="fake-chat",
    )

    assert messages.list_for_conversation(conversation.id) == [user, assistant]
    conversations.delete(conversation)
    assert messages.list_for_conversation(conversation.id) == []
