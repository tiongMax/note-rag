import json
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from note_rag.chat import ChatTurn
from note_rag.chunking import TokenChunker
from note_rag.generation import LearningMaterialService
from note_rag.persistence import (
    ChunkRepository,
    Course,
    CourseRepository,
    Database,
    Document,
    DocumentRepository,
    StudyItemRepository,
    TopicRepository,
    TopicState,
)


class QueuedProvider:
    model_name = "fake-learning-model"

    def __init__(self, *responses: dict | str) -> None:
        self.responses = list(responses)

    def generate(self, system_instruction: str, turns: list[ChatTurn]) -> str:
        del system_instruction, turns
        value = self.responses.pop(0)
        return value if isinstance(value, str) else json.dumps(value)

    def stream(self, system_instruction: str, turns: list[ChatTurn]) -> Iterator[str]:
        del system_instruction, turns
        return iter(())


def seeded_course(session: Session):
    document = DocumentRepository(session).add(
        Document(filename="biology.md", media_type="text/markdown")
    )
    chunk = ChunkRepository(session).add_from_chunks(
        document,
        TokenChunker(chunk_size=20, chunk_overlap=0).chunk(
            "Mitochondria produce ATP through cellular respiration."
        ),
    )[0]
    course = CourseRepository(session).add(Course(title="Biology"))
    CourseRepository(session).attach_document(course, document)
    return course, chunk


def test_generates_draft_curriculum_and_cited_items(
    database: Database, session: Session
) -> None:
    course, chunk = seeded_course(session)
    session.commit()
    provider = QueuedProvider(
        {
            "topics": [
                {
                    "title": "Cell energy",
                    "description": "How cells make usable energy",
                    "objectives": [{"title": "Explain ATP production"}],
                    "source_chunk_ids": [str(chunk.id)],
                }
            ]
        },
        {
            "items": [
                {
                    "item_type": "flashcard",
                    "prompt": "What do mitochondria produce?",
                    "answer": "ATP",
                    "explanation": "Mitochondria produce ATP during respiration.",
                    "difficulty": 2,
                    "source_chunk_ids": [str(chunk.id)],
                },
                {
                    "item_type": "multiple_choice",
                    "prompt": "Which molecule is produced?",
                    "answer": "ATP",
                    "explanation": "The source identifies ATP.",
                    "difficulty": 2,
                    "options": [
                        {"text": "ATP", "correct": True},
                        {"text": "DNA", "correct": False},
                    ],
                    "source_chunk_ids": [str(chunk.id)],
                },
                {
                    "item_type": "short_answer",
                    "prompt": "Name the energy molecule.",
                    "answer": "ATP",
                    "explanation": "ATP is the cited energy product.",
                    "difficulty": 3,
                    "source_chunk_ids": [str(chunk.id)],
                },
            ]
        },
        {
            "items": [
                {
                    "item_type": "flashcard",
                    "prompt": "What do mitochondria produce?",
                    "answer": "ATP",
                    "explanation": "The source supports ATP.",
                    "difficulty": 2,
                    "source_chunk_ids": [str(chunk.id)],
                }
            ]
        },
    )
    service = LearningMaterialService(database, provider)

    topics = service.generate_curriculum(course.id)
    items = service.generate_study_items(topics[0].id)
    topic_id = topics[0].id
    session.expire_all()
    stored_topic = TopicRepository(session).get(topic_id)
    assert stored_topic is not None
    stored_items = StudyItemRepository(session).list_for_topic(topic_id)

    assert stored_topic.state is TopicState.DRAFT
    assert len(stored_topic.objectives) == 1
    assert {item.item_type.value for item in items} == {
        "flashcard",
        "multiple_choice",
        "short_answer",
    }
    assert len(items) == len(stored_items)
    assert all(item.sources for item in stored_items)
    assert service.generate_study_items(topic_id) == []


def test_invalid_output_is_atomic_and_approved_topics_survive_regeneration(
    database: Database, session: Session
) -> None:
    course, chunk = seeded_course(session)
    approved = TopicRepository(session).add(
        course, title="Approved", state=TopicState.APPROVED
    )
    session.commit()
    service = LearningMaterialService(
        database,
        QueuedProvider(
            {"topics": [{"title": "Broken", "objectives": []}]},
            {
                "topics": [
                    {
                        "title": "Approved",
                        "objectives": [{"title": "Keep it"}],
                        "source_chunk_ids": [str(chunk.id)],
                    },
                    {
                        "title": "New draft",
                        "objectives": [{"title": "Learn it"}],
                        "source_chunk_ids": [str(chunk.id)],
                    },
                ]
            },
        ),
    )

    with pytest.raises(ValueError, match="invalid structured"):
        service.generate_curriculum(course.id)
    assert [
        topic.title for topic in TopicRepository(session).list_for_course(course.id)
    ] == ["Approved"]

    service.generate_curriculum(course.id)
    session.expire_all()
    stored_approved = TopicRepository(session).get(approved.id)
    assert stored_approved is not None
    assert stored_approved.state is TopicState.APPROVED
    assert {
        topic.title for topic in TopicRepository(session).list_for_course(course.id)
    } == {
        "Approved",
        "New draft",
    }
