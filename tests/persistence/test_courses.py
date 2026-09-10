import pytest
from sqlalchemy.orm import Session

from note_rag.chunking import TokenChunker
from note_rag.persistence import (
    ChunkRepository,
    Course,
    CourseRepository,
    Document,
    DocumentRepository,
    TopicRepository,
    TopicState,
)


def test_documents_can_belong_to_multiple_courses_and_survive_course_delete(
    session: Session,
) -> None:
    documents = DocumentRepository(session)
    document = documents.add(Document(filename="lesson.md", media_type="text/markdown"))
    courses = CourseRepository(session)
    first = courses.add(Course(title="Biology"))
    second = courses.add(Course(title="Exam review"))

    courses.attach_document(first, document)
    courses.attach_document(second, document)
    courses.attach_document(first, document)
    courses.delete(first)

    assert documents.get(document.id) is document
    assert [link.document_id for link in second.document_links] == [document.id]


def test_topics_are_hierarchical_ordered_and_cascade(session: Session) -> None:
    course = CourseRepository(session).add(Course(title="Physics"))
    topics = TopicRepository(session)
    mechanics = topics.add(course, title="Mechanics", state=TopicState.APPROVED)
    waves = topics.add(course, title="Waves")
    motion = topics.add(course, title="Motion", parent=mechanics)
    forces = topics.add(course, title="Forces", parent=mechanics, position=0)

    assert [(topic.title, topic.position) for topic in mechanics.children] == [
        ("Forces", 0),
        ("Motion", 1),
    ]
    topics.update(waves, position=0)
    assert [
        (topic.title, topic.position)
        for topic in topics.list_for_course(course.id)
        if topic.parent_id is None
    ] == [("Waves", 0), ("Mechanics", 1)]

    topics.delete(mechanics)
    assert topics.get(motion.id) is None
    assert topics.get(forces.id) is None


def test_topics_reject_cycles_and_sources_outside_course(session: Session) -> None:
    documents = DocumentRepository(session)
    attached = documents.add(Document(filename="inside.md", media_type="text/markdown"))
    outside = documents.add(Document(filename="outside.md", media_type="text/markdown"))
    chunks = ChunkRepository(session)
    inside_chunk = chunks.add_from_chunks(
        attached, TokenChunker(5, 0).chunk("inside source")
    )[0]
    outside_chunk = chunks.add_from_chunks(
        outside, TokenChunker(5, 0).chunk("outside source")
    )[0]
    courses = CourseRepository(session)
    course = courses.add(Course(title="Sources"))
    courses.attach_document(course, attached)
    topics = TopicRepository(session)
    parent = topics.add(course, title="Parent")
    child = topics.add(course, title="Child", parent=parent)

    topics.add_source(parent, inside_chunk)
    with pytest.raises(ValueError, match="not attached"):
        topics.add_source(parent, outside_chunk)
    with pytest.raises(ValueError, match="cycle"):
        topics.update(parent, parent=child)
