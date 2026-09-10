import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from note_rag.api.app import create_app
from note_rag.api.settings import ApiSettings
from note_rag.chunking import TokenChunker
from note_rag.ingest import LocalFileStorage
from note_rag.persistence import (
    ApprovalStatus,
    ChunkRepository,
    Course,
    CourseRepository,
    Database,
    Document,
    DocumentRepository,
    StudyItem,
    StudyItemRepository,
    StudyItemType,
    TopicRepository,
    TopicState,
)


class Clock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current


def test_complete_adaptive_study_workflow(database: Database, tmp_path: Path) -> None:
    with database.session() as session:
        document = DocumentRepository(session).add(
            Document(filename="cells.md", media_type="text/markdown")
        )
        chunk = ChunkRepository(session).add_from_chunks(
            document,
            TokenChunker(chunk_size=40, chunk_overlap=0).chunk(
                "Mitochondria generate ATP through cellular respiration."
            ),
        )[0]
        course = CourseRepository(session).add(Course(title="Biology"))
        CourseRepository(session).attach_document(course, document)
        topic = TopicRepository(session).add(
            course,
            title="Cell energy",
            state=TopicState.APPROVED,
            position=0,
        )
        specifications = [
            (
                StudyItemType.FLASHCARD,
                "What molecule stores cellular energy?",
                "ATP",
                [],
                2,
            ),
            (
                StudyItemType.MULTIPLE_CHOICE,
                "Which organelle generates ATP?",
                "Mitochondria",
                [
                    {"text": "Nucleus", "correct": False},
                    {"text": "Mitochondria", "correct": True},
                ],
                3,
            ),
            (
                StudyItemType.SHORT_ANSWER,
                "Explain what mitochondria generate.",
                "Mitochondria generate ATP",
                [],
                4,
            ),
        ]
        item_ids = []
        for item_type, prompt, answer, options, difficulty in specifications:
            study_item = StudyItemRepository(session).add(
                StudyItem(
                    topic=topic,
                    item_type=item_type,
                    prompt=prompt,
                    answer=answer,
                    explanation="Supported by the cited course passage.",
                    options=options,
                    difficulty=difficulty,
                    approval_status=ApprovalStatus.APPROVED,
                    fingerprint=hashlib.sha256(prompt.encode()).hexdigest(),
                ),
                [chunk.id],
            )
            item_ids.append(str(study_item.id))
        course_id = str(course.id)

    clock = Clock(datetime(2026, 9, 10, 12, tzinfo=UTC))
    client = TestClient(
        create_app(
            ApiSettings(
                storage_path=tmp_path,
                embedding_backend="deterministic",
                background_worker_enabled=False,
            ),
            database=database,
            storage=LocalFileStorage(tmp_path),
            clock=clock,
        )
    )

    queue = client.get(f"/api/v1/study/queue?course_id={course_id}")
    assert queue.status_code == 200
    assert [entry["difficulty"] for entry in queue.json()["items"]] == [4, 3, 2]
    assert all(
        entry["reason"] == "New approved item" for entry in queue.json()["items"]
    )

    created = client.post(
        "/api/v1/study/sessions",
        json={"course_id": course_id, "mode": "daily_review"},
    )
    assert created.status_code == 201
    session = created.json()
    assert len(session["items"]) == 3
    assert "answer" not in session["items"][0]
    session_id = session["id"]

    responses = []
    for index, question in enumerate(session["items"]):
        submitted = {
            "short_answer": "Mitochondria generate",
            "multiple_choice": "Nucleus",
            "flashcard": "ATP",
        }[question["item_type"]]
        payload = {
            "item_id": question["id"],
            "submitted_answer": submitted,
            "rating": "again" if question["item_type"] == "multiple_choice" else "good",
            "confidence": 3,
            "response_time_ms": 1200,
            "hint_used": False,
            "idempotency_key": f"answer-key-{index}",
        }
        response = client.post(
            f"/api/v1/study/sessions/{session_id}/answers", json=payload
        )
        assert response.status_code == 201
        responses.append(response.json())
        repeated = client.post(
            f"/api/v1/study/sessions/{session_id}/answers", json=payload
        )
        assert repeated.json()["id"] == response.json()["id"]

    short_answer = next(
        response for response in responses if response["study_item_id"] == item_ids[2]
    )
    assert short_answer["score"] > 0
    assert short_answer["sources"][0]["filename"] == "cells.md"
    assert short_answer["grading_details"]["missing_concepts"] == ["atp"]

    incorrect = next(response for response in responses if response["correct"] is False)
    overridden = client.post(
        f"/api/v1/review-attempts/{incorrect['id']}/override",
        json={"correct": True, "score": 1, "reason": "Accepted equivalent answer"},
    )
    assert overridden.status_code == 200
    assert overridden.json()["overridden_correct"] is True

    completed = client.post(f"/api/v1/study/sessions/{session_id}/complete")
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"

    clock.current += timedelta(days=7)
    memory = client.get(f"/api/v1/study-items/{item_ids[0]}/memory").json()
    assert memory["predicted_recall"] < 1
    updated_queue = client.get(f"/api/v1/study/queue?course_id={course_id}").json()
    assert "predicted recall" in updated_queue["items"][0]["reason"].lower()
