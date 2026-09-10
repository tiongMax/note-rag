import json
from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

from note_rag.api.app import create_app
from note_rag.api.settings import ApiSettings
from note_rag.chat import ChatTurn
from note_rag.chunking import TokenChunker
from note_rag.ingest import LocalFileStorage
from note_rag.persistence import (
    ChunkRepository,
    Course,
    CourseRepository,
    Database,
    Document,
    DocumentRepository,
)


class LearningProvider:
    model_name = "fake-learning"

    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id

    def generate(self, system_instruction: str, turns: list[ChatTurn]) -> str:
        del turns
        if "curriculum" in system_instruction:
            return json.dumps(
                {
                    "topics": [
                        {
                            "title": "Cell energy",
                            "objectives": [{"title": "Explain ATP"}],
                            "source_chunk_ids": [self.chunk_id],
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "items": [
                    {
                        "item_type": "flashcard",
                        "prompt": "What is produced?",
                        "answer": "ATP",
                        "explanation": "The cited passage says ATP.",
                        "difficulty": 2,
                        "source_chunk_ids": [self.chunk_id],
                    }
                ]
            }
        )

    def stream(self, system_instruction: str, turns: list[ChatTurn]) -> Iterator[str]:
        del system_instruction, turns
        return iter(())


def test_generated_material_review_workflow(database: Database, tmp_path: Path) -> None:
    with database.session() as session:
        document = DocumentRepository(session).add(
            Document(filename="cells.md", media_type="text/markdown")
        )
        chunk = ChunkRepository(session).add_from_chunks(
            document,
            TokenChunker(chunk_size=20, chunk_overlap=0).chunk(
                "Mitochondria produce ATP."
            ),
        )[0]
        course = CourseRepository(session).add(Course(title="Biology"))
        CourseRepository(session).attach_document(course, document)
        course_id, chunk_id = str(course.id), str(chunk.id)
    client = TestClient(
        create_app(
            ApiSettings(
                storage_path=tmp_path,
                embedding_backend="deterministic",
                background_worker_enabled=False,
            ),
            database=database,
            storage=LocalFileStorage(tmp_path),
            chat_provider=LearningProvider(chunk_id),
        )
    )

    job = client.post(
        f"/api/v1/courses/{course_id}/generate-curriculum",
        json={"idempotency_key": "curriculum-test-key"},
    )
    assert job.status_code == 202
    assert job.json()["status"] == "completed"
    repeated = client.post(
        f"/api/v1/courses/{course_id}/generate-curriculum",
        json={"idempotency_key": "curriculum-test-key"},
    )
    assert repeated.json()["id"] == job.json()["id"]
    topic = client.get(f"/api/v1/courses/{course_id}/topics").json()[0]
    assert topic["state"] == "draft"

    item_job = client.post(
        f"/api/v1/courses/{course_id}/topics/{topic['id']}/generate-items",
        json={},
    )
    assert item_job.json()["status"] == "completed"
    item = client.get(
        f"/api/v1/courses/{course_id}/topics/{topic['id']}/study-items"
    ).json()[0]
    assert item["approval_status"] == "draft"
    assert item["sources"][0]["chunk_id"] == chunk_id

    approved = client.patch(
        f"/api/v1/study-items/{item['id']}",
        json={"answer": "ATP", "approval_status": "approved"},
    )
    assert approved.status_code == 200
    assert approved.json()["approval_status"] == "approved"

    assert client.delete(f"/api/v1/study-items/{item['id']}").status_code == 204
    assert (
        client.get(
            f"/api/v1/courses/{course_id}/topics/{topic['id']}/study-items"
        ).json()
        == []
    )
