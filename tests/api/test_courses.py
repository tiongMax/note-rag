from pathlib import Path

from fastapi.testclient import TestClient

from note_rag.api.app import create_app
from note_rag.api.settings import ApiSettings
from note_rag.ingest import LocalFileStorage
from note_rag.persistence import Database, Document, DocumentRepository


def build_client(database: Database, tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            ApiSettings(
                storage_path=tmp_path,
                embedding_backend="deterministic",
                background_worker_enabled=False,
            ),
            database=database,
            storage=LocalFileStorage(tmp_path),
        )
    )


def test_course_document_and_topic_workflow(database: Database, tmp_path: Path) -> None:
    with database.session() as session:
        document = DocumentRepository(session).add(
            Document(filename="lesson.md", media_type="text/markdown")
        )
        document_id = str(document.id)
    client = build_client(database, tmp_path)

    created = client.post(
        "/api/v1/courses", json={"title": "Biology", "description": "Cells"}
    )
    assert created.status_code == 201
    course_id = created.json()["id"]
    attached = client.post(
        f"/api/v1/courses/{course_id}/documents", json={"document_ids": [document_id]}
    )
    assert attached.status_code == 200
    assert attached.json()["document_ids"] == [document_id]

    parent = client.post(
        f"/api/v1/courses/{course_id}/topics",
        json={"title": "Cells", "state": "approved"},
    )
    child = client.post(
        f"/api/v1/courses/{course_id}/topics",
        json={"title": "Organelles", "parent_id": parent.json()["id"]},
    )
    assert parent.status_code == 201
    assert child.status_code == 201
    assert child.json()["parent_id"] == parent.json()["id"]

    cycle = client.patch(
        f"/api/v1/courses/{course_id}/topics/{parent.json()['id']}",
        json={"parent_id": child.json()["id"]},
    )
    assert cycle.status_code == 422
    assert "cycle" in cycle.json()["error"]["message"]

    deleted = client.delete(f"/api/v1/courses/{course_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/documents/{document_id}").status_code == 200


def test_course_assignment_validates_references_atomically(
    database: Database, tmp_path: Path
) -> None:
    client = build_client(database, tmp_path)
    course = client.post("/api/v1/courses", json={"title": "Course"}).json()

    response = client.post(
        f"/api/v1/courses/{course['id']}/documents",
        json={"document_ids": ["00000000-0000-0000-0000-000000000000"]},
    )

    assert response.status_code == 404
    assert client.get(f"/api/v1/courses/{course['id']}").json()["document_ids"] == []
