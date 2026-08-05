from fastapi.testclient import TestClient

from note_rag.api.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chunk_text() -> None:
    response = client.post(
        "/api/v1/chunks",
        json={
            "text": "alpha beta gamma delta epsilon",
            "source_id": "note.txt",
            "chunk_size": 3,
            "chunk_overlap": 1,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "token_count": 5,
        "chunks": [
            {
                "text": "alpha beta gamma",
                "metadata": {
                    "index": 0,
                    "token_start": 0,
                    "token_end": 3,
                    "token_count": 3,
                    "char_start": 0,
                    "char_end": 16,
                    "source_id": "note.txt",
                },
            },
            {
                "text": "gamma delta epsilon",
                "metadata": {
                    "index": 1,
                    "token_start": 2,
                    "token_end": 5,
                    "token_count": 3,
                    "char_start": 11,
                    "char_end": 30,
                    "source_id": "note.txt",
                },
            },
        ],
    }


def test_rejects_overlap_equal_to_chunk_size() -> None:
    response = client.post(
        "/api/v1/chunks",
        json={"text": "some text", "chunk_size": 2, "chunk_overlap": 2},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "chunk_overlap must be smaller than chunk_size"
