import pytest
from sqlalchemy import select

from note_rag.persistence import Database, Document


def test_session_commits_on_success(database: Database) -> None:
    with database.session() as session:
        document = Document(filename="notes.txt", media_type="text/plain")
        session.add(document)
        session.flush()
        document_id = document.id

    with database.session() as session:
        stored = session.get(Document, document_id)
        assert stored is not None
        assert stored.filename == "notes.txt"


def test_session_rolls_back_on_failure(database: Database) -> None:
    with pytest.raises(RuntimeError):
        with database.session() as session:
            session.add(Document(filename="discard.txt", media_type="text/plain"))
            raise RuntimeError("stop transaction")

    with database.session() as session:
        assert list(session.scalars(select(Document))) == []
