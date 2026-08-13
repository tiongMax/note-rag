from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event

from note_rag.persistence import Base, Database


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'api-tests.sqlite3'}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    database = Database(engine=engine)
    yield database
    database.dispose()
