"""SQLAlchemy engine and transaction boundaries."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from note_rag.persistence.settings import DatabaseSettings


class Database:
    """Own the engine and produce short-lived transactional sessions."""

    def __init__(
        self,
        settings: DatabaseSettings | None = None,
        *,
        engine: Engine | None = None,
    ) -> None:
        if engine is None:
            resolved = settings or DatabaseSettings.from_env()
            engine = create_engine(
                resolved.url,
                echo=resolved.echo,
                pool_pre_ping=True,
            )
        self.engine = engine
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Commit on success and roll back the entire unit on failure."""

        db_session = self._session_factory()
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    def dispose(self) -> None:
        self.engine.dispose()
