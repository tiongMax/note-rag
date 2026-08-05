"""Dialect-aware vector and keyword candidate retrieval."""

import math
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, func, literal, literal_column, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from note_rag.persistence import ChunkRecord, Document
from note_rag.retrieval.models import SearchFilters

_TERM_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: ChunkRecord
    filename: str
    media_type: str
    score: float


class RetrievalRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def vector_search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[RankedChunk]:
        if self._is_postgresql:
            distance = ChunkRecord.embedding.cosine_distance(query_vector)
            score = (1.0 - distance).label("score")
            statement = (
                select(ChunkRecord, Document.filename, Document.media_type, score)
                .join(Document, ChunkRecord.document_id == Document.id)
                .where(ChunkRecord.embedding.is_not(None))
            )
            statement = self._apply_filters(statement, filters)
            rows = self.session.execute(
                statement.order_by(distance, ChunkRecord.id).limit(limit)
            )
            return [
                RankedChunk(chunk, filename, media_type, float(row_score))
                for chunk, filename, media_type, row_score in rows
            ]
        return self._python_vector_search(query_vector, limit=limit, filters=filters)

    def keyword_search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[RankedChunk]:
        if self._is_postgresql:
            language = literal_column("'english'::regconfig")
            document_vector = func.to_tsvector(language, ChunkRecord.text)
            query_vector = func.plainto_tsquery(language, query)
            score = func.ts_rank_cd(document_vector, query_vector).label("score")
            statement = (
                select(ChunkRecord, Document.filename, Document.media_type, score)
                .join(Document, ChunkRecord.document_id == Document.id)
                .where(document_vector.bool_op("@@")(query_vector))
            )
            statement = self._apply_filters(statement, filters)
            rows = self.session.execute(
                statement.order_by(score.desc(), ChunkRecord.id).limit(limit)
            )
            return [
                RankedChunk(chunk, filename, media_type, float(row_score))
                for chunk, filename, media_type, row_score in rows
            ]
        return self._python_keyword_search(query, limit=limit, filters=filters)

    def _base_rows(self, filters: SearchFilters) -> list[tuple[ChunkRecord, str, str]]:
        statement = (
            select(ChunkRecord, Document.filename, Document.media_type)
            .join(Document, ChunkRecord.document_id == Document.id)
        )
        statement = self._apply_filters(statement, filters)
        rows = [
            (row[0], row[1], row[2])
            for row in self.session.execute(statement)
        ]
        if filters.source_metadata and not self._is_postgresql:
            rows = [
                row
                for row in rows
                if all(
                    row[0].source_metadata.get(key) == value
                    for key, value in filters.source_metadata.items()
                )
            ]
        return rows

    def _apply_filters(
        self,
        statement: Select[Any],
        filters: SearchFilters,
    ) -> Select[Any]:
        if filters.document_ids:
            statement = statement.where(
                ChunkRecord.document_id.in_(filters.document_ids)
            )
        if filters.filenames:
            statement = statement.where(Document.filename.in_(filters.filenames))
        if filters.media_types:
            statement = statement.where(Document.media_type.in_(filters.media_types))
        if filters.source_metadata and self._is_postgresql:
            statement = statement.where(
                ChunkRecord.source_metadata.bool_op("@>")(
                    literal(filters.source_metadata, type_=JSONB)
                )
            )
        return statement

    @property
    def _is_postgresql(self) -> bool:
        return (
            self.session.bind is not None
            and self.session.bind.dialect.name == "postgresql"
        )

    def _python_vector_search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[RankedChunk]:
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        ranked = []
        for chunk, filename, media_type in self._base_rows(filters):
            if chunk.embedding is None:
                continue
            vector = list(chunk.embedding)
            vector_norm = math.sqrt(sum(value * value for value in vector))
            score = (
                sum(
                    left * right
                    for left, right in zip(query_vector, vector, strict=True)
                )
                / (query_norm * vector_norm)
                if query_norm and vector_norm
                else 0.0
            )
            ranked.append(RankedChunk(chunk, filename, media_type, score))
        ranked.sort(key=lambda item: (-item.score, str(item.chunk.id)))
        return ranked[:limit]

    def _python_keyword_search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[RankedChunk]:
        query_terms = set(_terms(query))
        ranked = []
        for chunk, filename, media_type in self._base_rows(filters):
            terms = _terms(chunk.text)
            if not terms:
                continue
            matches = sum(term in query_terms for term in terms)
            if matches:
                ranked.append(
                    RankedChunk(
                        chunk,
                        filename,
                        media_type,
                        matches / len(terms),
                    )
                )
        ranked.sort(key=lambda item: (-item.score, str(item.chunk.id)))
        return ranked[:limit]


def _terms(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TERM_PATTERN.finditer(text)]
