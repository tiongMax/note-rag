"""Dialect-aware vector, PostgreSQL FTS, and persisted BM25 retrieval."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import (
    Float,
    Select,
    cast,
    distinct,
    func,
    literal,
    literal_column,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from note_rag.chunking import lexical_terms
from note_rag.persistence import ChunkLexicalTerm, ChunkRecord, Document
from note_rag.retrieval.models import SearchFilters


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

    def bm25_search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters,
        k1: float,
        b: float,
    ) -> list[RankedChunk]:
        """Rank filtered chunks with exact Okapi BM25 corpus statistics."""

        query_terms = sorted(set(lexical_terms(query)))
        if not query_terms:
            return []
        if self._is_postgresql:
            return self._postgres_bm25_search(
                query_terms,
                limit=limit,
                filters=filters,
                k1=k1,
                b=b,
            )
        return self._python_bm25_search(
            query_terms,
            limit=limit,
            filters=filters,
            k1=k1,
            b=b,
        )

    def _postgres_bm25_search(
        self,
        query_terms: list[str],
        *,
        limit: int,
        filters: SearchFilters,
        k1: float,
        b: float,
    ) -> list[RankedChunk]:
        filtered = (
            select(
                ChunkRecord.id.label("chunk_id"),
                ChunkRecord.lexical_token_count.label("document_length"),
            )
            .join(Document, ChunkRecord.document_id == Document.id)
            .where(ChunkRecord.lexical_token_count > 0)
        )
        filtered = self._apply_filters(filtered, filters).cte("filtered_chunks")
        corpus = select(
            func.count(filtered.c.chunk_id).label("document_count"),
            func.avg(filtered.c.document_length).label("average_length"),
        ).cte("bm25_corpus")
        document_frequency = (
            select(
                ChunkLexicalTerm.term.label("term"),
                func.count(distinct(ChunkLexicalTerm.chunk_id)).label(
                    "document_frequency"
                ),
            )
            .join(
                filtered,
                filtered.c.chunk_id == ChunkLexicalTerm.chunk_id,
            )
            .where(ChunkLexicalTerm.term.in_(query_terms))
            .group_by(ChunkLexicalTerm.term)
            .cte("bm25_document_frequency")
        )
        length_ratio = cast(filtered.c.document_length, Float) / func.nullif(
            corpus.c.average_length,
            0,
        )
        inverse_document_frequency = func.ln(
            1
            + (
                corpus.c.document_count
                - document_frequency.c.document_frequency
                + 0.5
            )
            / (document_frequency.c.document_frequency + 0.5)
        )
        frequency = cast(ChunkLexicalTerm.term_frequency, Float)
        denominator = frequency + k1 * (1 - b + b * length_ratio)
        term_score = inverse_document_frequency * (
            frequency * (k1 + 1) / denominator
        )
        score = func.sum(term_score).label("score")
        statement = (
            select(ChunkRecord, Document.filename, Document.media_type, score)
            .join(Document, ChunkRecord.document_id == Document.id)
            .join(filtered, filtered.c.chunk_id == ChunkRecord.id)
            .join(ChunkLexicalTerm, ChunkLexicalTerm.chunk_id == ChunkRecord.id)
            .join(
                document_frequency,
                document_frequency.c.term == ChunkLexicalTerm.term,
            )
            .join(corpus, literal(True))
            .group_by(ChunkRecord.id, Document.filename, Document.media_type)
            .order_by(score.desc(), ChunkRecord.id)
            .limit(limit)
        )
        return [
            RankedChunk(chunk, filename, media_type, float(row_score))
            for chunk, filename, media_type, row_score in self.session.execute(
                statement
            )
        ]

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
        query_terms = set(lexical_terms(query))
        ranked = []
        for chunk, filename, media_type in self._base_rows(filters):
            terms = lexical_terms(chunk.text)
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

    def _python_bm25_search(
        self,
        query_terms: list[str],
        *,
        limit: int,
        filters: SearchFilters,
        k1: float,
        b: float,
    ) -> list[RankedChunk]:
        rows = self._base_rows(filters)
        documents = []
        document_frequency: Counter[str] = Counter()
        for chunk, filename, media_type in rows:
            frequencies = Counter(lexical_terms(chunk.text))
            document_frequency.update(frequencies.keys())
            documents.append((chunk, filename, media_type, frequencies))
        if not documents:
            return []
        average_length = sum(
            sum(frequencies.values()) for *_, frequencies in documents
        ) / len(documents)
        ranked = []
        for chunk, filename, media_type, frequencies in documents:
            length = sum(frequencies.values())
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (
                    1 - b + b * (length / average_length if average_length else 0)
                )
                score += idf * (frequency * (k1 + 1) / denominator)
            if score:
                ranked.append(RankedChunk(chunk, filename, media_type, score))
        ranked.sort(key=lambda item: (-item.score, str(item.chunk.id)))
        return ranked[:limit]
