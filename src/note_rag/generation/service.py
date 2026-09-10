"""Citation-grounded curriculum and study-item generation."""

import hashlib
import json
import re
import uuid
from difflib import SequenceMatcher

from pydantic import ValidationError
from sqlalchemy import select

from note_rag.chat import ChatProvider, ChatTurn
from note_rag.generation.models import (
    GeneratedCurriculum,
    GeneratedStudyItem,
    GeneratedStudyItems,
)
from note_rag.persistence import (
    ApprovalStatus,
    ChunkRecord,
    Course,
    CourseDocument,
    Database,
    LearningObjective,
    StudyItem,
    StudyItemRepository,
    StudyItemType,
    Topic,
    TopicRepository,
    TopicState,
)

PROMPT_VERSION = "learning-materials-v1"
_SPACE = re.compile(r"\s+")


class LearningMaterialService:
    def __init__(self, database: Database, provider: ChatProvider) -> None:
        self.database = database
        self.provider = provider

    @property
    def generation_version(self) -> str:
        return f"{PROMPT_VERSION}:{self.provider.model_name}"

    def generate_curriculum(self, course_id: uuid.UUID) -> list[Topic]:
        sources = self._course_sources(course_id)
        if not sources:
            raise ValueError("course has no chunked source material")
        payload = self._generate(
            "Return a grounded curriculum as JSON only. Each topic needs a title, "
            "description, one or more learning objectives, and source_chunk_ids "
            "selected only from the supplied sources.",
            {"sources": sources},
            GeneratedCurriculum,
        )
        allowed = {uuid.UUID(str(source["chunk_id"])) for source in sources}
        for topic in payload.topics:
            self._validate_citations(topic.source_chunk_ids, allowed)

        created: list[Topic] = []
        with self.database.session() as session:
            course = session.get(Course, course_id)
            if course is None:
                raise LookupError("course not found")
            topics = TopicRepository(session)
            existing = {
                item.title.strip().casefold()
                for item in topics.list_for_course(course_id)
            }
            for generated in payload.topics:
                if generated.title.strip().casefold() in existing:
                    continue
                topic = topics.add(
                    course,
                    title=generated.title.strip(),
                    description=generated.description.strip(),
                    state=TopicState.DRAFT,
                )
                for chunk_id in generated.source_chunk_ids:
                    chunk = session.get(ChunkRecord, chunk_id)
                    if chunk is None:
                        raise ValueError("generated curriculum cited a missing chunk")
                    topics.add_source(topic, chunk)
                for position, objective in enumerate(generated.objectives):
                    session.add(
                        LearningObjective(
                            topic=topic,
                            title=objective.title.strip(),
                            description=objective.description.strip(),
                            position=position,
                            generation_version=self.generation_version,
                        )
                    )
                created.append(topic)
                existing.add(generated.title.strip().casefold())
            session.flush()
        return created

    def generate_study_items(
        self,
        topic_id: uuid.UUID,
        *,
        only_type: StudyItemType | None = None,
        exclude_item_id: uuid.UUID | None = None,
    ) -> list[StudyItem]:
        topic, sources = self._topic_sources(topic_id)
        if not sources:
            raise ValueError("topic has no supporting source passages")
        type_instruction = (
            only_type.value
            if only_type
            else "flashcard, multiple_choice, and short_answer"
        )
        payload = self._generate(
            "Return cited study items as JSON only. Generate types: "
            f"{type_instruction}. Every answer and explanation must be supported "
            "by source_chunk_ids. Multiple-choice options must contain exactly one "
            "correct option and answer must equal it.",
            {"topic": topic, "sources": sources},
            GeneratedStudyItems,
        )
        allowed = {uuid.UUID(str(source["chunk_id"])) for source in sources}
        for item in payload.items:
            self._validate_citations(item.source_chunk_ids, allowed)
            if only_type is not None and item.item_type is not only_type:
                raise ValueError("provider returned an unexpected study item type")
        return self._persist_items(
            topic_id, payload.items, exclude_item_id=exclude_item_id
        )

    def regenerate_item(self, item_id: uuid.UUID) -> StudyItem:
        with self.database.session() as session:
            existing = StudyItemRepository(session).get(item_id)
            if existing is None:
                raise LookupError("study item not found")
            topic_id, item_type = existing.topic_id, existing.item_type
        generated = self.generate_study_items(
            topic_id, only_type=item_type, exclude_item_id=item_id
        )
        if not generated:
            raise ValueError("regeneration produced only duplicate items")
        with self.database.session() as session:
            existing = StudyItemRepository(session).get(item_id)
            if (
                existing is not None
                and existing.approval_status is not ApprovalStatus.APPROVED
            ):
                existing.approval_status = ApprovalStatus.ARCHIVED
            return StudyItemRepository(session).get(generated[0].id) or generated[0]

    def _course_sources(self, course_id: uuid.UUID) -> list[dict[str, str | int]]:
        with self.database.session() as session:
            rows = list(
                session.scalars(
                    select(ChunkRecord)
                    .join(
                        CourseDocument,
                        CourseDocument.document_id == ChunkRecord.document_id,
                    )
                    .where(CourseDocument.course_id == course_id)
                    .order_by(ChunkRecord.document_id, ChunkRecord.position)
                )
            )
            selected = self._representative(rows, 24)
            return [
                {"chunk_id": str(row.id), "position": row.position, "text": row.text}
                for row in selected
            ]

    def _topic_sources(
        self, topic_id: uuid.UUID
    ) -> tuple[dict[str, str], list[dict[str, str | int]]]:
        with self.database.session() as session:
            topic = session.get(Topic, topic_id)
            if topic is None:
                raise LookupError("topic not found")
            chunks = [source.chunk for source in topic.sources]
            return (
                {
                    "id": str(topic.id),
                    "title": topic.title,
                    "description": topic.description,
                },
                [
                    {
                        "chunk_id": str(chunk.id),
                        "position": chunk.position,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
            )

    def _persist_items(
        self,
        topic_id: uuid.UUID,
        generated: list[GeneratedStudyItem],
        *,
        exclude_item_id: uuid.UUID | None = None,
    ) -> list[StudyItem]:
        created: list[StudyItem] = []
        with self.database.session() as session:
            repository = StudyItemRepository(session)
            existing = repository.list_for_topic(topic_id, include_archived=True)
            prompts = [
                self._normalize(item.prompt)
                for item in existing
                if item.id != exclude_item_id
            ]
            for value in generated:
                normalized = self._normalize(value.prompt)
                if any(
                    SequenceMatcher(None, normalized, prompt).ratio() >= 0.9
                    for prompt in prompts
                ):
                    continue
                item = repository.add(
                    StudyItem(
                        topic_id=topic_id,
                        item_type=value.item_type,
                        prompt=value.prompt.strip(),
                        answer=value.answer.strip(),
                        explanation=value.explanation.strip(),
                        difficulty=value.difficulty,
                        options=[option.model_dump() for option in value.options],
                        approval_status=ApprovalStatus.DRAFT,
                        generation_version=self.generation_version,
                        fingerprint=hashlib.sha256(normalized.encode()).hexdigest(),
                    ),
                    value.source_chunk_ids,
                )
                created.append(item)
                prompts.append(normalized)
        return created

    def _generate(self, instruction: str, data: dict[str, object], contract):
        raw = self.provider.generate(
            instruction,
            [ChatTurn(role="user", content=json.dumps(data, ensure_ascii=False))],
        ).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return contract.model_validate_json(raw)
        except (ValidationError, ValueError) as error:
            raise ValueError(
                "provider returned invalid structured learning material"
            ) from error

    @staticmethod
    def _validate_citations(
        citations: list[uuid.UUID], allowed: set[uuid.UUID]
    ) -> None:
        if not citations or not set(citations) <= allowed:
            raise ValueError("generated content contains invalid source citations")

    @staticmethod
    def _representative(rows: list[ChunkRecord], limit: int) -> list[ChunkRecord]:
        if len(rows) <= limit:
            return rows
        return [
            rows[round(index * (len(rows) - 1) / (limit - 1))] for index in range(limit)
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        return _SPACE.sub(" ", text.strip().casefold())
