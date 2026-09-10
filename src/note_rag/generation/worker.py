"""Durable worker for curriculum and study-item generation jobs."""

import threading
import uuid

from note_rag.generation.service import LearningMaterialService
from note_rag.persistence import Database, GenerationJobRepository, GenerationKind


class GenerationWorker:
    def __init__(
        self,
        database: Database,
        service: LearningMaterialService,
        *,
        max_attempts: int = 3,
        poll_interval: float = 1.0,
    ) -> None:
        self.database = database
        self.service = service
        self.max_attempts = max_attempts
        self.poll_interval = poll_interval

    def run_once(self) -> bool:
        with self.database.session() as session:
            job = GenerationJobRepository(session).claim_next()
            job_id = job.id if job else None
        if job_id is None:
            return False
        self.run_job(job_id)
        return True

    def run_job(self, job_id: uuid.UUID) -> None:
        with self.database.session() as session:
            repository = GenerationJobRepository(session)
            job = repository.get(job_id)
            if job is None:
                raise LookupError("generation job not found")
            if job.status.value == "queued":
                job = repository.claim(job_id)
            if job is None:
                return
            kind, course_id, topic_id, item_id = (
                job.kind,
                job.course_id,
                job.topic_id,
                job.item_id,
            )
        try:
            if kind is GenerationKind.CURRICULUM:
                self.service.generate_curriculum(course_id)
            elif kind is GenerationKind.STUDY_ITEMS and topic_id is not None:
                self.service.generate_study_items(topic_id)
            elif kind is GenerationKind.STUDY_ITEM and item_id is not None:
                self.service.regenerate_item(item_id)
            else:
                raise ValueError("generation job target is invalid")
        except Exception as error:
            with self.database.session() as session:
                job = GenerationJobRepository(session).get(job_id)
                if job is not None:
                    GenerationJobRepository(session).fail(
                        job,
                        str(error) or error.__class__.__name__,
                        retry=job.attempts < self.max_attempts,
                    )
            return
        with self.database.session() as session:
            job = GenerationJobRepository(session).get(job_id)
            if job is not None:
                GenerationJobRepository(session).complete(job)

    def run_forever(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if not self.run_once():
                stop_event.wait(self.poll_interval)
