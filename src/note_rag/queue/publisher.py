"""Publishes ingestion job IDs to the Redis stream."""

from __future__ import annotations

import uuid

from note_rag.queue.redis_stream import RedisStreamQueue


class QueuePublisher:
    """Writes a job-ID payload to the ingestion stream.

    The API service uses this after creating an ``IngestionJob`` row so
    that an ingestion worker can pick it up asynchronously.
    """

    def __init__(
        self,
        queue: RedisStreamQueue,
        stream: str,
    ) -> None:
        self._queue = queue
        self._stream = stream

    def enqueue(self, job_id: uuid.UUID) -> bytes:
        """Push *job_id* onto the stream and return the Redis message ID."""
        return self._queue.publish(self._stream, {"job_id": str(job_id)})
