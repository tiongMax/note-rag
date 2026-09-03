"""Blocking consumer loop for the ingestion stream."""

from __future__ import annotations

import logging
import uuid

from note_rag.queue.redis_stream import RedisMsg, RedisStreamQueue

logger = logging.getLogger(__name__)


class QueueConsumer:
    """Wraps ``RedisStreamQueue`` to provide a typed, job-ID-oriented poll.

    The worker service calls ``poll()`` in a tight loop.  Each call blocks
    for up to 5 seconds (the XREADGROUP block timeout) and then returns
    either a ``(job_id, msg)`` pair or ``None``.

    The caller is responsible for calling ``ack(msg)`` after successful
    processing so Redis can remove the message from the pending-entries list.
    """

    def __init__(
        self,
        queue: RedisStreamQueue,
        stream: str,
        group: str,
        consumer: str,
    ) -> None:
        self._queue = queue
        self._stream = stream
        self._group = group
        self._consumer = consumer

    def start(self) -> None:
        """Ensure the consumer group exists and recover any pending messages."""
        self._queue.ensure_group(self._stream, self._group)
        pending = self._queue.recover_pending(
            self._stream,
            self._group,
            self._consumer,
        )
        if pending:
            logger.info(
                "Re-queued %d pending message(s) for processing", len(pending)
            )

    def poll(self) -> tuple[uuid.UUID, RedisMsg] | None:
        """Block up to 5 s and return the next ``(job_id, msg)``, or ``None``."""
        msg = self._queue.consume(self._stream, self._group, self._consumer)
        if msg is None:
            return None
        raw_id = msg.fields.get("job_id", "")
        try:
            job_id = uuid.UUID(raw_id)
        except (ValueError, AttributeError):
            logger.error("Received malformed job_id %r — discarding", raw_id)
            self._queue.ack(msg)
            return None
        return job_id, msg

    def ack(self, msg: RedisMsg) -> None:
        """Acknowledge *msg* — delegates to the underlying stream queue."""
        self._queue.ack(msg)
