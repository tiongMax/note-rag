"""Blocking consumer loop for the ingestion stream."""

from __future__ import annotations

import heapq
import itertools
import logging
import time
import uuid
from collections import deque

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
        *,
        recovery_idle_seconds: float = 300.0,
        recovery_interval_seconds: float = 5.0,
    ) -> None:
        if recovery_idle_seconds < 0:
            raise ValueError("recovery_idle_seconds cannot be negative")
        if recovery_interval_seconds <= 0:
            raise ValueError("recovery_interval_seconds must be greater than zero")
        self._queue = queue
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._recovery_idle_ms = int(recovery_idle_seconds * 1000)
        self._recovery_interval_seconds = recovery_interval_seconds
        self._ready: deque[RedisMsg] = deque()
        self._deferred: list[tuple[float, int, RedisMsg]] = []
        self._sequence = itertools.count()
        self._last_recovery_at = time.monotonic()

    def start(self) -> None:
        """Ensure the consumer group exists and recover any pending messages."""
        self._queue.ensure_group(self._stream, self._group)
        pending = self._recover_pending()
        self._ready.extend(pending)
        if pending:
            logger.info(
                "Re-queued %d pending message(s) for processing", len(pending)
            )

    def poll(self) -> tuple[uuid.UUID, RedisMsg] | None:
        """Block up to 5 s and return the next ``(job_id, msg)``, or ``None``."""
        self._promote_deferred()
        self._recover_if_due()
        if self._ready:
            return self._decode(self._ready.popleft())
        msg = self._queue.consume(self._stream, self._group, self._consumer)
        if msg is None:
            return None
        return self._decode(msg)

    def defer(self, msg: RedisMsg, *, delay_seconds: float) -> None:
        """Make an unacknowledged message locally available after a delay."""
        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative")
        heapq.heappush(
            self._deferred,
            (time.monotonic() + delay_seconds, next(self._sequence), msg),
        )

    def _decode(self, msg: RedisMsg) -> tuple[uuid.UUID, RedisMsg] | None:
        raw_id = msg.fields.get("job_id", "")
        try:
            job_id = uuid.UUID(raw_id)
        except (ValueError, AttributeError):
            logger.error("Received malformed job_id %r — discarding", raw_id)
            self._queue.ack(msg)
            return None
        return job_id, msg

    def _promote_deferred(self) -> None:
        now = time.monotonic()
        while self._deferred and self._deferred[0][0] <= now:
            _ready_at, _sequence, msg = heapq.heappop(self._deferred)
            self._ready.append(msg)

    def _recover_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_recovery_at < self._recovery_interval_seconds:
            return
        self._ready.extend(self._recover_pending())

    def _recover_pending(self) -> list[RedisMsg]:
        self._last_recovery_at = time.monotonic()
        return self._queue.recover_pending(
            self._stream,
            self._group,
            self._consumer,
            min_idle_ms=self._recovery_idle_ms,
        )

    def ack(self, msg: RedisMsg) -> None:
        """Acknowledge *msg* — delegates to the underlying stream queue."""
        self._queue.ack(msg)
