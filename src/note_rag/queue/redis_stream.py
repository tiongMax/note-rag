"""Low-level Redis Streams wrapper (mirrors RAGFlow's RedisDB pattern)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import redis

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedisMsg:
    """A single message delivered from a Redis Stream consumer group."""

    stream: str
    group: str
    consumer: str
    msg_id: bytes
    fields: dict[str, str]


class RedisStreamQueue:
    """Thin wrapper around redis-py that exposes only what note-rag needs.

    All stream operations use consumer groups so that:
    - Messages survive worker crashes (unacked → re-delivered after timeout).
    - Multiple worker replicas can share load without double-processing.
    """

    _BLOCK_MS = 5_000  # block up to 5 s on each XREADGROUP call

    def __init__(self, client: redis.Redis) -> None:  # type: ignore[type-arg]
        self._r = client

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------

    def ensure_group(self, stream: str, group: str) -> None:
        """Create the consumer group (and stream) if they don't exist.

        Safe to call on every worker start — BUSYGROUP is silenced.
        """
        try:
            self._r.xgroup_create(stream, group, id="0", mkstream=True)
            logger.info("Created consumer group %s on stream %s", group, stream)
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def publish(self, stream: str, payload: dict[str, Any]) -> bytes:
        """Append *payload* to *stream* and return the assigned message ID."""
        msg_id = self._r.xadd(stream, {k: str(v) for k, v in payload.items()})
        return msg_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
    ) -> RedisMsg | None:
        """Fetch the next undelivered message for *consumer* in *group*.

        Blocks for up to *_BLOCK_MS* milliseconds waiting for a message.
        Returns ``None`` if the block times out with no message.
        """
        results = self._r.xreadgroup(
            group,
            consumer,
            {stream: ">"},
            count=1,
            block=self._BLOCK_MS,
        )
        if not results:
            return None
        _stream, messages = results[0]
        msg_id, fields = messages[0]
        return RedisMsg(
            stream=stream,
            group=group,
            consumer=consumer,
            msg_id=msg_id,
            fields={k.decode(): v.decode() for k, v in fields.items()},
        )

    def ack(self, msg: RedisMsg) -> None:
        """Acknowledge *msg* so Redis removes it from the PEL."""
        self._r.xack(msg.stream, msg.group, msg.msg_id)

    # ------------------------------------------------------------------
    # Crash recovery — mirrors RAGFlow's get_unacked_iterator
    # ------------------------------------------------------------------

    def recover_pending(
        self,
        stream: str,
        group: str,
        consumer: str,
        *,
        min_idle_ms: int = 60_000,
    ) -> list[RedisMsg]:
        """Claim pending messages that have been idle for *min_idle_ms* ms.

        Call this on worker startup to reclaim any messages that were
        in-flight when the previous worker instance crashed.
        """
        # Inspect the entire group's PEL. Filtering by the new consumer name
        # only finds messages that are already owned by that consumer and can
        # never recover work left behind by a crashed replica.
        pending = self._r.xpending_range(
            stream,
            group,
            min="-",
            max="+",
            count=100,
        )
        if not pending:
            return []

        msg_ids = [entry["message_id"] for entry in pending]
        claimed = self._r.xclaim(
            stream,
            group,
            consumer,
            min_idle_time=min_idle_ms,
            message_ids=msg_ids,
        )
        recovered: list[RedisMsg] = []
        for msg_id, fields in claimed:
            recovered.append(
                RedisMsg(
                    stream=stream,
                    group=group,
                    consumer=consumer,
                    msg_id=msg_id,
                    fields={k.decode(): v.decode() for k, v in fields.items()},
                )
            )
        if recovered:
            logger.info(
                "Recovered %d pending message(s) on stream %s",
                len(recovered),
                stream,
            )
        return recovered
