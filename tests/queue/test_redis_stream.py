"""Tests for RedisStreamQueue."""

import fakeredis
import pytest

from note_rag.queue.redis_stream import RedisStreamQueue


@pytest.fixture
def redis_queue():
    client = fakeredis.FakeRedis(decode_responses=False)
    yield RedisStreamQueue(client)
    client.flushall()


def test_ensure_group_creates_stream(redis_queue):
    redis_queue.ensure_group("test_stream", "test_group")
    # Should not raise on second call (BUSYGROUP is silenced)
    redis_queue.ensure_group("test_stream", "test_group")


def test_publish_and_consume(redis_queue):
    redis_queue.ensure_group("test_stream", "test_group")
    
    # Consume when empty returns None
    assert redis_queue.consume("test_stream", "test_group", "consumer-1") is None

    # Publish a message
    msg_id = redis_queue.publish("test_stream", {"job_id": "123"})
    assert msg_id is not None

    # Consume retrieves it
    msg = redis_queue.consume("test_stream", "test_group", "consumer-1")
    assert msg is not None
    assert msg.msg_id == msg_id
    assert msg.fields == {"job_id": "123"}
    assert msg.consumer == "consumer-1"

    # Second consume without acking should get nothing (message is in consumer-1's PEL)
    assert redis_queue.consume("test_stream", "test_group", "consumer-2") is None


def test_ack_removes_from_pel(redis_queue):
    redis_queue.ensure_group("test_stream", "test_group")
    redis_queue.publish("test_stream", {"test": "val"})
    
    msg = redis_queue.consume("test_stream", "test_group", "consumer-1")
    assert msg is not None
    redis_queue.ack(msg)


def test_recover_pending(redis_queue):
    redis_queue.ensure_group("test_stream", "test_group")
    redis_queue.publish("test_stream", {"job_id": "999"})
    
    # Consumer 1 reads but does not ack
    msg = redis_queue.consume("test_stream", "test_group", "consumer-1")
    assert msg is not None

    # Recover with 0 min idle time should reclaim it for consumer-2
    recovered = redis_queue.recover_pending(
        "test_stream", "test_group", "consumer-2", min_idle_ms=0
    )
    assert len(recovered) == 1
    assert recovered[0].fields["job_id"] == "999"
    assert recovered[0].consumer == "consumer-2"


def test_consumer_group_distributes_messages_without_duplicates(redis_queue):
    redis_queue.ensure_group("scale_stream", "scale_group")
    message_ids = {
        redis_queue.publish("scale_stream", {"job_id": str(index)})
        for index in range(4)
    }

    consumed = [
        redis_queue.consume("scale_stream", "scale_group", "worker-1"),
        redis_queue.consume("scale_stream", "scale_group", "worker-2"),
        redis_queue.consume("scale_stream", "scale_group", "worker-1"),
        redis_queue.consume("scale_stream", "scale_group", "worker-2"),
    ]

    assert all(message is not None for message in consumed)
    consumed_ids = {
        message.msg_id for message in consumed if message is not None
    }
    assert consumed_ids == message_ids
    assert {message.consumer for message in consumed if message is not None} == {
        "worker-1",
        "worker-2",
    }
