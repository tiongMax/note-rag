"""Tests for QueuePublisher and QueueConsumer."""

import uuid
from unittest.mock import Mock

from note_rag.queue.consumer import QueueConsumer
from note_rag.queue.publisher import QueuePublisher
from note_rag.queue.redis_stream import RedisMsg


def test_queue_publisher():
    queue = Mock()
    queue.publish.return_value = b"123-0"
    publisher = QueuePublisher(queue, "test:stream")
    
    job_id = uuid.uuid4()
    msg_id = publisher.enqueue(job_id)
    
    assert msg_id == b"123-0"
    queue.publish.assert_called_once_with("test:stream", {"job_id": str(job_id)})


def test_queue_consumer_start_recovers_pending():
    queue = Mock()
    job_id = uuid.uuid4()
    recovered = RedisMsg(
        stream="stream1",
        group="grp1",
        consumer="con1",
        msg_id=b"1-0",
        fields={"job_id": str(job_id)},
    )
    queue.recover_pending.return_value = [recovered]
    
    consumer = QueueConsumer(
        queue,
        stream="stream1",
        group="grp1",
        consumer="con1",
        recovery_idle_seconds=0,
    )
    consumer.start()
    
    queue.ensure_group.assert_called_once_with("stream1", "grp1")
    queue.recover_pending.assert_called_once_with(
        "stream1", "grp1", "con1", min_idle_ms=0
    )
    assert consumer.poll() == (job_id, recovered)
    queue.consume.assert_not_called()


def test_queue_consumer_defers_retry_until_it_is_ready():
    queue = Mock()
    job_id = uuid.uuid4()
    msg = RedisMsg("s", "g", "c", b"1", {"job_id": str(job_id)})
    consumer = QueueConsumer(queue, "s", "g", "c")

    consumer.defer(msg, delay_seconds=0)

    assert consumer.poll() == (job_id, msg)
    queue.consume.assert_not_called()


def test_queue_consumer_poll_valid():
    queue = Mock()
    job_id = uuid.uuid4()
    msg = RedisMsg(
        stream="s", group="g", consumer="c", msg_id=b"1", fields={"job_id": str(job_id)}
    )
    queue.consume.return_value = msg
    
    consumer = QueueConsumer(queue, "s", "g", "c")
    result = consumer.poll()
    
    assert result is not None
    assert result[0] == job_id
    assert result[1] == msg
    queue.ack.assert_not_called()  # Caller must ack


def test_queue_consumer_poll_discards_malformed():
    queue = Mock()
    # Missing job_id field entirely
    msg = RedisMsg(
        stream="s", group="g", consumer="c", msg_id=b"1", fields={"wrong": "123"}
    )
    queue.consume.return_value = msg
    
    consumer = QueueConsumer(queue, "s", "g", "c")
    result = consumer.poll()
    
    assert result is None
    # Ensure it acks the malformed message so we don't get stuck on it
    queue.ack.assert_called_once_with(msg)
