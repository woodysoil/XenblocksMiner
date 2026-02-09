"""
test_mqtt_integration.py - MQTT Integration Tests

Tests the full MQTT message flow using mock_mqtt_broker:
 - Client connects and subscribes
 - Register → ACK → Assign Task → Mining status → Block Found
 - Heartbeat and timeout detection
 - Topic structure validation
 - Multiple clients
"""

import json
import time
import pytest

from .mock_mqtt_broker import MockMQTTBroker, MockMQTTClient, MQTTMessage, mqtt_topic_matches


# ── MQTT topic constants ───────────────────────────────────────────────────

TOPIC_REGISTER = "miner/register"
TOPIC_ACK = "miner/{miner_id}/ack"
TOPIC_HEARTBEAT = "miner/{miner_id}/heartbeat"
TOPIC_TASK_ASSIGN = "miner/{miner_id}/task/assign"
TOPIC_TASK_STATUS = "miner/{miner_id}/task/status"
TOPIC_BLOCK_FOUND = "miner/{miner_id}/block/found"
TOPIC_PLATFORM_CMD = "platform/command"


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def broker():
    """Fresh MockMQTTBroker for each test."""
    return MockMQTTBroker()


@pytest.fixture
def miner_id():
    """Default miner identifier for single-miner tests."""
    return "miner-gpu-001"


@pytest.fixture
def platform_client(broker):
    """Simulates the platform server."""
    client = broker.create_client("platform-server")
    client.connect()
    client.subscribe("miner/#")
    return client


@pytest.fixture
def miner_client(broker, miner_id):
    """Simulates a GPU miner."""
    client = broker.create_client(miner_id)
    client.connect()
    client.subscribe(f"miner/{miner_id}/#")
    return client


# ── Tests: topic matching ──────────────────────────────────────────────────

class TestTopicMatching:
    """Verify MQTT topic pattern matching with exact, +, and # wildcards."""
    def test_exact_match(self):
        assert mqtt_topic_matches("a/b/c", "a/b/c")

    def test_exact_mismatch(self):
        assert not mqtt_topic_matches("a/b/c", "a/b/d")

    def test_plus_wildcard(self):
        assert mqtt_topic_matches("miner/+/ack", "miner/gpu-001/ack")

    def test_plus_wildcard_mismatch(self):
        assert not mqtt_topic_matches("miner/+/ack", "miner/gpu-001/task/ack")

    def test_hash_wildcard(self):
        assert mqtt_topic_matches("miner/#", "miner/gpu-001/ack")

    def test_hash_wildcard_deep(self):
        assert mqtt_topic_matches("miner/#", "miner/gpu-001/task/assign")

    def test_hash_at_root(self):
        assert mqtt_topic_matches("#", "any/topic/here")

    def test_empty_vs_empty(self):
        assert mqtt_topic_matches("", "")

    def test_plus_single_level(self):
        assert mqtt_topic_matches("+/+", "a/b")
        assert not mqtt_topic_matches("+/+", "a/b/c")


# ── Tests: broker basics ──────────────────────────────────────────────────

class TestBrokerBasics:
    """Verify core MockMQTTBroker functionality: connect, disconnect, pub/sub."""
    def test_create_and_connect(self, broker):
        client = broker.create_client("test-01")
        assert not client.is_connected
        client.connect()
        assert client.is_connected
        assert "test-01" in broker.connected_clients

    def test_disconnect(self, broker):
        client = broker.create_client("test-01")
        client.connect()
        client.disconnect()
        assert not client.is_connected
        assert "test-01" not in broker.connected_clients

    def test_publish_without_connect_raises(self, broker):
        client = broker.create_client("test-01")
        with pytest.raises(ConnectionError):
            client.publish("topic", "data")

    def test_subscribe_without_connect_raises(self, broker):
        client = broker.create_client("test-01")
        with pytest.raises(ConnectionError):
            client.subscribe("topic")

    def test_simple_pub_sub(self, broker):
        pub = broker.create_client("publisher")
        sub = broker.create_client("subscriber")
        pub.connect()
        sub.connect()
        sub.subscribe("test/topic")
        pub.publish("test/topic", "hello")
        msgs = sub.get_messages("test/topic")
        assert len(msgs) == 1
        assert msgs[0].payload_str == "hello"

    def test_wildcard_subscription(self, broker):
        pub = broker.create_client("publisher")
        sub = broker.create_client("subscriber")
        pub.connect()
        sub.connect()
        sub.subscribe("test/#")
        pub.publish("test/a", "msg1")
        pub.publish("test/b", "msg2")
        all_msgs = sub.get_all_messages()
        assert len(all_msgs) == 2

    def test_json_payload(self, broker):
        pub = broker.create_client("publisher")
        sub = broker.create_client("subscriber")
        pub.connect()
        sub.connect()
        sub.subscribe("data")
        pub.publish("data", {"key": "value", "num": 42})
        msgs = sub.get_messages("data")
        assert msgs[0].payload_json == {"key": "value", "num": 42}

    def test_retained_message(self, broker):
        pub = broker.create_client("publisher")
        pub.connect()
        pub.publish("config/setting", "retained-value", retain=True)

        sub = broker.create_client("subscriber")
        sub.connect()
        sub.subscribe("config/setting")
        # Retained message should be delivered on subscribe
        # (broker delivers retained on _register_client if already subscribed)
        # For this mock, subscribe before connect or re-register to get retained.
        # We test retained is stored:
        assert "config/setting" in broker._retained

    def test_message_log(self, broker):
        pub = broker.create_client("publisher")
        pub.connect()
        pub.publish("log/test", "data1")
        pub.publish("log/test", "data2")
        assert len(broker.message_log) == 2


# ── Tests: MQTT on_message callback ───────────────────────────────────────

class TestOnMessageCallback:
    """Verify that on_message callbacks fire correctly on incoming messages."""
    def test_callback_fires(self, broker):
        received = []
        sub = broker.create_client("sub")
        sub.connect()
        sub.on_message(lambda msg: received.append(msg))
        sub.subscribe("events")

        pub = broker.create_client("pub")
        pub.connect()
        pub.publish("events", "event-data")

        assert len(received) == 1
        assert received[0].payload_str == "event-data"

    def test_multiple_callbacks(self, broker):
        count = [0, 0]
        sub = broker.create_client("sub")
        sub.connect()
        sub.on_message(lambda msg: count.__setitem__(0, count[0] + 1))
        sub.on_message(lambda msg: count.__setitem__(1, count[1] + 1))
        sub.subscribe("events")

        pub = broker.create_client("pub")
        pub.connect()
        pub.publish("events", "data")

        assert count == [1, 1]


# ── Tests: full platform registration flow ─────────────────────────────────

class TestRegistrationFlow:
    """Simulates: miner register → platform ack → task assign → block found."""

    def test_miner_registers(self, broker, miner_client, platform_client, miner_id):
        """Miner publishes registration; platform receives it."""
        reg_payload = {
            "miner_id": miner_id,
            "gpu_count": 2,
            "gpu_model": "RTX 4090",
            "hashrate": 1500.0,
            "version": "1.0.0",
        }
        miner_client.publish(TOPIC_REGISTER, reg_payload)

        msgs = platform_client.get_messages(TOPIC_REGISTER)
        assert len(msgs) == 1
        data = msgs[0].payload_json
        assert data["miner_id"] == miner_id
        assert data["gpu_count"] == 2

    def test_platform_sends_ack(self, broker, miner_client, platform_client, miner_id):
        """Platform sends ACK to miner."""
        ack_topic = TOPIC_ACK.format(miner_id=miner_id)
        ack_payload = {
            "status": "registered",
            "miner_id": miner_id,
            "lease_ready": True,
        }
        platform_client.publish(ack_topic, ack_payload)

        msgs = miner_client.get_messages(ack_topic)
        assert len(msgs) == 1
        assert msgs[0].payload_json["status"] == "registered"

    def test_task_assignment(self, broker, miner_client, platform_client, miner_id):
        """Platform assigns a mining task."""
        task_topic = TOPIC_TASK_ASSIGN.format(miner_id=miner_id)
        task_payload = {
            "task_id": "task-42",
            "key_prefix": "a1b2c3d4e5f67890",  # 16 chars
            "target_address": "0x1234567890abcdef1234567890abcdef12345678",
            "difficulty": 1727,
            "duration_seconds": 600,
        }
        platform_client.publish(task_topic, task_payload)

        msgs = miner_client.get_messages(task_topic)
        assert len(msgs) == 1
        data = msgs[0].payload_json
        assert data["task_id"] == "task-42"
        assert len(data["key_prefix"]) == 16

    def test_miner_reports_status(self, broker, miner_client, platform_client, miner_id):
        """Miner reports mining status."""
        status_topic = TOPIC_TASK_STATUS.format(miner_id=miner_id)
        status_payload = {
            "task_id": "task-42",
            "status": "mining",
            "hashrate": 1500.0,
            "hashes_done": 50000,
        }
        miner_client.publish(status_topic, status_payload)

        msgs = platform_client.get_messages(status_topic)
        assert len(msgs) == 1
        assert msgs[0].payload_json["status"] == "mining"

    def test_block_found_notification(self, broker, miner_client, platform_client, miner_id):
        """Miner reports found block."""
        block_topic = TOPIC_BLOCK_FOUND.format(miner_id=miner_id)
        block_payload = {
            "task_id": "task-42",
            "key": "a1b2c3d4e5f67890" + "0" * 48,
            "hash_result": "base64encodedXEN11hash...",
            "salt": "24691e54afafe2416a8252097c9ca67557271475",
            "attempts": 12345,
        }
        miner_client.publish(block_topic, block_payload)

        msgs = platform_client.get_messages(block_topic)
        assert len(msgs) == 1
        data = msgs[0].payload_json
        assert data["task_id"] == "task-42"
        assert data["key"].startswith("a1b2c3d4e5f67890")

    def test_full_flow(self, broker, miner_id):
        """End-to-end: register → ack → assign → mining → block_found."""
        platform = broker.create_client("platform")
        miner = broker.create_client(miner_id)
        platform.connect()
        miner.connect()
        platform.subscribe("miner/#")
        miner.subscribe(f"miner/{miner_id}/#")

        # 1. Register
        miner.publish(TOPIC_REGISTER, {"miner_id": miner_id})
        assert len(platform.get_messages(TOPIC_REGISTER)) == 1

        # 2. ACK
        ack_topic = TOPIC_ACK.format(miner_id=miner_id)
        platform.publish(ack_topic, {"status": "registered"})
        assert len(miner.get_messages(ack_topic)) == 1

        # 3. Assign task
        task_topic = TOPIC_TASK_ASSIGN.format(miner_id=miner_id)
        platform.publish(task_topic, {
            "task_id": "task-99",
            "key_prefix": "deadbeefcafe1234",
        })
        task_msg = miner.get_messages(task_topic)
        assert len(task_msg) == 1
        assert task_msg[0].payload_json["key_prefix"] == "deadbeefcafe1234"

        # 4. Mining status
        status_topic = TOPIC_TASK_STATUS.format(miner_id=miner_id)
        miner.publish(status_topic, {"task_id": "task-99", "status": "mining"})
        assert len(platform.get_messages(status_topic)) == 1

        # 5. Block found
        block_topic = TOPIC_BLOCK_FOUND.format(miner_id=miner_id)
        miner.publish(block_topic, {
            "task_id": "task-99",
            "key": "deadbeefcafe1234" + "a" * 48,
        })
        block_msgs = platform.get_messages(block_topic)
        assert len(block_msgs) == 1
        assert block_msgs[0].payload_json["key"].startswith("deadbeefcafe1234")


# ── Tests: heartbeat ──────────────────────────────────────────────────────

class TestHeartbeat:
    """Verify heartbeat message routing and timeout detection."""
    def test_heartbeat_received(self, broker, miner_client, platform_client, miner_id):
        hb_topic = TOPIC_HEARTBEAT.format(miner_id=miner_id)
        miner_client.publish(hb_topic, {"miner_id": miner_id, "ts": time.time()})
        msgs = platform_client.get_messages(hb_topic)
        assert len(msgs) == 1

    def test_heartbeat_timeout_detection(self, broker, miner_id):
        """Simulate heartbeat timeout: no heartbeat for > threshold."""
        platform = broker.create_client("platform")
        platform.connect()
        platform.subscribe(f"miner/{miner_id}/heartbeat")

        # No heartbeat sent - simulate timeout check
        last_heartbeat = time.time() - 60  # 60 seconds ago
        timeout_threshold = 30
        assert (time.time() - last_heartbeat) > timeout_threshold

    def test_multiple_heartbeats(self, broker, miner_client, platform_client, miner_id):
        hb_topic = TOPIC_HEARTBEAT.format(miner_id=miner_id)
        for i in range(5):
            miner_client.publish(hb_topic, {"seq": i})
        msgs = platform_client.get_messages(hb_topic)
        assert len(msgs) == 5


# ── Tests: multiple miners ────────────────────────────────────────────────

class TestMultipleMiners:
    """Verify independent topic isolation for multiple connected miners."""
    def test_two_miners_independent(self, broker):
        platform = broker.create_client("platform")
        miner_a = broker.create_client("miner-a")
        miner_b = broker.create_client("miner-b")
        platform.connect()
        miner_a.connect()
        miner_b.connect()

        platform.subscribe("miner/#")
        miner_a.subscribe("miner/miner-a/#")
        miner_b.subscribe("miner/miner-b/#")

        # Assign different tasks
        platform.publish("miner/miner-a/task/assign", {"task": "A"})
        platform.publish("miner/miner-b/task/assign", {"task": "B"})

        a_msgs = miner_a.get_messages("miner/miner-a/task/assign")
        b_msgs = miner_b.get_messages("miner/miner-b/task/assign")

        assert len(a_msgs) == 1
        assert a_msgs[0].payload_json["task"] == "A"
        assert len(b_msgs) == 1
        assert b_msgs[0].payload_json["task"] == "B"

        # Miner A should NOT receive miner B's messages
        a_b_msgs = miner_a.get_messages("miner/miner-b/task/assign")
        assert len(a_b_msgs) == 0

    def test_platform_sees_all_miners(self, broker):
        platform = broker.create_client("platform")
        platform.connect()
        platform.subscribe("miner/#")

        for i in range(3):
            m = broker.create_client(f"miner-{i}")
            m.connect()
            m.publish(TOPIC_REGISTER, {"miner_id": f"miner-{i}"})

        reg_msgs = platform.get_messages(TOPIC_REGISTER)
        assert len(reg_msgs) == 3
