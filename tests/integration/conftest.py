"""
Shared fixtures for XenblocksMiner integration tests.

Provides:
 - MockMQTTBroker instances and pre-wired platform/worker clients
 - WorkerSimulator: simulates a C++ worker's MQTT behavior per proto schemas
 - PlatformSimulator: simulates the platform server's MQTT behavior
 - Helper factories for building valid protocol messages
 - Constants matching proto/ and MiningCommon.h
"""

import json
import time
import uuid
import pytest
from typing import Any, Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unit.mock_mqtt_broker import MockMQTTBroker, MockMQTTClient, MQTTMessage


# ── Constants (from proto/README.md and MiningCommon.h) ───────────────────

PLATFORM_PREFIX_LENGTH = 16
HASH_LENGTH = 64
HEARTBEAT_INTERVAL_SEC = 30
QOS = 1

# Topic suffixes (from MqttClient.h)
TOPIC_REGISTER  = "register"
TOPIC_HEARTBEAT = "heartbeat"
TOPIC_STATUS    = "status"
TOPIC_BLOCK     = "block"
TOPIC_TASK      = "task"
TOPIC_CONTROL   = "control"

# Worker states (from PlatformManager.h PlatformState enum)
WORKER_STATES = ["IDLE", "AVAILABLE", "LEASED", "MINING", "COMPLETED", "ERROR", "offline"]


# ── Message Factories ─────────────────────────────────────────────────────

def make_register_msg(worker_id: str,
                      eth_address: str = "0x" + "ab" * 20,
                      gpu_count: int = 2,
                      total_memory_gb: int = 24,
                      gpus: Optional[List[Dict]] = None,
                      version: str = "2.0.0") -> dict:
    """Build a worker registration message per proto/worker_to_platform.json."""
    if gpus is None:
        gpus = [
            {"index": 0, "name": "NVIDIA GeForce RTX 4090", "memory_gb": 24, "bus_id": 1},
            {"index": 1, "name": "NVIDIA GeForce RTX 3080", "memory_gb": 10, "bus_id": 2},
        ][:gpu_count]
    return {
        "worker_id": worker_id,
        "eth_address": eth_address,
        "gpu_count": gpu_count,
        "total_memory_gb": total_memory_gb,
        "gpus": gpus,
        "version": version,
        "timestamp": int(time.time()),
    }


def make_heartbeat_msg(worker_id: str,
                       hashrate: float = 1250.75,
                       active_gpus: int = 2,
                       accepted_blocks: int = 0,
                       difficulty: int = 8,
                       uptime_sec: int = 0) -> dict:
    """Build a heartbeat message per proto/worker_to_platform.json."""
    return {
        "worker_id": worker_id,
        "hashrate": hashrate,
        "active_gpus": active_gpus,
        "accepted_blocks": accepted_blocks,
        "difficulty": difficulty,
        "uptime_sec": uptime_sec,
        "timestamp": int(time.time()),
    }


def make_status_msg(worker_id: str,
                    state: str,
                    lease_id: str = "",
                    detail: str = "") -> dict:
    """Build a status update message per proto/worker_to_platform.json."""
    msg = {
        "worker_id": worker_id,
        "state": state,
        "timestamp": int(time.time()),
    }
    if lease_id:
        msg["lease_id"] = lease_id
    if detail:
        msg["detail"] = detail
    return msg


def make_block_found_msg(worker_id: str,
                         lease_id: str,
                         hash_val: str = "0000" + "a1" * 30,
                         key: str = "",
                         account: str = "0x" + "cd" * 20,
                         attempts: int = 150000,
                         hashrate: str = "1250.75",
                         prefix: str = "") -> dict:
    """Build a block_found message per proto/worker_to_platform.json."""
    if not key:
        random_suffix = uuid.uuid4().hex[:HASH_LENGTH - len(prefix)]
        key = prefix + random_suffix
        key = key[:HASH_LENGTH].ljust(HASH_LENGTH, "0")
    return {
        "worker_id": worker_id,
        "lease_id": lease_id,
        "hash": hash_val,
        "key": key,
        "account": account,
        "attempts": attempts,
        "hashrate": hashrate,
        "timestamp": int(time.time()),
    }


def make_register_ack(accepted: bool = True,
                      reason: str = "") -> dict:
    """Build a register_ack message per proto/platform_to_worker.json."""
    msg: dict = {
        "command": "register_ack",
        "accepted": accepted,
    }
    if not accepted:
        msg["reason"] = reason or "unknown"
    return msg


def make_assign_task(lease_id: str = "",
                     consumer_id: str = "",
                     consumer_address: str = "0x" + "ee" * 20,
                     prefix: str = "a1b2c3d4e5f67890",
                     duration_sec: int = 3600) -> dict:
    """Build an assign_task message per proto/platform_to_worker.json."""
    return {
        "command": "assign_task",
        "lease_id": lease_id or f"lease-{uuid.uuid4()}",
        "consumer_id": consumer_id or f"consumer-{uuid.uuid4()}",
        "consumer_address": consumer_address,
        "prefix": prefix,
        "duration_sec": duration_sec,
    }


def make_release(lease_id: str = "") -> dict:
    """Build a release message per proto/platform_to_worker.json."""
    msg: dict = {"command": "release"}
    if lease_id:
        msg["lease_id"] = lease_id
    return msg


def make_control(action: str) -> dict:
    """Build a control message per proto/platform_to_worker.json."""
    assert action in ("pause", "resume", "shutdown")
    return {"action": action}


# ── WorkerSimulator ───────────────────────────────────────────────────────

class WorkerSimulator:
    """
    Simulates a C++ XenblocksMiner worker's MQTT protocol behavior.

    Mirrors the C++ PlatformManager + WorkerReporter message flow:
      1. connect → subscribe to task + control topics
      2. sendRegistration
      3. Receive register_ack, assign_task, release, control
      4. Send heartbeat, status, block_found
    """

    def __init__(self, broker: MockMQTTBroker, worker_id: str,
                 eth_address: str = "0x" + "ab" * 20,
                 gpu_count: int = 2):
        self.worker_id = worker_id
        self.eth_address = eth_address
        self.gpu_count = gpu_count
        self.broker = broker
        self.client = broker.create_client(f"worker-{worker_id}")
        self.state = "IDLE"
        self.lease_id: Optional[str] = None
        self.received_commands: List[dict] = []
        self._started = False

    def _build_topic(self, suffix: str) -> str:
        return f"xenminer/{self.worker_id}/{suffix}"

    def start(self):
        """Connect, subscribe, and send registration (mirrors PlatformManager::start)."""
        self.client.connect()
        self.client.subscribe(self._build_topic(TOPIC_TASK))
        self.client.subscribe(self._build_topic(TOPIC_CONTROL))
        self.client.on_message(self._on_message)

        # Send registration
        reg = make_register_msg(self.worker_id, self.eth_address, self.gpu_count)
        self.client.publish(self._build_topic(TOPIC_REGISTER), reg)
        self.state = "AVAILABLE"
        self._send_status()
        self._started = True

    def stop(self):
        """Disconnect (mirrors PlatformManager::stop)."""
        if self.lease_id:
            self.lease_id = None
        self._send_status("offline")
        self.client.disconnect()
        self.state = "IDLE"

    def send_heartbeat(self, hashrate: float = 1250.0,
                       active_gpus: int = 2,
                       accepted_blocks: int = 0):
        hb = make_heartbeat_msg(self.worker_id, hashrate, active_gpus, accepted_blocks)
        self.client.publish(self._build_topic(TOPIC_HEARTBEAT), hb)

    def send_block_found(self, hash_val: str, key: str, account: str,
                         attempts: int, hashrate: float):
        if self.state != "MINING" or not self.lease_id:
            return
        msg = make_block_found_msg(
            self.worker_id, self.lease_id, hash_val, key, account,
            attempts, f"{hashrate:.2f}"
        )
        self.client.publish(self._build_topic(TOPIC_BLOCK), msg)

    def _send_status(self, state: str = ""):
        st = state or self.state
        msg = make_status_msg(self.worker_id, st,
                              lease_id=self.lease_id or "")
        self.client.publish(self._build_topic(TOPIC_STATUS), msg)

    def _on_message(self, msg: MQTTMessage):
        try:
            data = msg.payload_json
        except Exception:
            return
        self.received_commands.append(data)

        command = data.get("command", "")
        if command == "register_ack":
            self._handle_register_ack(data)
        elif command == "assign_task":
            self._handle_assign_task(data)
        elif command == "release":
            self._handle_release(data)
        else:
            self._handle_control(data)

    def _handle_register_ack(self, data: dict):
        if data.get("accepted", False):
            self.state = "AVAILABLE"
        else:
            self.state = "ERROR"
        self._send_status()

    def _handle_assign_task(self, data: dict):
        if self.state != "AVAILABLE":
            return
        self.lease_id = data["lease_id"]
        self.state = "LEASED"
        self._send_status()
        # Immediately transition to MINING (simulating successful setup)
        self.state = "MINING"
        self._send_status()

    def _handle_release(self, data: dict):
        release_id = data.get("lease_id", "")
        if self.lease_id and release_id and release_id != self.lease_id:
            return
        self.state = "COMPLETED"
        self._send_status()
        self.lease_id = None
        self.state = "AVAILABLE"
        self._send_status()

    def _handle_control(self, data: dict):
        action = data.get("action", "")
        if action == "pause":
            self.lease_id = None
            self.state = "IDLE"
            self._send_status()
        elif action == "resume":
            if self.state == "IDLE":
                reg = make_register_msg(self.worker_id, self.eth_address, self.gpu_count)
                self.client.publish(self._build_topic(TOPIC_REGISTER), reg)
                self.state = "AVAILABLE"
                self._send_status()
        elif action == "shutdown":
            self.stop()


# ── PlatformSimulator ─────────────────────────────────────────────────────

class PlatformSimulator:
    """
    Simulates the platform server side of the MQTT protocol.

    Subscribes to all worker topics via wildcard and provides methods
    to send commands back to specific workers.
    """

    def __init__(self, broker: MockMQTTBroker):
        self.broker = broker
        self.client = broker.create_client("platform-server")
        self.registered_workers: Dict[str, dict] = {}
        self.heartbeats: Dict[str, List[dict]] = {}
        self.status_updates: Dict[str, List[dict]] = {}
        self.blocks_found: Dict[str, List[dict]] = {}
        self._all_messages: List[MQTTMessage] = []

    def start(self):
        self.client.connect()
        self.client.subscribe("xenminer/#")
        self.client.on_message(self._on_message)

    def stop(self):
        self.client.disconnect()

    def _worker_topic(self, worker_id: str, suffix: str) -> str:
        return f"xenminer/{worker_id}/{suffix}"

    # ── Send commands to workers ──

    def send_register_ack(self, worker_id: str, accepted: bool = True,
                          reason: str = ""):
        msg = make_register_ack(accepted, reason)
        topic = self._worker_topic(worker_id, TOPIC_TASK)
        self.client.publish(topic, msg)

    def send_assign_task(self, worker_id: str, lease_id: str = "",
                         consumer_id: str = "",
                         consumer_address: str = "0x" + "ee" * 20,
                         prefix: str = "a1b2c3d4e5f67890",
                         duration_sec: int = 3600) -> dict:
        msg = make_assign_task(lease_id, consumer_id, consumer_address,
                               prefix, duration_sec)
        topic = self._worker_topic(worker_id, TOPIC_TASK)
        self.client.publish(topic, msg)
        return msg

    def send_release(self, worker_id: str, lease_id: str = ""):
        msg = make_release(lease_id)
        topic = self._worker_topic(worker_id, TOPIC_TASK)
        self.client.publish(topic, msg)

    def send_control(self, worker_id: str, action: str):
        msg = make_control(action)
        topic = self._worker_topic(worker_id, TOPIC_CONTROL)
        self.client.publish(topic, msg)

    # ── Receive/query messages from workers ──

    def _on_message(self, msg: MQTTMessage):
        self._all_messages.append(msg)
        try:
            data = msg.payload_json
        except Exception:
            return

        parts = msg.topic.split("/")
        if len(parts) < 3:
            return
        worker_id = parts[1]
        msg_type = parts[2]

        if msg_type == TOPIC_REGISTER:
            self.registered_workers[worker_id] = data
        elif msg_type == TOPIC_HEARTBEAT:
            self.heartbeats.setdefault(worker_id, []).append(data)
        elif msg_type == TOPIC_STATUS:
            self.status_updates.setdefault(worker_id, []).append(data)
        elif msg_type == TOPIC_BLOCK:
            self.blocks_found.setdefault(worker_id, []).append(data)

    def get_worker_statuses(self, worker_id: str) -> List[dict]:
        return list(self.status_updates.get(worker_id, []))

    def get_latest_status(self, worker_id: str) -> Optional[dict]:
        statuses = self.status_updates.get(worker_id, [])
        return statuses[-1] if statuses else None

    def get_blocks(self, worker_id: str) -> List[dict]:
        return list(self.blocks_found.get(worker_id, []))

    def get_heartbeats(self, worker_id: str) -> List[dict]:
        return list(self.heartbeats.get(worker_id, []))


# ── Pytest Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def broker():
    """Fresh MockMQTTBroker for each test."""
    return MockMQTTBroker()


@pytest.fixture
def platform(broker):
    """Started PlatformSimulator."""
    p = PlatformSimulator(broker)
    p.start()
    return p


@pytest.fixture
def worker_id():
    return "test-worker-001"


@pytest.fixture
def worker(broker, worker_id):
    """Started WorkerSimulator."""
    w = WorkerSimulator(broker, worker_id)
    w.start()
    return w


@pytest.fixture
def worker_factory(broker):
    """Factory to create multiple WorkerSimulators."""
    workers = []

    def _create(worker_id: str, eth_address: str = "0x" + "ab" * 20,
                gpu_count: int = 2) -> WorkerSimulator:
        w = WorkerSimulator(broker, worker_id, eth_address, gpu_count)
        w.start()
        workers.append(w)
        return w

    yield _create

    for w in workers:
        if w.client.is_connected:
            w.stop()


@pytest.fixture
def sample_lease_id():
    return f"lease-{uuid.uuid4()}"


@pytest.fixture
def sample_consumer_address():
    return "0x" + "ee" * 20


@pytest.fixture
def sample_prefix():
    """A valid 16-char hex prefix."""
    return "a1b2c3d4e5f67890"
