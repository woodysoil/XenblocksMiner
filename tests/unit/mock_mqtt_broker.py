"""
mock_mqtt_broker.py - Local MQTT Broker Simulation

A lightweight in-process MQTT broker mock built on top of paho-mqtt's
client library. Instead of running a real broker, this module provides:

 - MockMQTTBroker: an in-process pub/sub message router
 - MockMQTTClient: a fake client that connects to the mock broker
 - Topic matching with wildcard support (+, #)

Usage in tests:
    broker = MockMQTTBroker()
    client = broker.create_client("miner-001")
    client.subscribe("platform/tasks/#")
    client.publish("platform/tasks/assign", payload)
    msgs = client.get_messages("platform/tasks/assign")
"""

import fnmatch
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class MQTTMessage:
    """Represents a single MQTT message."""
    topic: str
    payload: bytes
    qos: int = 0
    retain: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def payload_str(self) -> str:
        return self.payload.decode("utf-8", errors="replace")

    @property
    def payload_json(self) -> Any:
        return json.loads(self.payload)


def mqtt_topic_matches(pattern: str, topic: str) -> bool:
    """
    Match MQTT topic against subscription pattern.
    Supports + (single level) and # (multi level) wildcards.
    """
    pattern_parts = pattern.split("/")
    topic_parts = topic.split("/")

    pi = 0
    ti = 0
    while pi < len(pattern_parts) and ti < len(topic_parts):
        if pattern_parts[pi] == "#":
            return True  # matches everything after
        if pattern_parts[pi] == "+":
            pi += 1
            ti += 1
            continue
        if pattern_parts[pi] != topic_parts[ti]:
            return False
        pi += 1
        ti += 1

    # Both must be exhausted (unless pattern ends with #)
    if pi < len(pattern_parts) and pattern_parts[pi] == "#":
        return True
    return pi == len(pattern_parts) and ti == len(topic_parts)


class MockMQTTClient:
    """A mock MQTT client that connects to MockMQTTBroker."""

    def __init__(self, client_id: str, broker: "MockMQTTBroker"):
        self.client_id = client_id
        self._broker = broker
        self._subscriptions: Set[str] = set()
        self._messages: Dict[str, List[MQTTMessage]] = {}
        self._on_message_callbacks: List[Callable[[MQTTMessage], None]] = []
        self._connected = False
        self._lock = threading.Lock()

    def connect(self):
        self._connected = True
        self._broker._register_client(self)

    def disconnect(self):
        self._connected = False
        self._broker._unregister_client(self)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def subscribe(self, topic_pattern: str, qos: int = 0):
        if not self._connected:
            raise ConnectionError("Client not connected")
        self._subscriptions.add(topic_pattern)

    def unsubscribe(self, topic_pattern: str):
        self._subscriptions.discard(topic_pattern)

    def publish(self, topic: str, payload: Any, qos: int = 0, retain: bool = False):
        if not self._connected:
            raise ConnectionError("Client not connected")
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        elif isinstance(payload, dict):
            payload = json.dumps(payload).encode("utf-8")
        elif not isinstance(payload, bytes):
            payload = str(payload).encode("utf-8")

        msg = MQTTMessage(topic=topic, payload=payload, qos=qos, retain=retain)
        self._broker._route_message(msg, sender_id=self.client_id)

    def on_message(self, callback: Callable[[MQTTMessage], None]):
        self._on_message_callbacks.append(callback)

    def _deliver(self, msg: MQTTMessage):
        """Called by broker to deliver a message matching a subscription."""
        with self._lock:
            if msg.topic not in self._messages:
                self._messages[msg.topic] = []
            self._messages[msg.topic].append(msg)

        for cb in self._on_message_callbacks:
            cb(msg)

    def get_messages(self, topic: str) -> List[MQTTMessage]:
        """Get all received messages for a specific topic."""
        with self._lock:
            return list(self._messages.get(topic, []))

    def get_all_messages(self) -> List[MQTTMessage]:
        """Get all received messages across all topics."""
        with self._lock:
            all_msgs = []
            for msgs in self._messages.values():
                all_msgs.extend(msgs)
            all_msgs.sort(key=lambda m: m.timestamp)
            return all_msgs

    def clear_messages(self):
        with self._lock:
            self._messages.clear()

    def wait_for_message(self, topic: str, timeout: float = 2.0) -> Optional[MQTTMessage]:
        """Block until a message arrives on topic or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msgs = self.get_messages(topic)
            if msgs:
                return msgs[-1]
            time.sleep(0.01)
        return None


class MockMQTTBroker:
    """
    In-process MQTT broker mock.
    Routes messages between MockMQTTClient instances.
    """

    def __init__(self):
        self._clients: Dict[str, MockMQTTClient] = {}
        self._retained: Dict[str, MQTTMessage] = {}
        self._lock = threading.Lock()
        self._message_log: List[MQTTMessage] = []

    def create_client(self, client_id: str) -> MockMQTTClient:
        client = MockMQTTClient(client_id, self)
        return client

    def _register_client(self, client: MockMQTTClient):
        retained_to_deliver = []
        with self._lock:
            self._clients[client.client_id] = client
            # Collect retained messages to deliver outside the lock
            for pattern in client._subscriptions:
                for topic, msg in self._retained.items():
                    if mqtt_topic_matches(pattern, topic):
                        retained_to_deliver.append(msg)
        for msg in retained_to_deliver:
            client._deliver(msg)

    def _unregister_client(self, client: MockMQTTClient):
        with self._lock:
            self._clients.pop(client.client_id, None)

    def _route_message(self, msg: MQTTMessage, sender_id: str = ""):
        # Collect recipients under lock, then deliver outside to avoid
        # deadlock when callbacks publish new messages (re-entrant _route_message).
        recipients = []
        with self._lock:
            self._message_log.append(msg)

            if msg.retain:
                self._retained[msg.topic] = msg

            for cid, client in self._clients.items():
                for pattern in client._subscriptions:
                    if mqtt_topic_matches(pattern, msg.topic):
                        recipients.append(client)
                        break  # don't deliver same msg twice to same client

        for client in recipients:
            client._deliver(msg)

    @property
    def message_log(self) -> List[MQTTMessage]:
        with self._lock:
            return list(self._message_log)

    @property
    def connected_clients(self) -> List[str]:
        with self._lock:
            return list(self._clients.keys())
