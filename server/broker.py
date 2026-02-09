"""
broker.py - Embedded async MQTT broker using gmqtt for client connections.

This module wraps a lightweight pure-Python MQTT broker that listens on a
configurable port (default 1883).  Workers connect to it via standard MQTT.
Platform services interact with the broker through an internal publish API
so everything runs in a single asyncio event loop.

Implementation: We use asyncio TCP server + minimal MQTT 3.1.1 packet parsing
to avoid external broker dependencies.  This is sufficient for local testing
where we don't need TLS, persistence, or clustering.
"""

import asyncio
import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("broker")


# ---------------------------------------------------------------------------
# Minimal MQTT 3.1.1 packet codec
# ---------------------------------------------------------------------------

CONNECT = 1
CONNACK = 2
PUBLISH = 3
PUBACK = 4
SUBSCRIBE = 8
SUBACK = 9
UNSUBSCRIBE = 10
UNSUBACK = 11
PINGREQ = 12
PINGRESP = 13
DISCONNECT = 14


def _encode_remaining_length(length: int) -> bytes:
    out = bytearray()
    while True:
        byte = length % 128
        length //= 128
        if length > 0:
            byte |= 0x80
        out.append(byte)
        if length == 0:
            break
    return bytes(out)


def _decode_remaining_length(data: bytes, offset: int) -> Tuple[int, int]:
    """Returns (remaining_length, bytes_consumed)."""
    multiplier = 1
    value = 0
    idx = offset
    while True:
        if idx >= len(data):
            raise ValueError("Incomplete remaining length")
        encoded_byte = data[idx]
        value += (encoded_byte & 0x7F) * multiplier
        idx += 1
        if (encoded_byte & 0x80) == 0:
            break
        multiplier *= 128
    return value, idx - offset


def _decode_utf8_string(data: bytes, offset: int) -> Tuple[str, int]:
    if offset + 2 > len(data):
        raise ValueError("Incomplete UTF-8 string length")
    str_len = struct.unpack("!H", data[offset : offset + 2])[0]
    offset += 2
    if offset + str_len > len(data):
        raise ValueError("Incomplete UTF-8 string")
    s = data[offset : offset + str_len].decode("utf-8")
    return s, offset + str_len


def _encode_utf8_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("!H", len(encoded)) + encoded


def mqtt_topic_matches(pattern: str, topic: str) -> bool:
    """Match MQTT topic against subscription pattern with + and # wildcards."""
    pattern_parts = pattern.split("/")
    topic_parts = topic.split("/")
    pi = ti = 0
    while pi < len(pattern_parts) and ti < len(topic_parts):
        if pattern_parts[pi] == "#":
            return True
        if pattern_parts[pi] == "+":
            pi += 1
            ti += 1
            continue
        if pattern_parts[pi] != topic_parts[ti]:
            return False
        pi += 1
        ti += 1
    if pi < len(pattern_parts) and pattern_parts[pi] == "#":
        return True
    return pi == len(pattern_parts) and ti == len(topic_parts)


# ---------------------------------------------------------------------------
# Client session managed by the broker
# ---------------------------------------------------------------------------

@dataclass
class ClientSession:
    client_id: str
    writer: asyncio.StreamWriter
    subscriptions: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

# Callback type: async def handler(topic: str, payload: dict, client_id: str)
MessageHandler = Callable[[str, dict, str], Coroutine[Any, Any, None]]


class MQTTBroker:
    """Minimal async MQTT 3.1.1 broker for local testing."""

    def __init__(self, host: str = "0.0.0.0", port: int = 1883):
        self.host = host
        self.port = port
        self._clients: Dict[str, ClientSession] = {}
        self._server: Optional[asyncio.AbstractServer] = None
        self._handlers: List[MessageHandler] = []
        self._lock = asyncio.Lock()

    def on_message(self, handler: MessageHandler):
        """Register an async callback invoked on every PUBLISH from a client."""
        self._handlers.append(handler)

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        logger.info("MQTT broker listening on %s:%d", self.host, self.port)

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        async with self._lock:
            for session in self._clients.values():
                session.writer.close()
            self._clients.clear()

    async def publish(self, topic: str, payload: Any, qos: int = 1):
        """Publish a message from the platform (server-side) to matching clients."""
        if isinstance(payload, (dict, list)):
            raw = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            raw = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            raw = payload
        else:
            raw = str(payload).encode("utf-8")

        pkt = self._build_publish_packet(topic, raw, qos)
        async with self._lock:
            for session in list(self._clients.values()):
                for pattern in session.subscriptions:
                    if mqtt_topic_matches(pattern, topic):
                        try:
                            session.writer.write(pkt)
                            await session.writer.drain()
                        except Exception:
                            logger.debug("Failed to deliver to %s", session.client_id)
                        break

    @property
    def connected_client_ids(self) -> List[str]:
        return list(self._clients.keys())

    # -----------------------------------------------------------------------
    # Connection handler
    # -----------------------------------------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        session: Optional[ClientSession] = None
        try:
            session = await self._do_connect(reader, writer)
            if session is None:
                writer.close()
                return
            async with self._lock:
                # Disconnect any existing session with the same client_id
                old = self._clients.pop(session.client_id, None)
                if old:
                    try:
                        old.writer.close()
                    except Exception:
                        pass
                self._clients[session.client_id] = session
            logger.info("Client connected: %s", session.client_id)

            while True:
                header = await reader.read(1)
                if not header:
                    break
                pkt_type = (header[0] >> 4) & 0x0F
                flags = header[0] & 0x0F

                # Read remaining length
                remaining_bytes = bytearray()
                while True:
                    b = await reader.read(1)
                    if not b:
                        raise ConnectionError("Connection closed during remaining length")
                    remaining_bytes.append(b[0])
                    if (b[0] & 0x80) == 0:
                        break
                remaining_len, _ = _decode_remaining_length(bytes(remaining_bytes), 0)
                body = b""
                if remaining_len > 0:
                    body = await self._read_exact(reader, remaining_len)

                if pkt_type == PUBLISH:
                    await self._handle_publish(session, flags, body)
                elif pkt_type == SUBSCRIBE:
                    await self._handle_subscribe(session, body, writer)
                elif pkt_type == UNSUBSCRIBE:
                    await self._handle_unsubscribe(session, body, writer)
                elif pkt_type == PINGREQ:
                    writer.write(bytes([PINGRESP << 4, 0]))
                    await writer.drain()
                elif pkt_type == DISCONNECT:
                    break
                else:
                    pass  # Ignore unknown packets

        except (ConnectionError, asyncio.IncompleteReadError, OSError):
            pass
        except Exception:
            logger.exception("Error in client handler")
        finally:
            if session:
                async with self._lock:
                    self._clients.pop(session.client_id, None)
                logger.info("Client disconnected: %s", session.client_id)
            writer.close()

    async def _read_exact(self, reader: asyncio.StreamReader, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = await reader.read(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed while reading")
            data.extend(chunk)
        return bytes(data)

    async def _do_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> Optional[ClientSession]:
        """Handle CONNECT packet and send CONNACK."""
        header = await reader.read(1)
        if not header:
            return None
        pkt_type = (header[0] >> 4) & 0x0F
        if pkt_type != CONNECT:
            return None

        remaining_bytes = bytearray()
        while True:
            b = await reader.read(1)
            if not b:
                return None
            remaining_bytes.append(b[0])
            if (b[0] & 0x80) == 0:
                break
        remaining_len, _ = _decode_remaining_length(bytes(remaining_bytes), 0)
        body = await self._read_exact(reader, remaining_len)

        # Parse CONNECT variable header
        offset = 0
        protocol_name, offset = _decode_utf8_string(body, offset)
        protocol_level = body[offset]
        offset += 1
        connect_flags = body[offset]
        offset += 1
        keep_alive = struct.unpack("!H", body[offset : offset + 2])[0]
        offset += 2

        # Payload: client ID
        client_id, offset = _decode_utf8_string(body, offset)
        if not client_id:
            client_id = f"auto-{id(writer)}"

        # Send CONNACK (session present=0, return code=0 accepted)
        connack = bytes([CONNACK << 4, 2, 0, 0])
        writer.write(connack)
        await writer.drain()

        return ClientSession(client_id=client_id, writer=writer)

    async def _handle_publish(
        self, session: ClientSession, flags: int, body: bytes
    ):
        offset = 0
        topic, offset = _decode_utf8_string(body, offset)
        qos = (flags >> 1) & 0x03
        packet_id = None
        if qos > 0:
            packet_id = struct.unpack("!H", body[offset : offset + 2])[0]
            offset += 2
        payload_bytes = body[offset:]

        # Send PUBACK for QoS 1
        if qos == 1 and packet_id is not None:
            puback = bytes([PUBACK << 4, 2]) + struct.pack("!H", packet_id)
            try:
                session.writer.write(puback)
                await session.writer.drain()
            except Exception:
                pass

        # Parse JSON payload
        try:
            payload_dict = json.loads(payload_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload_dict = {}

        logger.debug("PUBLISH from %s on %s: %s", session.client_id, topic, payload_dict)

        # Route to subscribers (other clients)
        pkt = self._build_publish_packet(topic, payload_bytes, min(qos, 1))
        async with self._lock:
            for cid, csession in list(self._clients.items()):
                if cid == session.client_id:
                    continue
                for pattern in csession.subscriptions:
                    if mqtt_topic_matches(pattern, topic):
                        try:
                            csession.writer.write(pkt)
                            await csession.writer.drain()
                        except Exception:
                            pass
                        break

        # Invoke platform handlers
        for handler in self._handlers:
            try:
                await handler(topic, payload_dict, session.client_id)
            except Exception:
                logger.exception("Handler error for topic %s", topic)

    async def _handle_subscribe(
        self, session: ClientSession, body: bytes, writer: asyncio.StreamWriter
    ):
        offset = 0
        packet_id = struct.unpack("!H", body[offset : offset + 2])[0]
        offset += 2
        granted = []
        while offset < len(body):
            topic_filter, offset = _decode_utf8_string(body, offset)
            requested_qos = body[offset]
            offset += 1
            session.subscriptions.add(topic_filter)
            granted.append(min(requested_qos, 1))  # Max QoS 1
            logger.debug("Client %s subscribed to %s", session.client_id, topic_filter)

        # SUBACK
        suback = bytes([SUBACK << 4, 2 + len(granted)]) + struct.pack("!H", packet_id) + bytes(granted)
        writer.write(suback)
        await writer.drain()

    async def _handle_unsubscribe(
        self, session: ClientSession, body: bytes, writer: asyncio.StreamWriter
    ):
        offset = 0
        packet_id = struct.unpack("!H", body[offset : offset + 2])[0]
        offset += 2
        while offset < len(body):
            topic_filter, offset = _decode_utf8_string(body, offset)
            session.subscriptions.discard(topic_filter)

        unsuback = bytes([UNSUBACK << 4, 2]) + struct.pack("!H", packet_id)
        writer.write(unsuback)
        await writer.drain()

    @staticmethod
    def _build_publish_packet(topic: str, payload: bytes, qos: int = 0) -> bytes:
        flags = 0
        topic_bytes = _encode_utf8_string(topic)
        variable_header = topic_bytes
        if qos > 0:
            flags |= (qos << 1)
            # Use packet_id=0 for server-originated messages (simplification)
            variable_header += struct.pack("!H", 0)
        remaining = variable_header + payload
        first_byte = (PUBLISH << 4) | flags
        return bytes([first_byte]) + _encode_remaining_length(len(remaining)) + remaining
