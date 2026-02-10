"""
simulator.py - Python worker simulator that mimics the C++ miner's MQTT behavior.

Simulates one or more workers in a single process:
 - Connects to the MQTT broker
 - Sends registration with fake GPU info
 - Handles register_ack → transitions to AVAILABLE
 - Handles assign_task → transitions to LEASED → MINING
 - Periodically sends heartbeats
 - Simulates block discovery at configurable probability
 - Sends block_found with correct prefix from the lease
 - Handles release → transitions to COMPLETED → AVAILABLE

Usage:
    python -m server.simulator --workers 2 --mqtt-host localhost --mqtt-port 1883
    python server/simulator.py --workers 3 --block-interval 5

No GPU or C++ binary required.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import secrets
import struct
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

try:
    from http.client import HTTPConnection
except ImportError:
    pass

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-18s] %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulator")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLATFORM_PREFIX_LENGTH = 16
HEARTBEAT_INTERVAL = 5  # faster than real (30s) for testing
MINE_CHECK_INTERVAL = 2  # check for simulated block every N seconds


class WorkerState(str, Enum):
    IDLE = "IDLE"
    AVAILABLE = "AVAILABLE"
    LEASED = "LEASED"
    MINING = "MINING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Minimal async MQTT client (matches broker.py packet format)
# ---------------------------------------------------------------------------

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


def _encode_utf8(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("!H", len(b)) + b


def _decode_utf8(data: bytes, offset: int):
    slen = struct.unpack("!H", data[offset : offset + 2])[0]
    return data[offset + 2 : offset + 2 + slen].decode("utf-8"), offset + 2 + slen


class AsyncMQTTClient:
    """Minimal async MQTT 3.1.1 client for the simulator."""

    def __init__(self, client_id: str, host: str, port: int):
        self.client_id = client_id
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._packet_id = 0
        self._on_message = None
        self._read_task: Optional[asyncio.Task] = None

    def on_message(self, callback):
        """Register callback: async def handler(topic, payload_dict)"""
        self._on_message = callback

    def _next_packet_id(self) -> int:
        self._packet_id = (self._packet_id % 65535) + 1
        return self._packet_id

    async def connect(self):
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port
        )
        # CONNECT packet
        var_header = _encode_utf8("MQTT") + bytes([4, 2, 0, 60])
        payload = _encode_utf8(self.client_id)
        remaining = var_header + payload
        pkt = bytes([0x10]) + _encode_remaining_length(len(remaining)) + remaining
        self._writer.write(pkt)
        await self._writer.drain()

        # Read CONNACK
        header = await self._reader.read(1)
        if not header or (header[0] >> 4) != 2:
            raise ConnectionError("No CONNACK received")
        await self._reader.read(3)  # rest of CONNACK
        logger.debug("[%s] Connected to %s:%d", self.client_id, self.host, self.port)

        # Start background reader
        self._read_task = asyncio.create_task(self._read_loop())

    async def disconnect(self):
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            try:
                self._writer.write(bytes([0xE0, 0x00]))  # DISCONNECT
                await self._writer.drain()
                self._writer.close()
            except Exception:
                pass

    async def subscribe(self, topic: str, qos: int = 1):
        pid = self._next_packet_id()
        var_header = struct.pack("!H", pid)
        payload = _encode_utf8(topic) + bytes([qos])
        remaining = var_header + payload
        pkt = bytes([0x82]) + _encode_remaining_length(len(remaining)) + remaining
        self._writer.write(pkt)
        await self._writer.drain()
        logger.debug("[%s] Subscribed to %s", self.client_id, topic)

    async def publish(self, topic: str, payload: dict, qos: int = 1):
        pid = self._next_packet_id()
        payload_bytes = json.dumps(payload).encode("utf-8")
        var_header = _encode_utf8(topic)
        flags = 0
        if qos > 0:
            flags |= (qos << 1)
            var_header += struct.pack("!H", pid)
        remaining = var_header + payload_bytes
        pkt = bytes([(3 << 4) | flags]) + _encode_remaining_length(len(remaining)) + remaining
        self._writer.write(pkt)
        await self._writer.drain()

    async def _read_loop(self):
        try:
            while True:
                header = await self._reader.read(1)
                if not header:
                    break
                pkt_type = (header[0] >> 4) & 0x0F
                flags = header[0] & 0x0F

                # Read remaining length
                remaining_bytes = bytearray()
                while True:
                    b = await self._reader.read(1)
                    if not b:
                        return
                    remaining_bytes.append(b[0])
                    if (b[0] & 0x80) == 0:
                        break
                mult, val = 1, 0
                for byte in remaining_bytes:
                    val += (byte & 0x7F) * mult
                    mult *= 128
                body = b""
                if val > 0:
                    body = await self._read_exact(val)

                if pkt_type == 3:  # PUBLISH
                    await self._handle_publish(flags, body)
                elif pkt_type == 4:  # PUBACK
                    pass  # ignore
                elif pkt_type == 9:  # SUBACK
                    pass
                elif pkt_type == 13:  # PINGRESP
                    pass
        except (asyncio.CancelledError, ConnectionError, OSError):
            pass

    async def _read_exact(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = await self._reader.read(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data.extend(chunk)
        return bytes(data)

    async def _handle_publish(self, flags: int, body: bytes):
        offset = 0
        topic, offset = _decode_utf8(body, offset)
        qos = (flags >> 1) & 3
        if qos > 0:
            offset += 2  # skip packet_id
        payload_bytes = body[offset:]
        try:
            payload = json.loads(payload_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        if self._on_message:
            try:
                await self._on_message(topic, payload)
            except Exception:
                logger.exception("[%s] on_message error", self.client_id)


# ---------------------------------------------------------------------------
# Simulated Worker
# ---------------------------------------------------------------------------

# Fake GPU templates for variety
_GPU_TEMPLATES = [
    {"name": "NVIDIA GeForce RTX 4090", "memory_gb": 24},
    {"name": "NVIDIA GeForce RTX 3090", "memory_gb": 24},
    {"name": "NVIDIA GeForce RTX 3080", "memory_gb": 10},
    {"name": "NVIDIA GeForce RTX 4080", "memory_gb": 16},
    {"name": "NVIDIA GeForce RTX 3070", "memory_gb": 8},
]


@dataclass
class SimulatedWorker:
    worker_id: str
    eth_address: str
    mqtt_host: str
    mqtt_port: int
    block_interval: float  # avg seconds between simulated blocks
    gpu_count: int = 2
    api_port: int = 0  # Chain RPC port (0 = skip chain submission)

    # Runtime state
    state: WorkerState = WorkerState.IDLE
    lease_id: str = ""
    consumer_address: str = ""
    prefix: str = ""
    duration_sec: int = 0
    lease_start: float = 0.0
    blocks_found: int = 0
    total_blocks: int = 0
    hashrate: float = 0.0
    start_time: float = field(default_factory=time.time)

    _client: Optional[AsyncMQTTClient] = field(default=None, repr=False)
    _tasks: List[asyncio.Task] = field(default_factory=list, repr=False)
    _running: bool = False

    def _log(self, msg: str, *args):
        logger.info("[%-20s %s] " + msg, self.worker_id, self.state.value, *args)

    async def start(self):
        """Connect and begin worker lifecycle."""
        self._running = True
        self._client = AsyncMQTTClient(self.worker_id, self.mqtt_host, self.mqtt_port)
        self._client.on_message(self._on_message)
        await self._client.connect()
        self._log("Connected to MQTT broker")

        # Subscribe to platform topics
        await self._client.subscribe(f"xenminer/{self.worker_id}/task")
        await self._client.subscribe(f"xenminer/{self.worker_id}/control")

        # Send registration
        await self._send_register()

        # Start heartbeat loop
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        # Send offline status
        await self._send_status("offline")
        await self._client.disconnect()
        self._log("Disconnected")

    # -------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------

    async def _on_message(self, topic: str, payload: dict):
        command = payload.get("command", "")

        if command == "register_ack":
            accepted = payload.get("accepted", False)
            if accepted:
                self._log("Registration accepted")
                await self._transition(WorkerState.AVAILABLE)
            else:
                reason = payload.get("reason", "unknown")
                self._log("Registration rejected: %s", reason)
                await self._transition(WorkerState.ERROR)

        elif command == "assign_task":
            self.lease_id = payload.get("lease_id", "")
            self.consumer_address = payload.get("consumer_address", "")
            self.prefix = payload.get("prefix", "")
            self.duration_sec = payload.get("duration_sec", 3600)
            self.lease_start = time.time()
            self.blocks_found = 0
            self._log("Lease assigned: %s prefix=%s duration=%ds",
                       self.lease_id, self.prefix, self.duration_sec)
            await self._transition(WorkerState.LEASED)
            # Start mining
            await self._transition(WorkerState.MINING)
            self._tasks.append(asyncio.create_task(self._mining_loop()))

        elif command == "release":
            release_lease = payload.get("lease_id", "")
            if release_lease and release_lease != self.lease_id:
                self._log("Ignoring release for mismatched lease %s", release_lease)
                return
            self._log("Lease released: %s (blocks=%d)", self.lease_id, self.blocks_found)
            await self._transition(WorkerState.COMPLETED)
            self.lease_id = ""
            self.prefix = ""
            self.consumer_address = ""
            await self._transition(WorkerState.AVAILABLE)

        else:
            # Control messages
            action = payload.get("action", "")
            if action:
                self._log("Control action: %s", action)

    async def _transition(self, new_state: WorkerState):
        self.state = new_state
        await self._send_status(new_state.value)

    # -------------------------------------------------------------------
    # Message sending
    # -------------------------------------------------------------------

    async def _send_register(self):
        gpus = []
        for i in range(self.gpu_count):
            tmpl = _GPU_TEMPLATES[i % len(_GPU_TEMPLATES)]
            gpus.append({
                "index": i,
                "name": tmpl["name"],
                "memory_gb": tmpl["memory_gb"],
                "bus_id": i + 1,
            })
        total_mem = sum(g["memory_gb"] for g in gpus)
        await self._client.publish(f"xenminer/{self.worker_id}/register", {
            "worker_id": self.worker_id,
            "eth_address": self.eth_address,
            "gpu_count": self.gpu_count,
            "total_memory_gb": total_mem,
            "gpus": gpus,
            "version": "2.0.0",
            "timestamp": int(time.time()),
        })
        self._log("Sent registration (%d GPUs, %dGB)", self.gpu_count, total_mem)

    async def _send_status(self, state: str):
        msg = {
            "worker_id": self.worker_id,
            "state": state,
            "timestamp": int(time.time()),
        }
        if self.lease_id:
            msg["lease_id"] = self.lease_id
        await self._client.publish(f"xenminer/{self.worker_id}/status", msg)

    async def _send_heartbeat(self):
        await self._client.publish(f"xenminer/{self.worker_id}/heartbeat", {
            "worker_id": self.worker_id,
            "hashrate": round(self.hashrate, 2),
            "active_gpus": self.gpu_count if self.state == WorkerState.MINING else 0,
            "accepted_blocks": self.total_blocks,
            "difficulty": 8,
            "uptime_sec": int(time.time() - self.start_time),
            "timestamp": int(time.time()),
        })

    async def _send_block_found(self):
        # Generate key with correct prefix
        suffix = secrets.token_hex(24)  # 48 hex chars
        key = self.prefix + suffix
        # Generate a hash that contains XEN11 (so chain simulator accepts it)
        block_hash = secrets.token_hex(10) + "XEN11" + secrets.token_hex(19)
        attempts = random.randint(100000, 2000000)

        # Submit to chain RPC first (like real C++ miner does)
        chain_block_id = None
        if self.api_port > 0:
            chain_block_id = await self._submit_to_chain(
                block_hash, key, self.consumer_address, attempts,
            )

        # Then report via MQTT
        await self._client.publish(f"xenminer/{self.worker_id}/block", {
            "worker_id": self.worker_id,
            "lease_id": self.lease_id,
            "hash": block_hash,
            "key": key,
            "account": self.consumer_address,
            "attempts": attempts,
            "hashrate": f"{self.hashrate:.2f}",
            "timestamp": int(time.time()),
        })
        self.blocks_found += 1
        self.total_blocks += 1
        verified_str = f" chain_block=#{chain_block_id}" if chain_block_id else ""
        self._log("Block found! lease=%s total=%d%s", self.lease_id, self.blocks_found, verified_str)

    async def _submit_to_chain(self, block_hash: str, key: str, account: str, attempts: int) -> Optional[int]:
        """Submit block to chain simulator /verify endpoint (like real C++ miner)."""
        try:
            payload = json.dumps({
                "hash_to_verify": block_hash,
                "key": key,
                "account": account,
                "attempts": str(attempts),
                "hashes_per_second": f"{self.hashrate:.2f}",
                "worker": self.worker_id,
            })
            # Run blocking HTTP in executor to avoid stalling async loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._chain_post, payload)
            if result and result.get("status") == "success":
                return result.get("block_id")
            else:
                self._log("Chain submission rejected: %s", result.get("message", "unknown") if result else "no response")
                return None
        except Exception as e:
            self._log("Chain submission error: %s", e)
            return None

    def _chain_post(self, payload: str) -> Optional[dict]:
        """Blocking HTTP POST to chain /verify endpoint."""
        try:
            conn = HTTPConnection(self.mqtt_host, self.api_port, timeout=5)
            conn.request("POST", "/verify", payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read().decode())
            conn.close()
            return data
        except Exception:
            return None

    # -------------------------------------------------------------------
    # Background loops
    # -------------------------------------------------------------------

    async def _heartbeat_loop(self):
        try:
            while self._running:
                await self._send_heartbeat()
                await asyncio.sleep(HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            pass

    async def _mining_loop(self):
        """Simulate mining: periodically 'find' blocks."""
        self._log("Mining started (avg block every %.1fs)", self.block_interval)
        self.hashrate = random.uniform(800.0, 2000.0)
        try:
            while self._running and self.state == WorkerState.MINING:
                # Random delay until next block (exponential distribution)
                delay = random.expovariate(1.0 / self.block_interval)
                await asyncio.sleep(delay)

                if self.state != WorkerState.MINING:
                    break

                # Simulate varying hashrate
                self.hashrate += random.uniform(-50.0, 50.0)
                self.hashrate = max(200.0, self.hashrate)

                await self._send_block_found()
        except asyncio.CancelledError:
            pass
        self._log("Mining loop ended")


# ---------------------------------------------------------------------------
# Main: run multiple workers
# ---------------------------------------------------------------------------

async def run_simulator(
    num_workers: int,
    mqtt_host: str,
    mqtt_port: int,
    block_interval: float,
    gpu_count: int,
    api_port: int = 0,
):
    workers: List[SimulatedWorker] = []

    for i in range(num_workers):
        worker_id = f"sim-worker-{i+1:03d}"
        eth_address = f"0x{secrets.token_hex(20)}"
        worker = SimulatedWorker(
            worker_id=worker_id,
            eth_address=eth_address,
            mqtt_host=mqtt_host,
            mqtt_port=mqtt_port,
            block_interval=block_interval,
            gpu_count=gpu_count,
            api_port=api_port,
        )
        workers.append(worker)

    logger.info("Starting %d simulated workers (block_interval=%.1fs)", num_workers, block_interval)

    # Start all workers
    for w in workers:
        await w.start()
        await asyncio.sleep(0.2)  # stagger connections

    logger.info("All %d workers started. Press Ctrl+C to stop.", num_workers)

    # Wait forever
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down workers...")
        for w in workers:
            await w.stop()
        logger.info("All workers stopped.")


def main():
    parser = argparse.ArgumentParser(description="XenMiner Worker Simulator")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of simulated workers (default: 2)")
    parser.add_argument("--mqtt-host", default="localhost",
                        help="MQTT broker host (default: localhost)")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port (default: 1883)")
    parser.add_argument("--block-interval", type=float, default=5.0,
                        help="Average seconds between simulated blocks (default: 5.0)")
    parser.add_argument("--gpu-count", type=int, default=2,
                        help="GPUs per simulated worker (default: 2)")
    parser.add_argument("--api-port", type=int, default=0,
                        help="Platform API port for chain RPC submission (default: 0 = disabled)")
    args = parser.parse_args()

    try:
        asyncio.run(run_simulator(
            num_workers=args.workers,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            block_interval=args.block_interval,
            gpu_count=args.gpu_count,
            api_port=args.api_port,
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted.")


if __name__ == "__main__":
    main()
