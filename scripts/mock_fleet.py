#!/usr/bin/env python3
"""
mock_fleet.py - Standalone mock worker fleet simulator.

Spawns N simulated mining workers that connect to the server's MQTT broker,
send registration/heartbeat/block messages, and periodically go offline.

Usage:
    python scripts/mock_fleet.py --workers 5 --broker localhost --port 1883 --block-interval 60
"""

import argparse
import asyncio
import json
import logging
import random
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import List

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-10s] %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fleet")

GPU_PROFILES = [
    {"name": "NVIDIA GeForce RTX 4090", "memory_gb": 24, "hashrate_range": (150, 170)},
    {"name": "NVIDIA GeForce RTX 4080", "memory_gb": 16, "hashrate_range": (120, 140)},
    {"name": "NVIDIA GeForce RTX 4070 Ti", "memory_gb": 12, "hashrate_range": (95, 115)},
    {"name": "NVIDIA GeForce RTX 4070", "memory_gb": 12, "hashrate_range": (80, 100)},
    {"name": "NVIDIA GeForce RTX 3090 Ti", "memory_gb": 24, "hashrate_range": (100, 120)},
    {"name": "NVIDIA GeForce RTX 3090", "memory_gb": 24, "hashrate_range": (90, 110)},
    {"name": "NVIDIA GeForce RTX 3080 Ti", "memory_gb": 12, "hashrate_range": (80, 95)},
    {"name": "NVIDIA GeForce RTX 3080", "memory_gb": 10, "hashrate_range": (70, 90)},
    {"name": "NVIDIA GeForce RTX 3070", "memory_gb": 8, "hashrate_range": (55, 70)},
    {"name": "NVIDIA GeForce RTX 3060 Ti", "memory_gb": 8, "hashrate_range": (45, 60)},
    {"name": "NVIDIA A100", "memory_gb": 80, "hashrate_range": (200, 250)},
    {"name": "NVIDIA A100", "memory_gb": 40, "hashrate_range": (180, 220)},
    {"name": "NVIDIA H100", "memory_gb": 80, "hashrate_range": (280, 350)},
    {"name": "NVIDIA L40", "memory_gb": 48, "hashrate_range": (160, 200)},
    {"name": "NVIDIA RTX A6000", "memory_gb": 48, "hashrate_range": (140, 170)},
    {"name": "NVIDIA RTX A5000", "memory_gb": 24, "hashrate_range": (100, 130)},
    {"name": "AMD Radeon RX 7900 XTX", "memory_gb": 24, "hashrate_range": (110, 140)},
    {"name": "AMD Radeon RX 7900 XT", "memory_gb": 20, "hashrate_range": (90, 115)},
    {"name": "AMD Radeon RX 6900 XT", "memory_gb": 16, "hashrate_range": (70, 90)},
]

# Multi-GPU configurations (gpu_count, same_gpu probability)
MULTI_GPU_CONFIGS = [
    (1, 1.0),   # 50% single GPU
    (2, 0.8),   # 25% dual GPU (80% same type)
    (4, 0.9),   # 15% quad GPU (90% same type)
    (8, 1.0),   # 10% 8-GPU rig (always same type)
]


@dataclass
class SimWorker:
    worker_id: str
    eth_address: str
    gpus: list  # List of GPU profiles
    client: mqtt.Client = field(init=False, repr=False)
    online: bool = True
    blocks_found: int = 0
    current_hashrate: float = 0.0
    _connected: bool = False

    def __post_init__(self):
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"sim-{self.worker_id}",
            protocol=mqtt.MQTTv311,
        )
        # Sum hashrate from all GPUs
        total_hr = 0.0
        for gpu in self.gpus:
            lo, hi = gpu["hashrate_range"]
            total_hr += random.uniform(lo, hi)
        self.current_hashrate = total_hr

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_memory_gb(self) -> int:
        return sum(g["memory_gb"] for g in self.gpus)


def make_eth_address() -> str:
    return "0x" + uuid.uuid4().hex[:40]


class FleetSimulator:
    def __init__(self, n_workers: int, broker_host: str, broker_port: int,
                 block_interval: float, owner: str = ""):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.block_interval = block_interval
        self.workers: List[SimWorker] = []
        self._stop = False

        for i in range(n_workers):
            # Decide GPU count: 50% 1-GPU, 25% 2-GPU, 15% 4-GPU, 10% 8-GPU
            r = random.random()
            if r < 0.5:
                gpu_count, same_prob = 1, 1.0
            elif r < 0.75:
                gpu_count, same_prob = 2, 0.8
            elif r < 0.9:
                gpu_count, same_prob = 4, 0.9
            else:
                gpu_count, same_prob = 8, 1.0

            # Generate GPU list
            primary_gpu = random.choice(GPU_PROFILES)
            gpus = []
            for idx in range(gpu_count):
                if idx == 0 or random.random() < same_prob:
                    gpu = primary_gpu.copy()
                else:
                    gpu = random.choice(GPU_PROFILES).copy()
                gpu["index"] = idx
                gpu["bus_id"] = idx
                gpus.append(gpu)

            w = SimWorker(
                worker_id=f"sim-worker-{i:03d}",
                eth_address=owner if owner else make_eth_address(),
                gpus=gpus,
            )
            self.workers.append(w)

    def _topic(self, worker: SimWorker, msg_type: str) -> str:
        return f"xenminer/{worker.worker_id}/{msg_type}"

    def _connect_worker(self, w: SimWorker):
        try:
            w.client.connect(self.broker_host, self.broker_port, keepalive=60)
            w.client.loop_start()
            w._connected = True
        except Exception as e:
            logger.error("Failed to connect %s: %s", w.worker_id, e)
            w._connected = False

    def _disconnect_worker(self, w: SimWorker):
        try:
            w.client.loop_stop()
            w.client.disconnect()
        except Exception:
            pass
        w._connected = False

    def _send_register(self, w: SimWorker):
        if not w._connected:
            return
        msg = {
            "worker_id": w.worker_id,
            "eth_address": w.eth_address,
            "gpu_count": w.gpu_count,
            "total_memory_gb": w.total_memory_gb,
            "gpus": [{"index": g["index"], "name": g["name"],
                       "memory_gb": g["memory_gb"], "bus_id": g["bus_id"]} for g in w.gpus],
            "version": "2.0.0",
            "timestamp": int(time.time()),
        }
        w.client.publish(self._topic(w, "register"), json.dumps(msg), qos=1)

    def _send_heartbeat(self, w: SimWorker):
        if not w._connected or not w.online:
            return
        # Fluctuate hashrate slightly
        base_hr = sum((g["hashrate_range"][0] + g["hashrate_range"][1]) / 2 for g in w.gpus)
        w.current_hashrate += random.uniform(-5 * w.gpu_count, 5 * w.gpu_count)
        w.current_hashrate = max(base_hr * 0.85, min(base_hr * 1.15, w.current_hashrate))
        msg = {
            "worker_id": w.worker_id,
            "hashrate": round(w.current_hashrate, 2),
            "active_gpus": w.gpu_count,
            "accepted_blocks": w.blocks_found,
            "difficulty": 8,
            "uptime_sec": 0,
            "timestamp": int(time.time()),
        }
        w.client.publish(self._topic(w, "heartbeat"), json.dumps(msg), qos=0)

    def _send_block(self, w: SimWorker):
        if not w._connected or not w.online:
            return
        w.blocks_found += 1
        msg = {
            "worker_id": w.worker_id,
            "lease_id": "",
            "hash": "0000" + uuid.uuid4().hex[:60],
            "key": uuid.uuid4().hex[:64].ljust(64, "0"),
            "account": w.eth_address,
            "attempts": random.randint(50000, 500000),
            "hashrate": f"{w.current_hashrate:.2f}",
            "timestamp": int(time.time()),
        }
        w.client.publish(self._topic(w, "block"), json.dumps(msg), qos=1)
        logger.info("Block found by %s (total: %d)", w.worker_id, w.blocks_found)

    def _print_status(self):
        online = sum(1 for w in self.workers if w.online)
        total_hr = sum(w.current_hashrate for w in self.workers if w.online)
        total_blocks = sum(w.blocks_found for w in self.workers)
        logger.info(
            "Fleet: %d/%d online | %.1f H/s total | %d blocks",
            online, len(self.workers), total_hr, total_blocks,
        )

    async def run(self):
        # Connect all workers
        for w in self.workers:
            self._connect_worker(w)
            self._send_register(w)

        tick = 0
        try:
            while not self._stop:
                await asyncio.sleep(1)
                tick += 1

                # Heartbeats every 30s
                if tick % 30 == 0:
                    for w in self.workers:
                        if w.online:
                            self._send_heartbeat(w)

                # Block discovery (Poisson: lambda = 1/block_interval per worker per second)
                for w in self.workers:
                    if w.online and w._connected:
                        if random.random() < (1.0 / self.block_interval):
                            self._send_block(w)

                # Offline/online toggling: 10% chance every 5 min per worker
                if tick % 300 == 0:
                    for w in self.workers:
                        if random.random() < 0.1:
                            if w.online:
                                w.online = False
                                logger.info("%s going offline", w.worker_id)
                                offline_dur = random.randint(30, 120)
                                asyncio.get_event_loop().call_later(
                                    offline_dur, self._bring_online, w,
                                )
                # Status every 30s
                if tick % 30 == 0:
                    self._print_status()

        except asyncio.CancelledError:
            pass
        finally:
            for w in self.workers:
                self._disconnect_worker(w)

    def _bring_online(self, w: SimWorker):
        w.online = True
        if not w._connected:
            self._connect_worker(w)
            self._send_register(w)
        logger.info("%s back online", w.worker_id)

    def stop(self):
        self._stop = True


def main():
    parser = argparse.ArgumentParser(description="Mock mining fleet simulator")
    parser.add_argument("--workers", type=int, default=5, help="Number of simulated workers")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--block-interval", type=float, default=60,
                        help="Average seconds between blocks per worker")
    parser.add_argument("--owner", default="",
                        help="Ethereum address for all workers (default: random per worker)")
    args = parser.parse_args()

    fleet = FleetSimulator(args.workers, args.broker, args.port, args.block_interval, owner=args.owner)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler():
        logger.info("Shutting down fleet...")
        fleet.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("Starting fleet: %d workers -> %s:%d", args.workers, args.broker, args.port)
    try:
        loop.run_until_complete(fleet.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
