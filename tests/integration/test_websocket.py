"""
test_websocket.py - Integration tests for WebSocket dashboard endpoint.

Tests WebSocket connection, snapshot delivery, heartbeat/block relay,
multiple clients, and disconnection handling.
"""

import asyncio
import json
import time
import pytest
import pytest_asyncio

from server.storage import StorageManager
from server.ws import WSManager

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def storage():
    sm = StorageManager(":memory:")
    await sm.initialize()
    yield sm
    await sm.close()


@pytest_asyncio.fixture
async def ws_manager(storage):
    return WSManager(storage.workers, storage.blocks)


@pytest.fixture
def app(ws_manager):
    from fastapi import FastAPI, WebSocket
    _app = FastAPI()

    @_app.websocket("/ws/dashboard")
    async def ws_dashboard(ws: WebSocket):
        await ws_manager.handle_connection(ws)

    return _app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


async def _seed_worker(storage, worker_id, hashrate=100.0):
    gpus = [{"index": 0, "name": "RTX 4090", "memory_gb": 24, "bus_id": 0}]
    await storage.workers.upsert(
        worker_id=worker_id, eth_address="0x" + "ab" * 20,
        gpu_count=1, total_memory_gb=24, gpus=gpus, version="2.0.0",
    )
    now = time.time()
    await storage.workers._db.execute(
        "UPDATE workers SET hashrate = ?, last_heartbeat = ? WHERE worker_id = ?",
        (hashrate, now, worker_id),
    )
    await storage.workers._db.commit()


# ── Connection + Snapshot ─────────────────────────────────────────────────

class TestWebSocketConnection:

    async def test_connect_receives_snapshot(self, storage, client):
        await _seed_worker(storage, "w1", hashrate=150.0)
        with client.websocket_connect("/ws/dashboard") as ws:
            data = ws.receive_json()
            assert data["type"] == "snapshot"
            assert "workers" in data["data"]
            assert "recent_blocks" in data["data"]

    async def test_snapshot_contains_workers(self, storage, client):
        await _seed_worker(storage, "w1")
        await _seed_worker(storage, "w2")
        with client.websocket_connect("/ws/dashboard") as ws:
            data = ws.receive_json()
            assert len(data["data"]["workers"]) == 2

    async def test_empty_snapshot(self, client):
        with client.websocket_connect("/ws/dashboard") as ws:
            data = ws.receive_json()
            assert data["type"] == "snapshot"
            assert data["data"]["workers"] == []
            assert data["data"]["recent_blocks"] == []


# ── Broadcast events ─────────────────────────────────────────────────────

class TestBroadcastEvents:

    async def test_heartbeat_broadcast(self, storage, ws_manager, client):
        await _seed_worker(storage, "w1")
        with client.websocket_connect("/ws/dashboard") as ws:
            _ = ws.receive_json()  # snapshot

            await ws_manager.broadcast("heartbeat", {
                "worker_id": "w1", "hashrate": 155.0, "active_gpus": 1,
            })

            msg = ws.receive_json()
            assert msg["type"] == "heartbeat"
            assert msg["data"]["worker_id"] == "w1"
            assert msg["data"]["hashrate"] == 155.0

    async def test_block_broadcast(self, storage, ws_manager, client):
        await _seed_worker(storage, "w1")
        with client.websocket_connect("/ws/dashboard") as ws:
            _ = ws.receive_json()

            await ws_manager.broadcast("block", {
                "worker_id": "w1", "hash": "0000aabb", "lease_id": "",
            })

            msg = ws.receive_json()
            assert msg["type"] == "block"
            assert msg["data"]["worker_id"] == "w1"


# ── Multiple clients ──────────────────────────────────────────────────────

class TestMultipleClients:

    async def test_two_clients_receive_broadcast(self, storage, ws_manager, client):
        await _seed_worker(storage, "w1")
        with client.websocket_connect("/ws/dashboard") as ws1:
            _ = ws1.receive_json()
            with client.websocket_connect("/ws/dashboard") as ws2:
                _ = ws2.receive_json()

                await ws_manager.broadcast("heartbeat", {
                    "worker_id": "w1", "hashrate": 160.0, "active_gpus": 1,
                })

                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()
                assert msg1["type"] == "heartbeat"
                assert msg2["type"] == "heartbeat"


# ── Disconnection ─────────────────────────────────────────────────────────

class TestDisconnection:

    async def test_disconnect_cleans_up(self, ws_manager, client):
        with client.websocket_connect("/ws/dashboard") as ws:
            _ = ws.receive_json()
            assert len(ws_manager._clients) == 1
        # After context exit, client disconnects
        assert len(ws_manager._clients) == 0

    async def test_broadcast_after_disconnect_no_error(self, ws_manager, client):
        with client.websocket_connect("/ws/dashboard") as ws:
            _ = ws.receive_json()
        # Should not raise even with no clients
        await ws_manager.broadcast("heartbeat", {"test": True})
