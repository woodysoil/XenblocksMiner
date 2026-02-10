"""
test_monitoring_api.py - Integration tests for monitoring REST endpoints.

Tests /api/monitoring/* endpoints using FastAPI TestClient with
real repos backed by in-memory SQLite.
"""

import time
import pytest
import pytest_asyncio

from server.monitoring import MonitoringService, OFFLINE_THRESHOLD
from server.storage import StorageManager

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def storage():
    sm = StorageManager(":memory:")
    await sm.initialize()
    yield sm
    await sm.close()


@pytest_asyncio.fixture
async def monitoring(storage):
    return MonitoringService(storage.workers, storage.blocks, storage.snapshots)


@pytest.fixture
def app(storage, monitoring):
    from fastapi import FastAPI
    _app = FastAPI()

    @_app.get("/api/monitoring/fleet")
    async def fleet():
        return await monitoring.get_fleet_overview()

    @_app.get("/api/monitoring/stats")
    async def stats():
        return await monitoring.get_aggregated_stats()

    @_app.get("/api/monitoring/hashrate-history")
    async def hashrate_history(worker_id: str = None, hours: float = 1):
        return await monitoring.get_hashrate_history(worker_id=worker_id, hours=hours)

    @_app.get("/api/monitoring/blocks/recent")
    async def recent_blocks(limit: int = 20):
        return await monitoring.get_recent_blocks(limit=limit)

    return _app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


async def _seed_worker(storage, worker_id, hashrate=100.0, active_gpus=1,
                       hb_offset=0):
    gpus = [{"index": 0, "name": "RTX 4090", "memory_gb": 24, "bus_id": 0}]
    await storage.workers.upsert(
        worker_id=worker_id,
        eth_address="0x" + "ab" * 20,
        gpu_count=1,
        total_memory_gb=24,
        gpus=gpus,
        version="2.0.0",
    )
    now = time.time()
    await storage.workers._db.execute(
        "UPDATE workers SET hashrate = ?, active_gpus = ?, last_heartbeat = ? "
        "WHERE worker_id = ?",
        (hashrate, active_gpus, now - hb_offset, worker_id),
    )
    await storage.workers._db.commit()


async def _seed_block(storage, worker_id, created_offset=0):
    now = time.time()
    await storage.blocks.create(
        lease_id="",
        worker_id=worker_id,
        block_hash="0000" + "aa" * 30,
        key="bb" * 32,
        account="0x" + "cc" * 20,
    )
    if created_offset > 0:
        await storage.blocks._db.execute(
            "UPDATE blocks SET created_at = ? "
            "WHERE worker_id = ? AND id = (SELECT MAX(id) FROM blocks WHERE worker_id = ?)",
            (now - created_offset, worker_id, worker_id),
        )
        await storage.blocks._db.commit()


# ── /api/monitoring/fleet ─────────────────────────────────────────────────

class TestFleetEndpoint:

    async def test_empty(self, client):
        r = client.get("/api/monitoring/fleet")
        assert r.status_code == 200
        assert r.json() == []

    async def test_returns_workers(self, storage, client):
        await _seed_worker(storage, "w1", hashrate=150.0, hb_offset=5)
        await _seed_worker(storage, "w2", hashrate=200.0,
                           hb_offset=OFFLINE_THRESHOLD + 10)
        r = client.get("/api/monitoring/fleet")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        by_id = {w["worker_id"]: w for w in data}
        assert by_id["w1"]["online"] is True
        assert by_id["w2"]["online"] is False

    async def test_worker_fields_present(self, storage, client):
        await _seed_worker(storage, "w1")
        r = client.get("/api/monitoring/fleet")
        w = r.json()[0]
        for f in ("worker_id", "online", "hashrate", "gpu_count",
                  "last_heartbeat", "state"):
            assert f in w, f"Missing field: {f}"


# ── /api/monitoring/stats ─────────────────────────────────────────────────

class TestStatsEndpoint:

    async def test_empty(self, client):
        r = client.get("/api/monitoring/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_workers"] == 0
        assert data["total_hashrate"] == 0

    async def test_aggregated_values(self, storage, client):
        await _seed_worker(storage, "w1", hashrate=100.0, hb_offset=5)
        await _seed_worker(storage, "w2", hashrate=200.0, hb_offset=10)
        r = client.get("/api/monitoring/stats")
        data = r.json()
        assert data["total_workers"] == 2
        assert data["online"] == 2
        assert data["total_hashrate"] == 300.0

    async def test_blocks_counted(self, storage, client):
        await _seed_worker(storage, "w1", hb_offset=5)
        await _seed_block(storage, "w1")
        await _seed_block(storage, "w1")
        r = client.get("/api/monitoring/stats")
        assert r.json()["total_blocks"] == 2


# ── /api/monitoring/hashrate-history ──────────────────────────────────────

class TestHashrateHistoryEndpoint:

    async def test_empty(self, client):
        r = client.get("/api/monitoring/hashrate-history")
        assert r.status_code == 200
        assert r.json() == []

    async def test_returns_snapshots(self, storage, monitoring, client):
        await _seed_worker(storage, "w1", hashrate=150.0, hb_offset=5)
        await monitoring.record_hashrate_snapshot()
        r = client.get("/api/monitoring/hashrate-history?worker_id=w1&hours=1")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["hashrate"] == 150.0

    async def test_filter_by_worker(self, storage, monitoring, client):
        await _seed_worker(storage, "w1", hashrate=100.0, hb_offset=5)
        await _seed_worker(storage, "w2", hashrate=200.0, hb_offset=5)
        await monitoring.record_hashrate_snapshot()
        r = client.get("/api/monitoring/hashrate-history?worker_id=w1")
        assert len(r.json()) == 1
        r2 = client.get("/api/monitoring/hashrate-history")
        assert len(r2.json()) == 2


# ── /api/monitoring/blocks/recent ─────────────────────────────────────────

class TestRecentBlocksEndpoint:

    async def test_empty(self, client):
        r = client.get("/api/monitoring/blocks/recent")
        assert r.status_code == 200
        assert r.json() == []

    async def test_returns_blocks(self, storage, client):
        await _seed_worker(storage, "w1", hb_offset=5)
        await _seed_block(storage, "w1")
        await _seed_block(storage, "w1")
        r = client.get("/api/monitoring/blocks/recent")
        assert len(r.json()) == 2

    async def test_limit(self, storage, client):
        await _seed_worker(storage, "w1", hb_offset=5)
        for _ in range(5):
            await _seed_block(storage, "w1")
        r = client.get("/api/monitoring/blocks/recent?limit=3")
        assert len(r.json()) == 3

    async def test_ordered_descending(self, storage, client):
        await _seed_worker(storage, "w1", hb_offset=5)
        await _seed_block(storage, "w1")
        await _seed_block(storage, "w1")
        r = client.get("/api/monitoring/blocks/recent")
        data = r.json()
        assert data[0]["timestamp"] >= data[1]["timestamp"]


# ── Performance ───────────────────────────────────────────────────────────

class TestPerformance:

    async def test_fleet_response_time(self, storage, client):
        for i in range(100):
            await _seed_worker(storage, f"perf-{i:03d}", hashrate=float(i),
                               hb_offset=i % 2 * (OFFLINE_THRESHOLD + 1))
        start = time.time()
        r = client.get("/api/monitoring/fleet")
        elapsed_ms = (time.time() - start) * 1000
        assert r.status_code == 200
        assert len(r.json()) == 100
        assert elapsed_ms < 500  # generous for CI
