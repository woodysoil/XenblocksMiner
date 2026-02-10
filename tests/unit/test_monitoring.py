"""
test_monitoring.py - Unit tests for MonitoringService.

Tests fleet statistics aggregation, hashrate storage, worker health
detection, and block rate calculation in isolation using in-memory SQLite.
"""

import time
import pytest
import pytest_asyncio
import aiosqlite

from server.monitoring import MonitoringService, OFFLINE_THRESHOLD
from server.storage import (
    WorkerRepo, BlockRepo, SnapshotRepo, SCHEMA_SQL, SCHEMA_VERSION,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.executescript(SCHEMA_SQL)
    await conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, time.time()),
    )
    await conn.commit()
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def service(db):
    workers = WorkerRepo(db)
    blocks = BlockRepo(db)
    snapshots = SnapshotRepo(db)
    return MonitoringService(workers, blocks, snapshots)


async def _insert_worker(db, worker_id, hashrate=100.0, active_gpus=1,
                         last_hb_offset=0, state="AVAILABLE"):
    now = time.time()
    gpus = '[{"index":0,"name":"RTX 4090","memory_gb":24,"bus_id":0}]'
    await db.execute(
        "INSERT INTO workers "
        "(worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
        "version, state, hashrate, active_gpus, last_heartbeat, registered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (worker_id, "0x" + "ab" * 20, 1, 24, gpus, "2.0.0", state,
         hashrate, active_gpus, now - last_hb_offset, now),
    )
    await db.commit()


async def _insert_block(db, worker_id, created_offset=0):
    now = time.time()
    await db.execute(
        "INSERT INTO blocks "
        "(lease_id, worker_id, block_hash, key, account, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("", worker_id, "0000" + "aa" * 30, "bb" * 32, "0x" + "cc" * 20,
         now - created_offset),
    )
    await db.commit()


# ── Fleet overview ────────────────────────────────────────────────────────

class TestFleetOverview:

    async def test_empty_fleet(self, service):
        result = await service.get_fleet_overview()
        assert result == []

    async def test_online_worker(self, db, service):
        await _insert_worker(db, "w1", hashrate=150.0, last_hb_offset=10)
        fleet = await service.get_fleet_overview()
        assert len(fleet) == 1
        assert fleet[0]["worker_id"] == "w1"
        assert fleet[0]["online"] is True
        assert fleet[0]["hashrate"] == 150.0

    async def test_offline_worker(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=OFFLINE_THRESHOLD + 10)
        fleet = await service.get_fleet_overview()
        assert len(fleet) == 1
        assert fleet[0]["online"] is False

    async def test_mixed_online_offline(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=5)
        await _insert_worker(db, "w2", last_hb_offset=OFFLINE_THRESHOLD + 1)
        fleet = await service.get_fleet_overview()
        statuses = {w["worker_id"]: w["online"] for w in fleet}
        assert statuses["w1"] is True
        assert statuses["w2"] is False


# ── Aggregated stats ──────────────────────────────────────────────────────

class TestAggregatedStats:

    async def test_empty_stats(self, service):
        stats = await service.get_aggregated_stats()
        assert stats["total_workers"] == 0
        assert stats["online"] == 0
        assert stats["total_hashrate"] == 0

    async def test_online_workers_counted(self, db, service):
        await _insert_worker(db, "w1", hashrate=100.0, last_hb_offset=5)
        await _insert_worker(db, "w2", hashrate=200.0, last_hb_offset=10)
        stats = await service.get_aggregated_stats()
        assert stats["online"] == 2
        assert stats["offline"] == 0
        assert stats["total_hashrate"] == 300.0

    async def test_offline_workers_excluded_from_hashrate(self, db, service):
        await _insert_worker(db, "w1", hashrate=100.0, last_hb_offset=5)
        await _insert_worker(db, "w2", hashrate=200.0,
                             last_hb_offset=OFFLINE_THRESHOLD + 1)
        stats = await service.get_aggregated_stats()
        assert stats["online"] == 1
        assert stats["offline"] == 1
        assert stats["total_hashrate"] == 100.0

    async def test_blocks_last_hour(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=5)
        await _insert_block(db, "w1", created_offset=0)
        await _insert_block(db, "w1", created_offset=1800)
        await _insert_block(db, "w1", created_offset=7200)  # older than 1h
        stats = await service.get_aggregated_stats()
        assert stats["total_blocks"] == 3
        assert stats["blocks_last_hour"] == 2

    async def test_gpu_counts(self, db, service):
        await _insert_worker(db, "w1", active_gpus=2, last_hb_offset=5)
        await _insert_worker(db, "w2", active_gpus=1,
                             last_hb_offset=OFFLINE_THRESHOLD + 1)
        stats = await service.get_aggregated_stats()
        assert stats["total_gpus"] == 2  # gpu_count=1 each
        assert stats["active_gpus"] == 2  # only online w1's active_gpus


# ── Hashrate snapshots ────────────────────────────────────────────────────

class TestHashrateSnapshots:

    async def test_record_and_query(self, db, service):
        await _insert_worker(db, "w1", hashrate=150.0, last_hb_offset=5)
        await service.record_hashrate_snapshot()
        history = await service.get_hashrate_history(worker_id="w1", hours=1)
        assert len(history) == 1
        assert history[0]["worker_id"] == "w1"
        assert history[0]["hashrate"] == 150.0

    async def test_offline_workers_not_recorded(self, db, service):
        await _insert_worker(db, "w1", hashrate=150.0,
                             last_hb_offset=OFFLINE_THRESHOLD + 1)
        await service.record_hashrate_snapshot()
        history = await service.get_hashrate_history(worker_id="w1", hours=1)
        assert len(history) == 0

    async def test_query_all_workers(self, db, service):
        await _insert_worker(db, "w1", hashrate=100.0, last_hb_offset=5)
        await _insert_worker(db, "w2", hashrate=200.0, last_hb_offset=5)
        await service.record_hashrate_snapshot()
        history = await service.get_hashrate_history(hours=1)
        assert len(history) == 2

    async def test_cleanup_old_snapshots(self, db, service):
        await _insert_worker(db, "w1", hashrate=100.0, last_hb_offset=5)
        # Insert an old snapshot (25h ago) and a recent one (23h ago)
        now = time.time()
        await db.execute(
            "INSERT INTO hashrate_snapshots (worker_id, hashrate, active_gpus, timestamp) "
            "VALUES (?, ?, ?, ?)",
            ("w1", 50.0, 1, now - 25 * 3600),
        )
        await db.execute(
            "INSERT INTO hashrate_snapshots (worker_id, hashrate, active_gpus, timestamp) "
            "VALUES (?, ?, ?, ?)",
            ("w1", 80.0, 1, now - 23 * 3600),
        )
        await db.commit()
        await service.record_hashrate_snapshot()  # fresh one

        # Query within 24h bound returns the 23h + fresh snapshots
        history_before = await service.get_hashrate_history(worker_id="w1", hours=24)
        assert len(history_before) == 2

        await service.cleanup_old_snapshots()  # removes >24h (the 25h-old one)

        # Verify the 25h-old row was actually removed via raw count
        async with db.execute(
            "SELECT COUNT(*) FROM hashrate_snapshots WHERE worker_id = ?", ("w1",)
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 2  # 23h-old + fresh remain, 25h-old deleted


# ── Worker health ─────────────────────────────────────────────────────────

class TestWorkerHealth:

    async def test_all_healthy(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=10)
        await _insert_worker(db, "w2", last_hb_offset=30)
        unhealthy = await service.check_worker_health()
        assert len(unhealthy) == 0

    async def test_offline_detected(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=OFFLINE_THRESHOLD + 5)
        unhealthy = await service.check_worker_health()
        assert len(unhealthy) == 1
        assert unhealthy[0]["worker_id"] == "w1"
        assert unhealthy[0]["offline_sec"] >= OFFLINE_THRESHOLD

    async def test_mixed_health(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=10)
        await _insert_worker(db, "w2", last_hb_offset=OFFLINE_THRESHOLD + 20)
        await _insert_worker(db, "w3", last_hb_offset=OFFLINE_THRESHOLD + 100)
        unhealthy = await service.check_worker_health()
        ids = {u["worker_id"] for u in unhealthy}
        assert ids == {"w2", "w3"}


# ── Recent blocks ─────────────────────────────────────────────────────────

class TestRecentBlocks:

    async def test_empty(self, service):
        blocks = await service.get_recent_blocks()
        assert blocks == []

    async def test_returns_blocks(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=5)
        await _insert_block(db, "w1")
        await _insert_block(db, "w1")
        blocks = await service.get_recent_blocks(limit=10)
        assert len(blocks) == 2

    async def test_limit_respected(self, db, service):
        await _insert_worker(db, "w1", last_hb_offset=5)
        for _ in range(5):
            await _insert_block(db, "w1")
        blocks = await service.get_recent_blocks(limit=3)
        assert len(blocks) == 3
