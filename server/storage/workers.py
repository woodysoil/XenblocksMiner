import json
import logging
import time
from typing import List, Optional

import aiosqlite

logger = logging.getLogger("storage")


class WorkerRepo:
    """CRUD operations for the workers table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def upsert(
        self,
        worker_id: str,
        eth_address: str = "",
        gpu_count: int = 0,
        total_memory_gb: int = 0,
        gpus: Optional[list] = None,
        version: str = "",
        state: str = "AVAILABLE",
    ) -> dict:
        now = time.time()
        gpus_json = json.dumps(gpus or [])
        await self._db.execute(
            "INSERT INTO workers (worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
            "version, state, last_heartbeat, registered_at, last_online_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(worker_id) DO UPDATE SET "
            "eth_address=excluded.eth_address, gpu_count=excluded.gpu_count, "
            "total_memory_gb=excluded.total_memory_gb, gpus_json=excluded.gpus_json, "
            "version=excluded.version, state=excluded.state, last_heartbeat=excluded.last_heartbeat, "
            "last_online_at=excluded.last_online_at",
            (worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, version, state, now, now, now),
        )
        await self._db.commit()
        return await self.get(worker_id)

    async def get(self, worker_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
            "version, state, hashrate, active_gpus, last_heartbeat, registered_at, "
            "price_per_min, min_duration_sec, max_duration_sec, "
            "total_online_sec, last_online_at, self_blocks_found "
            "FROM workers WHERE worker_id = ?",
            (worker_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "worker_id": row[0],
            "eth_address": row[1],
            "gpu_count": row[2],
            "total_memory_gb": row[3],
            "gpus": json.loads(row[4]),
            "version": row[5],
            "state": row[6],
            "hashrate": row[7],
            "active_gpus": row[8],
            "last_heartbeat": row[9],
            "registered_at": row[10],
            "price_per_min": row[11],
            "min_duration_sec": row[12],
            "max_duration_sec": row[13],
            "total_online_sec": row[14] or 0.0,
            "last_online_at": row[15],
            "self_blocks_found": row[16],
        }

    async def update_heartbeat(self, worker_id: str, hashrate: float, active_gpus: int):
        now = time.time()
        async with self._db.execute(
            "SELECT last_online_at FROM workers WHERE worker_id = ?", (worker_id,)
        ) as cursor:
            row = await cursor.fetchone()
        delta = 0.0
        if row and row[0] is not None:
            delta = min(now - row[0], 60.0)
            if delta < 0:
                delta = 0.0
        await self._db.execute(
            "UPDATE workers SET hashrate = ?, active_gpus = ?, last_heartbeat = ?, "
            "total_online_sec = total_online_sec + ?, last_online_at = ? "
            "WHERE worker_id = ?",
            (hashrate, active_gpus, now, delta, now, worker_id),
        )
        await self._db.commit()

    async def update_state(self, worker_id: str, state: str):
        await self._db.execute(
            "UPDATE workers SET state = ? WHERE worker_id = ?",
            (state, worker_id),
        )
        await self._db.commit()

    async def update_pricing(
        self, worker_id: str, price_per_min: float, min_duration_sec: int, max_duration_sec: int
    ):
        await self._db.execute(
            "UPDATE workers SET price_per_min = ?, min_duration_sec = ?, max_duration_sec = ? "
            "WHERE worker_id = ?",
            (price_per_min, min_duration_sec, max_duration_sec, worker_id),
        )
        await self._db.commit()

    async def increment_self_blocks(self, worker_id: str):
        await self._db.execute(
            "UPDATE workers SET self_blocks_found = self_blocks_found + 1 WHERE worker_id = ?",
            (worker_id,),
        )
        await self._db.commit()

    async def list_all(self) -> List[dict]:
        results = []
        async with self._db.execute(
            "SELECT worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
            "version, state, hashrate, active_gpus, last_heartbeat, registered_at, "
            "price_per_min, min_duration_sec, max_duration_sec, "
            "total_online_sec, last_online_at, self_blocks_found "
            "FROM workers"
        ) as cursor:
            async for row in cursor:
                results.append({
                    "worker_id": row[0],
                    "eth_address": row[1],
                    "gpu_count": row[2],
                    "total_memory_gb": row[3],
                    "gpus": json.loads(row[4]),
                    "version": row[5],
                    "state": row[6],
                    "hashrate": row[7],
                    "active_gpus": row[8],
                    "last_heartbeat": row[9],
                    "registered_at": row[10],
                    "price_per_min": row[11],
                    "min_duration_sec": row[12],
                    "max_duration_sec": row[13],
                    "total_online_sec": row[14] or 0.0,
                    "last_online_at": row[15],
                    "self_blocks_found": row[16],
                })
        return results

    async def count(self) -> int:
        async with self._db.execute("SELECT COUNT(*) FROM workers") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def find_available(self, exclude_worker_ids: Optional[List[str]] = None) -> Optional[dict]:
        exclude = set(exclude_worker_ids or [])
        async with self._db.execute(
            "SELECT worker_id FROM workers WHERE state = 'AVAILABLE' ORDER BY registered_at"
        ) as cursor:
            async for row in cursor:
                if row[0] not in exclude:
                    return await self.get(row[0])
        return None
