import time
from typing import List, Optional

import aiosqlite


class SnapshotRepo:

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def insert_batch(self, rows: List[tuple]):
        now = time.time()
        await self._db.executemany(
            "INSERT INTO hashrate_snapshots (worker_id, hashrate, active_gpus, timestamp) "
            "VALUES (?, ?, ?, ?)",
            [(wid, hr, gpus, now) for wid, hr, gpus in rows],
        )
        await self._db.commit()

    async def query(self, worker_id: Optional[str] = None, hours: float = 1) -> List[dict]:
        cutoff = time.time() - hours * 3600
        if worker_id:
            sql = ("SELECT worker_id, hashrate, active_gpus, timestamp "
                   "FROM hashrate_snapshots WHERE worker_id = ? AND timestamp >= ? "
                   "ORDER BY timestamp")
            params: tuple = (worker_id, cutoff)
        else:
            sql = ("SELECT worker_id, hashrate, active_gpus, timestamp "
                   "FROM hashrate_snapshots WHERE timestamp >= ? "
                   "ORDER BY timestamp")
            params = (cutoff,)
        results = []
        async with self._db.execute(sql, params) as cursor:
            async for row in cursor:
                results.append({
                    "worker_id": row[0],
                    "hashrate": row[1],
                    "active_gpus": row[2],
                    "timestamp": row[3],
                })
        return results

    async def delete_older_than(self, hours: float = 24):
        cutoff = time.time() - hours * 3600
        await self._db.execute(
            "DELETE FROM hashrate_snapshots WHERE timestamp < ?", (cutoff,)
        )
        await self._db.commit()
