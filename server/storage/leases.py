import time
from typing import List, Optional

import aiosqlite


class LeaseRepo:
    """CRUD operations for the leases table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(
        self,
        lease_id: str,
        worker_id: str,
        consumer_id: str,
        consumer_address: str,
        prefix: str,
        duration_sec: int,
        price_per_sec: float = 0.01,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO leases (lease_id, worker_id, consumer_id, consumer_address, prefix, "
            "duration_sec, price_per_sec, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)",
            (lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, price_per_sec, now),
        )
        await self._db.commit()
        return await self.get(lease_id)

    async def get(self, lease_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, "
            "price_per_sec, state, created_at, ended_at, blocks_found, "
            "total_hashrate_samples, hashrate_count "
            "FROM leases WHERE lease_id = ?",
            (lease_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        created_at = row[8]
        ended_at = row[9]
        hashrate_count = row[12]
        total_hashrate_samples = row[11]
        elapsed = (ended_at or time.time()) - created_at
        avg_hashrate = (total_hashrate_samples / hashrate_count) if hashrate_count > 0 else 0.0
        return {
            "lease_id": row[0],
            "worker_id": row[1],
            "consumer_id": row[2],
            "consumer_address": row[3],
            "prefix": row[4],
            "duration_sec": row[5],
            "price_per_sec": row[6],
            "state": row[7],
            "created_at": created_at,
            "ended_at": ended_at,
            "blocks_found": row[10],
            "total_hashrate_samples": total_hashrate_samples,
            "hashrate_count": hashrate_count,
            "avg_hashrate": avg_hashrate,
            "elapsed_sec": elapsed,
        }

    async def update_state(self, lease_id: str, state: str, ended_at: Optional[float] = None):
        if ended_at is not None:
            await self._db.execute(
                "UPDATE leases SET state = ?, ended_at = ? WHERE lease_id = ?",
                (state, ended_at, lease_id),
            )
        else:
            await self._db.execute(
                "UPDATE leases SET state = ? WHERE lease_id = ?",
                (state, lease_id),
            )
        await self._db.commit()

    async def increment_blocks(self, lease_id: str):
        await self._db.execute(
            "UPDATE leases SET blocks_found = blocks_found + 1 WHERE lease_id = ?",
            (lease_id,),
        )
        await self._db.commit()

    async def update_hashrate_stats(self, lease_id: str, hashrate: float):
        await self._db.execute(
            "UPDATE leases SET total_hashrate_samples = total_hashrate_samples + ?, "
            "hashrate_count = hashrate_count + 1 WHERE lease_id = ?",
            (hashrate, lease_id),
        )
        await self._db.commit()

    async def get_active_lease_for_worker(self, worker_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT lease_id FROM leases WHERE worker_id = ? AND state = 'active'",
            (worker_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return await self.get(row[0])

    async def find_expired(self) -> List[dict]:
        now = time.time()
        results = []
        async with self._db.execute(
            "SELECT lease_id FROM leases WHERE state = 'active' AND (created_at + duration_sec) < ?",
            (now,),
        ) as cursor:
            async for row in cursor:
                lease = await self.get(row[0])
                if lease:
                    results.append(lease)
        return results

    async def list_all(self, state: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        cols = ("lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, "
                "price_per_sec, state, created_at, ended_at, blocks_found, "
                "total_hashrate_samples, hashrate_count")
        if state:
            query = f"SELECT {cols} FROM leases WHERE state = ? ORDER BY created_at DESC"
            params: tuple = (state,)
        else:
            query = f"SELECT {cols} FROM leases ORDER BY created_at DESC"
            params = ()
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = params + (limit, offset)
        results = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                created_at = row[8]
                ended_at = row[9]
                hashrate_count = row[12]
                total_hashrate_samples = row[11]
                elapsed = (ended_at or time.time()) - created_at
                avg_hashrate = (total_hashrate_samples / hashrate_count) if hashrate_count > 0 else 0.0
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "consumer_id": row[2],
                    "consumer_address": row[3],
                    "prefix": row[4],
                    "duration_sec": row[5],
                    "price_per_sec": row[6],
                    "state": row[7],
                    "created_at": created_at,
                    "ended_at": ended_at,
                    "blocks_found": row[10],
                    "total_hashrate_samples": total_hashrate_samples,
                    "hashrate_count": hashrate_count,
                    "avg_hashrate": avg_hashrate,
                    "elapsed_sec": elapsed,
                })
        return results

    async def count(self, state: Optional[str] = None) -> int:
        if state:
            async with self._db.execute(
                "SELECT COUNT(*) FROM leases WHERE state = ?", (state,)
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self._db.execute("SELECT COUNT(*) FROM leases") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def list_for_consumer(
        self, consumer_id: str, state: Optional[str] = None,
        limit: int = 50, offset: int = 0,
    ) -> List[dict]:
        cols = ("lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, "
                "price_per_sec, state, created_at, ended_at, blocks_found, "
                "total_hashrate_samples, hashrate_count")
        query = f"SELECT {cols} FROM leases WHERE consumer_id = ?"
        params: tuple = (consumer_id,)
        if state:
            query += " AND state = ?"
            params += (state,)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += (limit, offset)
        results = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                created_at, ended_at = row[8], row[9]
                hrc = row[12]
                elapsed = (ended_at or time.time()) - created_at
                avg_hr = (row[11] / hrc) if hrc > 0 else 0.0
                results.append({
                    "lease_id": row[0], "worker_id": row[1],
                    "consumer_id": row[2], "consumer_address": row[3],
                    "prefix": row[4], "duration_sec": row[5],
                    "price_per_sec": row[6], "state": row[7],
                    "created_at": created_at, "ended_at": ended_at,
                    "blocks_found": row[10], "avg_hashrate": avg_hr,
                    "elapsed_sec": elapsed,
                })
        return results

    async def count_for_consumer(self, consumer_id: str, state: Optional[str] = None) -> int:
        if state:
            async with self._db.execute(
                "SELECT COUNT(*) FROM leases WHERE consumer_id = ? AND state = ?",
                (consumer_id, state),
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self._db.execute(
                "SELECT COUNT(*) FROM leases WHERE consumer_id = ?", (consumer_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0
