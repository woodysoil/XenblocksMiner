import time
from typing import List, Optional

import aiosqlite


class SettlementRepo:
    """CRUD operations for the settlements table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(
        self,
        lease_id: str,
        consumer_id: str,
        worker_id: str,
        duration_sec: float,
        blocks_found: int,
        total_cost: float,
        provider_payout: float,
        platform_fee: float,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO settlements (lease_id, consumer_id, worker_id, duration_sec, blocks_found, "
            "total_cost, provider_payout, platform_fee, settled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (lease_id, consumer_id, worker_id, duration_sec, blocks_found, total_cost, provider_payout, platform_fee, now),
        )
        await self._db.commit()
        return {
            "lease_id": lease_id,
            "consumer_id": consumer_id,
            "worker_id": worker_id,
            "duration_sec": round(duration_sec, 2),
            "blocks_found": blocks_found,
            "total_cost": round(total_cost, 4),
            "provider_payment": round(provider_payout, 4),
            "platform_fee": round(platform_fee, 4),
        }

    async def get(self, lease_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT lease_id, consumer_id, worker_id, duration_sec, blocks_found, "
            "total_cost, provider_payout, platform_fee, settled_at "
            "FROM settlements WHERE lease_id = ?",
            (lease_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "lease_id": row[0],
            "consumer_id": row[1],
            "worker_id": row[2],
            "duration_sec": round(row[3], 2),
            "blocks_found": row[4],
            "total_cost": round(row[5], 4),
            "provider_payment": round(row[6], 4),
            "platform_fee": round(row[7], 4),
        }

    async def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        results = []
        query = ("SELECT lease_id, consumer_id, worker_id, duration_sec, blocks_found, "
                 "total_cost, provider_payout, platform_fee, settled_at "
                 "FROM settlements ORDER BY settled_at DESC")
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "consumer_id": row[1],
                    "worker_id": row[2],
                    "duration_sec": round(row[3], 2),
                    "blocks_found": row[4],
                    "total_cost": round(row[5], 4),
                    "provider_payment": round(row[6], 4),
                    "platform_fee": round(row[7], 4),
                })
        return results

    async def count(self) -> int:
        async with self._db.execute("SELECT COUNT(*) FROM settlements") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0
