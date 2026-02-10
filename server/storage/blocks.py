import time
from typing import List, Optional

import aiosqlite


class BlockRepo:
    """CRUD operations for the blocks table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(
        self,
        lease_id: str,
        worker_id: str,
        block_hash: str,
        key: str,
        account: str = "",
        attempts: int = 0,
        hashrate: str = "0.0",
        prefix_valid: bool = True,
        chain_verified: bool = False,
        chain_block_id: int = None,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO blocks (lease_id, worker_id, block_hash, key, account, attempts, hashrate, prefix_valid, chain_verified, chain_block_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (lease_id, worker_id, block_hash, key, account, attempts, hashrate, int(prefix_valid), int(chain_verified), chain_block_id, now),
        )
        await self._db.commit()
        return {
            "lease_id": lease_id,
            "worker_id": worker_id,
            "hash": block_hash,
            "key": key,
            "account": account,
            "attempts": attempts,
            "hashrate": hashrate,
            "prefix_valid": prefix_valid,
            "chain_verified": chain_verified,
            "chain_block_id": chain_block_id,
            "timestamp": now,
        }

    async def get_for_lease(self, lease_id: str) -> List[dict]:
        results = []
        async with self._db.execute(
            "SELECT lease_id, worker_id, block_hash, key, account, attempts, hashrate, prefix_valid, chain_verified, chain_block_id, created_at "
            "FROM blocks WHERE lease_id = ? ORDER BY created_at",
            (lease_id,),
        ) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "hash": row[2],
                    "key": row[3],
                    "account": row[4],
                    "attempts": row[5],
                    "hashrate": row[6],
                    "prefix_valid": bool(row[7]),
                    "chain_verified": bool(row[8]),
                    "chain_block_id": row[9],
                    "timestamp": row[10],
                })
        return results

    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        results = []
        query = ("SELECT lease_id, worker_id, block_hash, key, account, attempts, hashrate, "
                 "prefix_valid, chain_verified, chain_block_id, created_at "
                 "FROM blocks ORDER BY created_at DESC")
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "hash": row[2],
                    "key": row[3],
                    "account": row[4],
                    "attempts": row[5],
                    "hashrate": row[6],
                    "prefix_valid": bool(row[7]),
                    "chain_verified": bool(row[8]),
                    "chain_block_id": row[9],
                    "timestamp": row[10],
                })
        return results

    async def get_self_mined(self, worker_id: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        query = ("SELECT lease_id, worker_id, block_hash, key, account, attempts, hashrate, "
                 "prefix_valid, chain_verified, chain_block_id, created_at "
                 "FROM blocks WHERE lease_id = ''")
        params: tuple = ()
        if worker_id:
            query += " AND worker_id = ?"
            params = (worker_id,)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = params + (limit, offset)
        results = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "hash": row[2],
                    "key": row[3],
                    "account": row[4],
                    "attempts": row[5],
                    "hashrate": row[6],
                    "prefix_valid": bool(row[7]),
                    "chain_verified": bool(row[8]),
                    "chain_block_id": row[9],
                    "timestamp": row[10],
                })
        return results

    async def count(self, lease_id: Optional[str] = None) -> int:
        if lease_id:
            async with self._db.execute(
                "SELECT COUNT(*) FROM blocks WHERE lease_id = ?", (lease_id,)
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self._db.execute("SELECT COUNT(*) FROM blocks") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_self_mined(self, worker_id: Optional[str] = None) -> int:
        if worker_id:
            async with self._db.execute(
                "SELECT COUNT(*) FROM blocks WHERE lease_id = '' AND worker_id = ?", (worker_id,)
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self._db.execute(
                "SELECT COUNT(*) FROM blocks WHERE lease_id = ''"
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_since(self, since_ts: float) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM blocks WHERE created_at >= ?", (since_ts,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0
