import time
from typing import List, Optional

import aiosqlite


class WalletSnapshotRepo:
    """Hourly/daily snapshots per wallet address for historical charts."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def insert(
        self,
        eth_address: str,
        interval_type: str,
        total_hashrate: float,
        online_workers: int,
        total_workers: int,
        blocks_found: int,
        cumulative_blocks: int,
        earnings: float,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO wallet_snapshots (eth_address, timestamp, interval_type, total_hashrate, "
            "online_workers, total_workers, blocks_found, cumulative_blocks, earnings) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (eth_address, now, interval_type, total_hashrate, online_workers, total_workers,
             blocks_found, cumulative_blocks, earnings),
        )
        await self._db.commit()
        return {
            "eth_address": eth_address,
            "timestamp": now,
            "interval_type": interval_type,
            "total_hashrate": total_hashrate,
            "online_workers": online_workers,
            "total_workers": total_workers,
            "blocks_found": blocks_found,
            "cumulative_blocks": cumulative_blocks,
            "earnings": earnings,
        }

    async def query(
        self,
        eth_address: str,
        hours: float = 24,
        interval_type: str = "hourly",
    ) -> List[dict]:
        cutoff = time.time() - hours * 3600
        results = []
        async with self._db.execute(
            "SELECT timestamp, total_hashrate, online_workers, total_workers, "
            "blocks_found, cumulative_blocks, earnings "
            "FROM wallet_snapshots "
            "WHERE eth_address = ? COLLATE NOCASE AND timestamp >= ? AND interval_type = ? "
            "ORDER BY timestamp",
            (eth_address, cutoff, interval_type),
        ) as cursor:
            async for row in cursor:
                results.append({
                    "timestamp": row[0],
                    "hashrate": row[1],
                    "online_workers": row[2],
                    "total_workers": row[3],
                    "blocks": row[4],
                    "cumulative_blocks": row[5],
                    "earnings": row[6],
                })
        return results

    async def get_latest(self, eth_address: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT timestamp, total_hashrate, online_workers, total_workers, "
            "blocks_found, cumulative_blocks, earnings "
            "FROM wallet_snapshots "
            "WHERE eth_address = ? COLLATE NOCASE "
            "ORDER BY timestamp DESC LIMIT 1",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "timestamp": row[0],
            "hashrate": row[1],
            "online_workers": row[2],
            "total_workers": row[3],
            "blocks": row[4],
            "cumulative_blocks": row[5],
            "earnings": row[6],
        }

    async def get_achievements(self, eth_address: str) -> dict:
        async with self._db.execute(
            "SELECT MAX(total_hashrate) FROM wallet_snapshots WHERE eth_address = ? COLLATE NOCASE",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        peak_hashrate = row[0] if row and row[0] else 0

        async with self._db.execute(
            "SELECT cumulative_blocks FROM wallet_snapshots "
            "WHERE eth_address = ? COLLATE NOCASE ORDER BY timestamp DESC LIMIT 1",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        total_blocks = row[0] if row and row[0] else 0

        async with self._db.execute(
            "SELECT SUM(earnings) FROM wallet_snapshots WHERE eth_address = ? COLLATE NOCASE",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        total_earnings = row[0] if row and row[0] else 0

        async with self._db.execute(
            "SELECT MIN(timestamp) FROM wallet_snapshots WHERE eth_address = ? COLLATE NOCASE",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        first_seen = row[0] if row and row[0] else None

        return {
            "peak_hashrate": peak_hashrate,
            "total_blocks": total_blocks,
            "total_earnings": round(total_earnings, 4),
            "first_seen": first_seen,
        }

    async def cleanup(self, hourly_retain_hours: float = 168, daily_retain_days: float = 90):
        now = time.time()
        await self._db.execute(
            "DELETE FROM wallet_snapshots WHERE interval_type = 'hourly' AND timestamp < ?",
            (now - hourly_retain_hours * 3600,),
        )
        await self._db.execute(
            "DELETE FROM wallet_snapshots WHERE interval_type = 'daily' AND timestamp < ?",
            (now - daily_retain_days * 86400,),
        )
        await self._db.commit()
