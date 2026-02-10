"""Wallet snapshot service - periodic aggregation of per-address stats."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.storage import StorageManager

logger = logging.getLogger("snapshot")

OFFLINE_THRESHOLD = 90  # seconds


class WalletSnapshotService:
    """Periodically creates wallet snapshots for historical charts."""

    def __init__(
        self,
        storage: "StorageManager",
        interval_sec: int = 3600,  # default: hourly
    ):
        self.storage = storage
        self.interval_sec = interval_sec
        self._task: asyncio.Task | None = None
        self._last_blocks: dict[str, int] = {}  # eth_address -> cumulative blocks

    async def start(self):
        """Start the background snapshot task."""
        self._task = asyncio.create_task(self._run())
        logger.info("Wallet snapshot service started (interval: %ds)", self.interval_sec)

    async def stop(self):
        """Stop the background task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Wallet snapshot service stopped")

    async def _run(self):
        """Background loop: create snapshots at interval."""
        while True:
            try:
                await asyncio.sleep(self.interval_sec)
                await self.create_snapshots()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Snapshot error: %s", e)

    async def create_snapshots(self, interval_type: str = "hourly"):
        """Create a snapshot for each wallet address with workers."""
        now = time.time()
        workers = await self.storage.workers.list_all()

        # Group workers by eth_address
        by_address: dict[str, list[dict]] = {}
        for w in workers:
            addr = w.get("eth_address", "")
            if not addr:
                continue
            addr_lower = addr.lower()
            if addr_lower not in by_address:
                by_address[addr_lower] = []
            by_address[addr_lower].append(w)

        # Get settlements for earnings calculation
        settlements = await self.storage.settlements.list_all()
        earnings_by_worker: dict[str, float] = {}
        for s in settlements:
            wid = s.get("worker_id", "")
            earnings_by_worker[wid] = earnings_by_worker.get(wid, 0) + s.get("provider_payout", 0)

        count = 0
        for addr, addr_workers in by_address.items():
            online = 0
            total_hashrate = 0.0
            total_workers = len(addr_workers)
            cumulative_blocks = 0
            total_earnings = 0.0

            for w in addr_workers:
                if (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD:
                    online += 1
                    total_hashrate += w.get("hashrate", 0)
                cumulative_blocks += w.get("self_blocks_found", 0)
                total_earnings += earnings_by_worker.get(w["worker_id"], 0)

            # Calculate blocks found since last snapshot
            prev_blocks = self._last_blocks.get(addr, 0)
            blocks_found = max(0, cumulative_blocks - prev_blocks)
            self._last_blocks[addr] = cumulative_blocks

            await self.storage.wallet_snapshots.insert(
                eth_address=addr,
                interval_type=interval_type,
                total_hashrate=total_hashrate,
                online_workers=online,
                total_workers=total_workers,
                blocks_found=blocks_found,
                cumulative_blocks=cumulative_blocks,
                earnings=total_earnings,
            )
            count += 1

        # Cleanup old snapshots
        await self.storage.wallet_snapshots.cleanup()

        logger.info("Created %d wallet snapshots", count)
        return count

    async def create_snapshot_for_address(self, eth_address: str) -> dict | None:
        """Create an immediate snapshot for a specific address (on-demand)."""
        now = time.time()
        workers = await self.storage.workers.list_all()

        # Filter workers for this address
        addr_lower = eth_address.lower()
        addr_workers = [w for w in workers if w.get("eth_address", "").lower() == addr_lower]

        if not addr_workers:
            return None

        online = 0
        total_hashrate = 0.0
        cumulative_blocks = 0

        for w in addr_workers:
            if (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD:
                online += 1
                total_hashrate += w.get("hashrate", 0)
            cumulative_blocks += w.get("self_blocks_found", 0)

        # Get earnings
        settlements = await self.storage.settlements.list_all()
        worker_ids = {w["worker_id"] for w in addr_workers}
        total_earnings = sum(
            s.get("provider_payout", 0) for s in settlements if s.get("worker_id") in worker_ids
        )

        # Calculate blocks since last snapshot
        prev_blocks = self._last_blocks.get(addr_lower, 0)
        blocks_found = max(0, cumulative_blocks - prev_blocks)
        self._last_blocks[addr_lower] = cumulative_blocks

        return await self.storage.wallet_snapshots.insert(
            eth_address=eth_address,
            interval_type="hourly",
            total_hashrate=total_hashrate,
            online_workers=online,
            total_workers=len(addr_workers),
            blocks_found=blocks_found,
            cumulative_blocks=cumulative_blocks,
            earnings=total_earnings,
        )
