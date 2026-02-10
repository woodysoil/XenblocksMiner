"""
monitoring.py - Fleet monitoring and aggregation service.
"""

import logging
import time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from server.storage import WorkerRepo, BlockRepo, SnapshotRepo

logger = logging.getLogger("monitoring")

OFFLINE_THRESHOLD = 90  # seconds without heartbeat
MAX_HISTORY_HOURS = 24.0
MAX_RECENT_BLOCKS = 200


class MonitoringService:
    def __init__(
        self,
        worker_repo: "WorkerRepo",
        block_repo: "BlockRepo",
        snapshot_repo: "SnapshotRepo",
    ):
        self._workers = worker_repo
        self._blocks = block_repo
        self._snapshots = snapshot_repo

    async def get_fleet_overview(self) -> List[dict]:
        now = time.time()
        workers = await self._workers.list_all()
        result = []
        for w in workers:
            online = (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD
            result.append({
                "worker_id": w["worker_id"],
                "eth_address": w["eth_address"],
                "state": w["state"],
                "online": online,
                "hashrate": w["hashrate"],
                "gpu_count": w["gpu_count"],
                "active_gpus": w["active_gpus"],
                "total_memory_gb": w["total_memory_gb"],
                "gpus": w["gpus"],
                "version": w["version"],
                "last_heartbeat": w["last_heartbeat"],
                "self_blocks_found": w["self_blocks_found"],
                "price_per_min": w["price_per_min"],
                "total_online_sec": w["total_online_sec"],
            })
        return result

    async def get_aggregated_stats(self) -> dict:
        now = time.time()
        workers = await self._workers.list_all()
        total_hashrate = 0.0
        online = 0
        offline = 0
        total_gpus = 0
        active_gpus = 0
        for w in workers:
            is_online = (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD
            if is_online:
                online += 1
                total_hashrate += w["hashrate"]
                active_gpus += w["active_gpus"]
            else:
                offline += 1
            total_gpus += w["gpu_count"]

        total_blocks = await self._blocks.count()
        one_hour_ago = now - 3600
        blocks_last_hour = await self._blocks.count_since(one_hour_ago)

        return {
            "total_workers": len(workers),
            "online": online,
            "offline": offline,
            "total_hashrate": round(total_hashrate, 2),
            "total_gpus": total_gpus,
            "active_gpus": active_gpus,
            "total_blocks": total_blocks,
            "blocks_last_hour": blocks_last_hour,
        }

    async def record_hashrate_snapshot(self):
        workers = await self._workers.list_all()
        now = time.time()
        rows = []
        for w in workers:
            if (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD:
                rows.append((w["worker_id"], w["hashrate"], w["active_gpus"]))
        if rows:
            await self._snapshots.insert_batch(rows)

    async def get_hashrate_history(
        self, worker_id: Optional[str] = None, hours: float = 1
    ) -> List[dict]:
        bounded = max(0.0167, min(float(hours), MAX_HISTORY_HOURS))
        return await self._snapshots.query(worker_id=worker_id, hours=bounded)

    async def check_worker_health(self) -> List[dict]:
        now = time.time()
        workers = await self._workers.list_all()
        unhealthy = []
        for w in workers:
            if (now - w["last_heartbeat"]) >= OFFLINE_THRESHOLD:
                unhealthy.append({
                    "worker_id": w["worker_id"],
                    "last_heartbeat": w["last_heartbeat"],
                    "offline_sec": round(now - w["last_heartbeat"], 1),
                })
        return unhealthy

    async def get_recent_blocks(self, limit: int = 20) -> List[dict]:
        bounded = max(1, min(int(limit), MAX_RECENT_BLOCKS))
        return await self._blocks.get_all(limit=bounded)

    async def cleanup_old_snapshots(self):
        await self._snapshots.delete_older_than(hours=24)
