"""
matcher.py - Order matching engine.

Matches consumer rent requests to available workers and manages active leases.
Backed by StorageManager's WorkerRepo and LeaseRepo.
"""

import asyncio
import logging
import secrets
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from server.broker import MQTTBroker
    from server.account import AccountService
    from server.storage import WorkerRepo, LeaseRepo

logger = logging.getLogger("matcher")

PLATFORM_PREFIX_LENGTH = 16


class LeaseState(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MatchingEngine:
    """Matches consumer rent requests to available workers."""

    def __init__(
        self,
        broker: "MQTTBroker",
        accounts: "AccountService",
        worker_repo: "WorkerRepo",
        lease_repo: "LeaseRepo",
    ):
        self.broker = broker
        self.accounts = accounts
        self._workers = worker_repo
        self._leases = lease_repo
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------
    # Worker management
    # -------------------------------------------------------------------

    async def register_worker(self, msg: dict) -> bool:
        """Register a new worker and send an acknowledgement via MQTT."""
        worker_id = msg["worker_id"]
        await self._workers.upsert(
            worker_id=worker_id,
            eth_address=msg.get("eth_address", ""),
            gpu_count=msg.get("gpu_count", 0),
            total_memory_gb=msg.get("total_memory_gb", 0),
            gpus=msg.get("gpus", []),
            version=msg.get("version", "unknown"),
            state="AVAILABLE",
        )
        # Ensure provider account exists
        await self.accounts.get_or_create_provider(worker_id, msg.get("eth_address", ""))

        # Send register_ack
        await self.broker.publish(
            f"xenminer/{worker_id}/task",
            {"command": "register_ack", "accepted": True},
        )
        logger.info("Worker %s registered (%d GPUs, %dGB)",
                     worker_id, msg.get("gpu_count", 0), msg.get("total_memory_gb", 0))
        return True

    async def update_heartbeat(self, msg: dict):
        """Update worker heartbeat and lease hashrate statistics."""
        worker_id = msg.get("worker_id", "")
        hashrate = msg.get("hashrate", 0.0)
        active_gpus = msg.get("active_gpus", 0)
        await self._workers.update_heartbeat(worker_id, hashrate, active_gpus)
        # Update lease hashrate stats
        lease = await self._leases.get_active_lease_for_worker(worker_id)
        if lease and lease["state"] == "active":
            await self._leases.update_hashrate_stats(lease["lease_id"], hashrate)

    async def update_worker_state(self, msg: dict):
        """Update a worker's state from an incoming status message."""
        worker_id = msg.get("worker_id", "")
        state = msg.get("state", "")
        await self._workers.update_state(worker_id, state)
        logger.debug("Worker %s state -> %s", worker_id, state)

    async def get_available_workers(self) -> List[dict]:
        """Return all registered workers with their current status."""
        workers = await self._workers.list_all()
        return [
            {
                "worker_id": w["worker_id"],
                "eth_address": w["eth_address"],
                "gpu_count": w["gpu_count"],
                "total_memory_gb": w["total_memory_gb"],
                "gpus": w["gpus"],
                "state": w["state"],
                "hashrate": w["hashrate"],
                "active_gpus": w["active_gpus"],
                "last_heartbeat": w["last_heartbeat"],
                "price_per_min": w.get("price_per_min", 0.60),
                "min_duration_sec": w.get("min_duration_sec", 60),
                "max_duration_sec": w.get("max_duration_sec", 86400),
            }
            for w in workers
        ]

    # -------------------------------------------------------------------
    # Lease management
    # -------------------------------------------------------------------

    async def rent_hashpower(
        self,
        consumer_id: str,
        consumer_address: str,
        duration_sec: int = 3600,
        worker_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Create a lease: match consumer to an available worker."""
        async with self._lock:
            # Find an available worker
            target = None
            if worker_id:
                w = await self._workers.get(worker_id)
                active_lease = await self._leases.get_active_lease_for_worker(worker_id)
                if w and w["state"] == "AVAILABLE" and active_lease is None:
                    target = w
            else:
                # Find any available worker without active lease
                all_workers = await self._workers.list_all()
                for w in all_workers:
                    if w["state"] == "AVAILABLE":
                        active_lease = await self._leases.get_active_lease_for_worker(w["worker_id"])
                        if active_lease is None:
                            target = w
                            break

            if target is None:
                logger.warning("No available workers for rent request from %s", consumer_id)
                return None

            # Generate prefix (16 hex chars)
            prefix = secrets.token_hex(PLATFORM_PREFIX_LENGTH // 2)
            lease_id = f"lease-{uuid.uuid4()}"

            # Use worker's pricing (convert price_per_min to price_per_sec)
            price_per_min = target.get("price_per_min", 0.60)
            price_per_sec = price_per_min / 60.0

            lease = await self._leases.create(
                lease_id=lease_id,
                worker_id=target["worker_id"],
                consumer_id=consumer_id,
                consumer_address=consumer_address,
                prefix=prefix,
                duration_sec=duration_sec,
                price_per_sec=price_per_sec,
            )
            await self._workers.update_state(target["worker_id"], "LEASED")

        # Send assign_task to worker
        await self.broker.publish(
            f"xenminer/{target['worker_id']}/task",
            {
                "command": "assign_task",
                "lease_id": lease_id,
                "consumer_id": consumer_id,
                "consumer_address": consumer_address,
                "prefix": prefix,
                "duration_sec": duration_sec,
            },
        )
        logger.info("Lease %s created: worker=%s consumer=%s duration=%ds prefix=%s",
                     lease_id, target["worker_id"], consumer_id, duration_sec, prefix)
        return lease

    async def stop_lease(self, lease_id: str) -> Optional[dict]:
        """Stop a lease early by sending release to the worker."""
        async with self._lock:
            lease = await self._leases.get(lease_id)
            if lease is None or lease["state"] != "active":
                return None
            await self._leases.update_state(lease_id, "completed", ended_at=time.time())
            await self._workers.update_state(lease["worker_id"], "AVAILABLE")

        await self.broker.publish(
            f"xenminer/{lease['worker_id']}/task",
            {"command": "release", "lease_id": lease_id},
        )
        logger.info("Lease %s stopped (worker=%s)", lease_id, lease["worker_id"])
        # Return updated lease
        return await self._leases.get(lease_id)

    async def check_expired_leases(self) -> List[dict]:
        """Check for and complete expired leases. Returns list of newly expired."""
        expired_leases = await self._leases.find_expired()
        completed = []
        for lease in expired_leases:
            await self._leases.update_state(lease["lease_id"], "completed", ended_at=time.time())
            await self._workers.update_state(lease["worker_id"], "AVAILABLE")
            await self.broker.publish(
                f"xenminer/{lease['worker_id']}/task",
                {"command": "release", "lease_id": lease["lease_id"]},
            )
            # Re-fetch with updated state
            updated = await self._leases.get(lease["lease_id"])
            completed.append(updated)
            logger.info("Lease %s expired (worker=%s, blocks=%d)",
                         lease["lease_id"], lease["worker_id"], lease["blocks_found"])
        return completed

    async def get_lease(self, lease_id: str) -> Optional[dict]:
        """Retrieve a lease by ID."""
        return await self._leases.get(lease_id)

    async def get_active_lease_for_worker(self, worker_id: str) -> Optional[dict]:
        """Return the active lease for a worker, if any."""
        return await self._leases.get_active_lease_for_worker(worker_id)

    async def list_leases(self, state: Optional[str] = None) -> List[dict]:
        """List leases, optionally filtered by state."""
        leases = await self._leases.list_all(state=state)
        return [
            {
                "lease_id": l["lease_id"],
                "worker_id": l["worker_id"],
                "consumer_id": l["consumer_id"],
                "consumer_address": l["consumer_address"],
                "prefix": l["prefix"],
                "duration_sec": l["duration_sec"],
                "state": l["state"],
                "created_at": l["created_at"],
                "ended_at": l["ended_at"],
                "blocks_found": l["blocks_found"],
                "avg_hashrate": l["avg_hashrate"],
                "elapsed_sec": l["elapsed_sec"],
            }
            for l in leases
        ]
