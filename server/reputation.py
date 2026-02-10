"""
reputation.py - Provider reputation engine.

Calculates a 0-100 reputation score for each worker based on:
 - Block History   (40%): verified blocks mined
 - Completion Rate (30%): leases completed successfully
 - Uptime          (20%): online time vs total time since registration
 - Account Age     (10%): days since first registration

Score maps to 0-5 stars on the dashboard (score / 20).
"""

import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from server.storage import BlockRepo, LeaseRepo, WorkerRepo

logger = logging.getLogger("reputation")

# Score weights (must sum to 100)
WEIGHT_BLOCKS = 40.0
WEIGHT_COMPLETION = 30.0
WEIGHT_UPTIME = 20.0
WEIGHT_AGE = 10.0

# Scaling parameters
BLOCKS_FOR_MAX_SCORE = 100       # 100 verified blocks = full block score
AGE_DAYS_FOR_MAX_SCORE = 90.0    # 90 days = full age score


class ReputationEngine:
    """Calculates reputation scores for workers."""

    def __init__(
        self,
        worker_repo: "WorkerRepo",
        lease_repo: "LeaseRepo",
        block_repo: "BlockRepo",
    ):
        self._workers = worker_repo
        self._leases = lease_repo
        self._blocks = block_repo

    async def get_score(self, worker_id: str) -> Optional[dict]:
        """
        Calculate full reputation breakdown for a worker.
        Returns None if worker not found.
        """
        worker = await self._workers.get(worker_id)
        if worker is None:
            return None

        breakdown = await self._calculate_breakdown(worker)
        total = (
            breakdown["block_score"]
            + breakdown["completion_score"]
            + breakdown["uptime_score"]
            + breakdown["age_score"]
        )
        total = round(min(total, 100.0), 2)
        stars = round(total / 20.0, 1)

        return {
            "worker_id": worker_id,
            "score": total,
            "stars": stars,
            "breakdown": breakdown,
        }

    async def get_score_value(self, worker_id: str) -> float:
        """Quick helper: return just the numeric score (0-100), or 0 if not found."""
        result = await self.get_score(worker_id)
        return result["score"] if result else 0.0

    async def _calculate_breakdown(self, worker: dict) -> dict:
        worker_id = worker["worker_id"]
        now = time.time()

        # --- Block History (40%) ---
        verified_blocks = await self._count_verified_blocks(worker_id)
        block_ratio = min(verified_blocks / BLOCKS_FOR_MAX_SCORE, 1.0)
        block_score = round(block_ratio * WEIGHT_BLOCKS, 2)

        # --- Completion Rate (30%) ---
        completed, total_leases = await self._count_leases(worker_id)
        if total_leases > 0:
            completion_ratio = completed / total_leases
        else:
            # No leases yet: give benefit of the doubt
            completion_ratio = 1.0
        completion_score = round(completion_ratio * WEIGHT_COMPLETION, 2)

        # --- Uptime (20%) ---
        registered_at = worker.get("registered_at", now)
        total_time = max(now - registered_at, 1.0)  # avoid division by zero
        total_online = worker.get("total_online_sec", 0.0)
        uptime_ratio = min(total_online / total_time, 1.0)
        uptime_score = round(uptime_ratio * WEIGHT_UPTIME, 2)

        # --- Account Age (10%) ---
        days_since_registration = (now - registered_at) / 86400.0
        age_ratio = min(days_since_registration / AGE_DAYS_FOR_MAX_SCORE, 1.0)
        age_score = round(age_ratio * WEIGHT_AGE, 2)

        return {
            "block_score": block_score,
            "block_detail": {
                "verified_blocks": verified_blocks,
                "max_blocks": BLOCKS_FOR_MAX_SCORE,
                "weight": WEIGHT_BLOCKS,
            },
            "completion_score": completion_score,
            "completion_detail": {
                "completed_leases": completed,
                "total_leases": total_leases,
                "weight": WEIGHT_COMPLETION,
            },
            "uptime_score": uptime_score,
            "uptime_detail": {
                "online_seconds": round(total_online, 1),
                "total_seconds": round(total_time, 1),
                "uptime_pct": round(uptime_ratio * 100, 1),
                "weight": WEIGHT_UPTIME,
            },
            "age_score": age_score,
            "age_detail": {
                "days": round(days_since_registration, 1),
                "max_days": AGE_DAYS_FOR_MAX_SCORE,
                "weight": WEIGHT_AGE,
            },
        }

    async def _count_verified_blocks(self, worker_id: str) -> int:
        """Count chain-verified blocks for this worker across all leases."""
        all_blocks = await self._blocks.get_all()
        return sum(
            1 for b in all_blocks
            if b.get("worker_id") == worker_id and b.get("chain_verified", False)
        )

    async def _count_leases(self, worker_id: str) -> tuple:
        """Return (completed_count, total_count) for this worker."""
        all_leases = await self._leases.list_all()
        worker_leases = [l for l in all_leases if l.get("worker_id") == worker_id]
        total = len(worker_leases)
        completed = sum(1 for l in worker_leases if l.get("state") == "completed")
        return completed, total
