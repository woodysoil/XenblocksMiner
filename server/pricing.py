"""
pricing.py - Provider pricing engine.

Allows providers to set and manage hashpower pricing. Helps consumers
browse the marketplace, filter by GPU/price/hashrate, and estimate costs.

Pricing data is stored in the workers table (price_per_min, min_duration_sec,
max_duration_sec columns).
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from server.storage import WorkerRepo

logger = logging.getLogger("pricing")

# Default pricing when a provider has not set their own
DEFAULT_PRICE_PER_MIN = 0.60  # $0.60/min ($0.01/sec like existing settlement)
DEFAULT_MIN_DURATION = 60     # 1 minute minimum
DEFAULT_MAX_DURATION = 86400  # 24 hours maximum

# GPU-based suggested pricing (price per minute per GPU)
GPU_PRICING_SUGGESTIONS: Dict[str, float] = {
    "RTX 4090": 1.20,
    "RTX 4080": 0.90,
    "RTX 4070": 0.60,
    "RTX 3090": 0.80,
    "RTX 3080": 0.55,
    "RTX 3070": 0.40,
    "RTX 3060": 0.25,
    "A100": 2.00,
    "H100": 3.50,
}


class PricingEngine:
    """Manages provider pricing and consumer price discovery."""

    def __init__(self, worker_repo: "WorkerRepo"):
        self._workers = worker_repo

    # -------------------------------------------------------------------
    # Provider pricing
    # -------------------------------------------------------------------

    async def set_pricing(
        self,
        worker_id: str,
        price_per_min: float,
        min_duration_sec: int = DEFAULT_MIN_DURATION,
        max_duration_sec: int = DEFAULT_MAX_DURATION,
    ) -> Optional[dict]:
        """Set pricing for a worker. Returns updated worker or None if not found."""
        worker = await self._workers.get(worker_id)
        if worker is None:
            return None

        if price_per_min < 0:
            raise ValueError("price_per_min must be non-negative")
        if min_duration_sec < 1:
            raise ValueError("min_duration_sec must be at least 1")
        if max_duration_sec < min_duration_sec:
            raise ValueError("max_duration_sec must be >= min_duration_sec")

        await self._workers.update_pricing(
            worker_id, price_per_min, min_duration_sec, max_duration_sec
        )
        logger.info(
            "Worker %s pricing updated: $%.4f/min, duration %d-%ds",
            worker_id, price_per_min, min_duration_sec, max_duration_sec,
        )
        return await self._workers.get(worker_id)

    async def get_pricing(self, worker_id: str) -> Optional[dict]:
        """Get pricing info for a specific worker."""
        worker = await self._workers.get(worker_id)
        if worker is None:
            return None
        return _extract_pricing(worker)

    async def suggest_pricing(self, worker_id: str) -> Optional[dict]:
        """Suggest pricing based on GPU specs."""
        worker = await self._workers.get(worker_id)
        if worker is None:
            return None

        suggested = _suggest_price_for_worker(worker)
        return {
            "worker_id": worker_id,
            "suggested_price_per_min": round(suggested, 4),
            "gpu_count": worker["gpu_count"],
            "total_memory_gb": worker["total_memory_gb"],
            "current_price_per_min": worker.get("price_per_min", DEFAULT_PRICE_PER_MIN),
        }

    # -------------------------------------------------------------------
    # Marketplace / price discovery
    # -------------------------------------------------------------------

    async def browse_marketplace(
        self,
        sort_by: str = "price",
        gpu_type: Optional[str] = None,
        min_hashrate: Optional[float] = None,
        max_price: Optional[float] = None,
        min_gpus: Optional[int] = None,
        available_only: bool = True,
    ) -> List[dict]:
        """Browse available hashpower with filters. Returns sorted listing."""
        all_workers = await self._workers.list_all()
        results = []

        for w in all_workers:
            # Filter: available only
            if available_only and w["state"] != "AVAILABLE":
                continue

            # Filter: GPU type (substring match on GPU names)
            if gpu_type:
                gpu_names = " ".join(
                    g.get("name", "") for g in (w.get("gpus") or [])
                ).lower()
                if gpu_type.lower() not in gpu_names:
                    continue

            # Filter: minimum hashrate
            if min_hashrate is not None and w["hashrate"] < min_hashrate:
                continue

            # Filter: minimum GPU count
            if min_gpus is not None and w["gpu_count"] < min_gpus:
                continue

            price = w.get("price_per_min", DEFAULT_PRICE_PER_MIN)

            # Filter: max price
            if max_price is not None and price > max_price:
                continue

            results.append({
                "worker_id": w["worker_id"],
                "state": w["state"],
                "gpu_count": w["gpu_count"],
                "total_memory_gb": w["total_memory_gb"],
                "gpus": w.get("gpus", []),
                "hashrate": w["hashrate"],
                "eth_address": w["eth_address"],
                "price_per_min": price,
                "min_duration_sec": w.get("min_duration_sec", DEFAULT_MIN_DURATION),
                "max_duration_sec": w.get("max_duration_sec", DEFAULT_MAX_DURATION),
                "last_heartbeat": w["last_heartbeat"],
            })

        # Sort
        if sort_by == "price":
            results.sort(key=lambda x: x["price_per_min"])
        elif sort_by == "hashrate":
            results.sort(key=lambda x: x["hashrate"], reverse=True)
        elif sort_by == "gpus":
            results.sort(key=lambda x: x["gpu_count"], reverse=True)
        elif sort_by == "memory":
            results.sort(key=lambda x: x["total_memory_gb"], reverse=True)

        return results

    async def estimate_cost(
        self,
        duration_sec: int,
        worker_id: Optional[str] = None,
        min_hashrate: Optional[float] = None,
    ) -> dict:
        """Estimate cost for a given duration. Uses specific worker or cheapest available."""
        if worker_id:
            worker = await self._workers.get(worker_id)
            if worker is None:
                return {"error": "Worker not found"}
            price_per_min = worker.get("price_per_min", DEFAULT_PRICE_PER_MIN)
            min_dur = worker.get("min_duration_sec", DEFAULT_MIN_DURATION)
            max_dur = worker.get("max_duration_sec", DEFAULT_MAX_DURATION)
            hashrate = worker["hashrate"]
            target_worker = worker_id
        else:
            # Find cheapest available worker meeting hashrate requirement
            listings = await self.browse_marketplace(
                sort_by="price",
                min_hashrate=min_hashrate,
                available_only=True,
            )
            if not listings:
                return {"error": "No available workers matching criteria"}
            cheapest = listings[0]
            price_per_min = cheapest["price_per_min"]
            min_dur = cheapest["min_duration_sec"]
            max_dur = cheapest["max_duration_sec"]
            hashrate = cheapest["hashrate"]
            target_worker = cheapest["worker_id"]

        duration_min = duration_sec / 60.0
        total_cost = price_per_min * duration_min
        provider_payout = total_cost * 0.95
        platform_fee = total_cost * 0.05

        result = {
            "worker_id": target_worker,
            "duration_sec": duration_sec,
            "price_per_min": round(price_per_min, 4),
            "total_cost": round(total_cost, 4),
            "provider_payout": round(provider_payout, 4),
            "platform_fee": round(platform_fee, 4),
            "hashrate": hashrate,
            "min_duration_sec": min_dur,
            "max_duration_sec": max_dur,
        }

        # Duration validation
        if duration_sec < min_dur:
            result["warning"] = f"Duration below minimum ({min_dur}s)"
        elif duration_sec > max_dur:
            result["warning"] = f"Duration above maximum ({max_dur}s)"

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pricing(worker: dict) -> dict:
    """Extract pricing fields from a worker dict."""
    return {
        "worker_id": worker["worker_id"],
        "price_per_min": worker.get("price_per_min", DEFAULT_PRICE_PER_MIN),
        "min_duration_sec": worker.get("min_duration_sec", DEFAULT_MIN_DURATION),
        "max_duration_sec": worker.get("max_duration_sec", DEFAULT_MAX_DURATION),
        "gpu_count": worker["gpu_count"],
        "total_memory_gb": worker["total_memory_gb"],
        "hashrate": worker["hashrate"],
    }


def _suggest_price_for_worker(worker: dict) -> float:
    """Suggest a price per minute based on GPU specs."""
    gpus = worker.get("gpus") or []
    if not gpus:
        # Fallback: use memory as a rough proxy
        mem_gb = worker.get("total_memory_gb", 0)
        if mem_gb >= 80:
            return 2.00 * worker.get("gpu_count", 1)
        elif mem_gb >= 24:
            return 0.80 * worker.get("gpu_count", 1)
        elif mem_gb >= 12:
            return 0.40 * worker.get("gpu_count", 1)
        return DEFAULT_PRICE_PER_MIN

    total = 0.0
    for gpu in gpus:
        name = gpu.get("name", "").upper()
        matched = False
        for key, price in GPU_PRICING_SUGGESTIONS.items():
            if key.upper() in name:
                total += price
                matched = True
                break
        if not matched:
            # Fallback by memory
            mem = gpu.get("memory_gb", gpu.get("totalMemory", 0))
            if isinstance(mem, str):
                try:
                    mem = float(mem)
                except ValueError:
                    mem = 0
            if mem >= 80:
                total += 2.00
            elif mem >= 24:
                total += 0.80
            elif mem >= 12:
                total += 0.40
            else:
                total += 0.20

    return total if total > 0 else DEFAULT_PRICE_PER_MIN
