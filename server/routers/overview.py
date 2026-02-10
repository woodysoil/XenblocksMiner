"""Overview router — /api/overview/* platform-wide stats endpoints."""

import time

from fastapi import APIRouter, Query
from starlette.requests import Request

from server.deps import get_server

router = APIRouter()

OFFLINE_THRESHOLD = 90


@router.get("/api/overview/stats")
async def overview_stats(request: Request):
    srv = get_server(request)
    now = time.time()

    # Accounts
    accounts = await srv.accounts.list_accounts()
    total_users = len(accounts)
    total_providers = sum(1 for a in accounts.values() if a["role"] == "provider")
    total_consumers = sum(1 for a in accounts.values() if a["role"] == "consumer")

    # Workers
    workers = await srv.storage.workers.list_all()
    total_workers = len(workers)
    online_workers = sum(1 for w in workers if (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD)
    total_hashrate = sum(w["hashrate"] for w in workers if (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD)

    # Blocks
    total_blocks = await srv.storage.blocks.count()
    blocks_24h = await srv.storage.blocks.count_since(now - 86400)

    # Leases
    total_leases = await srv.storage.leases.count()
    active_leases = await srv.storage.leases.count(state="active")

    # Settlements
    all_settlements = await srv.settlement.list_settlements()
    total_settled = len(all_settlements)
    platform_revenue = sum(s.get("platform_fee", 0.0) for s in all_settlements)

    return {
        "total_users": total_users,
        "total_providers": total_providers,
        "total_consumers": total_consumers,
        "total_workers": total_workers,
        "online_workers": online_workers,
        "total_hashrate": round(total_hashrate, 2),
        "total_blocks": total_blocks,
        "blocks_24h": blocks_24h,
        "total_leases": total_leases,
        "active_leases": active_leases,
        "total_settled": total_settled,
        "platform_revenue": round(platform_revenue, 4),
    }


@router.get("/api/overview/activity")
async def overview_activity(
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    srv = get_server(request)
    activity = []

    # Recent blocks
    blocks = await srv.watcher.get_all_blocks(limit=200)
    for b in blocks:
        activity.append({
            "type": "block",
            "timestamp": b.get("timestamp", 0),
            "details": {
                "worker_id": b.get("worker_id", ""),
                "hash": b.get("hash", ""),
                "lease_id": b.get("lease_id", ""),
            },
        })

    # Recent leases
    leases = await srv.storage.leases.list_all(limit=200)
    for l in leases:
        if l["state"] in ("completed", "cancelled") and l.get("ended_at"):
            activity.append({
                "type": "lease_completed",
                "timestamp": l["ended_at"],
                "details": {
                    "lease_id": l["lease_id"],
                    "worker_id": l["worker_id"],
                    "consumer_id": l["consumer_id"],
                    "blocks_found": l["blocks_found"],
                },
            })
        activity.append({
            "type": "lease_started",
            "timestamp": l["created_at"],
            "details": {
                "lease_id": l["lease_id"],
                "worker_id": l["worker_id"],
                "consumer_id": l["consumer_id"],
                "duration_sec": l["duration_sec"],
            },
        })

    activity.sort(key=lambda x: x["timestamp"], reverse=True)
    total = len(activity)
    start = (page - 1) * limit
    items = activity[start:start + limit]
    total_pages = (total + limit - 1) // limit
    return {"items": items, "total": total, "page": page, "limit": limit, "total_pages": total_pages}


@router.get("/api/overview/network")
async def overview_network(request: Request):
    srv = get_server(request)
    now = time.time()
    workers = await srv.storage.workers.list_all()
    total_workers = len(workers)
    total_blocks = await srv.storage.blocks.count()
    chain_blocks = 0
    difficulty = 0
    if srv.chain:
        stats = srv.chain.get_stats()
        difficulty = stats["difficulty"]
        chain_blocks = stats["total_blocks"]
    return {
        "difficulty": difficulty,
        "total_workers": total_workers,
        "total_blocks": total_blocks,
        "chain_blocks": chain_blocks,
    }
