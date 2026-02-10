"""Provider router — /api/provider/* endpoints."""

import time

from fastapi import APIRouter, Header, HTTPException, Query
from starlette.requests import Request

from server.deps import get_server
from server.models import PricingRequest

router = APIRouter()

OFFLINE_THRESHOLD = 90


async def _resolve_provider_workers(srv, provider_id: str) -> list:
    workers = await srv.storage.workers.list_all()
    # Direct address match (0x-prefixed)
    if provider_id.startswith("0x"):
        return [w for w in workers if w.get("eth_address", "").lower() == provider_id.lower()]
    matched = [w for w in workers if w["worker_id"] == provider_id]
    if matched:
        return matched
    acct = await srv.accounts.get_account(provider_id)
    if acct and acct.get("eth_address"):
        return [w for w in workers if w.get("eth_address", "") == acct["eth_address"]]
    return []


async def _get_provider_id(srv, provider_id: str, authorization: str) -> str:
    """Resolve provider_id: use query param if given, else JWT eth_address."""
    if provider_id:
        return provider_id
    if authorization.startswith("Bearer ") and srv.auth:
        claims = srv.auth.decode_jwt(authorization[7:])
        if claims and claims.get("sub"):
            return claims["sub"]
    raise HTTPException(
        status_code=400,
        detail="provider_id query param or wallet JWT required",
    )


@router.get("/api/provider/dashboard")
async def provider_dashboard(
    request: Request,
    provider_id: str = Query(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    pid = await _get_provider_id(srv, provider_id, authorization)
    provider_workers = await _resolve_provider_workers(srv, pid)

    worker_count = len(provider_workers)
    if not provider_workers:
        return {
            "provider_id": pid, "worker_count": 0, "total_earned": 0.0,
            "active_leases": 0, "total_blocks_mined": 0, "avg_hashrate": 0.0,
        }

    total_hashrate = sum(w["hashrate"] for w in provider_workers)
    avg_hashrate = total_hashrate / worker_count
    total_blocks_mined = sum(w.get("self_blocks_found", 0) for w in provider_workers)

    worker_ids = {w["worker_id"] for w in provider_workers}
    all_leases = await srv.storage.leases.list_all()
    active_leases = [l for l in all_leases if l["worker_id"] in worker_ids and l["state"] == "active"]

    all_settlements = await srv.settlement.list_settlements()
    provider_settlements = [s for s in all_settlements if s["worker_id"] in worker_ids]
    total_earned = sum(s.get("provider_payment", 0.0) for s in provider_settlements)

    return {
        "provider_id": pid,
        "worker_count": worker_count,
        "total_earned": round(total_earned, 4),
        "active_leases": len(active_leases),
        "total_blocks_mined": total_blocks_mined,
        "avg_hashrate": round(avg_hashrate, 2),
    }


@router.get("/api/provider/earnings")
async def provider_earnings(
    request: Request,
    provider_id: str = Query(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    pid = await _get_provider_id(srv, provider_id, authorization)
    provider_workers = await _resolve_provider_workers(srv, pid)
    worker_ids = {w["worker_id"] for w in provider_workers}
    all_settlements = await srv.settlement.list_settlements()
    earnings = [s for s in all_settlements if s["worker_id"] in worker_ids]
    return {"provider_id": pid, "earnings": earnings}


@router.get("/api/provider/workers")
async def provider_workers(
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    provider_id: str = Query(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    pid = await _get_provider_id(srv, provider_id, authorization)
    now = time.time()
    matched = await _resolve_provider_workers(srv, pid)
    all_items = []
    for w in matched:
        online = (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD
        gpus = w.get("gpus") or []
        memory_gb = w.get("total_memory_gb", 0)

        # Format GPU name for display (handles multi-GPU)
        if not gpus:
            gpu_name = "GPU"
        else:
            from collections import Counter
            counts = Counter()
            for g in gpus:
                short = (g.get("name") or "GPU").replace("NVIDIA ", "").replace("GeForce ", "")
                counts[short] += 1
            if len(counts) == 1:
                name, cnt = list(counts.items())[0]
                gpu_name = f"{cnt}x {name}" if cnt > 1 else name
            elif len(counts) == 2 and len(gpus) <= 4:
                gpu_name = " + ".join(f"{c}x {n}" if c > 1 else n for n, c in counts.items())
            else:
                gpu_name = f"{len(gpus)}x (mixed)"

        all_items.append({
            "worker_id": w["worker_id"],
            "state": w["state"],
            "online": online,
            "hashrate": w["hashrate"],
            "gpu_count": w["gpu_count"],
            "active_gpus": w["active_gpus"],
            "gpu_name": gpu_name,
            "memory_gb": memory_gb,
            "price_per_min": w["price_per_min"],
            "self_blocks_found": w["self_blocks_found"],
            "total_online_sec": w["total_online_sec"],
        })
    total = len(all_items)
    start = (page - 1) * limit
    items = all_items[start:start + limit]
    total_pages = (total + limit - 1) // limit
    return {"provider_id": pid, "items": items, "total": total, "page": page, "limit": limit, "total_pages": total_pages}


@router.put("/api/provider/workers/{worker_id}/pricing")
async def set_provider_worker_pricing(
    request: Request,
    worker_id: str,
    req: PricingRequest,
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    caller = None
    if srv.auth and (x_api_key or authorization):
        caller = await srv.auth.resolve_account(x_api_key, authorization)
    if caller and caller["role"] not in ("provider", "admin"):
        raise HTTPException(status_code=403, detail="Provider account required")
    if caller and caller["role"] == "provider" and caller["account_id"] != worker_id:
        raise HTTPException(status_code=403, detail="You can only set pricing for your own worker")
    try:
        result = await srv.pricing.set_pricing(
            worker_id=worker_id,
            price_per_min=req.price_per_min,
            min_duration_sec=req.min_duration_sec,
            max_duration_sec=req.max_duration_sec,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {
        "worker_id": result["worker_id"],
        "price_per_min": result["price_per_min"],
        "min_duration_sec": result["min_duration_sec"],
        "max_duration_sec": result["max_duration_sec"],
    }
