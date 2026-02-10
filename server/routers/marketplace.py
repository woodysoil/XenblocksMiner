"""Marketplace router — /api/marketplace/* and /api/workers/* read endpoints."""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from starlette.requests import Request

from server.deps import get_server
from server.models import PricingRequest

router = APIRouter()


@router.get("/api/workers")
async def list_workers(request: Request):
    srv = get_server(request)
    workers = await srv.matcher.get_available_workers()
    for w in workers:
        rep = await srv.reputation.get_score(w["worker_id"])
        w["reputation"] = rep
    return workers


@router.get("/api/marketplace")
async def browse_marketplace(
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=18, ge=1, le=100),
    sort_by: str = "price",
    gpu_type: Optional[str] = None,
    min_hashrate: Optional[float] = None,
    max_price: Optional[float] = None,
    min_gpus: Optional[int] = None,
    available_only: bool = True,
):
    srv = get_server(request)
    all_items = await srv.pricing.browse_marketplace(
        sort_by=sort_by,
        gpu_type=gpu_type,
        min_hashrate=min_hashrate,
        max_price=max_price,
        min_gpus=min_gpus,
        available_only=available_only,
    )
    total = len(all_items)
    start = (page - 1) * limit
    items = all_items[start:start + limit]
    total_pages = (total + limit - 1) // limit
    return {"items": items, "total": total, "page": page, "limit": limit, "total_pages": total_pages}


@router.get("/api/marketplace/estimate")
async def estimate_cost(
    request: Request,
    duration_sec: int,
    worker_id: Optional[str] = None,
    min_hashrate: Optional[float] = None,
):
    srv = get_server(request)
    result = await srv.pricing.estimate_cost(
        duration_sec=duration_sec,
        worker_id=worker_id,
        min_hashrate=min_hashrate,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/api/workers/{worker_id}/pricing")
async def get_worker_pricing(request: Request, worker_id: str):
    srv = get_server(request)
    result = await srv.pricing.get_pricing(worker_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    return result


@router.get("/api/workers/{worker_id}/pricing/suggest")
async def suggest_worker_pricing(request: Request, worker_id: str):
    srv = get_server(request)
    result = await srv.pricing.suggest_pricing(worker_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    return result


@router.get("/api/workers/{worker_id}/reputation")
async def get_worker_reputation(request: Request, worker_id: str):
    srv = get_server(request)
    result = await srv.reputation.get_score(worker_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    return result


@router.put("/api/workers/{worker_id}/pricing")
async def set_worker_pricing(
    request: Request, worker_id: str, req: PricingRequest, x_api_key: str = Header(default=""),
):
    srv = get_server(request)
    caller = None
    if srv.auth and x_api_key:
        caller = await srv.auth.resolve_account(x_api_key)
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
