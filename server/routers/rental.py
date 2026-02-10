"""Rental router — /api/rent, /api/stop, /api/leases/*, /api/blocks/* endpoints."""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from starlette.requests import Request

from server.deps import get_server
from server.models import RentRequest, StopRequest

router = APIRouter()


async def _optional_account(srv, x_api_key: str) -> Optional[dict]:
    if srv.auth is None or not x_api_key:
        return None
    return await srv.auth.resolve_account(x_api_key)


@router.post("/api/rent")
async def rent_hashpower(request: Request, req: RentRequest, x_api_key: str = Header(default="")):
    srv = get_server(request)
    caller = await _optional_account(srv, x_api_key)
    if caller and caller["role"] not in ("consumer", "admin"):
        raise HTTPException(status_code=403, detail="Consumer account required to rent")
    lease = await srv.matcher.rent_hashpower(
        consumer_id=req.consumer_id,
        consumer_address=req.consumer_address,
        duration_sec=req.duration_sec,
        worker_id=req.worker_id,
    )
    if lease is None:
        raise HTTPException(status_code=404, detail="No available workers")
    return {
        "lease_id": lease["lease_id"],
        "worker_id": lease["worker_id"],
        "prefix": lease["prefix"],
        "duration_sec": lease["duration_sec"],
        "consumer_id": lease["consumer_id"],
        "consumer_address": lease["consumer_address"],
        "created_at": lease["created_at"],
    }


@router.post("/api/stop")
async def stop_lease(request: Request, req: StopRequest, x_api_key: str = Header(default="")):
    srv = get_server(request)
    caller = await _optional_account(srv, x_api_key)
    lease = await srv.matcher.stop_lease(req.lease_id)
    if lease is None:
        raise HTTPException(status_code=404, detail="Lease not found or not active")
    if caller and caller["role"] != "admin":
        if caller["role"] == "consumer" and caller["account_id"] != lease["consumer_id"]:
            raise HTTPException(status_code=403, detail="You can only stop your own leases")
    record = await srv.settlement.settle_lease(lease)
    result = {
        "lease_id": lease["lease_id"],
        "state": lease["state"],
        "blocks_found": lease["blocks_found"],
    }
    if record:
        result["settlement"] = record
    return result


# Aliases
@router.post("/api/rental/start")
async def rental_start(request: Request, req: RentRequest, x_api_key: str = Header(default="")):
    return await rent_hashpower(request, req, x_api_key)


@router.post("/api/rental/stop")
async def rental_stop(request: Request, req: StopRequest, x_api_key: str = Header(default="")):
    return await stop_lease(request, req, x_api_key)


@router.get("/api/leases")
async def list_leases(request: Request, state: Optional[str] = None, limit: int = 50, offset: int = 0):
    srv = get_server(request)
    items = await srv.matcher.list_leases(state=state, limit=limit, offset=offset)
    total = await srv.storage.leases.count(state=state)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/api/leases/{lease_id}")
async def get_lease(request: Request, lease_id: str):
    srv = get_server(request)
    lease = await srv.matcher.get_lease(lease_id)
    if lease is None:
        raise HTTPException(status_code=404, detail="Lease not found")
    result = {
        "lease_id": lease["lease_id"],
        "worker_id": lease["worker_id"],
        "consumer_id": lease["consumer_id"],
        "consumer_address": lease["consumer_address"],
        "prefix": lease["prefix"],
        "duration_sec": lease["duration_sec"],
        "state": lease["state"],
        "created_at": lease["created_at"],
        "ended_at": lease["ended_at"],
        "blocks_found": lease["blocks_found"],
        "avg_hashrate": lease["avg_hashrate"],
        "elapsed_sec": lease["elapsed_sec"],
    }
    blocks = await srv.watcher.get_blocks_for_lease(lease_id)
    if blocks:
        result["blocks"] = blocks
    settlement = await srv.settlement.get_settlement(lease_id)
    if settlement:
        result["settlement"] = settlement
    return result


@router.get("/api/blocks")
async def list_blocks(request: Request, lease_id: Optional[str] = None, limit: int = 50, offset: int = 0):
    srv = get_server(request)
    if lease_id:
        return await srv.watcher.get_blocks_for_lease(lease_id)
    items = await srv.watcher.get_all_blocks(limit=limit, offset=offset)
    total = await srv.storage.blocks.count()
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/api/blocks/self-mined")
async def list_self_mined_blocks(request: Request, worker_id: Optional[str] = None, limit: int = 50, offset: int = 0):
    srv = get_server(request)
    items = await srv.watcher.get_self_mined_blocks(worker_id=worker_id, limit=limit, offset=offset)
    total = await srv.watcher.count_self_mined(worker_id=worker_id)
    return {"items": items, "total": total, "limit": limit, "offset": offset}
