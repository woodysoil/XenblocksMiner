"""Monitoring router — /api/monitoring/* endpoints."""

from typing import Optional

from fastapi import APIRouter, Query
from starlette.requests import Request

from server.deps import get_server

router = APIRouter()


@router.get("/api/monitoring/fleet")
async def monitoring_fleet(request: Request):
    srv = get_server(request)
    return await srv.monitoring.get_fleet_overview()


@router.get("/api/monitoring/stats")
async def monitoring_stats(request: Request):
    srv = get_server(request)
    return await srv.monitoring.get_aggregated_stats()


@router.get("/api/monitoring/hashrate-history")
async def monitoring_hashrate_history(
    request: Request,
    worker_id: Optional[str] = Query(default=None, max_length=128),
    hours: float = Query(default=1.0, ge=0.0167, le=24.0),
):
    srv = get_server(request)
    return await srv.monitoring.get_hashrate_history(
        worker_id=worker_id, hours=hours,
    )


@router.get("/api/monitoring/blocks/recent")
async def monitoring_recent_blocks(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
):
    srv = get_server(request)
    return await srv.monitoring.get_recent_blocks(limit=limit)
