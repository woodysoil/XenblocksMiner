"""
ws.py - WebSocket connection manager for real-time dashboard updates.
"""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, List

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from server.storage import WorkerRepo, BlockRepo

logger = logging.getLogger("ws")

MAX_WS_CLIENTS = 200
SEND_TIMEOUT = 2.0


class WSManager:
    def __init__(self, worker_repo: "WorkerRepo", block_repo: "BlockRepo"):
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._workers = worker_repo
        self._blocks = block_repo

    async def connect(self, ws: WebSocket) -> bool:
        await ws.accept()
        async with self._lock:
            if len(self._clients) >= MAX_WS_CLIENTS:
                logger.warning("WebSocket capacity reached (%d)", MAX_WS_CLIENTS)
                await ws.close(code=1013, reason="Server overloaded")
                return False
            self._clients.append(ws)
            total = len(self._clients)
        logger.info("WebSocket client connected (%d total)", total)
        await self._send_snapshot(ws)
        return True

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            try:
                self._clients.remove(ws)
            except ValueError:
                pass
            total = len(self._clients)
        logger.info("WebSocket client disconnected (%d total)", total)

    async def broadcast(self, event_type: str, data: Any):
        if not self._clients:
            return
        msg = json.dumps({"type": event_type, "data": data, "ts": time.time()})
        async with self._lock:
            clients = list(self._clients)
        stale: List[WebSocket] = []
        if clients:
            await asyncio.gather(*(self._safe_send(ws, msg, stale) for ws in clients))
        if stale:
            async with self._lock:
                for ws in stale:
                    try:
                        self._clients.remove(ws)
                    except ValueError:
                        pass

    async def _safe_send(self, ws: WebSocket, msg: str, stale: List[WebSocket]):
        try:
            await asyncio.wait_for(ws.send_text(msg), timeout=SEND_TIMEOUT)
        except Exception:
            stale.append(ws)

    async def _send_snapshot(self, ws: WebSocket):
        try:
            workers = await self._workers.list_all()
            blocks = await self._blocks.get_all(limit=20)
            snapshot = {
                "type": "snapshot",
                "data": {
                    "workers": workers,
                    "recent_blocks": blocks,
                },
                "ts": time.time(),
            }
            await ws.send_text(json.dumps(snapshot))
        except Exception:
            logger.exception("Failed to send snapshot")

    async def handle_connection(self, ws: WebSocket):
        if not await self.connect(ws):
            return
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await self.disconnect(ws)
