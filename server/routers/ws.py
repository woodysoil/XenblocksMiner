"""WebSocket router — /ws/dashboard endpoint."""

from fastapi import FastAPI, WebSocket


def register(app: FastAPI):
    @app.websocket("/ws/dashboard")
    async def ws_dashboard(ws: WebSocket):
        srv = app.state.server
        if srv.ws_manager is None:
            await ws.close(code=1013, reason="Service unavailable")
            return
        await srv.ws_manager.handle_connection(ws)
