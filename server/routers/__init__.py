"""Router package — collects all API routers and registers them on the FastAPI app."""

from fastapi import FastAPI

from server.routers import (
    overview,
    monitoring,
    marketplace,
    rental,
    provider,
    account,
    admin,
    ws as ws_router,
)


def register_all_routers(app: FastAPI):
    app.include_router(overview.router)
    app.include_router(monitoring.router)
    app.include_router(marketplace.router)
    app.include_router(rental.router)
    app.include_router(provider.router)
    app.include_router(account.router)
    app.include_router(admin.router)
    ws_router.register(app)
