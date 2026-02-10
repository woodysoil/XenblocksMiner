"""Dependency helpers for router modules."""

from starlette.requests import Request


def get_server(request: Request):
    return request.app.state.server
