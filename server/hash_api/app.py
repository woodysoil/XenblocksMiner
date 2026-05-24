"""Standalone FastAPI app for local Hash API access."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from server.hash_api.client import HashCliClient, HashCommandResult
from server.hash_api.models import HashBatchRequest, HashBenchmarkRequest, HashOneRequest, HashRequestBase
from server.hash_api.validation import validate_hash_payload


def _payload(model: HashRequestBase) -> dict[str, Any]:
    return model.model_dump(by_alias=False)


def _error_response(error: str, exit_code: int | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "exit_code": exit_code,
        "result": payload or {},
    }


def _command_response(result: HashCommandResult) -> dict[str, Any]:
    ok = result.exit_code == 0 and bool(result.payload.get("ok")) and not result.timed_out
    error = result.error or result.payload.get("error", "")
    return {
        "ok": ok,
        "error": "" if ok else error,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "result": result.payload,
    }


def create_app(client: HashCliClient | None = None) -> FastAPI:
    app = FastAPI(title="Xenblocks Local Hash API", version="0.1.0")
    hash_client = client or HashCliClient.from_env()
    app.state.hash_client = hash_client

    @app.get("/hash/v1/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "xenblocks-local-hash-api",
            "version": "v1",
        }

    @app.get("/hash/v1/backends")
    def backends() -> dict[str, Any]:
        return {
            "ok": True,
            "binary": str(hash_client.binary),
            "timeout_seconds": hash_client.timeout_seconds,
            "max_concurrency": hash_client.max_concurrency,
            "backends": hash_client.backends(),
        }

    @app.post("/hash/v1/validate")
    def validate(request: HashRequestBase) -> dict[str, Any]:
        payload = _payload(request)
        errors = validate_hash_payload(payload)
        return {
            "ok": not errors,
            "errors": errors,
        }

    @app.post("/hash/v1/hash-one")
    def hash_one(request: HashOneRequest) -> dict[str, Any]:
        payload = _payload(request)
        errors = validate_hash_payload(payload, require_key=True)
        if errors:
            return _error_response("; ".join(errors), payload={"errors": errors})
        return _command_response(hash_client.run("hash-one", payload))

    @app.post("/hash/v1/batch")
    def batch(request: HashBatchRequest) -> dict[str, Any]:
        payload = _payload(request)
        errors = validate_hash_payload(payload)
        if errors:
            return _error_response("; ".join(errors), payload={"errors": errors})
        return _command_response(hash_client.run("hash-batch", payload))

    @app.post("/hash/v1/benchmark")
    def benchmark(request: HashBenchmarkRequest) -> dict[str, Any]:
        payload = _payload(request)
        errors = validate_hash_payload(payload)
        if request.seconds <= 0:
            errors.append("seconds must be greater than zero")
        if errors:
            return _error_response("; ".join(errors), payload={"errors": errors})
        return _command_response(hash_client.run("hash-benchmark", payload))

    return app


app = create_app()

