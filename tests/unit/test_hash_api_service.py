"""Tests for the standalone local Hash API service."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from server.hash_api.app import create_app
from server.hash_api.client import HashCliClient, HashCommandResult
from server.hash_api.validation import validate_hash_payload


class FakeHashClient(HashCliClient):
    def __init__(self):
        super().__init__(binary="fake-hashapi-cli", timeout_seconds=1, max_concurrency=1)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, command: str, payload: dict[str, Any]) -> HashCommandResult:
        self.calls.append((command, payload))
        return HashCommandResult(
            exit_code=0,
            payload={
                "ok": True,
                "error": "",
                "backend": payload.get("backend", "cpu"),
                "attempts": payload.get("batch_size", 1),
                "matches": [],
            },
        )


def test_validation_matches_hash_api_contract():
    errors = validate_hash_payload(
        {
            "algorithm": "argon2id-xen",
            "backend": "cpu",
            "salt_hex": "invalid",
            "key_prefix": "",
            "target_pattern": "XEN11",
            "difficulty": 1,
            "batch_size": 1,
            "device_id": 0,
        }
    )
    assert "salt_hex must contain an even number of hex characters" in errors
    assert "salt_hex must contain only hex characters" in errors


def test_health_and_backends_are_independent_hash_routes():
    client = TestClient(create_app(FakeHashClient()))

    health = client.get("/hash/v1/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    backends = client.get("/hash/v1/backends")
    assert backends.status_code == 200
    assert {item["name"] for item in backends.json()["backends"]} >= {"cpu", "reference", "cuda"}


def test_validate_endpoint_does_not_spawn_cli():
    fake = FakeHashClient()
    client = TestClient(create_app(fake))

    response = client.post(
        "/hash/v1/validate",
        json={"salt": "aabbccddeeff0011", "backend": "cpu", "difficulty": 1, "batch_size": 1},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "errors": []}
    assert fake.calls == []


def test_batch_endpoint_invokes_hash_batch_command():
    fake = FakeHashClient()
    client = TestClient(create_app(fake))

    response = client.post(
        "/hash/v1/batch",
        json={
            "salt": "aabbccddeeff0011",
            "backend": "cpu",
            "prefix": "deadbeef",
            "pattern": "stub",
            "difficulty": 1,
            "batch_size": 3,
            "allow_xuni": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["result"]["attempts"] == 3
    assert fake.calls[0][0] == "hash-batch"
    assert fake.calls[0][1]["key_prefix"] == "deadbeef"
    assert fake.calls[0][1]["allow_xuni"] is False


def test_batch_endpoint_forwards_gpu_first_blocks_flag():
    fake = FakeHashClient()
    client = TestClient(create_app(fake))

    response = client.post(
        "/hash/v1/batch",
        json={
            "salt": "aabbccddeeff0011",
            "backend": "cuda",
            "difficulty": 8,
            "batch_size": 3,
            "gpu_first_blocks": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert fake.calls[0][1]["gpu_first_blocks"] is True


def test_validate_rejects_gpu_first_blocks_for_cpu():
    errors = validate_hash_payload(
        {
            "algorithm": "argon2id-xen",
            "backend": "cpu",
            "salt_hex": "aabbccddeeff0011",
            "key_prefix": "",
            "target_pattern": "XEN11",
            "difficulty": 8,
            "batch_size": 1,
            "device_id": 0,
            "gpu_first_blocks": True,
        }
    )

    assert "gpu_first_blocks requires backend=cuda" in errors


def test_hash_one_validation_requires_key():
    client = TestClient(create_app(FakeHashClient()))

    response = client.post(
        "/hash/v1/hash-one",
        json={"salt": "aabbccddeeff0011", "backend": "cpu", "difficulty": 1},
    )

    assert response.status_code == 422


def test_benchmark_rejects_non_positive_seconds_before_cli():
    fake = FakeHashClient()
    client = TestClient(create_app(fake))

    response = client.post(
        "/hash/v1/benchmark",
        json={
            "salt": "aabbccddeeff0011",
            "backend": "cpu",
            "difficulty": 1,
            "batch_size": 1,
            "seconds": 0,
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert "seconds must be greater than zero" in response.json()["error"]
    assert fake.calls == []


def test_missing_binary_returns_structured_error(tmp_path):
    client = HashCliClient(tmp_path / "missing-hashapi-cli", timeout_seconds=1)

    result = client.run(
        "hash-batch",
        {
            "salt_hex": "aabbccddeeff0011",
            "backend": "cpu",
            "target_pattern": "XEN11",
            "difficulty": 1,
            "batch_size": 1,
            "device_id": 0,
        },
    )

    assert result.exit_code is None
    assert "hash command failed to start" in result.error
