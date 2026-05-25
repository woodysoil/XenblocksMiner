"""Subprocess client for the standalone local Hash API service."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HashCommandResult:
    exit_code: int | None
    payload: dict[str, Any]
    error: str = ""
    timed_out: bool = False
    stderr: str = ""


class HashCliClient:
    def __init__(self, binary: str | Path, timeout_seconds: float = 60.0, max_concurrency: int = 1):
        self.binary = Path(binary)
        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    @classmethod
    def from_env(cls) -> "HashCliClient":
        binary = os.environ.get("XENBLOCKS_HASH_API_BINARY", "hashapi-cli")
        timeout = float(os.environ.get("XENBLOCKS_HASH_API_TIMEOUT", "60"))
        concurrency = int(os.environ.get("XENBLOCKS_HASH_API_CONCURRENCY", "1"))
        return cls(binary=binary, timeout_seconds=timeout, max_concurrency=concurrency)

    def backends(self) -> list[dict[str, Any]]:
        return [
            {"name": "cpu", "available": True},
            {"name": "reference", "available": True},
            {"name": "cuda", "available": True, "requires_full_miner_build": True},
        ]

    def run(self, command: str, payload: dict[str, Any]) -> HashCommandResult:
        if not self._semaphore.acquire(blocking=False):
            return HashCommandResult(
                exit_code=None,
                payload={},
                error="hash service concurrency limit reached",
            )

        try:
            completed = subprocess.run(
                self._build_command(command, payload),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except OSError as exc:
            return HashCommandResult(
                exit_code=None,
                payload={},
                error=f"hash command failed to start: {exc}",
            )
        except subprocess.TimeoutExpired:
            return HashCommandResult(
                exit_code=None,
                payload={},
                error=f"hash command timed out after {self.timeout_seconds:g} seconds",
                timed_out=True,
            )
        finally:
            self._semaphore.release()

        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return HashCommandResult(
                exit_code=completed.returncode,
                payload={},
                error="hash command did not emit valid JSON",
                stderr=completed.stderr,
            )

        return HashCommandResult(
            exit_code=completed.returncode,
            payload=parsed,
            error=parsed.get("error", ""),
            stderr=completed.stderr,
        )

    def _build_command(self, command: str, payload: dict[str, Any]) -> list[str]:
        args = [
            str(self.binary),
            command,
            "--backend",
            str(payload.get("backend", "cpu")),
            "--salt",
            str(payload.get("salt_hex", "")),
            "--pattern",
            str(payload.get("target_pattern", "XEN11")),
            "--batch-size",
            str(payload.get("batch_size", 1)),
            "--difficulty",
            str(payload.get("difficulty", 42069)),
            "--device",
            str(payload.get("device_id", 0)),
            "--json",
        ]

        if payload.get("request_id"):
            args.extend(["--request-id", str(payload["request_id"])])
        if payload.get("key_prefix"):
            args.extend(["--prefix", str(payload["key_prefix"])])
        if payload.get("key"):
            args.extend(["--key", str(payload["key"])])
        if command == "hash-benchmark":
            args.extend(["--seconds", str(payload.get("seconds", 30))])
        if payload.get("allow_xuni") is False:
            args.append("--no-xuni")
        if payload.get("gpu_first_blocks") is True:
            args.append("--gpu-first-blocks")

        return args
