"""Validation helpers mirroring the C++ Hash API contract."""

from __future__ import annotations

import re
from typing import Any


KEY_LENGTH = 64
MAX_TARGET_PATTERN_LENGTH = 128
MAX_CPU_BATCH_SIZE = 10000
SUPPORTED_ALGORITHMS = {"argon2id-xen"}
SUPPORTED_BACKENDS = {"cpu", "reference", "cuda"}


def normalize_hex(value: str) -> str:
    text = value or ""
    if text.startswith(("0x", "0X")):
        text = text[2:]
    return text.lower()


def _is_hex(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]*", value))


def validate_hash_payload(payload: dict[str, Any], require_key: bool = False) -> list[str]:
    errors: list[str] = []

    algorithm = payload.get("algorithm", "argon2id-xen")
    if algorithm not in SUPPORTED_ALGORITHMS:
        errors.append(f"unsupported algorithm: {algorithm}")

    backend = payload.get("backend", "cpu")
    if backend not in SUPPORTED_BACKENDS:
        errors.append(f"unsupported backend: {backend}")

    salt = normalize_hex(str(payload.get("salt_hex", "")))
    if not salt:
        errors.append("salt_hex is required")
    else:
        if len(salt) % 2 != 0:
            errors.append("salt_hex must contain an even number of hex characters")
        if len(salt) < 16:
            errors.append("salt_hex must be at least 16 hex characters")
        if not _is_hex(salt):
            errors.append("salt_hex must contain only hex characters")

    prefix = normalize_hex(str(payload.get("key_prefix", "")))
    if prefix:
        if len(prefix) > KEY_LENGTH:
            errors.append("key_prefix cannot exceed 64 hex characters")
        if not _is_hex(prefix):
            errors.append("key_prefix must contain only hex characters")

    key = normalize_hex(str(payload.get("key", "")))
    if require_key and not key:
        errors.append("key is required")
    if key:
        if len(key) != KEY_LENGTH:
            errors.append("key must contain exactly 64 hex characters")
        if not _is_hex(key):
            errors.append("key must contain only hex characters")
        if prefix and not key.startswith(prefix):
            errors.append("key must start with key_prefix when both are provided")

    target_pattern = str(payload.get("target_pattern", ""))
    if not target_pattern:
        errors.append("target_pattern is required")
    if len(target_pattern) > MAX_TARGET_PATTERN_LENGTH:
        errors.append("target_pattern is too long")

    difficulty = int(payload.get("difficulty", 0))
    if difficulty <= 0:
        errors.append("difficulty must be greater than zero")

    batch_size = int(payload.get("batch_size", 0))
    if batch_size <= 0:
        errors.append("batch_size must be greater than zero")
    if backend in {"cpu", "reference"} and batch_size > MAX_CPU_BATCH_SIZE:
        errors.append("cpu batch_size exceeds safe limit")

    device_id = int(payload.get("device_id", 0))
    if device_id < 0:
        errors.append("device_id must be non-negative")

    if payload.get("gpu_first_blocks") and backend != "cuda":
        errors.append("gpu_first_blocks requires backend=cuda")

    return errors
