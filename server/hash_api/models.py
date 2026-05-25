"""Pydantic models for the standalone local Hash API service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HashRequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = ""
    algorithm: str = "argon2id-xen"
    backend: str = "cpu"
    salt_hex: str = Field("", alias="salt")
    key_prefix: str = Field("", alias="prefix")
    target_pattern: str = Field("XEN11", alias="pattern")
    difficulty: int = 42069
    batch_size: int = 1
    device_id: int = Field(0, alias="device")
    allow_xuni: bool = True
    gpu_first_blocks: bool = False


class HashOneRequest(HashRequestBase):
    key: str


class HashBatchRequest(HashRequestBase):
    pass


class HashBenchmarkRequest(HashRequestBase):
    seconds: int = 30
