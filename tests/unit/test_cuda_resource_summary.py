"""Tests for public-safe CUDA resource summary parsing."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.cuda_resource_summary as resources


def test_parse_resource_usage_extracts_public_kernel_rows():
    text = """
Fatbin ptx code:
================
arch = sm_75
Function _Z26argon2_first_blocks_kernelP7block_gPKhjS2_jjjjjjjjy:
  REG:255 STACK:496 SHARED:0 LOCAL:0 CONSTANT[0]:424 TEXTURE:0 SURFACE:0 SAMPLER:0
Function _Z21argon2_kernel_oneshotP7block_gj:
  REG:52 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:364 TEXTURE:0 SURFACE:0 SAMPLER:0

Fatbin ptx code:
================
arch = sm_86
Function _Z21argon2_kernel_oneshotP7block_gj:
  REG:40 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:364 TEXTURE:0 SURFACE:0 SAMPLER:0
"""

    summary = resources.parse_resource_usage(text)

    assert summary["schema"] == "xenblocks.cuda.resource_summary.v1"
    assert summary["source"] == "cuobjdump --dump-resource-usage"
    assert summary["kernels"] == [
        {
            "arch": "sm_75",
            "kernel": "argon2_first_blocks_kernel",
            "registers": 255,
            "stack": 496,
            "shared": 0,
            "local": 0,
            "constant0": 424,
            "texture": 0,
            "surface": 0,
            "sampler": 0,
        },
        {
            "arch": "sm_75",
            "kernel": "argon2_kernel_oneshot",
            "registers": 52,
            "stack": 0,
            "shared": 1024,
            "local": 0,
            "constant0": 364,
            "texture": 0,
            "surface": 0,
            "sampler": 0,
        },
        {
            "arch": "sm_86",
            "kernel": "argon2_kernel_oneshot",
            "registers": 40,
            "stack": 0,
            "shared": 1024,
            "local": 0,
            "constant0": 364,
            "texture": 0,
            "surface": 0,
            "sampler": 0,
        },
    ]


def test_main_writes_summary_without_binary_path(monkeypatch, tmp_path, capsys):
    def fake_run_cuobjdump(binary: Path, cuobjdump: str) -> str:
        assert binary == Path("private/build/miner.exe")
        assert cuobjdump == "fake-cuobjdump"
        return """
arch = sm_75
Function _Z21argon2_kernel_oneshotP7block_gj:
  REG:52 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:364 TEXTURE:0 SURFACE:0 SAMPLER:0
"""

    monkeypatch.setattr(resources, "run_cuobjdump", fake_run_cuobjdump)
    output = tmp_path / "resource-summary.json"

    exit_code = resources.main(
        [
            "--binary",
            "private/build/miner.exe",
            "--cuobjdump",
            "fake-cuobjdump",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    written = output.read_text(encoding="utf-8")
    assert "private/build/miner.exe" not in stdout
    assert "private/build/miner.exe" not in written
    parsed = json.loads(written)
    assert parsed["kernels"][0]["kernel"] == "argon2_kernel_oneshot"
