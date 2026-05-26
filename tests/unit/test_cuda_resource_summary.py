"""Tests for public-safe CUDA resource summary parsing."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.cuda_resource_summary as resources


def _summary(registers: int = 53, stack: int = 0, local: int = 0) -> dict:
    return {
        "schema": "xenblocks.cuda.resource_summary.v1",
        "source": "cuobjdump --dump-resource-usage",
        "kernels": [
            {
                "arch": "sm_75",
                "kernel": "argon2_kernel_oneshot",
                "registers": registers,
                "stack": stack,
                "shared": 1024,
                "local": local,
                "constant0": 364,
                "texture": 0,
                "surface": 0,
                "sampler": 0,
            }
        ],
    }


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


def test_compare_resource_summaries_allows_equal_or_lower_resources():
    comparison = resources.compare_resource_summaries(_summary(registers=53), _summary(registers=52))

    assert comparison["schema"] == "xenblocks.cuda.resource_compare.v1"
    assert comparison["ok"] is True
    assert comparison["regressions"] == []
    row = comparison["comparisons"][0]
    assert row["status"] == "changed"
    assert row["resources"]["registers"] == {"before": 53, "after": 52, "delta": -1}
    assert row["resources"]["local"] == {"before": 0, "after": 0, "delta": 0}


def test_compare_resource_summaries_flags_register_stack_and_local_regressions():
    comparison = resources.compare_resource_summaries(
        _summary(registers=53, stack=0, local=0),
        _summary(registers=54, stack=8, local=16),
    )

    assert comparison["ok"] is False
    assert comparison["regressions"] == [
        "sm_75:argon2_kernel_oneshot:registers+1",
        "sm_75:argon2_kernel_oneshot:stack+8",
        "sm_75:argon2_kernel_oneshot:local+16",
    ]


def test_main_compares_summaries_without_private_paths(tmp_path, capsys):
    before = tmp_path / "private-before.json"
    after = tmp_path / "private-after.json"
    output = tmp_path / "resource-compare.json"
    before.write_text(json.dumps(_summary(registers=53)), encoding="utf-8")
    after.write_text(json.dumps(_summary(registers=52)), encoding="utf-8")

    exit_code = resources.main(
        [
            "--compare-before",
            str(before),
            "--compare-after",
            str(after),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    written = output.read_text(encoding="utf-8")
    assert str(before) not in stdout
    assert str(after) not in stdout
    assert str(before) not in written
    assert str(after) not in written
    parsed = json.loads(written)
    assert parsed["ok"] is True
    assert parsed["comparisons"][0]["resources"]["registers"]["delta"] == -1


def test_main_fail_on_regression_returns_two(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_summary(registers=53)), encoding="utf-8")
    after.write_text(json.dumps(_summary(registers=54)), encoding="utf-8")

    exit_code = resources.main(
        [
            "--compare-before",
            str(before),
            "--compare-after",
            str(after),
            "--fail-on-regression",
        ]
    )

    assert exit_code == 2


def test_main_requires_compare_args_to_be_paired(tmp_path, capsys):
    before = tmp_path / "before.json"
    before.write_text(json.dumps(_summary()), encoding="utf-8")

    exit_code = resources.main(["--compare-before", str(before)])

    assert exit_code == 1
    assert "--compare-before and --compare-after must be used together" in capsys.readouterr().err


def test_main_requires_binary_without_compare_mode(capsys):
    exit_code = resources.main([])

    assert exit_code == 1
    assert "--binary is required unless --compare-before and --compare-after are used" in capsys.readouterr().err
