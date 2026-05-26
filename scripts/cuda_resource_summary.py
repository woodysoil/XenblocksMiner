"""Summarize CUDA kernel resource usage without machine-specific metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ARCH_RE = re.compile(r"^arch = (?P<arch>sm_\d+)\s*$")
FUNCTION_RE = re.compile(r"^\s*Function (?P<name>[^:]+):\s*$")
RESOURCE_RE = re.compile(
    r"^\s*"
    r"REG:(?P<registers>\d+)\s+"
    r"STACK:(?P<stack>\d+)\s+"
    r"SHARED:(?P<shared>\d+)\s+"
    r"LOCAL:(?P<local>\d+)\s+"
    r"CONSTANT\[0\]:(?P<constant0>\d+)\s+"
    r"TEXTURE:(?P<texture>\d+)\s+"
    r"SURFACE:(?P<surface>\d+)\s+"
    r"SAMPLER:(?P<sampler>\d+)\s*$"
)

KERNEL_ALIASES = {
    "_Z21argon2_kernel_oneshotP7block_gj": "argon2_kernel_oneshot",
    "_Z26argon2_first_blocks_kernelP7block_gPKhjS2_jjjjjjjjy": "argon2_first_blocks_kernel",
}


def public_kernel_name(raw_name: str) -> str:
    return KERNEL_ALIASES.get(raw_name, raw_name)


def parse_resource_usage(text: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    current_arch = ""
    current_function = ""

    for line in text.splitlines():
        arch_match = ARCH_RE.match(line)
        if arch_match:
            current_arch = arch_match.group("arch")
            current_function = ""
            continue

        function_match = FUNCTION_RE.match(line)
        if function_match:
            current_function = function_match.group("name")
            continue

        resource_match = RESOURCE_RE.match(line)
        if not resource_match or not current_arch or not current_function:
            continue

        values = {key: int(value) for key, value in resource_match.groupdict().items()}
        rows.append(
            {
                "arch": current_arch,
                "kernel": public_kernel_name(current_function),
                **values,
            }
        )

    return {
        "schema": "xenblocks.cuda.resource_summary.v1",
        "source": "cuobjdump --dump-resource-usage",
        "kernels": rows,
    }


def run_cuobjdump(binary: Path, cuobjdump: str) -> str:
    completed = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(binary)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 and not completed.stdout:
        raise RuntimeError(completed.stderr.strip() or f"{cuobjdump} exited with {completed.returncode}")
    return completed.stdout


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resource_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("arch", "")), str(row.get("kernel", ""))


def _resource_map(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = summary.get("kernels") or []
    return {
        _resource_key(row): row
        for row in rows
        if isinstance(row, dict) and all(_resource_key(row))
    }


def compare_resource_summaries(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_map = _resource_map(before)
    after_map = _resource_map(after)
    resource_fields = ["registers", "stack", "shared", "local", "constant0", "texture", "surface", "sampler"]
    comparisons: list[dict[str, Any]] = []
    regressions: list[str] = []

    for key in sorted(set(before_map) | set(after_map)):
        before_row = before_map.get(key)
        after_row = after_map.get(key)
        arch, kernel = key
        if before_row is None:
            comparisons.append({"arch": arch, "kernel": kernel, "status": "added"})
            regressions.append(f"{arch}:{kernel}:added")
            continue
        if after_row is None:
            comparisons.append({"arch": arch, "kernel": kernel, "status": "removed"})
            regressions.append(f"{arch}:{kernel}:removed")
            continue

        deltas: dict[str, dict[str, int]] = {}
        for field in resource_fields:
            before_value = int(before_row.get(field, 0) or 0)
            after_value = int(after_row.get(field, 0) or 0)
            deltas[field] = {
                "before": before_value,
                "after": after_value,
                "delta": after_value - before_value,
            }

        if deltas["registers"]["delta"] > 0:
            regressions.append(f"{arch}:{kernel}:registers+{deltas['registers']['delta']}")
        for field in ("stack", "local"):
            if deltas[field]["delta"] > 0:
                regressions.append(f"{arch}:{kernel}:{field}+{deltas[field]['delta']}")

        comparisons.append(
            {
                "arch": arch,
                "kernel": kernel,
                "status": "changed" if any(item["delta"] != 0 for item in deltas.values()) else "unchanged",
                "resources": deltas,
            }
        )

    return {
        "schema": "xenblocks.cuda.resource_compare.v1",
        "ok": not regressions,
        "regressions": regressions,
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize CUDA kernel resource usage as public-safe JSON.")
    parser.add_argument("--binary", type=Path, help="CUDA binary to inspect.")
    parser.add_argument("--cuobjdump", default="cuobjdump", help="cuobjdump executable.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--compare-before", type=Path, help="Optional previous resource summary JSON to compare.")
    parser.add_argument("--compare-after", type=Path, help="Optional next resource summary JSON to compare.")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero if resource comparison finds regressions.")
    args = parser.parse_args(argv)

    try:
        if args.compare_before or args.compare_after:
            if not args.compare_before or not args.compare_after:
                raise ValueError("--compare-before and --compare-after must be used together")
            summary = compare_resource_summaries(load_summary(args.compare_before), load_summary(args.compare_after))
        else:
            if args.binary is None:
                raise ValueError("--binary is required unless --compare-before and --compare-after are used")
            summary = parse_resource_usage(run_cuobjdump(args.binary, args.cuobjdump))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.fail_on_regression and not bool(summary.get("ok", True)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
