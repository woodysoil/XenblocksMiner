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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize CUDA kernel resource usage as public-safe JSON.")
    parser.add_argument("--binary", required=True, type=Path, help="CUDA binary to inspect.")
    parser.add_argument("--cuobjdump", default="cuobjdump", help="cuobjdump executable.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    try:
        summary = parse_resource_usage(run_cuobjdump(args.binary, args.cuobjdump))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
