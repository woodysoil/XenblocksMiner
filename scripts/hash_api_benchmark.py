"""Run reproducible Hash API benchmark scenarios and emit aggregate JSON."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SALT = "aabbccddeeff0011"


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    backend: str
    difficulty: int
    batch_size: int
    seconds: int
    prefix: str = ""
    pattern: str = "XEN11"
    device: int = 0


def parse_scenario(text: str) -> BenchmarkScenario:
    parts = dict(part.split("=", 1) for part in text.split(",") if part)
    name = parts.get("name") or f"{parts.get('backend', 'cpu')}-d{parts.get('difficulty', '1')}-b{parts.get('batch_size', '1')}"
    return BenchmarkScenario(
        name=name,
        backend=parts.get("backend", "cpu"),
        difficulty=int(parts.get("difficulty", "1")),
        batch_size=int(parts.get("batch_size", "1")),
        seconds=int(parts.get("seconds", "5")),
        prefix=parts.get("prefix", ""),
        pattern=parts.get("pattern", "XEN11"),
        device=int(parts.get("device", "0")),
    )


def default_scenarios(seconds: int, backend: str, device: int) -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name=f"{backend}-smoke-b1-d1",
            backend=backend,
            difficulty=1,
            batch_size=1,
            seconds=seconds,
            device=device,
        ),
        BenchmarkScenario(
            name=f"{backend}-batch-b8-d1",
            backend=backend,
            difficulty=1,
            batch_size=8,
            seconds=seconds,
            device=device,
        ),
    ]


def run_metadata_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "error": str(exc),
        }

    return {
        "available": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def collect_hardware_metadata() -> dict[str, Any]:
    return {
        "nvidia_smi": run_metadata_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": run_metadata_command(["nvcc", "--version"]),
    }


def summarize_result(scenario: BenchmarkScenario, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "backend": result.get("backend", scenario.backend),
        "device_id": result.get("device_id", scenario.device),
        "difficulty": scenario.difficulty,
        "batch_size": result.get("batch_size", scenario.batch_size),
        "attempts": result.get("attempts", 0),
        "elapsed_ms": result.get("elapsed_ms", 0.0),
        "hashrate": result.get("hashrate", 0.0),
        "matches": len(result.get("matches", [])),
        "ok": bool(result.get("ok")),
        "error": result.get("error", ""),
    }


def run_scenario(binary: Path, salt: str, scenario: BenchmarkScenario) -> dict[str, Any]:
    command = [
        str(binary),
        "hash-benchmark",
        "--backend",
        scenario.backend,
        "--salt",
        salt,
        "--pattern",
        scenario.pattern,
        "--batch-size",
        str(scenario.batch_size),
        "--difficulty",
        str(scenario.difficulty),
        "--seconds",
        str(scenario.seconds),
        "--device",
        str(scenario.device),
        "--json",
    ]
    if scenario.prefix:
        command.extend(["--prefix", scenario.prefix])

    started_at = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed_ms = (time.time() - started_at) * 1000.0

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "ok": False,
            "error": "hash-benchmark did not emit valid JSON",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    return {
        "scenario": asdict(scenario),
        "summary": summarize_result(scenario, result),
        "command": command,
        "exit_code": completed.returncode,
        "wall_elapsed_ms": elapsed_ms,
        "result": result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Path to xenblocksMiner or hashapi-cli.")
    parser.add_argument("--salt", default=DEFAULT_SALT, help="Hex salt used by all benchmark scenarios.")
    parser.add_argument("--backend", default="cpu", help="Default backend for built-in scenarios.")
    parser.add_argument("--device", default=0, type=int, help="Default device id for built-in scenarios.")
    parser.add_argument("--seconds", default=5, type=int, help="Seconds per built-in scenario.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario as comma-separated key=value pairs, e.g. name=cpu1,backend=cpu,difficulty=1,batch_size=4,seconds=3.",
    )
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    scenarios = [parse_scenario(item) for item in args.scenario]
    if not scenarios:
        scenarios = default_scenarios(args.seconds, args.backend, args.device)

    report = {
        "schema": "xenblocks.hashapi.benchmark.v1",
        "created_at_unix": time.time(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "hardware": collect_hardware_metadata(),
        "binary": str(args.binary),
        "salt": args.salt,
        "runs": [run_scenario(args.binary, args.salt, scenario) for scenario in scenarios],
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(run["exit_code"] == 0 and run["result"].get("ok") for run in report["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
