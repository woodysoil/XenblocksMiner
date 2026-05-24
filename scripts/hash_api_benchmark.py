"""Run reproducible Hash API benchmark scenarios and emit aggregate JSON."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SALT = "aabbccddeeff0011"
PRESET_NAMES = ("smoke", "warm-short", "cuda-compare")


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
    warmup: int = 0
    repeat: int = 1


def parse_scenario(text: str, default_warmup: int = 0, default_repeat: int = 1) -> BenchmarkScenario:
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
        warmup=int(parts.get("warmup", str(default_warmup))),
        repeat=max(1, int(parts.get("repeat", str(default_repeat)))),
    )


def default_scenarios(seconds: int, backend: str, device: int, warmup: int, repeat: int) -> list[BenchmarkScenario]:
    return preset_scenarios("smoke", seconds, backend, device, warmup, repeat)


def preset_scenarios(preset: str, seconds: int, backend: str, device: int, warmup: int, repeat: int) -> list[BenchmarkScenario]:
    if preset == "smoke":
        return [
            BenchmarkScenario(
                name=f"{backend}-smoke-b1-d1",
                backend=backend,
                difficulty=1,
                batch_size=1,
                seconds=seconds,
                device=device,
                warmup=warmup,
                repeat=repeat,
            ),
            BenchmarkScenario(
                name=f"{backend}-batch-b8-d1",
                backend=backend,
                difficulty=1,
                batch_size=8,
                seconds=seconds,
                device=device,
                warmup=warmup,
                repeat=repeat,
            ),
        ]

    if preset == "warm-short":
        pairs = [(1, 1), (1, 64), (8, 64)]
    elif preset == "cuda-compare":
        pairs = [(1, 64), (8, 64), (64, 128), (256, 256)]
    else:
        raise ValueError(f"unknown benchmark preset: {preset}")

    return [
        BenchmarkScenario(
            name=f"{backend}-{preset}-d{difficulty}-b{batch_size}",
            backend=backend,
            difficulty=difficulty,
            batch_size=batch_size,
            seconds=seconds,
            device=device,
            warmup=warmup,
            repeat=repeat,
        )
        for difficulty, batch_size in pairs
    ]


def ensure_unique_scenario_names(scenarios: list[BenchmarkScenario]) -> None:
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.name in seen:
            raise ValueError(f"duplicate benchmark scenario name: {scenario.name}")
        seen.add(scenario.name)


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


def summarize_timings(timings: Any) -> dict[str, float]:
    if not isinstance(timings, dict):
        return {}

    summary: dict[str, float] = {}
    for key, value in timings.items():
        try:
            summary[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return summary


def median_timings(summaries: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for summary in summaries
            for key in summary.get("timings", {})
        }
    )
    medians: dict[str, float] = {}
    for key in keys:
        values = [
            float(summary["timings"][key])
            for summary in summaries
            if key in summary.get("timings", {})
        ]
        if values:
            medians[key] = statistics.median(values)
    return medians


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
        "timings": summarize_timings(result.get("timings", {})),
        "matches": len(result.get("matches", [])),
        "ok": bool(result.get("ok")),
        "error": result.get("error", ""),
    }


def summarize_iterations(scenario: BenchmarkScenario, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ok_summaries = [item for item in summaries if item["ok"]]
    hashrates = [float(item["hashrate"]) for item in ok_summaries]
    errors = [item["error"] for item in summaries if item["error"]]
    aggregate = {
        "name": scenario.name,
        "backend": summaries[0]["backend"] if summaries else scenario.backend,
        "device_id": summaries[0]["device_id"] if summaries else scenario.device,
        "difficulty": scenario.difficulty,
        "batch_size": summaries[0]["batch_size"] if summaries else scenario.batch_size,
        "attempts": sum(int(item["attempts"]) for item in ok_summaries),
        "elapsed_ms": sum(float(item["elapsed_ms"]) for item in ok_summaries),
        "hashrate": statistics.median(hashrates) if hashrates else 0.0,
        "median_hashrate": statistics.median(hashrates) if hashrates else 0.0,
        "min_hashrate": min(hashrates) if hashrates else 0.0,
        "max_hashrate": max(hashrates) if hashrates else 0.0,
        "timings": median_timings(ok_summaries),
        "matches": sum(int(item["matches"]) for item in ok_summaries),
        "ok": len(ok_summaries) == len(summaries) and bool(summaries),
        "error": "; ".join(errors),
        "warmup": scenario.warmup,
        "repeat": scenario.repeat,
    }
    return aggregate


def build_hash_command(binary: Path, salt: str, scenario: BenchmarkScenario) -> list[str]:
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
    return command


def run_hash_command(command: list[str]) -> dict[str, Any]:
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
        "exit_code": completed.returncode,
        "wall_elapsed_ms": elapsed_ms,
        "result": result,
    }


def run_scenario(binary: Path, salt: str, scenario: BenchmarkScenario) -> dict[str, Any]:
    command = build_hash_command(binary, salt, scenario)
    warmup_runs = [run_hash_command(command) for _ in range(scenario.warmup)]
    iterations = [run_hash_command(command) for _ in range(scenario.repeat)]
    iteration_summaries = [summarize_result(scenario, item["result"]) for item in iterations]
    aggregate = summarize_iterations(scenario, iteration_summaries)
    selected_index = 0
    if iteration_summaries:
        selected_index = max(
            range(len(iteration_summaries)),
            key=lambda index: iteration_summaries[index]["hashrate"] if iteration_summaries[index]["ok"] else -1,
        )
    selected_result = iterations[selected_index]["result"] if iterations else {}
    all_runs = warmup_runs + iterations
    ok = bool(all_runs) and all(item["exit_code"] == 0 and item["result"].get("ok") for item in all_runs)

    return {
        "scenario": asdict(scenario),
        "summary": aggregate,
        "aggregate": aggregate,
        "command": command,
        "exit_code": 0 if ok else 2,
        "wall_elapsed_ms": sum(float(item["wall_elapsed_ms"]) for item in all_runs),
        "warmup_runs": warmup_runs,
        "iterations": iterations,
        "iteration_summaries": iteration_summaries,
        "result": selected_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Path to xenblocksMiner or hashapi-cli.")
    parser.add_argument("--salt", default=DEFAULT_SALT, help="Hex salt used by all benchmark scenarios.")
    parser.add_argument("--backend", default="cpu", help="Default backend for built-in scenarios.")
    parser.add_argument("--device", default=0, type=int, help="Default device id for built-in scenarios.")
    parser.add_argument("--seconds", default=5, type=int, help="Seconds per built-in scenario.")
    parser.add_argument("--warmup", default=0, type=int, help="Warm-up runs per scenario before measured repeats.")
    parser.add_argument("--repeat", default=1, type=int, help="Measured repeats per scenario.")
    parser.add_argument("--output", type=Path, help="Optional path to write the aggregate JSON report.")
    parser.add_argument(
        "--preset",
        action="append",
        choices=PRESET_NAMES,
        default=[],
        help="Add a reusable scenario preset. Can be provided more than once.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario as comma-separated key=value pairs, e.g. name=cpu1,backend=cpu,difficulty=1,batch_size=4,seconds=3.",
    )
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = [
            scenario
            for preset in args.preset
            for scenario in preset_scenarios(preset, args.seconds, args.backend, args.device, args.warmup, args.repeat)
        ]
        scenarios.extend(
            parse_scenario(item, default_warmup=args.warmup, default_repeat=args.repeat) for item in args.scenario
        )
        if not scenarios:
            scenarios = default_scenarios(args.seconds, args.backend, args.device, args.warmup, args.repeat)
        ensure_unique_scenario_names(scenarios)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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
        "presets": args.preset,
        "runs": [run_scenario(args.binary, args.salt, scenario) for scenario in scenarios],
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if all(run["exit_code"] == 0 and run["result"].get("ok") for run in report["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
