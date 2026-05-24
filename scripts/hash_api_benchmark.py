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
PRESET_NAMES = ("smoke", "warm-short", "cuda-compare", "batch-scan")
DEFAULT_STABLE_SPREAD_PCT = 10.0


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


def scan_scenarios(
    difficulties: list[int],
    batch_sizes: list[int],
    seconds: int,
    backend: str,
    device: int,
    warmup: int,
    repeat: int,
) -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name=f"{backend}-scan-d{difficulty}-b{batch_size}",
            backend=backend,
            difficulty=difficulty,
            batch_size=batch_size,
            seconds=seconds,
            device=device,
            warmup=warmup,
            repeat=repeat,
        )
        for difficulty in difficulties
        for batch_size in batch_sizes
    ]


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
    elif preset == "batch-scan":
        pairs = [
            (1, 64),
            (1, 128),
            (1, 256),
            (1, 512),
            (8, 64),
            (8, 128),
            (8, 256),
            (8, 512),
        ]
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


def timing_analysis(timings: dict[str, float]) -> dict[str, Any]:
    total_ms = float(timings.get("total_ms", 0.0) or 0.0)
    shares: dict[str, float] = {}
    for key, value in timings.items():
        if key == "total_ms" or total_ms <= 0.0:
            continue
        shares[key] = float(value) / total_ms * 100.0

    dominant_stage = ""
    dominant_stage_ms = 0.0
    dominant_stage_pct = 0.0
    stage_values = {key: value for key, value in timings.items() if key != "total_ms"}
    if stage_values:
        dominant_stage = max(stage_values, key=stage_values.get)
        dominant_stage_ms = float(timings.get(dominant_stage, 0.0) or 0.0)
        dominant_stage_pct = shares.get(dominant_stage, 0.0)

    return {
        "dominant_stage": dominant_stage,
        "dominant_stage_ms": dominant_stage_ms,
        "dominant_stage_pct": dominant_stage_pct,
        "stage_pct": shares,
    }


def hashrate_spread_pct(min_hashrate: float, max_hashrate: float, median_hashrate: float) -> float:
    if median_hashrate <= 0.0:
        return 0.0
    return (max_hashrate - min_hashrate) / median_hashrate * 100.0


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
    median_hashrate = statistics.median(hashrates) if hashrates else 0.0
    min_hashrate = min(hashrates) if hashrates else 0.0
    max_hashrate = max(hashrates) if hashrates else 0.0
    timings = median_timings(ok_summaries)
    aggregate = {
        "name": scenario.name,
        "backend": summaries[0]["backend"] if summaries else scenario.backend,
        "device_id": summaries[0]["device_id"] if summaries else scenario.device,
        "difficulty": scenario.difficulty,
        "batch_size": summaries[0]["batch_size"] if summaries else scenario.batch_size,
        "attempts": sum(int(item["attempts"]) for item in ok_summaries),
        "elapsed_ms": sum(float(item["elapsed_ms"]) for item in ok_summaries),
        "hashrate": median_hashrate,
        "median_hashrate": median_hashrate,
        "min_hashrate": min_hashrate,
        "max_hashrate": max_hashrate,
        "hashrate_spread_pct": hashrate_spread_pct(min_hashrate, max_hashrate, median_hashrate),
        "timings": timings,
        "timing_analysis": timing_analysis(timings),
        "matches": sum(int(item["matches"]) for item in ok_summaries),
        "ok": len(ok_summaries) == len(summaries) and bool(summaries),
        "error": "; ".join(errors),
        "warmup": scenario.warmup,
        "repeat": scenario.repeat,
    }
    return aggregate


def build_recommendations(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for run in runs:
        summary = run.get("summary") or {}
        if not summary.get("ok"):
            continue
        key = (
            str(summary.get("backend", "")),
            int(summary.get("device_id", 0)),
            int(summary.get("difficulty", 0)),
        )
        hashrate = float(summary.get("median_hashrate", summary.get("hashrate", 0.0)) or 0.0)
        current = best_by_key.get(key)
        current_hashrate = 0.0
        if current is not None:
            current_hashrate = float(current.get("median_hashrate", current.get("hashrate", 0.0)) or 0.0)
        if current is None or hashrate > current_hashrate:
            best_by_key[key] = summary

    batch_size_by_difficulty = [
        {
            "backend": backend,
            "device_id": device_id,
            "difficulty": difficulty,
            "batch_size": int(summary.get("batch_size", 0)),
            "median_hashrate": float(summary.get("median_hashrate", summary.get("hashrate", 0.0)) or 0.0),
            "hashrate_spread_pct": float(summary.get("hashrate_spread_pct", 0.0) or 0.0),
            "stable": float(summary.get("hashrate_spread_pct", 0.0) or 0.0) <= DEFAULT_STABLE_SPREAD_PCT,
            "dominant_stage": str((summary.get("timing_analysis") or {}).get("dominant_stage", "")),
            "dominant_stage_pct": float((summary.get("timing_analysis") or {}).get("dominant_stage_pct", 0.0) or 0.0),
            "scenario": str(summary.get("name", "")),
        }
        for (backend, device_id, difficulty), summary in sorted(best_by_key.items())
    ]
    return {
        "stable_spread_pct": DEFAULT_STABLE_SPREAD_PCT,
        "batch_size_by_difficulty": batch_size_by_difficulty,
    }


def sanitize_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    safe_keys = (
        "name",
        "backend",
        "difficulty",
        "batch_size",
        "seconds",
        "device",
        "warmup",
        "repeat",
        "pattern",
    )
    sanitized = {key: scenario[key] for key in safe_keys if key in scenario}
    prefix = str(scenario.get("prefix", ""))
    sanitized["prefix_length"] = len(prefix)
    return sanitized


def build_sanitized_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "xenblocks.hashapi.benchmark-summary.v1",
        "source_schema": report.get("schema", ""),
        "created_at_unix": report.get("created_at_unix"),
        "privacy": {
            "sanitized": True,
            "omitted_fields": [
                "binary",
                "command",
                "hardware",
                "host",
                "iterations",
                "prefix",
                "raw result",
                "salt",
                "warmup_runs",
            ],
        },
        "presets": report.get("presets", []),
        "recommendations": report.get("recommendations", {}),
        "runs": [
            {
                "scenario": sanitize_scenario(run.get("scenario", {})),
                "summary": run.get("summary", {}),
            }
            for run in report.get("runs", [])
        ],
    }


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
        "--sanitized-output",
        type=Path,
        help="Optional path to write a public-safe summary without local paths, hardware details, commands, raw results, salts, or prefixes.",
    )
    parser.add_argument("--recommendations-only", action="store_true", help="Print only report recommendations as JSON.")
    parser.add_argument(
        "--scan-difficulty",
        action="append",
        type=int,
        default=[],
        help="Add a difficulty value for generated batch-size scan scenarios. Requires --scan-batch-size.",
    )
    parser.add_argument(
        "--scan-batch-size",
        action="append",
        type=int,
        default=[],
        help="Add a batch size for generated scan scenarios. Requires --scan-difficulty.",
    )
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
        if args.scan_difficulty or args.scan_batch_size:
            if not args.scan_difficulty or not args.scan_batch_size:
                raise ValueError("--scan-difficulty and --scan-batch-size must be used together")
            scenarios.extend(
                scan_scenarios(
                    args.scan_difficulty,
                    args.scan_batch_size,
                    args.seconds,
                    args.backend,
                    args.device,
                    args.warmup,
                    args.repeat,
                )
            )
        if not scenarios:
            scenarios = default_scenarios(args.seconds, args.backend, args.device, args.warmup, args.repeat)
        ensure_unique_scenario_names(scenarios)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runs = [run_scenario(args.binary, args.salt, scenario) for scenario in scenarios]
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
        "recommendations": build_recommendations(runs),
        "runs": runs,
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if args.sanitized_output:
        args.sanitized_output.parent.mkdir(parents=True, exist_ok=True)
        sanitized_output = json.dumps(build_sanitized_report(report), indent=2, sort_keys=True)
        args.sanitized_output.write_text(sanitized_output + "\n", encoding="utf-8")
    if args.recommendations_only:
        print(json.dumps(report["recommendations"], indent=2, sort_keys=True))
    else:
        print(output)
    return 0 if all(run["exit_code"] == 0 and run["result"].get("ok") for run in report["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
