"""Run reproducible Hash API benchmark scenarios and emit aggregate JSON."""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SALT = "aabbccddeeff0011"
PRESET_NAMES = ("smoke", "warm-short", "cuda-compare", "batch-scan", "difficulty-sequence", "isolation")
DEFAULT_STABLE_SPREAD_PCT = 10.0
NESTED_TIMING_FIELDS = frozenset(
    {
        "kernel_ms",
        "host_to_device_ms",
        "gpu_first_block_ms",
        "device_to_host_ms",
        "setup_normalize_cpu_ms",
        "setup_activate_cpu_ms",
        "setup_device_info_cpu_ms",
        "setup_params_cpu_ms",
        "setup_backend_init_cpu_ms",
        "first_block_initial_hash_cpu_ms",
        "first_block_digest_cpu_ms",
        "first_block_max_worker_ms",
        "first_block_thread_launch_ms",
        "first_block_max_worker_start_ms",
        "first_block_worker_start_span_ms",
        "first_block_max_worker_finish_ms",
        "first_block_worker_finish_span_ms",
        "finalize_hash_ms",
        "argon2_finalize_ms",
        "base64_ms",
        "match_ms",
    }
)
NESTED_TIMING_PARENTS = {
    "kernel_ms": "compute_ms",
    "host_to_device_ms": "compute_ms",
    "gpu_first_block_ms": "compute_ms",
    "device_to_host_ms": "compute_ms",
    "setup_normalize_cpu_ms": "setup_ms",
    "setup_activate_cpu_ms": "setup_ms",
    "setup_device_info_cpu_ms": "setup_ms",
    "setup_params_cpu_ms": "setup_ms",
    "setup_backend_init_cpu_ms": "setup_ms",
    "first_block_initial_hash_cpu_ms": "first_block_ms",
    "first_block_digest_cpu_ms": "first_block_ms",
    "first_block_max_worker_ms": "first_block_ms",
    "first_block_thread_launch_ms": "first_block_ms",
    "first_block_max_worker_start_ms": "first_block_ms",
    "first_block_worker_start_span_ms": "first_block_ms",
    "first_block_max_worker_finish_ms": "first_block_ms",
    "first_block_worker_finish_span_ms": "first_block_ms",
    "finalize_hash_ms": "finalize_ms",
    "argon2_finalize_ms": "finalize_ms",
    "base64_ms": "finalize_ms",
    "match_ms": "finalize_ms",
}


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    backend: str
    difficulty: int
    batch_size: int
    seconds: int
    difficulty_sequence: tuple[int, ...] = ()
    batch_size_sequence: tuple[int, ...] = ()
    prefix: str = ""
    key: str = ""
    pattern: str = "XEN11"
    device: int = 0
    warmup: int = 0
    repeat: int = 1
    allow_xuni: bool = True
    detailed_timings: bool = False
    first_block_workers: int = 0
    first_block_dynamic_chunk_size: int = 0
    first_block_dynamic_chunk_auto: bool = False
    gpu_first_blocks: bool = False
    auto_batch_size: bool = False


def parse_difficulty_sequence(text: str) -> tuple[int, ...]:
    normalized = text.replace("|", ",").replace(";", ",")
    values = []
    for token in normalized.split(","):
        token = token.strip()
        if token == "":
            raise ValueError("difficulty sequence cannot contain empty values")
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError("difficulty sequence values must be integers") from exc
        if value <= 0:
            raise ValueError("difficulty sequence values must be greater than zero")
        values.append(value)
    if not values:
        raise ValueError("difficulty sequence must not be empty")
    return tuple(values)


def parse_batch_size_sequence(text: str) -> tuple[int, ...]:
    normalized = text.replace("|", ",").replace(";", ",")
    values = []
    for token in normalized.split(","):
        token = token.strip()
        if token == "":
            raise ValueError("batch-size sequence cannot contain empty values")
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError("batch-size sequence values must be integers") from exc
        if value <= 0:
            raise ValueError("batch-size sequence values must be greater than zero")
        values.append(value)
    if not values:
        raise ValueError("batch-size sequence must not be empty")
    return tuple(values)


def difficulty_sequence_label(sequence: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in sequence)


def batch_size_sequence_label(sequence: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in sequence)


def difficulty_change_count(sequence: tuple[int, ...]) -> int:
    if len(sequence) < 2:
        return 0
    return sum(1 for index in range(1, len(sequence)) if sequence[index] != sequence[index - 1])


def batch_size_change_count(sequence: tuple[int, ...]) -> int:
    if len(sequence) < 2:
        return 0
    return sum(1 for index in range(1, len(sequence)) if sequence[index] != sequence[index - 1])


def parse_scenario(text: str, default_warmup: int = 0, default_repeat: int = 1) -> BenchmarkScenario:
    parts: dict[str, str] = {}
    for part in text.split(","):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                "scenario fields must be key=value pairs; use difficulty_sequence=1|8|1|8 inside --scenario"
            )
        key, value = part.split("=", 1)
        parts[key] = value
    difficulty_sequence = parse_difficulty_sequence(parts["difficulty_sequence"]) if "difficulty_sequence" in parts else ()
    batch_size_sequence = parse_batch_size_sequence(parts["batch_size_sequence"]) if "batch_size_sequence" in parts else ()
    difficulty = int(parts.get("difficulty", str(difficulty_sequence[0] if difficulty_sequence else 1)))
    batch_size = int(parts.get("batch_size", str(batch_size_sequence[0] if batch_size_sequence else 1)))
    name = parts.get("name")
    if not name:
        difficulty_label = f"seq-d{difficulty_sequence_label(difficulty_sequence)}" if difficulty_sequence else f"d{difficulty}"
        batch_label = f"bseq-{batch_size_sequence_label(batch_size_sequence)}" if batch_size_sequence else f"b{batch_size}"
        name = f"{parts.get('backend', 'cpu')}-{difficulty_label}-{batch_label}"
    return BenchmarkScenario(
        name=name,
        backend=parts.get("backend", "cpu"),
        difficulty=difficulty,
        batch_size=batch_size,
        seconds=int(parts.get("seconds", "5")),
        difficulty_sequence=difficulty_sequence,
        batch_size_sequence=batch_size_sequence,
        prefix=parts.get("prefix", ""),
        key=parts.get("key", ""),
        pattern=parts.get("pattern", "XEN11"),
        device=int(parts.get("device", "0")),
        warmup=int(parts.get("warmup", str(default_warmup))),
        repeat=max(1, int(parts.get("repeat", str(default_repeat)))),
        allow_xuni=parts.get("allow_xuni", "true").lower() not in {"0", "false", "no"},
        detailed_timings=parts.get("detailed_timings", "false").lower() in {"1", "true", "yes"},
        first_block_workers=max(0, int(parts.get("first_block_workers", "0"))),
        first_block_dynamic_chunk_size=max(0, int(parts.get("first_block_dynamic_chunk_size", "0"))),
        first_block_dynamic_chunk_auto=parts.get("first_block_dynamic_chunk_auto", "false").lower()
        in {"1", "true", "yes"},
        gpu_first_blocks=parts.get("gpu_first_blocks", "false").lower() in {"1", "true", "yes"},
        auto_batch_size=parts.get("auto_batch_size", "false").lower() in {"1", "true", "yes"},
    )


def default_scenarios(seconds: int, backend: str, device: int, warmup: int, repeat: int) -> list[BenchmarkScenario]:
    return preset_scenarios("smoke", seconds, backend, device, warmup, repeat)


def enable_gpu_first_blocks(scenarios: list[BenchmarkScenario]) -> list[BenchmarkScenario]:
    enabled: list[BenchmarkScenario] = []
    for scenario in scenarios:
        if scenario.gpu_first_blocks:
            enabled.append(scenario)
            continue
        enabled.append(
            BenchmarkScenario(
                **{
                    **asdict(scenario),
                    "name": f"{scenario.name}-gfb",
                    "gpu_first_blocks": True,
                }
            )
        )
    return enabled


def scan_scenarios(
    difficulties: list[int],
    batch_sizes: list[int],
    first_block_workers: list[int],
    first_block_dynamic_chunk_sizes: list[int],
    detailed_timings: bool,
    seconds: int,
    backend: str,
    device: int,
    warmup: int,
    repeat: int,
    first_block_dynamic_chunk_auto: bool = False,
    scan_gpu_first_blocks: bool = False,
) -> list[BenchmarkScenario]:
    worker_caps = first_block_workers or [0]
    dynamic_chunk_sizes = first_block_dynamic_chunk_sizes or [0]
    gpu_first_block_modes = [False, True] if scan_gpu_first_blocks else [False]
    return [
        BenchmarkScenario(
            name=f"{backend}-scan-d{difficulty}-b{'auto' if batch_size == 0 else batch_size}"
            + ("" if worker_cap == 0 else f"-fbw{worker_cap}")
            + ("" if dynamic_chunk_size == 0 else f"-fbd{dynamic_chunk_size}")
            + ("-fbda" if first_block_dynamic_chunk_auto else "")
            + ("-gfb" if gpu_first_blocks else ""),
            backend=backend,
            difficulty=difficulty,
            batch_size=batch_size,
            seconds=seconds,
            device=device,
            warmup=warmup,
            repeat=repeat,
            first_block_workers=worker_cap,
            first_block_dynamic_chunk_size=dynamic_chunk_size,
            first_block_dynamic_chunk_auto=first_block_dynamic_chunk_auto,
            gpu_first_blocks=gpu_first_blocks,
            auto_batch_size=batch_size == 0,
            detailed_timings=detailed_timings,
        )
        for difficulty in difficulties
        for batch_size in batch_sizes
        for worker_cap in worker_caps
        for dynamic_chunk_size in dynamic_chunk_sizes
        for gpu_first_blocks in gpu_first_block_modes
    ]


def difficulty_sequence_scenarios(
    sequences: list[tuple[int, ...]],
    batch_sizes: list[int],
    detailed_timings: bool,
    first_block_dynamic_chunk_auto: bool,
    seconds: int,
    backend: str,
    device: int,
    warmup: int,
    repeat: int,
) -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name=f"{backend}-difficulty-sequence-d{difficulty_sequence_label(sequence)}-b{batch_size}",
            backend=backend,
            difficulty=sequence[0],
            batch_size=batch_size,
            seconds=seconds,
            difficulty_sequence=sequence,
            device=device,
            warmup=warmup,
            repeat=repeat,
            detailed_timings=detailed_timings,
            first_block_dynamic_chunk_auto=first_block_dynamic_chunk_auto,
        )
        for sequence in sequences
        for batch_size in batch_sizes
    ]


def automatic_batch_difficulty_sequence_scenarios(
    sequences: list[tuple[int, ...]],
    detailed_timings: bool,
    first_block_dynamic_chunk_auto: bool,
    seconds: int,
    backend: str,
    device: int,
    warmup: int,
    repeat: int,
) -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name=f"{backend}-difficulty-sequence-d{difficulty_sequence_label(sequence)}-bauto",
            backend=backend,
            difficulty=sequence[0],
            batch_size=0,
            seconds=seconds,
            difficulty_sequence=sequence,
            device=device,
            warmup=warmup,
            repeat=repeat,
            detailed_timings=detailed_timings,
            first_block_dynamic_chunk_auto=first_block_dynamic_chunk_auto,
            auto_batch_size=True,
        )
        for sequence in sequences
    ]


def paired_sequence_scenarios(
    difficulty_sequences: list[tuple[int, ...]],
    batch_size_sequences: list[tuple[int, ...]],
    detailed_timings: bool,
    first_block_dynamic_chunk_auto: bool,
    seconds: int,
    backend: str,
    device: int,
    warmup: int,
    repeat: int,
) -> list[BenchmarkScenario]:
    for difficulty_sequence in difficulty_sequences:
        for batch_size_sequence in batch_size_sequences:
            if (
                len(difficulty_sequence) != len(batch_size_sequence)
                and len(difficulty_sequence) != 1
                and len(batch_size_sequence) != 1
            ):
                raise ValueError(
                    "difficulty sequence and batch-size sequence lengths must match unless one sequence has length 1"
                )
    return [
        BenchmarkScenario(
            name=(
                f"{backend}-difficulty-sequence-d{difficulty_sequence_label(difficulty_sequence)}"
                f"-bseq-{batch_size_sequence_label(batch_size_sequence)}"
            ),
            backend=backend,
            difficulty=difficulty_sequence[0],
            batch_size=batch_size_sequence[0],
            seconds=seconds,
            difficulty_sequence=difficulty_sequence,
            batch_size_sequence=batch_size_sequence,
            device=device,
            warmup=warmup,
            repeat=repeat,
            detailed_timings=detailed_timings,
            first_block_dynamic_chunk_auto=first_block_dynamic_chunk_auto,
        )
        for difficulty_sequence in difficulty_sequences
        for batch_size_sequence in batch_size_sequences
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
                difficulty_sequence=(),
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
                difficulty_sequence=(),
                device=device,
                warmup=warmup,
                repeat=repeat,
            ),
        ]

    if preset == "difficulty-sequence":
        return difficulty_sequence_scenarios(
            [(1, 1, 1, 1), (1, 8, 1, 8), (8, 64, 8, 64)],
            [512],
            False,
            False,
            seconds,
            backend,
            device,
            warmup,
            repeat,
        )

    if preset == "isolation":
        return [
            BenchmarkScenario(
                name=f"{backend}-isolation-generated-d8-b2048",
                backend=backend,
                difficulty=8,
                batch_size=2048,
                seconds=seconds,
                device=device,
                warmup=warmup,
                repeat=repeat,
            ),
            BenchmarkScenario(
                name=f"{backend}-isolation-fixed-d8-b1",
                backend=backend,
                difficulty=8,
                batch_size=1,
                seconds=seconds,
                key="0" * 64,
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
            difficulty_sequence=(),
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


def run_metadata_command(command: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
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


def parse_cmake_cache(cache_path: Path) -> dict[str, str]:
    cache_file = cache_path if cache_path.name == "CMakeCache.txt" else cache_path / "CMakeCache.txt"
    if not cache_file.exists():
        return {}

    values: dict[str, str] = {}
    for line in cache_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        values[key] = value.strip()
    return values


def parse_nvcc_version_output(output: str) -> dict[str, str]:
    release_match = re.search(r"release\s+([^,\s]+)", output)
    version_match = re.search(r"\bV(\d+(?:\.\d+)+)\b", output)
    metadata: dict[str, str] = {}
    if release_match:
        metadata["release"] = release_match.group(1)
    if version_match:
        metadata["version"] = version_match.group(1)
    return metadata


def cuda_compiler_metadata(compiler: str) -> dict[str, Any]:
    if not compiler:
        return {}

    metadata: dict[str, Any] = {
        "basename": compiler.replace("\\", "/").rsplit("/", 1)[-1],
    }
    try:
        completed = subprocess.run([compiler, "--version"], text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        metadata["available"] = False
        return metadata

    metadata["available"] = completed.returncode == 0
    if completed.returncode == 0:
        metadata.update(parse_nvcc_version_output(f"{completed.stdout}\n{completed.stderr}"))
    return metadata


def collect_build_metadata(cache_path: Path | None) -> dict[str, Any]:
    if cache_path is None:
        return {"provided": False}

    cache = parse_cmake_cache(cache_path)
    if not cache:
        return {
            "provided": True,
            "available": False,
            "error": "CMakeCache.txt not found",
        }

    raw_architectures = cache.get("CMAKE_CUDA_ARCHITECTURES", "")
    metadata: dict[str, Any] = {
        "provided": True,
        "available": True,
    }
    for output_key, cache_key in (
        ("generator", "CMAKE_GENERATOR"),
        ("build_type", "CMAKE_BUILD_TYPE"),
        ("configuration_types", "CMAKE_CONFIGURATION_TYPES"),
        ("vcpkg_target_triplet", "VCPKG_TARGET_TRIPLET"),
    ):
        value = cache.get(cache_key, "")
        if value:
            metadata[output_key] = value
    if raw_architectures:
        metadata["cuda_architectures_raw"] = raw_architectures
        metadata["cuda_architectures"] = [
            item
            for item in re.split(r"[;,]\s*", raw_architectures)
            if item
        ]
    compiler = cuda_compiler_metadata(cache.get("CMAKE_CUDA_COMPILER", ""))
    if compiler:
        metadata["cuda_compiler"] = compiler
    return metadata


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


def collect_environment_metadata() -> dict[str, Any]:
    if platform.system() != "Windows":
        return {
            "available": False,
            "reason": "unsupported_platform",
        }

    completed = run_metadata_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average",
        ],
        timeout=5,
    )
    if not completed.get("available") or completed.get("exit_code") != 0:
        return {
            "available": False,
            "reason": "cpu_load_unavailable",
        }

    try:
        cpu_load_pct = float(str(completed.get("stdout", "")).strip())
    except ValueError:
        return {
            "available": False,
            "reason": "cpu_load_parse_failed",
        }

    high_cpu_load = cpu_load_pct >= 90.0
    return {
        "available": True,
        "cpu_load_pct": cpu_load_pct,
        "high_cpu_load": high_cpu_load,
        "benchmark_trust": "low" if high_cpu_load else "normal",
    }


def combine_environment_metadata(*samples: dict[str, Any]) -> dict[str, Any]:
    available_samples = [sample for sample in samples if sample.get("available")]
    if not available_samples:
        return samples[0] if samples else {"available": False, "reason": "environment_unavailable"}

    cpu_loads = [float(sample.get("cpu_load_pct", 0.0) or 0.0) for sample in available_samples]
    high_cpu_load = any(bool(sample.get("high_cpu_load")) for sample in available_samples) or max(cpu_loads) >= 90.0
    return {
        "available": True,
        "cpu_load_pct": max(cpu_loads),
        "start_cpu_load_pct": cpu_loads[0],
        "end_cpu_load_pct": cpu_loads[-1],
        "sample_count": len(available_samples),
        "high_cpu_load": high_cpu_load,
        "benchmark_trust": "low" if high_cpu_load else "normal",
    }


def sample_environment(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample = collect_environment_metadata()
    samples.append(sample)
    return sample


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


def _median_int(values: list[Any]) -> int:
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(int(value))
        except (TypeError, ValueError):
            continue
    return int(statistics.median(numeric_values)) if numeric_values else 0


def timing_analysis(timings: dict[str, float]) -> dict[str, Any]:
    total_ms = float(timings.get("total_ms", 0.0) or 0.0)
    shares: dict[str, float] = {}
    nested_shares: dict[str, float] = {}
    for key, value in timings.items():
        if key == "total_ms" or key in NESTED_TIMING_FIELDS or total_ms <= 0.0:
            continue
        shares[key] = float(value) / total_ms * 100.0

    for key, parent in NESTED_TIMING_PARENTS.items():
        value = float(timings.get(key, 0.0) or 0.0)
        parent_value = float(timings.get(parent, 0.0) or 0.0)
        if parent_value <= 0.0:
            continue
        nested_shares[key] = value / parent_value * 100.0

    dominant_stage = ""
    dominant_stage_ms = 0.0
    dominant_stage_pct = 0.0
    stage_values = {key: value for key, value in timings.items() if key != "total_ms" and key not in NESTED_TIMING_FIELDS}
    if stage_values:
        dominant_stage = max(stage_values, key=stage_values.get)
        dominant_stage_ms = float(timings.get(dominant_stage, 0.0) or 0.0)
        dominant_stage_pct = shares.get(dominant_stage, 0.0)

    first_block_wall_ms = float(timings.get("first_block_ms", 0.0) or 0.0)
    first_block_initial_ms = float(timings.get("first_block_initial_hash_cpu_ms", 0.0) or 0.0)
    first_block_digest_ms = float(timings.get("first_block_digest_cpu_ms", 0.0) or 0.0)
    first_block_worker_wall_ms = float(timings.get("first_block_max_worker_ms", 0.0) or 0.0)
    first_block_worker_finish_ms = float(timings.get("first_block_max_worker_finish_ms", 0.0) or 0.0)
    input_ms = float(timings.get("input_ms", 0.0) or 0.0)
    keygen_ms = float(timings.get("keygen_ms", 0.0) or 0.0)
    first_block_cpu_sum_ms = first_block_initial_ms + first_block_digest_ms
    input_explained_ms = keygen_ms + first_block_wall_ms
    input_residual_ms = (
        input_ms - input_explained_ms
        if input_ms > 0.0 and input_explained_ms > 0.0
        else 0.0
    )
    input_residual_ms = max(0.0, input_residual_ms)
    input_explained_to_input = (
        input_explained_ms / input_ms
        if input_ms > 0.0 and input_explained_ms > 0.0
        else 0.0
    )
    input_residual_pct = (
        input_residual_ms / input_ms * 100.0
        if input_ms > 0.0 and input_residual_ms > 0.0
        else 0.0
    )
    first_block_cpu_sum_to_wall = (
        first_block_cpu_sum_ms / first_block_wall_ms if first_block_wall_ms > 0.0 and first_block_cpu_sum_ms > 0.0 else 0.0
    )
    first_block_worker_wall_to_wall = (
        first_block_worker_wall_ms / first_block_wall_ms
        if first_block_wall_ms > 0.0 and first_block_worker_wall_ms > 0.0
        else 0.0
    )
    first_block_scheduling_overhead_ms = (
        max(0.0, first_block_wall_ms - first_block_worker_wall_ms)
        if first_block_wall_ms > 0.0 and first_block_worker_wall_ms > 0.0
        else 0.0
    )
    first_block_finish_wall_to_wall = (
        first_block_worker_finish_ms / first_block_wall_ms
        if first_block_wall_ms > 0.0 and first_block_worker_finish_ms > 0.0
        else 0.0
    )
    first_block_post_worker_overhead_ms = (
        max(0.0, first_block_wall_ms - first_block_worker_finish_ms)
        if first_block_wall_ms > 0.0 and first_block_worker_finish_ms > 0.0
        else 0.0
    )

    return {
        "dominant_stage": dominant_stage,
        "dominant_stage_ms": dominant_stage_ms,
        "dominant_stage_pct": dominant_stage_pct,
        "stage_pct": shares,
        "nested_stage_pct": nested_shares,
        "input_explained_ms": input_explained_ms,
        "input_residual_ms": input_residual_ms,
        "input_explained_to_input": input_explained_to_input,
        "input_residual_pct": input_residual_pct,
        "first_block_cpu_sum_ms": first_block_cpu_sum_ms,
        "first_block_cpu_sum_to_wall": first_block_cpu_sum_to_wall,
        "first_block_worker_wall_to_wall": first_block_worker_wall_to_wall,
        "first_block_scheduling_overhead_ms": first_block_scheduling_overhead_ms,
        "first_block_finish_wall_to_wall": first_block_finish_wall_to_wall,
        "first_block_post_worker_overhead_ms": first_block_post_worker_overhead_ms,
    }


def timing_per_attempt(timings: dict[str, float], attempts: int) -> dict[str, float]:
    if attempts <= 0:
        return {}
    return {key: value / attempts for key, value in timings.items()}


def median_timing_per_attempt(summaries: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for summary in summaries
            for key in summary.get("timings", {})
        }
    )
    medians: dict[str, float] = {}
    for key in keys:
        values = []
        for summary in summaries:
            attempts = int(summary.get("attempts", 0) or 0)
            per_attempt = timing_per_attempt(summary.get("timings", {}), attempts)
            if key in per_attempt:
                values.append(per_attempt[key])
        if values:
            medians[key] = statistics.median(values)
    return medians


def hashrate_spread_pct(min_hashrate: float, max_hashrate: float, median_hashrate: float) -> float:
    if median_hashrate <= 0.0:
        return 0.0
    return (max_hashrate - min_hashrate) / median_hashrate * 100.0


def summarize_result(scenario: BenchmarkScenario, result: dict[str, Any]) -> dict[str, Any]:
    batch_size = result.get("batch_size", scenario.batch_size)
    return {
        "name": scenario.name,
        "backend": result.get("backend", scenario.backend),
        "device_id": result.get("device_id", scenario.device),
        "difficulty": scenario.difficulty,
        "difficulty_sequence": list(scenario.difficulty_sequence),
        "difficulty_mode": "sequence" if scenario.difficulty_sequence else "fixed",
        "difficulty_changes": difficulty_change_count(scenario.difficulty_sequence),
        "key_mode": "fixed" if scenario.key else "generated",
        "batch_size": batch_size,
        "auto_batch_size": scenario.auto_batch_size,
        "batch_size_sequence": list(scenario.batch_size_sequence),
        "batch_size_mode": "sequence" if scenario.batch_size_sequence else "fixed",
        "batch_size_changes": batch_size_change_count(scenario.batch_size_sequence),
        "batch_size_min": result.get("batch_size_min", batch_size),
        "batch_size_max": result.get("batch_size_max", batch_size),
        "attempts": result.get("attempts", 0),
        "first_block_workers": scenario.first_block_workers,
        "first_block_dynamic_chunk_size": result.get(
            "first_block_dynamic_chunk_size",
            scenario.first_block_dynamic_chunk_size,
        ),
        "first_block_dynamic_chunk_auto": result.get(
            "first_block_dynamic_chunk_auto",
            scenario.first_block_dynamic_chunk_auto,
        ),
        "first_block_worker_count": result.get("first_block_worker_count", 0),
        "first_block_chunk_size": result.get("first_block_chunk_size", 0),
        "first_block_dynamic_chunk_size_min": result.get(
            "first_block_dynamic_chunk_size_min",
            result.get("first_block_dynamic_chunk_size", 0),
        ),
        "first_block_dynamic_chunk_size_max": result.get(
            "first_block_dynamic_chunk_size_max",
            result.get("first_block_dynamic_chunk_size", 0),
        ),
        "first_block_chunk_size_min": result.get(
            "first_block_chunk_size_min",
            result.get("first_block_chunk_size", 0),
        ),
        "first_block_chunk_size_max": result.get(
            "first_block_chunk_size_max",
            result.get("first_block_chunk_size", 0),
        ),
        "gpu_first_blocks": result.get("gpu_first_blocks", scenario.gpu_first_blocks),
        "elapsed_ms": result.get("elapsed_ms", 0.0),
        "hashrate": result.get("hashrate", 0.0),
        "timings": summarize_timings(result.get("timings", {})),
        "matches": len(result.get("matches", [])),
        "ok": bool(result.get("ok")),
        "error": result.get("error", ""),
        "process_exit_code": result.get("process_exit_code", 0),
    }


def summarize_iterations(scenario: BenchmarkScenario, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ok_summaries = [item for item in summaries if item["ok"]]
    hashrates = [float(item["hashrate"]) for item in ok_summaries]
    errors = [item["error"] for item in summaries if item["error"]]
    median_hashrate = statistics.median(hashrates) if hashrates else 0.0
    min_hashrate = min(hashrates) if hashrates else 0.0
    max_hashrate = max(hashrates) if hashrates else 0.0
    spread_pct = hashrate_spread_pct(min_hashrate, max_hashrate, median_hashrate)
    timings = median_timings(ok_summaries)
    attempts = sum(int(item["attempts"]) for item in ok_summaries)
    elapsed_ms = sum(float(item["elapsed_ms"]) for item in ok_summaries)
    aggregate = {
        "name": scenario.name,
        "backend": summaries[0]["backend"] if summaries else scenario.backend,
        "device_id": summaries[0]["device_id"] if summaries else scenario.device,
        "difficulty": scenario.difficulty,
        "difficulty_sequence": list(scenario.difficulty_sequence),
        "difficulty_mode": "sequence" if scenario.difficulty_sequence else "fixed",
        "difficulty_changes": difficulty_change_count(scenario.difficulty_sequence),
        "key_mode": "fixed" if scenario.key else "generated",
        "batch_size": _median_int([item.get("batch_size", scenario.batch_size) for item in ok_summaries])
        or (summaries[0]["batch_size"] if summaries else scenario.batch_size),
        "auto_batch_size": scenario.auto_batch_size,
        "batch_size_sequence": list(scenario.batch_size_sequence),
        "batch_size_mode": "sequence" if scenario.batch_size_sequence else "fixed",
        "batch_size_changes": batch_size_change_count(scenario.batch_size_sequence),
        "batch_size_min": _median_int(
            [item.get("batch_size_min", item.get("batch_size", scenario.batch_size)) for item in ok_summaries]
        ),
        "batch_size_max": _median_int(
            [item.get("batch_size_max", item.get("batch_size", scenario.batch_size)) for item in ok_summaries]
        ),
        "attempts": attempts,
        "first_block_workers": scenario.first_block_workers,
        "first_block_dynamic_chunk_size": _median_int(
            [item.get("first_block_dynamic_chunk_size", 0) for item in ok_summaries]
        ),
        "first_block_dynamic_chunk_auto": scenario.first_block_dynamic_chunk_auto,
        "first_block_worker_count": _median_int([item.get("first_block_worker_count", 0) for item in ok_summaries]),
        "first_block_chunk_size": _median_int([item.get("first_block_chunk_size", 0) for item in ok_summaries]),
        "first_block_dynamic_chunk_size_min": _median_int(
            [item.get("first_block_dynamic_chunk_size_min", item.get("first_block_dynamic_chunk_size", 0)) for item in ok_summaries]
        ),
        "first_block_dynamic_chunk_size_max": _median_int(
            [item.get("first_block_dynamic_chunk_size_max", item.get("first_block_dynamic_chunk_size", 0)) for item in ok_summaries]
        ),
        "first_block_chunk_size_min": _median_int(
            [item.get("first_block_chunk_size_min", item.get("first_block_chunk_size", 0)) for item in ok_summaries]
        ),
        "first_block_chunk_size_max": _median_int(
            [item.get("first_block_chunk_size_max", item.get("first_block_chunk_size", 0)) for item in ok_summaries]
        ),
        "gpu_first_blocks": scenario.gpu_first_blocks,
        "elapsed_ms": elapsed_ms,
        "ms_per_attempt": elapsed_ms / attempts if attempts > 0 else 0.0,
        "hashrate": median_hashrate,
        "median_hashrate": median_hashrate,
        "min_hashrate": min_hashrate,
        "max_hashrate": max_hashrate,
        "hashrate_spread_pct": spread_pct,
        "stable": len(ok_summaries) == len(summaries) and bool(summaries) and spread_pct <= DEFAULT_STABLE_SPREAD_PCT,
        "stable_spread_pct": DEFAULT_STABLE_SPREAD_PCT,
        "timings": timings,
        "timing_per_attempt": median_timing_per_attempt(ok_summaries),
        "timing_analysis": timing_analysis(timings),
        "matches": sum(int(item["matches"]) for item in ok_summaries),
        "ok": len(ok_summaries) == len(summaries) and bool(summaries),
        "error": "; ".join(errors),
        "warmup": scenario.warmup,
        "repeat": scenario.repeat,
        "sample_count": len(summaries),
        "ok_sample_count": len(ok_summaries),
    }
    return aggregate


def build_recommendations(runs: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_scenarios: list[str] = []
    cold_scenarios: list[str] = []
    unstable_scenarios: list[str] = []
    valid_run_count = 0
    warm_evidence_run_count = 0
    stable_run_count = 0
    candidates_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for run in runs:
        summary = run.get("summary") or {}
        scenario = run.get("scenario") or {}
        scenario_name = str(summary.get("name") or scenario.get("name") or "")
        run_ok = int(run.get("exit_code", 0) or 0) == 0 and bool(summary.get("ok"))
        if run_ok:
            valid_run_count += 1
            warmup = int(summary.get("warmup", scenario.get("warmup", 0)) or 0)
            repeat = int(summary.get("repeat", scenario.get("repeat", 1)) or 1)
            if warmup >= 1 and repeat >= 2:
                warm_evidence_run_count += 1
            else:
                cold_scenarios.append(scenario_name)
            spread_pct = float(summary.get("hashrate_spread_pct", 0.0) or 0.0)
            if bool(summary.get("stable", spread_pct <= DEFAULT_STABLE_SPREAD_PCT)) and spread_pct <= DEFAULT_STABLE_SPREAD_PCT:
                stable_run_count += 1
            else:
                unstable_scenarios.append(scenario_name)
        else:
            invalid_scenarios.append(scenario_name)
        if not summary.get("ok"):
            continue
        if summary.get("difficulty_sequence") or summary.get("batch_size_sequence"):
            continue
        if summary.get("key_mode") == "fixed":
            continue
        key = (
            str(summary.get("backend", "")),
            int(summary.get("device_id", 0)),
            int(summary.get("difficulty", 0)),
        )
        candidates_by_key.setdefault(key, []).append(summary)

    selected_by_key: dict[tuple[str, int, int], tuple[dict[str, Any], str]] = {}
    for key, candidates in candidates_by_key.items():
        stable_candidates = [
            summary
            for summary in candidates
            if float(summary.get("hashrate_spread_pct", 0.0) or 0.0) <= DEFAULT_STABLE_SPREAD_PCT
        ]
        selection_pool = stable_candidates or candidates
        selection_reason = "best_stable_median" if stable_candidates else "no_stable_candidate"
        selected = max(
            selection_pool,
            key=lambda summary: float(summary.get("median_hashrate", summary.get("hashrate", 0.0)) or 0.0),
        )
        selected_by_key[key] = (selected, selection_reason)

    def recommendation_entry(summary: dict[str, Any], selection_reason: str = "") -> dict[str, Any]:
        return {
            "backend": str(summary.get("backend", "")),
            "device_id": int(summary.get("device_id", 0)),
            "difficulty": int(summary.get("difficulty", 0)),
            "batch_size": int(summary.get("batch_size", 0)),
            "batch_size_min": int(summary.get("batch_size_min", summary.get("batch_size", 0)) or 0),
            "batch_size_max": int(summary.get("batch_size_max", summary.get("batch_size", 0)) or 0),
            "first_block_workers": int(summary.get("first_block_workers", 0) or 0),
            "first_block_dynamic_chunk_size": int(summary.get("first_block_dynamic_chunk_size", 0) or 0),
            "first_block_dynamic_chunk_auto": bool(summary.get("first_block_dynamic_chunk_auto", False)),
            "first_block_worker_count": int(summary.get("first_block_worker_count", 0) or 0),
            "first_block_chunk_size": int(summary.get("first_block_chunk_size", 0) or 0),
            "first_block_dynamic_chunk_size_min": int(
                summary.get("first_block_dynamic_chunk_size_min", summary.get("first_block_dynamic_chunk_size", 0)) or 0
            ),
            "first_block_dynamic_chunk_size_max": int(
                summary.get("first_block_dynamic_chunk_size_max", summary.get("first_block_dynamic_chunk_size", 0)) or 0
            ),
            "first_block_chunk_size_min": int(
                summary.get("first_block_chunk_size_min", summary.get("first_block_chunk_size", 0)) or 0
            ),
            "first_block_chunk_size_max": int(
                summary.get("first_block_chunk_size_max", summary.get("first_block_chunk_size", 0)) or 0
            ),
            "gpu_first_blocks": bool(summary.get("gpu_first_blocks", False)),
            "median_hashrate": float(summary.get("median_hashrate", summary.get("hashrate", 0.0)) or 0.0),
            "min_hashrate": float(summary.get("min_hashrate", summary.get("hashrate", 0.0)) or 0.0),
            "max_hashrate": float(summary.get("max_hashrate", summary.get("hashrate", 0.0)) or 0.0),
            "hashrate_spread_pct": float(summary.get("hashrate_spread_pct", 0.0) or 0.0),
            "ms_per_attempt": float(summary.get("ms_per_attempt", 0.0) or 0.0),
            "stable": float(summary.get("hashrate_spread_pct", 0.0) or 0.0) <= DEFAULT_STABLE_SPREAD_PCT,
            "warm_evidence": int(summary.get("warmup", 0) or 0) >= 1 and int(summary.get("repeat", 1) or 1) >= 2,
            "selection_reason": selection_reason,
            "dominant_stage": str((summary.get("timing_analysis") or {}).get("dominant_stage", "")),
            "dominant_stage_pct": float((summary.get("timing_analysis") or {}).get("dominant_stage_pct", 0.0) or 0.0),
            "scenario": str(summary.get("name", "")),
        }

    batch_size_by_difficulty = [
        recommendation_entry(summary, selection_reason)
        for (backend, device_id, difficulty), (summary, selection_reason) in sorted(selected_by_key.items())
    ]
    candidates_by_difficulty = [
        {
            "backend": backend,
            "device_id": device_id,
            "difficulty": difficulty,
            "candidates": [
                recommendation_entry(summary)
                for summary in sorted(
                    candidates,
                    key=lambda item: int(item.get("batch_size", 0)),
                )
            ],
        }
        for (backend, device_id, difficulty), candidates in sorted(candidates_by_key.items())
    ]
    return {
        "report_ok": len(invalid_scenarios) == 0,
        "run_count": len(runs),
        "valid_run_count": valid_run_count,
        "warm_evidence_run_count": warm_evidence_run_count,
        "stable_run_count": stable_run_count,
        "invalid_run_count": len(invalid_scenarios),
        "invalid_scenarios": invalid_scenarios,
        "cold_scenarios": cold_scenarios,
        "unstable_scenarios": unstable_scenarios,
        "stable_spread_pct": DEFAULT_STABLE_SPREAD_PCT,
        "batch_size_by_difficulty": batch_size_by_difficulty,
        "candidates_by_difficulty": candidates_by_difficulty,
    }


def add_recommendation_quality(recommendations: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(recommendations)
    benchmark_trust = str(environment.get("benchmark_trust") or "unknown")
    high_cpu_load = bool(environment.get("high_cpu_load", False))
    environment_available = bool(environment.get("available", False))
    try:
        sample_count = int(environment.get("sample_count", 0) or 0)
    except (TypeError, ValueError):
        sample_count = 0

    annotated["benchmark_trust"] = benchmark_trust
    annotated["environment_available"] = environment_available
    annotated["environment_sample_count"] = sample_count
    annotated["high_cpu_load"] = high_cpu_load
    run_count = int(annotated.get("run_count", 0) or 0)
    warm_evidence_run_count = int(annotated.get("warm_evidence_run_count", run_count) or 0)
    stable_run_count = int(annotated.get("stable_run_count", run_count) or 0)
    warm_evidence_ok = run_count == 0 or warm_evidence_run_count == run_count
    stable_runs_ok = run_count == 0 or stable_run_count == run_count
    failure_reasons: list[str] = []
    if not bool(annotated.get("report_ok", True)):
        failure_reasons.append("invalid_runs")
    if benchmark_trust == "low":
        failure_reasons.append("low_benchmark_trust")
    if high_cpu_load:
        failure_reasons.append("high_cpu_load")
    if not warm_evidence_ok:
        failure_reasons.append("missing_warm_evidence")
    if not stable_runs_ok:
        failure_reasons.append("unstable_runs")
    annotated["report_quality_ok"] = (
        bool(annotated.get("report_ok", True))
        and benchmark_trust != "low"
        and not high_cpu_load
        and warm_evidence_ok
        and stable_runs_ok
    )
    annotated["report_quality_failure_reasons"] = failure_reasons
    return annotated


def build_empty_recommendations() -> dict[str, Any]:
    return {
        "report_ok": True,
        "run_count": 0,
        "valid_run_count": 0,
        "warm_evidence_run_count": 0,
        "stable_run_count": 0,
        "invalid_run_count": 0,
        "invalid_scenarios": [],
        "cold_scenarios": [],
        "unstable_scenarios": [],
        "stable_spread_pct": DEFAULT_STABLE_SPREAD_PCT,
        "batch_size_by_difficulty": [],
        "candidates_by_difficulty": [],
    }


def effective_preflight_stable_samples(wait_seconds: float, stable_samples: int) -> int:
    if stable_samples > 0:
        return stable_samples
    return 2 if wait_seconds > 0.0 else 1


def preflight_environment_quality(
    wait_seconds: float,
    wait_interval: float,
    stable_samples: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + max(0.0, wait_seconds)
    samples: list[dict[str, Any]] = []
    stable_streak: list[dict[str, Any]] = []
    required_stable_samples = max(1, stable_samples)
    while True:
        sample = collect_environment_metadata()
        samples.append(sample)
        sample_recommendations = add_recommendation_quality(build_empty_recommendations(), sample)
        if bool(sample_recommendations.get("report_quality_ok", False)):
            stable_streak.append(sample)
            if len(stable_streak) >= required_stable_samples:
                environment = combine_environment_metadata(*stable_streak[-required_stable_samples:])
                recommendations = add_recommendation_quality(build_empty_recommendations(), environment)
                recommendations["preflight_sample_count"] = len(samples)
                recommendations["preflight_stable_samples_required"] = required_stable_samples
                recommendations["preflight_stable_samples_observed"] = len(stable_streak)
                return environment, recommendations
        else:
            stable_streak = []
        if time.monotonic() >= deadline:
            environment = combine_environment_metadata(*samples)
            recommendations = add_recommendation_quality(build_empty_recommendations(), environment)
            recommendations["preflight_sample_count"] = len(samples)
            recommendations["preflight_stable_samples_required"] = required_stable_samples
            recommendations["preflight_stable_samples_observed"] = len(stable_streak)
            if len(stable_streak) < required_stable_samples:
                recommendations["report_quality_ok"] = False
            return environment, recommendations
        time.sleep(max(0.1, wait_interval))


def sanitize_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    safe_keys = (
        "name",
        "backend",
        "difficulty",
        "difficulty_sequence",
        "batch_size",
        "batch_size_sequence",
        "seconds",
        "device",
        "warmup",
        "repeat",
        "pattern",
        "key_mode",
        "detailed_timings",
        "first_block_workers",
        "first_block_dynamic_chunk_size",
        "first_block_dynamic_chunk_auto",
        "gpu_first_blocks",
    )
    sanitized = {key: scenario[key] for key in safe_keys if key in scenario}
    if "key_mode" not in sanitized:
        sanitized["key_mode"] = "fixed" if scenario.get("key") else "generated"
    prefix = str(scenario.get("prefix", ""))
    sanitized["prefix_length"] = len(prefix)
    return sanitized


def sanitize_build_metadata(build: Any) -> dict[str, Any]:
    if not isinstance(build, dict):
        return {"provided": False}

    safe: dict[str, Any] = {}
    for key in (
        "provided",
        "available",
        "error",
        "generator",
        "build_type",
        "configuration_types",
        "vcpkg_target_triplet",
        "cuda_architectures",
        "cuda_architectures_raw",
    ):
        if key in build:
            safe[key] = build[key]

    compiler = build.get("cuda_compiler")
    if isinstance(compiler, dict):
        safe_compiler = {
            key: compiler[key]
            for key in ("basename", "available", "release", "version")
            if key in compiler
        }
        if safe_compiler:
            safe["cuda_compiler"] = safe_compiler
    return safe or {"provided": False}


def build_sanitized_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "xenblocks.hashapi.benchmark-summary.v1",
        "source_schema": report.get("schema", ""),
        "created_at_unix": report.get("created_at_unix"),
        "build": sanitize_build_metadata(report.get("build")),
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
        "environment": report.get("environment", {}),
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
        "--difficulty",
        str(scenario.difficulty),
        "--seconds",
        str(scenario.seconds),
        "--device",
        str(scenario.device),
        "--json",
    ]
    if scenario.auto_batch_size:
        command.append("--auto-batch-size")
        if scenario.batch_size > 0:
            command.extend(["--batch-size", str(scenario.batch_size)])
    else:
        command.extend(["--batch-size", str(scenario.batch_size)])
    if scenario.key:
        command.extend(["--key", scenario.key])
    if scenario.difficulty_sequence:
        command.extend(["--difficulty-sequence", ",".join(str(value) for value in scenario.difficulty_sequence)])
    if scenario.batch_size_sequence:
        command.extend(["--batch-size-sequence", ",".join(str(value) for value in scenario.batch_size_sequence)])
    if scenario.prefix:
        command.extend(["--prefix", scenario.prefix])
    if not scenario.allow_xuni:
        command.append("--no-xuni")
    if scenario.detailed_timings:
        command.append("--detailed-timings")
    if scenario.first_block_workers > 0:
        command.extend(["--first-block-workers", str(scenario.first_block_workers)])
    if scenario.first_block_dynamic_chunk_size > 0:
        command.extend(["--first-block-dynamic-chunk-size", str(scenario.first_block_dynamic_chunk_size)])
    if scenario.first_block_dynamic_chunk_auto:
        command.append("--first-block-dynamic-chunk-auto")
    if scenario.gpu_first_blocks:
        command.append("--gpu-first-blocks")
    return command


def skipped_quality_result(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "benchmark report quality preflight failed",
        "preflight_skipped": True,
        "environment": environment,
    }


def run_hash_command(
    command: list[str],
    environment_samples: list[dict[str, Any]] | None = None,
    preflight_wait_seconds: float = 0.0,
    preflight_wait_interval: float = 5.0,
    preflight_stable_samples: int = 1,
) -> dict[str, Any]:
    if preflight_wait_seconds > 0.0 and environment_samples is not None:
        environment, recommendations = preflight_environment_quality(
            preflight_wait_seconds,
            preflight_wait_interval,
            preflight_stable_samples,
        )
        environment_samples.append(environment)
        if not bool(recommendations.get("report_quality_ok", False)):
            return {
                "exit_code": 2,
                "wall_elapsed_ms": 0.0,
                "result": skipped_quality_result(environment),
            }
    if environment_samples is not None:
        start_environment = sample_environment(environment_samples)
        if preflight_wait_seconds > 0.0:
            start_recommendations = add_recommendation_quality(build_empty_recommendations(), start_environment)
            if not bool(start_recommendations.get("report_quality_ok", False)):
                return {
                    "exit_code": 2,
                    "wall_elapsed_ms": 0.0,
                    "result": skipped_quality_result(start_environment),
                }
    started_at = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed_ms = (time.time() - started_at) * 1000.0
    if environment_samples is not None:
        sample_environment(environment_samples)

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "ok": False,
            "error": "hash-benchmark did not emit valid JSON",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    if completed.returncode != 0:
        result = {
            **result,
            "ok": False,
            "error": str(result.get("error") or f"process exited with code {completed.returncode}"),
            "process_exit_code": completed.returncode,
        }

    return {
        "exit_code": completed.returncode,
        "wall_elapsed_ms": elapsed_ms,
        "result": result,
    }


def run_hash_command_with_preflight_retries(
    command: list[str],
    environment_samples: list[dict[str, Any]] | None = None,
    preflight_wait_seconds: float = 0.0,
    preflight_wait_interval: float = 5.0,
    preflight_stable_samples: int = 1,
    preflight_skip_retries: int = 0,
) -> dict[str, Any]:
    skipped_retries = 0
    max_retries = max(0, preflight_skip_retries)
    while True:
        attempt_environment_samples: list[dict[str, Any]] | None = [] if environment_samples is not None else None
        run = run_hash_command(
            command,
            attempt_environment_samples,
            preflight_wait_seconds,
            preflight_wait_interval,
            preflight_stable_samples,
        )
        result = run.get("result") or {}
        preflight_skipped = bool(result.get("preflight_skipped", False))
        if preflight_skipped and skipped_retries < max_retries:
            skipped_retries += 1
            continue

        if environment_samples is not None and attempt_environment_samples is not None:
            environment_samples.extend(attempt_environment_samples)
        if skipped_retries > 0:
            run["preflight_skip_retries"] = skipped_retries
            run["result"] = {
                **result,
                "preflight_skip_retries": skipped_retries,
            }
        return run


def run_failure_errors(runs: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for run in runs:
        exit_code = int(run.get("exit_code", 0) or 0)
        result = run.get("result") or {}
        if exit_code != 0:
            errors.append(str(result.get("error") or f"process exited with code {exit_code}"))
        elif not result.get("ok"):
            errors.append(str(result.get("error") or "hash-benchmark failed"))
    return errors


def run_scenario(
    binary: Path,
    salt: str,
    scenario: BenchmarkScenario,
    preflight_wait_seconds: float = 0.0,
    preflight_wait_interval: float = 5.0,
    preflight_stable_samples: int = 1,
    preflight_skip_retries: int = 0,
) -> dict[str, Any]:
    command = build_hash_command(binary, salt, scenario)
    environment_samples: list[dict[str, Any]] = []
    warmup_runs = [
        run_hash_command_with_preflight_retries(
            command,
            environment_samples,
            preflight_wait_seconds,
            preflight_wait_interval,
            preflight_stable_samples,
            preflight_skip_retries,
        )
        for _ in range(scenario.warmup)
    ]
    iterations = [
        run_hash_command_with_preflight_retries(
            command,
            environment_samples,
            preflight_wait_seconds,
            preflight_wait_interval,
            preflight_stable_samples,
            preflight_skip_retries,
        )
        for _ in range(scenario.repeat)
    ]
    iteration_summaries = [summarize_result(scenario, item["result"]) for item in iterations]
    aggregate = summarize_iterations(scenario, iteration_summaries)
    selected_index = 0
    ok_indices = [index for index, summary in enumerate(iteration_summaries) if summary["ok"]]
    if ok_indices:
        median_hashrate = float(aggregate.get("median_hashrate", aggregate.get("hashrate", 0.0)) or 0.0)
        selected_index = min(
            ok_indices,
            key=lambda index: abs(float(iteration_summaries[index]["hashrate"]) - median_hashrate),
        )
    selected_result = iterations[selected_index]["result"] if iterations else {}
    all_runs = warmup_runs + iterations
    ok = bool(all_runs) and all(item["exit_code"] == 0 and item["result"].get("ok") for item in all_runs)
    if not ok:
        aggregate["ok"] = False
        errors = [str(aggregate.get("error") or ""), *run_failure_errors(all_runs)]
        aggregate["error"] = "; ".join(dict.fromkeys(error for error in errors if error))
        aggregate["process_exit_codes"] = [
            int(item["exit_code"])
            for item in all_runs
            if int(item.get("exit_code", 0) or 0) != 0
        ]

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
        "environment": combine_environment_metadata(*environment_samples),
        "environment_samples": environment_samples,
    }


def report_environment_metadata(runs: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        sample
        for run in runs
        for sample in run.get("environment_samples", [])
        if isinstance(sample, dict)
    ]
    if samples:
        return combine_environment_metadata(*samples)

    run_environments = [
        environment
        for run in runs
        for environment in [run.get("environment")]
        if isinstance(environment, dict) and environment
    ]
    if run_environments:
        return combine_environment_metadata(*run_environments)

    return collect_environment_metadata()


def build_report(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
    environment: dict[str, Any],
    recommendations: dict[str, Any],
    include_hardware: bool = True,
) -> dict[str, Any]:
    return {
        "schema": "xenblocks.hashapi.benchmark.v1",
        "created_at_unix": time.time(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "build": collect_build_metadata(args.build_cache),
        "hardware": collect_hardware_metadata() if include_hardware else {},
        "environment": environment,
        "binary": str(args.binary),
        "salt": args.salt,
        "presets": args.preset,
        "recommendations": recommendations,
        "runs": runs,
    }


def emit_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Path to xenblocksMiner or hashapi-cli.")
    parser.add_argument(
        "--build-cache",
        type=Path,
        help="Optional CMake build directory or CMakeCache.txt used to record public-safe build metadata.",
    )
    parser.add_argument("--salt", default=DEFAULT_SALT, help="Hex salt used by all benchmark scenarios.")
    parser.add_argument("--backend", default="cpu", help="Default backend for built-in scenarios.")
    parser.add_argument("--device", default=0, type=int, help="Default device id for built-in scenarios.")
    parser.add_argument("--seconds", default=5, type=int, help="Seconds per built-in scenario.")
    parser.add_argument("--warmup", default=0, type=int, help="Warm-up runs per scenario before measured repeats.")
    parser.add_argument("--repeat", default=1, type=int, help="Measured repeats per scenario.")
    parser.add_argument("--no-xuni", action="store_true", help="Disable secondary XUNI matching in generated scenarios.")
    parser.add_argument(
        "--gpu-first-blocks",
        action="store_true",
        help="Enable explicit CUDA device-side first-block preparation for generated scenarios.",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the aggregate JSON report.")
    parser.add_argument(
        "--sanitized-output",
        type=Path,
        help="Optional path to write a public-safe summary without local paths, hardware details, commands, raw results, salts, or prefixes.",
    )
    parser.add_argument("--recommendations-only", action="store_true", help="Print only report recommendations as JSON.")
    parser.add_argument(
        "--fail-on-report-quality",
        action="store_true",
        help="Return a non-zero exit code when report quality is low, for example under high CPU load.",
    )
    parser.add_argument(
        "--preflight-report-quality",
        action="store_true",
        help="Check report quality before running scenarios and skip benchmarks when the environment is already low-trust.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only run benchmark report quality preflight, emit an empty report, and never launch benchmark subprocesses.",
    )
    parser.add_argument(
        "--preflight-wait-seconds",
        type=float,
        default=0.0,
        help="When preflight quality is enabled, wait up to this many seconds for a normal-trust environment before skipping.",
    )
    parser.add_argument(
        "--preflight-wait-interval",
        type=float,
        default=5.0,
        help="Seconds between preflight quality samples while waiting.",
    )
    parser.add_argument(
        "--subprocess-preflight-wait-seconds",
        type=float,
        default=None,
        help=(
            "When preflight quality is enabled, wait up to this many seconds before each benchmark subprocess. "
            "Defaults to --preflight-wait-seconds."
        ),
    )
    parser.add_argument(
        "--preflight-skip-retries",
        type=int,
        default=0,
        help=(
            "Retry each benchmark subprocess this many times when it is skipped by the report-quality "
            "preflight gate before launch. Retries still require a normal-trust preflight sample before "
            "any benchmark subprocess can run."
        ),
    )
    parser.add_argument(
        "--preflight-stable-samples",
        type=int,
        default=0,
        help=(
            "Consecutive normal-trust samples required before preflight allows a benchmark. "
            "Defaults to 2 when --preflight-wait-seconds is positive, otherwise 1."
        ),
    )
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
        "--scan-first-block-workers",
        action="append",
        type=int,
        default=[],
        help="Add a CUDA first-block worker cap for generated scan scenarios. Use 0 for automatic worker count.",
    )
    parser.add_argument(
        "--scan-first-block-dynamic-chunk-size",
        action="append",
        type=int,
        default=[],
        help="Add a CUDA first-block dynamic chunk size for generated scan scenarios. Use 0 for static chunking.",
    )
    parser.add_argument(
        "--scan-first-block-dynamic-chunk-auto",
        action="store_true",
        help="Enable backend-selected first-block dynamic chunks for generated scan scenarios.",
    )
    parser.add_argument(
        "--scan-gpu-first-blocks",
        action="store_true",
        help="Add both default and explicit GPU first-block variants to generated scan scenarios.",
    )
    parser.add_argument(
        "--scan-detailed-timings",
        action="store_true",
        help="Enable detailed timing diagnostics on generated scan scenarios.",
    )
    parser.add_argument(
        "--difficulty-sequence",
        action="append",
        default=[],
        help="Add a comma-separated difficulty sequence for generated variable-difficulty scenarios. Requires --sequence-batch-size or --batch-size-sequence.",
    )
    parser.add_argument(
        "--sequence-batch-size",
        action="append",
        type=int,
        default=[],
        help="Add a batch size for generated variable-difficulty scenarios. Requires --difficulty-sequence.",
    )
    parser.add_argument(
        "--sequence-auto-batch-size",
        action="store_true",
        help="Use the Hash API CUDA automatic batch-size selector for generated variable-difficulty scenarios. Requires --difficulty-sequence.",
    )
    parser.add_argument(
        "--batch-size-sequence",
        action="append",
        default=[],
        help="Add a comma-separated batch-size sequence for generated variable-shape scenarios. Requires --difficulty-sequence.",
    )
    parser.add_argument(
        "--sequence-detailed-timings",
        action="store_true",
        help="Enable detailed timing diagnostics on generated variable-difficulty scenarios.",
    )
    parser.add_argument(
        "--sequence-first-block-dynamic-chunk-auto",
        action="store_true",
        help="Enable backend-selected first-block dynamic chunks on generated variable-difficulty scenarios.",
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
        if args.gpu_first_blocks and args.scan_gpu_first_blocks:
            raise ValueError("--gpu-first-blocks and --scan-gpu-first-blocks cannot be used together")
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
                    args.scan_first_block_workers,
                    args.scan_first_block_dynamic_chunk_size,
                    args.scan_detailed_timings,
                    args.seconds,
                    args.backend,
                    args.device,
                    args.warmup,
                    args.repeat,
                    args.scan_first_block_dynamic_chunk_auto,
                    args.scan_gpu_first_blocks,
                )
            )
        if args.difficulty_sequence or args.sequence_batch_size or args.sequence_auto_batch_size or args.batch_size_sequence:
            if args.sequence_batch_size and not args.difficulty_sequence:
                raise ValueError("--sequence-batch-size requires --difficulty-sequence")
            if args.sequence_auto_batch_size and not args.difficulty_sequence:
                raise ValueError("--sequence-auto-batch-size requires --difficulty-sequence")
            if args.batch_size_sequence and not args.difficulty_sequence:
                raise ValueError("--batch-size-sequence requires --difficulty-sequence")
            if (
                args.difficulty_sequence
                and not args.sequence_batch_size
                and not args.sequence_auto_batch_size
                and not args.batch_size_sequence
            ):
                raise ValueError(
                    "--difficulty-sequence requires --sequence-batch-size, --sequence-auto-batch-size, or --batch-size-sequence"
                )
            difficulty_sequences = [parse_difficulty_sequence(item) for item in args.difficulty_sequence]
            if args.sequence_auto_batch_size:
                scenarios.extend(
                    automatic_batch_difficulty_sequence_scenarios(
                        difficulty_sequences,
                        args.sequence_detailed_timings,
                        args.sequence_first_block_dynamic_chunk_auto,
                        args.seconds,
                        args.backend,
                        args.device,
                        args.warmup,
                        args.repeat,
                    )
                )
            if args.sequence_batch_size:
                scenarios.extend(
                    difficulty_sequence_scenarios(
                        difficulty_sequences,
                        args.sequence_batch_size,
                        args.sequence_detailed_timings,
                        args.sequence_first_block_dynamic_chunk_auto,
                        args.seconds,
                        args.backend,
                        args.device,
                        args.warmup,
                        args.repeat,
                    )
                )
            if args.batch_size_sequence:
                scenarios.extend(
                    paired_sequence_scenarios(
                        difficulty_sequences,
                        [parse_batch_size_sequence(item) for item in args.batch_size_sequence],
                        args.sequence_detailed_timings,
                        args.sequence_first_block_dynamic_chunk_auto,
                        args.seconds,
                        args.backend,
                        args.device,
                        args.warmup,
                        args.repeat,
                    )
                )
        if not scenarios:
            scenarios = default_scenarios(args.seconds, args.backend, args.device, args.warmup, args.repeat)
        if args.gpu_first_blocks:
            scenarios = enable_gpu_first_blocks(scenarios)
        if args.no_xuni:
            scenarios = [BenchmarkScenario(**{**asdict(scenario), "allow_xuni": False}) for scenario in scenarios]
        ensure_unique_scenario_names(scenarios)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    preflight_stable_samples = effective_preflight_stable_samples(
        args.preflight_wait_seconds,
        args.preflight_stable_samples,
    )
    subprocess_preflight_wait_seconds = (
        args.preflight_wait_seconds
        if args.subprocess_preflight_wait_seconds is None
        else max(0.0, args.subprocess_preflight_wait_seconds)
    )
    if args.preflight_only:
        environment, recommendations = preflight_environment_quality(
            args.preflight_wait_seconds,
            args.preflight_wait_interval,
            preflight_stable_samples,
        )
        report = build_report(args, [], environment, recommendations, include_hardware=False)
        emit_report(args, report)
        if not bool(recommendations.get("report_quality_ok", False)):
            print("benchmark report quality preflight failed", file=sys.stderr)
            return 2
        return 0

    if args.preflight_report_quality:
        environment, recommendations = preflight_environment_quality(
            args.preflight_wait_seconds,
            args.preflight_wait_interval,
            preflight_stable_samples,
        )
        if not bool(recommendations.get("report_quality_ok", False)):
            report = build_report(args, [], environment, recommendations)
            emit_report(args, report)
            print("benchmark report quality preflight failed", file=sys.stderr)
            return 2

    if args.preflight_report_quality and args.preflight_wait_seconds > 0.0:
        runs = [
            run_scenario(
                args.binary,
                args.salt,
                scenario,
                subprocess_preflight_wait_seconds,
                args.preflight_wait_interval,
                preflight_stable_samples,
                max(0, args.preflight_skip_retries),
            )
            for scenario in scenarios
        ]
    else:
        runs = [run_scenario(args.binary, args.salt, scenario) for scenario in scenarios]
    environment = report_environment_metadata(runs)
    recommendations = add_recommendation_quality(build_recommendations(runs), environment)
    report = build_report(args, runs, environment, recommendations)
    emit_report(args, report)
    if args.fail_on_report_quality and not bool(recommendations.get("report_quality_ok", False)):
        print("benchmark report quality check failed", file=sys.stderr)
        return 2
    return 0 if all(run["exit_code"] == 0 and run["result"].get("ok") for run in report["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
