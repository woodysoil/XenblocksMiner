"""Compare two Hash API benchmark reports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any


Report = dict[str, Any]
RunKey = str | tuple[Any, ...]
RunMap = dict[RunKey, dict[str, Any]]
DEFAULT_STABLE_SPREAD_PCT = 10.0


def _numeric_analysis_metrics(analysis: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in analysis.items():
        if isinstance(value, bool) or isinstance(value, (dict, list, str)):
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def load_report(path: Path) -> Report:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_for(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("summary") or run.get("aggregate") or {}


def _scenario_for(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("scenario") or {}


def _float_value(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(data: dict[str, Any], key: str, default: int = 0) -> int:
    value = data.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_value(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _run_name(run: dict[str, Any]) -> str:
    summary = _summary_for(run)
    scenario = _scenario_for(run)
    return str(summary.get("name") or scenario.get("name") or "")


def _run_has_warm_evidence(run: dict[str, Any]) -> bool:
    summary = _summary_for(run)
    scenario = _scenario_for(run)
    warmup = _int_value(summary, "warmup", _int_value(scenario, "warmup", 0))
    repeat = _int_value(summary, "repeat", _int_value(scenario, "repeat", 1))
    return warmup >= 1 and repeat >= 2


def _derive_warm_evidence(report: Report) -> tuple[int, list[str]]:
    runs = report.get("runs") or []
    if not isinstance(runs, list):
        return 0, []
    warm_count = 0
    cold_scenarios: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if _run_has_warm_evidence(run):
            warm_count += 1
        else:
            cold_scenarios.append(_run_name(run))
    return warm_count, cold_scenarios


def _run_has_stable_evidence(run: dict[str, Any], max_spread_pct: float = DEFAULT_STABLE_SPREAD_PCT) -> bool:
    summary = _summary_for(run)
    run_ok = _bool_value(summary, "ok", False) and _float_value(
        summary,
        "median_hashrate",
        _float_value(summary, "hashrate", 0.0),
    ) > 0.0
    spread_pct = _float_value(summary, "hashrate_spread_pct", 0.0)
    stable = _bool_value(summary, "stable", spread_pct <= max_spread_pct)
    return run_ok and stable and spread_pct <= max_spread_pct


def _derive_stable_evidence(report: Report, max_spread_pct: float = DEFAULT_STABLE_SPREAD_PCT) -> tuple[int, list[str]]:
    runs = report.get("runs") or []
    if not isinstance(runs, list):
        return 0, []
    stable_count = 0
    unstable_scenarios: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if _run_has_stable_evidence(run, max_spread_pct):
            stable_count += 1
        else:
            unstable_scenarios.append(_run_name(run))
    return stable_count, unstable_scenarios


def report_quality(report: Report, label: str) -> dict[str, Any]:
    recommendations = report.get("recommendations") or {}
    environment = report.get("environment") or {}
    report_ok_value = recommendations.get("report_ok")
    report_quality_ok_value = recommendations.get("report_quality_ok")
    report_ok_available = isinstance(report_ok_value, bool)
    report_quality_ok_available = isinstance(report_quality_ok_value, bool)
    report_ok = bool(report_ok_value) if report_ok_available else True
    report_quality_ok = bool(report_quality_ok_value) if report_quality_ok_available else True
    run_count = _int_value(recommendations, "run_count", len(report.get("runs") or []))
    invalid_run_count = _int_value(recommendations, "invalid_run_count", 0)
    derived_warm_count, derived_cold_scenarios = _derive_warm_evidence(report)
    warm_evidence_run_count = _int_value(recommendations, "warm_evidence_run_count", derived_warm_count)
    derived_stable_count, derived_unstable_scenarios = _derive_stable_evidence(report)
    stable_run_count = _int_value(recommendations, "stable_run_count", derived_stable_count)
    cold_scenarios = recommendations.get("cold_scenarios")
    if not isinstance(cold_scenarios, list):
        cold_scenarios = derived_cold_scenarios
    if not cold_scenarios and derived_cold_scenarios:
        cold_scenarios = derived_cold_scenarios
    unstable_scenarios = recommendations.get("unstable_scenarios")
    if not isinstance(unstable_scenarios, list):
        unstable_scenarios = derived_unstable_scenarios
    if not unstable_scenarios and derived_unstable_scenarios:
        unstable_scenarios = derived_unstable_scenarios
    benchmark_trust = str(environment.get("benchmark_trust") or "unknown")
    high_cpu_load = _bool_value(environment, "high_cpu_load", False)
    environment_available = _bool_value(environment, "available", False)

    reasons: list[str] = []
    if report_ok_available and not report_ok:
        reasons.append("report_ok=false")
    if report_quality_ok_available and not report_quality_ok:
        reasons.append("report_quality_ok=false")
    if invalid_run_count > 0:
        reasons.append("invalid_run_count>0")
    if run_count > 0 and warm_evidence_run_count < run_count:
        reasons.append("warm_evidence_incomplete")
    if run_count > 0 and stable_run_count < run_count:
        reasons.append("stable_evidence_incomplete")
    if benchmark_trust == "low" or high_cpu_load:
        reasons.append("benchmark_trust=low")

    return {
        "label": label,
        "acceptable": not reasons,
        "report_ok": report_ok,
        "report_ok_available": report_ok_available,
        "report_quality_ok": report_quality_ok,
        "report_quality_ok_available": report_quality_ok_available,
        "run_count": run_count,
        "warm_evidence_run_count": warm_evidence_run_count,
        "stable_run_count": stable_run_count,
        "invalid_run_count": invalid_run_count,
        "cold_scenarios": cold_scenarios,
        "unstable_scenarios": unstable_scenarios,
        "benchmark_trust": benchmark_trust,
        "environment_available": environment_available,
        "high_cpu_load": high_cpu_load,
        "sample_count": _int_value(environment, "sample_count", 0),
        "reasons": reasons,
    }


def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = _summary_for(run)
    scenario = _scenario_for(run)
    timing_analysis = summary.get("timing_analysis") or {}
    name = str(summary.get("name") or scenario.get("name") or "")
    if not name:
        raise ValueError("benchmark run is missing a scenario name")

    return {
        "name": name,
        "backend": str(summary.get("backend") or scenario.get("backend") or ""),
        "device_id": _int_value(summary, "device_id", _int_value(scenario, "device", 0)),
        "difficulty": _int_value(summary, "difficulty", _int_value(scenario, "difficulty", 0)),
        "batch_size": _int_value(summary, "batch_size", _int_value(scenario, "batch_size", 0)),
        "seconds": _int_value(scenario, "seconds", 0),
        "warmup": _int_value(summary, "warmup", _int_value(scenario, "warmup", 0)),
        "repeat": _int_value(summary, "repeat", _int_value(scenario, "repeat", 1)),
        "allow_xuni": _bool_value(summary, "allow_xuni", _bool_value(scenario, "allow_xuni", True)),
        "detailed_timings": _bool_value(
            summary,
            "detailed_timings",
            _bool_value(scenario, "detailed_timings", False),
        ),
        "first_block_workers": _int_value(scenario, "first_block_workers", _int_value(summary, "first_block_workers", 0)),
        "first_block_dynamic_chunk_size": _int_value(
            scenario,
            "first_block_dynamic_chunk_size",
            _int_value(summary, "first_block_dynamic_chunk_size", 0),
        ),
        "first_block_dynamic_chunk_auto": _bool_value(
            scenario,
            "first_block_dynamic_chunk_auto",
            _bool_value(summary, "first_block_dynamic_chunk_auto", False),
        ),
        "first_block_worker_count": _int_value(summary, "first_block_worker_count", 0),
        "first_block_chunk_size": _int_value(summary, "first_block_chunk_size", 0),
        "first_block_dynamic_chunk_size_min": _int_value(
            summary,
            "first_block_dynamic_chunk_size_min",
            _int_value(summary, "first_block_dynamic_chunk_size", 0),
        ),
        "first_block_dynamic_chunk_size_max": _int_value(
            summary,
            "first_block_dynamic_chunk_size_max",
            _int_value(summary, "first_block_dynamic_chunk_size", 0),
        ),
        "first_block_chunk_size_min": _int_value(
            summary,
            "first_block_chunk_size_min",
            _int_value(summary, "first_block_chunk_size", 0),
        ),
        "first_block_chunk_size_max": _int_value(
            summary,
            "first_block_chunk_size_max",
            _int_value(summary, "first_block_chunk_size", 0),
        ),
        "gpu_first_blocks": _bool_value(summary, "gpu_first_blocks", _bool_value(scenario, "gpu_first_blocks", False)),
        "attempts": _int_value(summary, "attempts", 0),
        "elapsed_ms": _float_value(summary, "elapsed_ms", 0.0),
        "hashrate": _float_value(summary, "hashrate", 0.0),
        "median_hashrate": _float_value(summary, "median_hashrate", _float_value(summary, "hashrate", 0.0)),
        "min_hashrate": _float_value(summary, "min_hashrate", _float_value(summary, "hashrate", 0.0)),
        "max_hashrate": _float_value(summary, "max_hashrate", _float_value(summary, "hashrate", 0.0)),
        "hashrate_spread_pct": _float_value(summary, "hashrate_spread_pct", 0.0),
        "difficulty_mode": str(summary.get("difficulty_mode") or scenario.get("difficulty_mode") or "fixed"),
        "difficulty_sequence": summary.get("difficulty_sequence") or scenario.get("difficulty_sequence") or [],
        "difficulty_changes": _int_value(summary, "difficulty_changes", _int_value(scenario, "difficulty_changes", 0)),
        "key_mode": str(summary.get("key_mode") or scenario.get("key_mode") or "generated"),
        "batch_size_sequence": summary.get("batch_size_sequence") or scenario.get("batch_size_sequence") or [],
        "batch_size_mode": str(summary.get("batch_size_mode") or scenario.get("batch_size_mode") or "fixed"),
        "batch_size_changes": _int_value(summary, "batch_size_changes", _int_value(scenario, "batch_size_changes", 0)),
        "batch_size_min": _int_value(summary, "batch_size_min", _int_value(summary, "batch_size", 0)),
        "batch_size_max": _int_value(summary, "batch_size_max", _int_value(summary, "batch_size", 0)),
        "timings": summary.get("timings", {}),
        "timing_per_attempt": summary.get("timing_per_attempt", {}),
        "stage_pct": timing_analysis.get("stage_pct", {}),
        "nested_stage_pct": timing_analysis.get("nested_stage_pct", {}),
        "analysis_metrics": _numeric_analysis_metrics(timing_analysis),
        "matches": _int_value(summary, "matches", 0),
        "ok": bool(summary.get("ok")),
        "error": str(summary.get("error") or ""),
    }


def run_config_key(run: dict[str, Any], ignore_detailed_timings: bool = False) -> tuple[Any, ...]:
    return (
        run["backend"],
        run["device_id"],
        run["difficulty"],
        run["difficulty_mode"],
        tuple(run["difficulty_sequence"]),
        run["difficulty_changes"],
        run["key_mode"],
        run["batch_size"],
        run["batch_size_mode"],
        tuple(run["batch_size_sequence"]),
        run["batch_size_changes"],
        run["seconds"],
        run["warmup"],
        run["repeat"],
        run["allow_xuni"],
        False if ignore_detailed_timings else run["detailed_timings"],
        run["first_block_workers"],
        run["first_block_dynamic_chunk_size"],
        run["first_block_dynamic_chunk_auto"],
        run["gpu_first_blocks"],
    )


def run_key(run: dict[str, Any], match_by: str, ignore_detailed_timings: bool = False) -> RunKey:
    if match_by == "name":
        return run["name"]
    if match_by == "config":
        return run_config_key(run, ignore_detailed_timings=ignore_detailed_timings)
    raise ValueError(f"unsupported match mode: {match_by}")


def format_run_key(key: RunKey) -> str:
    if isinstance(key, str):
        return key
    (
        backend,
        device_id,
        difficulty,
        difficulty_mode,
        difficulty_sequence,
        difficulty_changes,
        key_mode,
        batch_size,
        batch_size_mode,
        batch_size_sequence,
        batch_size_changes,
        seconds,
        warmup,
        repeat,
        allow_xuni,
        detailed_timings,
        first_block_workers,
        first_block_dynamic_chunk_size,
        first_block_dynamic_chunk_auto,
        gpu_first_blocks,
    ) = key
    if difficulty_mode == "sequence":
        difficulty_label = "x".join(str(item) for item in difficulty_sequence)
    else:
        difficulty_label = str(difficulty)
    if batch_size_mode == "sequence":
        batch_size_label = "x".join(str(item) for item in batch_size_sequence)
    else:
        batch_size_label = str(batch_size)
    xuni_label = "xuni" if allow_xuni else "no-xuni"
    detail_label = "detailed" if detailed_timings else "default-timing"
    return (
        f"{backend}:dev{device_id}:d{difficulty_label}:b{batch_size_label}:"
        f"{key_mode}:dchanges{difficulty_changes}:bchanges{batch_size_changes}:s{seconds}:w{warmup}:r{repeat}:"
        f"{xuni_label}:{detail_label}:fbw{first_block_workers}:fbd{first_block_dynamic_chunk_size}:"
        f"fbda{int(first_block_dynamic_chunk_auto)}:gfb{int(gpu_first_blocks)}"
    )


def display_name(key: RunKey, before: dict[str, Any] | None, after: dict[str, Any] | None, match_by: str) -> str:
    if match_by == "name":
        return str(key)
    before_name = str((before or {}).get("name") or "")
    after_name = str((after or {}).get("name") or "")
    if before_name and after_name and before_name != after_name:
        return f"{before_name} -> {after_name}"
    return before_name or after_name or format_run_key(key)


def index_runs(report: Report, match_by: str = "name", ignore_detailed_timings: bool = False) -> RunMap:
    indexed: RunMap = {}
    for run in report.get("runs", []):
        normalized = normalize_run(run)
        key = run_key(normalized, match_by, ignore_detailed_timings=ignore_detailed_timings)
        if key in indexed:
            label = "scenario name" if match_by == "name" else "scenario config"
            raise ValueError(f"duplicate {label} in report: {format_run_key(key)}")
        indexed[key] = normalized
    return indexed


def _percent_change(before: float, after: float) -> float | None:
    if before <= 0.0:
        return None
    return ((after - before) / before) * 100.0


def compare_timings(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, dict[str, float | None]]:
    before_timings = (before or {}).get("timings") or {}
    after_timings = (after or {}).get("timings") or {}
    comparison: dict[str, dict[str, float | None]] = {}

    for key in sorted(set(before_timings) | set(after_timings)):
        before_value = _float_value(before_timings, key, 0.0)
        after_value = _float_value(after_timings, key, 0.0)
        comparison[key] = {
            "before_ms": before_value,
            "after_ms": after_value,
            "delta_ms": after_value - before_value,
            "change_pct": _percent_change(before_value, after_value),
        }

    return comparison


def compare_timing_per_attempt(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    before_timings = (before or {}).get("timing_per_attempt") or {}
    after_timings = (after or {}).get("timing_per_attempt") or {}
    comparison: dict[str, dict[str, float | None]] = {}

    for key in sorted(set(before_timings) | set(after_timings)):
        before_value = _float_value(before_timings, key, 0.0)
        after_value = _float_value(after_timings, key, 0.0)
        comparison[key] = {
            "before_ms_per_attempt": before_value,
            "after_ms_per_attempt": after_value,
            "delta_ms_per_attempt": after_value - before_value,
            "change_pct": _percent_change(before_value, after_value),
        }

    return comparison


def compare_stage_pct(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    before_percentages = (before or {}).get("stage_pct") or {}
    after_percentages = (after or {}).get("stage_pct") or {}
    comparison: dict[str, dict[str, float | None]] = {}

    for key in sorted(set(before_percentages) | set(after_percentages)):
        before_value = _float_value(before_percentages, key, 0.0)
        after_value = _float_value(after_percentages, key, 0.0)
        comparison[key] = {
            "before_pct": before_value,
            "after_pct": after_value,
            "delta_pct_points": after_value - before_value,
            "change_pct": _percent_change(before_value, after_value),
        }

    return comparison


def compare_nested_stage_pct(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    before_percentages = (before or {}).get("nested_stage_pct") or {}
    after_percentages = (after or {}).get("nested_stage_pct") or {}
    comparison: dict[str, dict[str, float | None]] = {}

    for key in sorted(set(before_percentages) | set(after_percentages)):
        before_value = _float_value(before_percentages, key, 0.0)
        after_value = _float_value(after_percentages, key, 0.0)
        comparison[key] = {
            "before_pct": before_value,
            "after_pct": after_value,
            "delta_pct_points": after_value - before_value,
            "change_pct": _percent_change(before_value, after_value),
        }

    return comparison


def compare_analysis_metrics(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    before_metrics = (before or {}).get("analysis_metrics") or {}
    after_metrics = (after or {}).get("analysis_metrics") or {}
    comparison: dict[str, dict[str, float | None]] = {}

    for key in sorted(set(before_metrics) | set(after_metrics)):
        before_value = _float_value(before_metrics, key, 0.0)
        after_value = _float_value(after_metrics, key, 0.0)
        comparison[key] = {
            "before_value": before_value,
            "after_value": after_value,
            "delta_value": after_value - before_value,
            "change_pct": _percent_change(before_value, after_value),
        }

    return comparison


def _status(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    change_pct: float | None,
    min_change_pct: float,
    max_spread_pct: float,
) -> str:
    if before is None:
        return "missing-before"
    if after is None:
        return "missing-after"
    if not before["ok"] or not after["ok"]:
        return "invalid"
    if change_pct is None:
        return "unrated"
    noisy = (
        _float_value(before, "hashrate_spread_pct", 0.0) > max_spread_pct or
        _float_value(after, "hashrate_spread_pct", 0.0) > max_spread_pct
    )
    if change_pct > min_change_pct:
        return "noisy-improved" if noisy else "improved"
    if change_pct < -min_change_pct:
        return "noisy-regressed" if noisy else "regressed"
    return "noisy-unchanged" if noisy else "unchanged"


def compare_reports(
    before_report: Report,
    after_report: Report,
    min_change_pct: float = 0.0,
    max_spread_pct: float = 10.0,
    match_by: str = "name",
    ignore_detailed_timings: bool = False,
    min_difficulty: int = 0,
) -> Report:
    before_quality = report_quality(before_report, "before")
    after_quality = report_quality(after_report, "after")
    before_runs = index_runs(
        before_report,
        match_by=match_by,
        ignore_detailed_timings=ignore_detailed_timings,
    )
    after_runs = index_runs(
        after_report,
        match_by=match_by,
        ignore_detailed_timings=ignore_detailed_timings,
    )
    comparisons: list[dict[str, Any]] = []

    for key in sorted(set(before_runs) | set(after_runs), key=format_run_key):
        before = before_runs.get(key)
        after = after_runs.get(key)
        run_shape = after or before or {}
        if min_difficulty > 0:
            difficulty_values = list(run_shape.get("difficulty_sequence") or [])
            if not difficulty_values:
                difficulty_values = [_int_value(run_shape, "difficulty", 0)]
            if min(difficulty_values or [0]) < min_difficulty:
                continue
        before_rate = before["median_hashrate"] if before else 0.0
        after_rate = after["median_hashrate"] if after else 0.0
        change_pct = _percent_change(before_rate, after_rate) if before and after else None
        status = _status(before, after, change_pct, min_change_pct, max_spread_pct)
        comparisons.append(
            {
                "name": display_name(key, before, after, match_by),
                "match_key": format_run_key(key),
                "before_name": (before or {}).get("name", ""),
                "after_name": (after or {}).get("name", ""),
                "status": status,
                "before_hashrate": before_rate,
                "after_hashrate": after_rate,
                "delta_hashrate": after_rate - before_rate,
                "change_pct": change_pct,
                "max_spread_pct": max_spread_pct,
                "before_spread_pct": _float_value(before or {}, "hashrate_spread_pct", 0.0),
                "after_spread_pct": _float_value(after or {}, "hashrate_spread_pct", 0.0),
                "backend": (after or before or {}).get("backend", ""),
                "device_id": (after or before or {}).get("device_id", 0),
                "difficulty": (after or before or {}).get("difficulty", 0),
                "difficulty_mode": (after or before or {}).get("difficulty_mode", "fixed"),
                "difficulty_sequence": (after or before or {}).get("difficulty_sequence", []),
                "difficulty_changes": (after or before or {}).get("difficulty_changes", 0),
                "key_mode": (after or before or {}).get("key_mode", "generated"),
                "batch_size": (after or before or {}).get("batch_size", 0),
                "batch_size_mode": (after or before or {}).get("batch_size_mode", "fixed"),
                "batch_size_sequence": (after or before or {}).get("batch_size_sequence", []),
                "batch_size_changes": (after or before or {}).get("batch_size_changes", 0),
                "batch_size_min": (after or before or {}).get(
                    "batch_size_min",
                    (after or before or {}).get("batch_size", 0),
                ),
                "batch_size_max": (after or before or {}).get(
                    "batch_size_max",
                    (after or before or {}).get("batch_size", 0),
                ),
                "seconds": (after or before or {}).get("seconds", 0),
                "warmup": (after or before or {}).get("warmup", 0),
                "repeat": (after or before or {}).get("repeat", 1),
                "first_block_workers": (after or before or {}).get("first_block_workers", 0),
                "first_block_dynamic_chunk_size": (after or before or {}).get("first_block_dynamic_chunk_size", 0),
                "first_block_dynamic_chunk_auto": (after or before or {}).get("first_block_dynamic_chunk_auto", False),
                "first_block_worker_count": (after or before or {}).get("first_block_worker_count", 0),
                "first_block_chunk_size": (after or before or {}).get("first_block_chunk_size", 0),
                "first_block_dynamic_chunk_size_min": (after or before or {}).get(
                    "first_block_dynamic_chunk_size_min",
                    (after or before or {}).get("first_block_dynamic_chunk_size", 0),
                ),
                "first_block_dynamic_chunk_size_max": (after or before or {}).get(
                    "first_block_dynamic_chunk_size_max",
                    (after or before or {}).get("first_block_dynamic_chunk_size", 0),
                ),
                "first_block_chunk_size_min": (after or before or {}).get(
                    "first_block_chunk_size_min",
                    (after or before or {}).get("first_block_chunk_size", 0),
                ),
                "first_block_chunk_size_max": (after or before or {}).get(
                    "first_block_chunk_size_max",
                    (after or before or {}).get("first_block_chunk_size", 0),
                ),
                "gpu_first_blocks": (after or before or {}).get("gpu_first_blocks", False),
                "timing_deltas": compare_timings(before, after),
                "timing_per_attempt_deltas": compare_timing_per_attempt(before, after),
                "stage_pct_deltas": compare_stage_pct(before, after),
                "nested_stage_pct_deltas": compare_nested_stage_pct(before, after),
                "analysis_metric_deltas": compare_analysis_metrics(before, after),
                "before": before,
                "after": after,
            }
        )

    statuses = [item["status"] for item in comparisons]
    return {
        "schema": "xenblocks.hashapi.compare.v1",
        "min_change_pct": min_change_pct,
        "max_spread_pct": max_spread_pct,
        "match_by": match_by,
        "ignore_detailed_timings": ignore_detailed_timings,
        "min_difficulty": min_difficulty,
        "quality": {
            "ok": bool(before_quality["acceptable"] and after_quality["acceptable"]),
            "before": before_quality,
            "after": after_quality,
        },
        "summary": {
            "total": len(comparisons),
            "improved": statuses.count("improved"),
            "regressed": statuses.count("regressed"),
            "noisy_improved": statuses.count("noisy-improved"),
            "noisy_regressed": statuses.count("noisy-regressed"),
            "noisy_unchanged": statuses.count("noisy-unchanged"),
            "unchanged": statuses.count("unchanged"),
            "invalid": statuses.count("invalid"),
            "missing_before": statuses.count("missing-before"),
            "missing_after": statuses.count("missing-after"),
        },
        "comparisons": comparisons,
    }


def format_text(report: Report) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "scenario",
            "status",
            "before_hashrate",
            "after_hashrate",
            "delta_hashrate",
            "change_pct",
            "backend",
            "difficulty",
            "difficulty_mode",
            "difficulty_changes",
            "batch_size",
            "batch_size_mode",
            "batch_size_changes",
            "batch_size_min",
            "batch_size_max",
            "before_spread_pct",
            "after_spread_pct",
            "seconds",
            "warmup",
            "repeat",
            "first_block_workers",
            "first_block_dynamic_chunk_size",
            "first_block_dynamic_chunk_auto",
            "first_block_worker_count",
            "first_block_chunk_size",
            "first_block_dynamic_chunk_size_min",
            "first_block_dynamic_chunk_size_max",
            "first_block_chunk_size_min",
            "first_block_chunk_size_max",
            "gpu_first_blocks",
            "dominant_timing_delta",
            "dominant_timing_per_attempt_delta",
            "dominant_stage_pct_delta",
            "dominant_nested_stage_pct_delta",
            "dominant_analysis_metric_delta",
        ]
    )
    for item in report["comparisons"]:
        change = "" if item["change_pct"] is None else f"{item['change_pct']:.3f}"
        timing_deltas = item.get("timing_deltas") or {}
        dominant_timing_delta = ""
        if timing_deltas:
            dominant_stage, dominant_delta = max(
                timing_deltas.items(),
                key=lambda pair: abs(float(pair[1].get("delta_ms") or 0.0)),
            )
            dominant_timing_delta = f"{dominant_stage}:{float(dominant_delta.get('delta_ms') or 0.0):.3f}ms"
        per_attempt_deltas = item.get("timing_per_attempt_deltas") or {}
        dominant_per_attempt_delta = ""
        if per_attempt_deltas:
            dominant_stage, dominant_delta = max(
                per_attempt_deltas.items(),
                key=lambda pair: abs(float(pair[1].get("delta_ms_per_attempt") or 0.0)),
            )
            dominant_per_attempt_delta = (
                f"{dominant_stage}:{float(dominant_delta.get('delta_ms_per_attempt') or 0.0):.6f}ms/attempt"
            )
        stage_pct_deltas = item.get("stage_pct_deltas") or {}
        dominant_stage_pct_delta = ""
        if stage_pct_deltas:
            dominant_stage, dominant_delta = max(
                stage_pct_deltas.items(),
                key=lambda pair: abs(float(pair[1].get("delta_pct_points") or 0.0)),
            )
            dominant_stage_pct_delta = (
                f"{dominant_stage}:{float(dominant_delta.get('delta_pct_points') or 0.0):.3f}pp"
            )
        nested_pct_deltas = item.get("nested_stage_pct_deltas") or {}
        dominant_nested_pct_delta = ""
        if nested_pct_deltas:
            dominant_stage, dominant_delta = max(
                nested_pct_deltas.items(),
                key=lambda pair: abs(float(pair[1].get("delta_pct_points") or 0.0)),
            )
            dominant_nested_pct_delta = (
                f"{dominant_stage}:{float(dominant_delta.get('delta_pct_points') or 0.0):.3f}pp"
            )
        analysis_metric_deltas = item.get("analysis_metric_deltas") or {}
        dominant_analysis_metric_delta = ""
        if analysis_metric_deltas:
            dominant_stage, dominant_delta = max(
                analysis_metric_deltas.items(),
                key=lambda pair: abs(float(pair[1].get("delta_value") or 0.0)),
            )
            dominant_analysis_metric_delta = (
                f"{dominant_stage}:{float(dominant_delta.get('delta_value') or 0.0):.6f}"
            )
        writer.writerow(
            [
                item["name"],
                item["status"],
                f"{item['before_hashrate']:.6f}",
                f"{item['after_hashrate']:.6f}",
                f"{item['delta_hashrate']:.6f}",
                change,
                str(item["backend"]),
                str(item["difficulty"]),
                str(item["difficulty_mode"]),
                str(item["difficulty_changes"]),
                str(item["batch_size"]),
                str(item["batch_size_mode"]),
                str(item["batch_size_changes"]),
                str(item["batch_size_min"]),
                str(item["batch_size_max"]),
                f"{item['before_spread_pct']:.3f}",
                f"{item['after_spread_pct']:.3f}",
                str(item["seconds"]),
                str(item["warmup"]),
                str(item["repeat"]),
                str(item["first_block_workers"]),
                str(item["first_block_dynamic_chunk_size"]),
                "true" if item["first_block_dynamic_chunk_auto"] else "false",
                str(item["first_block_worker_count"]),
                str(item["first_block_chunk_size"]),
                str(item["first_block_dynamic_chunk_size_min"]),
                str(item["first_block_dynamic_chunk_size_max"]),
                str(item["first_block_chunk_size_min"]),
                str(item["first_block_chunk_size_max"]),
                "true" if item["gpu_first_blocks"] else "false",
                dominant_timing_delta,
                dominant_per_attempt_delta,
                dominant_stage_pct_delta,
                dominant_nested_pct_delta,
                dominant_analysis_metric_delta,
            ]
        )
    return output.getvalue().rstrip("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Baseline benchmark report JSON.")
    parser.add_argument("after", type=Path, help="Candidate benchmark report JSON.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    parser.add_argument(
        "--min-change-pct",
        type=float,
        default=0.0,
        help="Absolute percent threshold used to classify improved/regressed status.",
    )
    parser.add_argument(
        "--max-spread-pct",
        type=float,
        default=10.0,
        help="Maximum before/after hashrate spread accepted before marking a changed scenario as noisy.",
    )
    parser.add_argument(
        "--match-by",
        choices=("name", "config"),
        default="name",
        help="Match benchmark runs by scenario name or by comparable scenario settings.",
    )
    parser.add_argument(
        "--ignore-detailed-timings",
        action="store_true",
        help="When matching by config, treat detailed and default timing reports as the same scenario.",
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        default=0,
        help=(
            "Only compare fixed-difficulty runs at or above this difficulty, and "
            "difficulty-sequence runs whose minimum difficulty is at or above this value."
        ),
    )
    parser.add_argument(
        "--fail-on-report-quality",
        action="store_true",
        help="Exit with code 2 if either report is partial, invalid, or marked with low benchmark trust.",
    )
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit with code 2 if any scenario regresses.")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_reports(
            load_report(args.before),
            load_report(args.after),
            min_change_pct=args.min_change_pct,
            max_spread_pct=args.max_spread_pct,
            match_by=args.match_by,
            ignore_detailed_timings=args.ignore_detailed_timings,
            min_difficulty=args.min_difficulty,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text(report))

    if args.fail_on_regression and (report["summary"]["regressed"] > 0 or report["summary"]["noisy_regressed"] > 0):
        return 2
    if args.fail_on_report_quality and not report["quality"]["ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
