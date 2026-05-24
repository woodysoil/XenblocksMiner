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
RunMap = dict[str, dict[str, Any]]


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


def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = _summary_for(run)
    scenario = _scenario_for(run)
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
        "attempts": _int_value(summary, "attempts", 0),
        "elapsed_ms": _float_value(summary, "elapsed_ms", 0.0),
        "hashrate": _float_value(summary, "hashrate", 0.0),
        "median_hashrate": _float_value(summary, "median_hashrate", _float_value(summary, "hashrate", 0.0)),
        "min_hashrate": _float_value(summary, "min_hashrate", _float_value(summary, "hashrate", 0.0)),
        "max_hashrate": _float_value(summary, "max_hashrate", _float_value(summary, "hashrate", 0.0)),
        "timings": summary.get("timings", {}),
        "matches": _int_value(summary, "matches", 0),
        "ok": bool(summary.get("ok")),
        "error": str(summary.get("error") or ""),
    }


def index_runs(report: Report) -> RunMap:
    indexed: RunMap = {}
    for run in report.get("runs", []):
        normalized = normalize_run(run)
        name = normalized["name"]
        if name in indexed:
            raise ValueError(f"duplicate scenario name in report: {name}")
        indexed[name] = normalized
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


def _status(before: dict[str, Any] | None, after: dict[str, Any] | None, change_pct: float | None, min_change_pct: float) -> str:
    if before is None:
        return "missing-before"
    if after is None:
        return "missing-after"
    if not before["ok"] or not after["ok"]:
        return "invalid"
    if change_pct is None:
        return "unrated"
    if change_pct > min_change_pct:
        return "improved"
    if change_pct < -min_change_pct:
        return "regressed"
    return "unchanged"


def compare_reports(before_report: Report, after_report: Report, min_change_pct: float = 0.0) -> Report:
    before_runs = index_runs(before_report)
    after_runs = index_runs(after_report)
    comparisons: list[dict[str, Any]] = []

    for name in sorted(set(before_runs) | set(after_runs)):
        before = before_runs.get(name)
        after = after_runs.get(name)
        before_rate = before["median_hashrate"] if before else 0.0
        after_rate = after["median_hashrate"] if after else 0.0
        change_pct = _percent_change(before_rate, after_rate) if before and after else None
        status = _status(before, after, change_pct, min_change_pct)
        comparisons.append(
            {
                "name": name,
                "status": status,
                "before_hashrate": before_rate,
                "after_hashrate": after_rate,
                "delta_hashrate": after_rate - before_rate,
                "change_pct": change_pct,
                "backend": (after or before or {}).get("backend", ""),
                "device_id": (after or before or {}).get("device_id", 0),
                "difficulty": (after or before or {}).get("difficulty", 0),
                "batch_size": (after or before or {}).get("batch_size", 0),
                "seconds": (after or before or {}).get("seconds", 0),
                "warmup": (after or before or {}).get("warmup", 0),
                "repeat": (after or before or {}).get("repeat", 1),
                "timing_deltas": compare_timings(before, after),
                "before": before,
                "after": after,
            }
        )

    statuses = [item["status"] for item in comparisons]
    return {
        "schema": "xenblocks.hashapi.compare.v1",
        "min_change_pct": min_change_pct,
        "summary": {
            "total": len(comparisons),
            "improved": statuses.count("improved"),
            "regressed": statuses.count("regressed"),
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
            "batch_size",
            "seconds",
            "warmup",
            "repeat",
            "dominant_timing_delta",
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
                str(item["batch_size"]),
                str(item["seconds"]),
                str(item["warmup"]),
                str(item["repeat"]),
                dominant_timing_delta,
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
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit with code 2 if any scenario regresses.")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_reports(load_report(args.before), load_report(args.after), min_change_pct=args.min_change_pct)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text(report))

    if args.fail_on_regression and report["summary"]["regressed"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
