"""Tests for Hash API benchmark report comparison."""

from __future__ import annotations

import json
import csv
import io

import scripts.hash_api_compare as compare


def _run(
    name: str,
    hashrate: float,
    ok: bool = True,
    timings: dict | None = None,
    timing_per_attempt: dict | None = None,
    stage_pct: dict | None = None,
    nested_stage_pct: dict | None = None,
    analysis_metrics: dict | None = None,
    spread_pct: float = 2.0,
    difficulty_sequence: list[int] | None = None,
    batch_size: int = 64,
    batch_size_sequence: list[int] | None = None,
    batch_size_min: int | None = None,
    batch_size_max: int | None = None,
    difficulty: int = 1,
    key_mode: str = "generated",
    first_block_workers: int = 0,
    first_block_dynamic_chunk_size: int = 0,
    first_block_dynamic_chunk_auto: bool = False,
    detailed_timings: bool = False,
    first_block_worker_count: int = 0,
    first_block_chunk_size: int = 0,
    first_block_dynamic_chunk_size_min: int | None = None,
    first_block_dynamic_chunk_size_max: int | None = None,
    first_block_chunk_size_min: int | None = None,
    first_block_chunk_size_max: int | None = None,
    gpu_first_blocks: bool = False,
) -> dict:
    sequence = difficulty_sequence or []
    batch_sequence = batch_size_sequence or []
    dynamic_min = first_block_dynamic_chunk_size if first_block_dynamic_chunk_size_min is None else first_block_dynamic_chunk_size_min
    dynamic_max = first_block_dynamic_chunk_size if first_block_dynamic_chunk_size_max is None else first_block_dynamic_chunk_size_max
    chunk_min = first_block_chunk_size if first_block_chunk_size_min is None else first_block_chunk_size_min
    chunk_max = first_block_chunk_size if first_block_chunk_size_max is None else first_block_chunk_size_max
    batch_min = batch_size if batch_size_min is None else batch_size_min
    batch_max = batch_size if batch_size_max is None else batch_size_max
    return {
        "scenario": {
            "name": name,
            "backend": "cuda",
            "difficulty": difficulty,
            "difficulty_sequence": sequence,
            "key_mode": key_mode,
            "batch_size": batch_size,
            "batch_size_sequence": batch_sequence,
            "seconds": 3,
            "device": 0,
            "warmup": 1,
            "repeat": 3,
            "first_block_workers": first_block_workers,
            "first_block_dynamic_chunk_size": first_block_dynamic_chunk_size,
            "first_block_dynamic_chunk_auto": first_block_dynamic_chunk_auto,
            "gpu_first_blocks": gpu_first_blocks,
            "detailed_timings": detailed_timings,
        },
        "summary": {
            "name": name,
            "backend": "cuda",
            "device_id": 0,
            "difficulty": difficulty,
            "batch_size": batch_size,
            "batch_size_sequence": batch_sequence,
            "batch_size_mode": "sequence" if batch_sequence else "fixed",
            "batch_size_changes": sum(
                1 for index in range(1, len(batch_sequence)) if batch_sequence[index] != batch_sequence[index - 1]
            ),
            "batch_size_min": batch_min,
            "batch_size_max": batch_max,
            "attempts": 128,
            "elapsed_ms": 3000.0,
            "hashrate": hashrate,
            "median_hashrate": hashrate,
            "min_hashrate": hashrate - 1,
            "max_hashrate": hashrate + 1,
            "hashrate_spread_pct": spread_pct,
            "difficulty_mode": "sequence" if sequence else "fixed",
            "difficulty_sequence": sequence,
            "difficulty_changes": sum(1 for index in range(1, len(sequence)) if sequence[index] != sequence[index - 1]),
            "key_mode": key_mode,
            "matches": 0,
            "ok": ok,
            "error": "" if ok else "failed",
            "warmup": 1,
            "repeat": 3,
            "timings": timings or {},
            "timing_per_attempt": timing_per_attempt or {},
            "timing_analysis": {
                "stage_pct": stage_pct or {},
                "nested_stage_pct": nested_stage_pct or {},
                **(analysis_metrics or {}),
            },
            "first_block_workers": first_block_workers,
            "first_block_dynamic_chunk_size": first_block_dynamic_chunk_size,
            "first_block_dynamic_chunk_auto": first_block_dynamic_chunk_auto,
            "first_block_worker_count": first_block_worker_count,
            "first_block_chunk_size": first_block_chunk_size,
            "first_block_dynamic_chunk_size_min": dynamic_min,
            "first_block_dynamic_chunk_size_max": dynamic_max,
            "first_block_chunk_size_min": chunk_min,
            "first_block_chunk_size_max": chunk_max,
            "gpu_first_blocks": gpu_first_blocks,
        },
    }


def _report(
    *runs: dict,
    report_ok: bool = True,
    report_quality_ok: bool | None = None,
    invalid_run_count: int = 0,
    warm_evidence_run_count: int | None = None,
    cold_scenarios: list[str] | None = None,
    stable_run_count: int | None = None,
    unstable_scenarios: list[str] | None = None,
    benchmark_trust: str = "normal",
    high_cpu_load: bool = False,
) -> dict:
    run_count = len(runs)
    if report_quality_ok is None:
        report_quality_ok = (
            report_ok
            and invalid_run_count == 0
            and not cold_scenarios
            and not unstable_scenarios
            and benchmark_trust != "low"
            and not high_cpu_load
        )
    if warm_evidence_run_count is None:
        warm_evidence_run_count = 0 if cold_scenarios else run_count
    if stable_run_count is None:
        stable_run_count = 0 if unstable_scenarios else run_count
    return {
        "schema": "xenblocks.hashapi.benchmark.v1",
        "environment": {
            "available": True,
            "benchmark_trust": benchmark_trust,
            "high_cpu_load": high_cpu_load,
            "sample_count": 2,
        },
        "recommendations": {
            "report_ok": report_ok,
            "report_quality_ok": report_quality_ok,
            "run_count": run_count,
            "warm_evidence_run_count": warm_evidence_run_count,
            "stable_run_count": stable_run_count,
            "invalid_run_count": invalid_run_count,
            "cold_scenarios": cold_scenarios or [],
            "unstable_scenarios": unstable_scenarios or [],
        },
        "runs": list(runs),
    }


def test_compare_reports_classifies_improvement_and_regression():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0), _run("cuda-b", 100.0)),
        _report(_run("cuda-a", 120.0), _run("cuda-b", 90.0)),
        min_change_pct=1.0,
    )

    by_name = {item["name"]: item for item in result["comparisons"]}
    assert by_name["cuda-a"]["status"] == "improved"
    assert by_name["cuda-a"]["change_pct"] == 20.0
    assert by_name["cuda-b"]["status"] == "regressed"
    assert by_name["cuda-b"]["change_pct"] == -10.0
    assert result["summary"]["improved"] == 1
    assert result["summary"]["regressed"] == 1


def test_compare_reports_marks_changed_noisy_runs():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0, spread_pct=25.0), _run("cuda-b", 100.0)),
        _report(_run("cuda-a", 120.0), _run("cuda-b", 80.0, spread_pct=25.0)),
        min_change_pct=1.0,
        max_spread_pct=10.0,
    )

    by_name = {item["name"]: item for item in result["comparisons"]}
    assert by_name["cuda-a"]["status"] == "noisy-improved"
    assert by_name["cuda-a"]["before_spread_pct"] == 25.0
    assert by_name["cuda-b"]["status"] == "noisy-regressed"
    assert by_name["cuda-b"]["after_spread_pct"] == 25.0
    assert result["summary"]["noisy_improved"] == 1
    assert result["summary"]["noisy_regressed"] == 1


def test_compare_reports_marks_unchanged_noisy_runs():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0, spread_pct=25.0), _run("cuda-b", 100.0)),
        _report(_run("cuda-a", 100.5), _run("cuda-b", 99.5, spread_pct=25.0)),
        min_change_pct=1.0,
        max_spread_pct=10.0,
    )

    by_name = {item["name"]: item for item in result["comparisons"]}
    assert by_name["cuda-a"]["status"] == "noisy-unchanged"
    assert by_name["cuda-a"]["before_spread_pct"] == 25.0
    assert by_name["cuda-b"]["status"] == "noisy-unchanged"
    assert by_name["cuda-b"]["after_spread_pct"] == 25.0
    assert result["summary"]["noisy_unchanged"] == 2
    assert result["summary"]["unchanged"] == 0


def test_compare_reports_includes_timing_deltas():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0, timings={"input_ms": 10.0, "compute_ms": 5.0})),
        _report(_run("cuda-a", 120.0, timings={"input_ms": 8.0, "compute_ms": 6.0})),
    )

    timing_deltas = result["comparisons"][0]["timing_deltas"]

    assert timing_deltas["input_ms"]["before_ms"] == 10.0
    assert timing_deltas["input_ms"]["after_ms"] == 8.0
    assert timing_deltas["input_ms"]["delta_ms"] == -2.0
    assert timing_deltas["input_ms"]["change_pct"] == -20.0
    assert timing_deltas["compute_ms"]["delta_ms"] == 1.0


def test_compare_reports_includes_timing_per_attempt_deltas():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0, timing_per_attempt={"input_ms": 0.010, "compute_ms": 0.005})),
        _report(_run("cuda-a", 120.0, timing_per_attempt={"input_ms": 0.008, "compute_ms": 0.006})),
    )

    timing_deltas = result["comparisons"][0]["timing_per_attempt_deltas"]

    assert timing_deltas["input_ms"]["before_ms_per_attempt"] == 0.010
    assert timing_deltas["input_ms"]["after_ms_per_attempt"] == 0.008
    assert timing_deltas["input_ms"]["delta_ms_per_attempt"] == -0.002
    assert timing_deltas["input_ms"]["change_pct"] == -20.0
    assert timing_deltas["compute_ms"]["delta_ms_per_attempt"] == 0.001


def test_compare_reports_includes_nested_stage_percentage_deltas():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0, nested_stage_pct={"first_block_digest_cpu_ms": 300.0})),
        _report(_run("cuda-a", 120.0, nested_stage_pct={"first_block_digest_cpu_ms": 240.0})),
    )

    timing_deltas = result["comparisons"][0]["nested_stage_pct_deltas"]

    assert timing_deltas["first_block_digest_cpu_ms"]["before_pct"] == 300.0
    assert timing_deltas["first_block_digest_cpu_ms"]["after_pct"] == 240.0
    assert timing_deltas["first_block_digest_cpu_ms"]["delta_pct_points"] == -60.0
    assert timing_deltas["first_block_digest_cpu_ms"]["change_pct"] == -20.0


def test_compare_reports_includes_top_level_stage_percentage_deltas():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0, stage_pct={"setup_ms": 20.0, "input_ms": 60.0})),
        _report(_run("cuda-a", 120.0, stage_pct={"setup_ms": 10.0, "input_ms": 70.0})),
    )

    timing_deltas = result["comparisons"][0]["stage_pct_deltas"]

    assert timing_deltas["setup_ms"]["before_pct"] == 20.0
    assert timing_deltas["setup_ms"]["after_pct"] == 10.0
    assert timing_deltas["setup_ms"]["delta_pct_points"] == -10.0
    assert timing_deltas["setup_ms"]["change_pct"] == -50.0
    assert timing_deltas["input_ms"]["delta_pct_points"] == 10.0


def test_compare_reports_includes_analysis_metric_deltas():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0, analysis_metrics={"input_residual_pct": 4.0})),
        _report(_run("cuda-a", 120.0, analysis_metrics={"input_residual_pct": 1.0})),
    )

    timing_deltas = result["comparisons"][0]["analysis_metric_deltas"]

    assert timing_deltas["input_residual_pct"]["before_value"] == 4.0
    assert timing_deltas["input_residual_pct"]["after_value"] == 1.0
    assert timing_deltas["input_residual_pct"]["delta_value"] == -3.0
    assert timing_deltas["input_residual_pct"]["change_pct"] == -75.0


def test_compare_reports_includes_difficulty_sequence_metadata():
    result = compare.compare_reports(
        _report(_run("cuda-seq", 100.0, difficulty_sequence=[1, 8, 1, 8])),
        _report(_run("cuda-seq", 120.0, difficulty_sequence=[1, 8, 1, 8])),
    )

    item = result["comparisons"][0]

    assert item["difficulty_mode"] == "sequence"
    assert item["difficulty_sequence"] == [1, 8, 1, 8]
    assert item["difficulty_changes"] == 3


def test_compare_reports_includes_batch_size_sequence_metadata():
    result = compare.compare_reports(
        _report(
            _run(
                "cuda-variable-shape",
                100.0,
                difficulty_sequence=[1, 8, 64],
                batch_size=3072,
                batch_size_sequence=[2048, 3072, 3072],
                batch_size_min=2048,
                batch_size_max=3072,
            )
        ),
        _report(
            _run(
                "cuda-variable-shape",
                120.0,
                difficulty_sequence=[1, 8, 64],
                batch_size=3072,
                batch_size_sequence=[2048, 3072, 3072],
                batch_size_min=2048,
                batch_size_max=3072,
            )
        ),
    )

    item = result["comparisons"][0]

    assert item["batch_size_mode"] == "sequence"
    assert item["batch_size_sequence"] == [2048, 3072, 3072]
    assert item["batch_size_changes"] == 1
    assert item["batch_size_min"] == 2048
    assert item["batch_size_max"] == 3072


def test_compare_reports_includes_key_mode_metadata():
    result = compare.compare_reports(
        _report(_run("cuda-fixed", 100.0, key_mode="fixed")),
        _report(_run("cuda-fixed", 120.0, key_mode="fixed")),
    )

    assert result["comparisons"][0]["key_mode"] == "fixed"


def test_compare_reports_includes_report_quality_metadata():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0)),
        _report(_run("cuda-a", 105.0), benchmark_trust="low", high_cpu_load=True),
    )

    assert result["quality"]["ok"] is False
    assert result["quality"]["before"]["acceptable"] is True
    assert result["quality"]["after"]["acceptable"] is False
    assert result["quality"]["after"]["benchmark_trust"] == "low"
    assert result["quality"]["after"]["reasons"] == ["report_quality_ok=false", "benchmark_trust=low"]


def test_compare_reports_marks_partial_report_quality():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0), report_ok=False, invalid_run_count=1),
        _report(_run("cuda-a", 105.0)),
    )

    assert result["quality"]["ok"] is False
    assert result["quality"]["before"]["acceptable"] is False
    assert result["quality"]["before"]["reasons"] == ["report_ok=false", "report_quality_ok=false", "invalid_run_count>0"]


def test_compare_reports_marks_cold_report_quality():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0)),
        _report(
            _run("cuda-a", 105.0),
            report_quality_ok=False,
            warm_evidence_run_count=0,
            cold_scenarios=["cuda-a"],
        ),
    )

    assert result["quality"]["ok"] is False
    assert result["quality"]["after"]["acceptable"] is False
    assert result["quality"]["after"]["warm_evidence_run_count"] == 0
    assert result["quality"]["after"]["cold_scenarios"] == ["cuda-a"]
    assert result["quality"]["after"]["reasons"] == ["report_quality_ok=false", "warm_evidence_incomplete"]


def test_compare_reports_marks_unstable_report_quality():
    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0)),
        _report(
            _run("cuda-a", 105.0, spread_pct=29.0),
            report_quality_ok=False,
            stable_run_count=0,
            unstable_scenarios=["cuda-a"],
        ),
    )

    assert result["quality"]["ok"] is False
    assert result["quality"]["after"]["acceptable"] is False
    assert result["quality"]["after"]["stable_run_count"] == 0
    assert result["quality"]["after"]["unstable_scenarios"] == ["cuda-a"]
    assert result["quality"]["after"]["reasons"] == ["report_quality_ok=false", "stable_evidence_incomplete"]


def test_compare_reports_derives_cold_quality_for_legacy_reports():
    cold_report = _report(_run("cuda-a", 105.0), report_quality_ok=True)
    cold_report["runs"][0]["summary"]["warmup"] = 0
    cold_report["runs"][0]["summary"]["repeat"] = 1
    cold_report["runs"][0]["scenario"]["warmup"] = 0
    cold_report["runs"][0]["scenario"]["repeat"] = 1
    cold_report["recommendations"].pop("warm_evidence_run_count")
    cold_report["recommendations"].pop("cold_scenarios")

    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0)),
        cold_report,
    )

    assert result["quality"]["ok"] is False
    assert result["quality"]["after"]["acceptable"] is False
    assert result["quality"]["after"]["warm_evidence_run_count"] == 0
    assert result["quality"]["after"]["cold_scenarios"] == ["cuda-a"]
    assert result["quality"]["after"]["reasons"] == ["warm_evidence_incomplete"]


def test_compare_reports_derives_unstable_quality_for_legacy_reports():
    unstable_report = _report(_run("cuda-a", 105.0, spread_pct=29.0), report_quality_ok=True)
    unstable_report["runs"][0]["summary"]["stable"] = False
    unstable_report["recommendations"].pop("stable_run_count")
    unstable_report["recommendations"].pop("unstable_scenarios")

    result = compare.compare_reports(
        _report(_run("cuda-a", 100.0)),
        unstable_report,
    )

    assert result["quality"]["ok"] is False
    assert result["quality"]["after"]["acceptable"] is False
    assert result["quality"]["after"]["stable_run_count"] == 0
    assert result["quality"]["after"]["unstable_scenarios"] == ["cuda-a"]
    assert result["quality"]["after"]["reasons"] == ["stable_evidence_incomplete"]


def test_compare_reports_reports_missing_scenarios():
    result = compare.compare_reports(
        _report(_run("before-only", 100.0)),
        _report(_run("after-only", 110.0)),
    )

    by_name = {item["name"]: item for item in result["comparisons"]}
    assert by_name["before-only"]["status"] == "missing-after"
    assert by_name["after-only"]["status"] == "missing-before"
    assert result["summary"]["missing_after"] == 1
    assert result["summary"]["missing_before"] == 1


def test_compare_reports_can_match_by_config_when_names_differ():
    result = compare.compare_reports(
        _report(_run("before-label", 100.0)),
        _report(_run("after-label", 110.0)),
        min_change_pct=1.0,
        match_by="config",
    )

    assert result["match_by"] == "config"
    assert len(result["comparisons"]) == 1
    item = result["comparisons"][0]
    assert item["name"] == "before-label -> after-label"
    assert item["before_name"] == "before-label"
    assert item["after_name"] == "after-label"
    assert item["status"] == "improved"
    assert item["change_pct"] == 10.0
    assert result["summary"]["missing_after"] == 0
    assert result["summary"]["missing_before"] == 0


def test_compare_reports_config_match_separates_different_settings():
    result = compare.compare_reports(
        _report(_run("same-shape", 100.0)),
        _report(_run("different-sequence", 110.0, difficulty_sequence=[1, 8])),
        match_by="config",
    )

    statuses = sorted(item["status"] for item in result["comparisons"])
    assert statuses == ["missing-after", "missing-before"]


def test_compare_reports_config_match_separates_batch_size_sequences():
    result = compare.compare_reports(
        _report(_run("fixed-shape", 100.0, batch_size=3072)),
        _report(
            _run(
                "variable-shape",
                110.0,
                batch_size=3072,
                batch_size_sequence=[2048, 3072, 3072],
                batch_size_min=2048,
                batch_size_max=3072,
            )
        ),
        match_by="config",
    )

    statuses = sorted(item["status"] for item in result["comparisons"])
    assert statuses == ["missing-after", "missing-before"]
    assert {item["batch_size_mode"] for item in result["comparisons"]} == {"fixed", "sequence"}
    assert any("b2048x3072x3072" in item["match_key"] for item in result["comparisons"])


def test_compare_reports_config_match_separates_first_block_workers():
    result = compare.compare_reports(
        _report(_run("auto-workers", 100.0, first_block_workers=0)),
        _report(_run("four-workers", 110.0, first_block_workers=4)),
        match_by="config",
    )

    statuses = sorted(item["status"] for item in result["comparisons"])
    assert statuses == ["missing-after", "missing-before"]
    assert {item["first_block_workers"] for item in result["comparisons"]} == {0, 4}
    assert any("fbw4" in item["match_key"] for item in result["comparisons"])


def test_compare_reports_config_match_separates_first_block_dynamic_chunk_size():
    result = compare.compare_reports(
        _report(_run("static-chunks", 100.0, first_block_dynamic_chunk_size=0)),
        _report(_run("dynamic-chunks", 110.0, first_block_dynamic_chunk_size=64)),
        match_by="config",
    )

    statuses = sorted(item["status"] for item in result["comparisons"])
    assert statuses == ["missing-after", "missing-before"]
    assert {item["first_block_dynamic_chunk_size"] for item in result["comparisons"]} == {0, 64}
    assert any("fbd64" in item["match_key"] for item in result["comparisons"])


def test_compare_reports_config_match_separates_first_block_dynamic_chunk_auto():
    result = compare.compare_reports(
        _report(_run("manual-chunks", 100.0, first_block_dynamic_chunk_size=32)),
        _report(
            _run(
                "auto-chunks",
                110.0,
                first_block_dynamic_chunk_size=32,
                first_block_dynamic_chunk_auto=True,
            )
        ),
        match_by="config",
    )

    statuses = sorted(item["status"] for item in result["comparisons"])
    assert statuses == ["missing-after", "missing-before"]
    assert {item["first_block_dynamic_chunk_auto"] for item in result["comparisons"]} == {False, True}
    assert any("fbda1" in item["match_key"] for item in result["comparisons"])


def test_compare_reports_config_match_separates_gpu_first_blocks():
    result = compare.compare_reports(
        _report(_run("cpu-first-blocks", 100.0)),
        _report(_run("gpu-first-blocks", 110.0, gpu_first_blocks=True)),
        match_by="config",
    )

    statuses = sorted(item["status"] for item in result["comparisons"])
    assert statuses == ["missing-after", "missing-before"]
    assert {item["gpu_first_blocks"] for item in result["comparisons"]} == {False, True}
    assert any("gfb1" in item["match_key"] for item in result["comparisons"])


def test_compare_reports_config_match_uses_requested_dynamic_chunk_for_auto_runs():
    before = _run(
        "auto-before",
        100.0,
        first_block_dynamic_chunk_size=0,
        first_block_dynamic_chunk_auto=True,
        first_block_dynamic_chunk_size_min=0,
        first_block_dynamic_chunk_size_max=16,
    )
    after = _run(
        "auto-after",
        105.0,
        first_block_dynamic_chunk_size=0,
        first_block_dynamic_chunk_auto=True,
        first_block_dynamic_chunk_size_min=0,
        first_block_dynamic_chunk_size_max=16,
    )
    before["summary"]["first_block_dynamic_chunk_size"] = 16
    after["summary"]["first_block_dynamic_chunk_size"] = 0

    result = compare.compare_reports(
        _report(before),
        _report(after),
        match_by="config",
        min_change_pct=1.0,
    )

    assert len(result["comparisons"]) == 1
    item = result["comparisons"][0]
    assert item["status"] == "improved"
    assert item["first_block_dynamic_chunk_size"] == 0
    assert item["first_block_dynamic_chunk_size_min"] == 0
    assert item["first_block_dynamic_chunk_size_max"] == 16


def test_compare_reports_can_filter_by_min_difficulty():
    result = compare.compare_reports(
        _report(_run("low-diff", 100.0, difficulty=8), _run("high-diff", 200.0, difficulty=4096)),
        _report(_run("low-diff", 110.0, difficulty=8), _run("high-diff", 220.0, difficulty=4096)),
        min_difficulty=4096,
    )

    assert result["min_difficulty"] == 4096
    assert [item["name"] for item in result["comparisons"]] == ["high-diff"]


def test_compare_reports_min_difficulty_uses_sequence_minimum():
    result = compare.compare_reports(
        _report(
            _run("mixed-seq", 100.0, difficulty_sequence=[8, 4096]),
            _run("high-seq", 200.0, difficulty_sequence=[4096, 8192]),
        ),
        _report(
            _run("mixed-seq", 110.0, difficulty_sequence=[8, 4096]),
            _run("high-seq", 220.0, difficulty_sequence=[4096, 8192]),
        ),
        min_difficulty=4096,
    )

    assert [item["name"] for item in result["comparisons"]] == ["high-seq"]
    assert result["comparisons"][0]["difficulty_sequence"] == [4096, 8192]


def test_compare_reports_config_match_separates_detailed_timing_mode_by_default():
    result = compare.compare_reports(
        _report(_run("default-timing", 100.0, detailed_timings=False)),
        _report(_run("detailed-timing", 110.0, detailed_timings=True)),
        match_by="config",
    )

    statuses = sorted(item["status"] for item in result["comparisons"])
    assert statuses == ["missing-after", "missing-before"]
    assert any("detailed" in item["match_key"] for item in result["comparisons"])


def test_compare_reports_can_ignore_detailed_timing_mode_for_config_match():
    result = compare.compare_reports(
        _report(_run("default-timing", 100.0, detailed_timings=False)),
        _report(_run("detailed-timing", 110.0, detailed_timings=True)),
        match_by="config",
        ignore_detailed_timings=True,
        min_change_pct=1.0,
    )

    assert result["ignore_detailed_timings"] is True
    assert len(result["comparisons"]) == 1
    item = result["comparisons"][0]
    assert item["name"] == "default-timing -> detailed-timing"
    assert item["status"] == "improved"
    assert item["change_pct"] == 10.0
    assert "default-timing" in item["match_key"]


def test_compare_reports_rejects_duplicate_names():
    report = _report(_run("same", 100.0), _run("same", 110.0))

    try:
        compare.compare_reports(report, _report(_run("same", 120.0)))
    except ValueError as exc:
        assert "duplicate scenario name" in str(exc)
    else:
        raise AssertionError("expected duplicate scenario rejection")


def test_compare_reports_rejects_duplicate_configs_when_matching_by_config():
    report = _report(_run("same-shape-a", 100.0), _run("same-shape-b", 110.0))

    try:
        compare.compare_reports(report, _report(_run("same-shape-c", 120.0)), match_by="config")
    except ValueError as exc:
        assert "duplicate scenario config" in str(exc)
    else:
        raise AssertionError("expected duplicate config rejection")


def test_main_outputs_json_and_fails_on_regression(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_report(_run("cuda-a", 100.0))), encoding="utf-8")
    after.write_text(json.dumps(_report(_run("cuda-a", 90.0))), encoding="utf-8")

    exit_code = compare.main([str(before), str(after), "--format", "json", "--fail-on-regression"])

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "xenblocks.hashapi.compare.v1"
    assert output["summary"]["regressed"] == 1
    assert output["quality"]["ok"] is True


def test_main_can_fail_on_report_quality(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_report(_run("cuda-a", 100.0))), encoding="utf-8")
    after.write_text(json.dumps(_report(_run("cuda-a", 101.0), benchmark_trust="low")), encoding="utf-8")

    exit_code = compare.main([str(before), str(after), "--format", "json", "--fail-on-report-quality"])

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["quality"]["ok"] is False
    assert output["summary"]["regressed"] == 0


def test_main_can_match_by_config(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_report(_run("before-label", 100.0))), encoding="utf-8")
    after.write_text(json.dumps(_report(_run("after-label", 110.0))), encoding="utf-8")

    exit_code = compare.main([str(before), str(after), "--match-by", "config", "--format", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["match_by"] == "config"
    assert output["comparisons"][0]["status"] == "improved"


def test_main_can_filter_by_min_difficulty(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps(_report(_run("low-diff", 100.0, difficulty=8), _run("high-diff", 200.0, difficulty=4096))),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(_report(_run("low-diff", 110.0, difficulty=8), _run("high-diff", 220.0, difficulty=4096))),
        encoding="utf-8",
    )

    exit_code = compare.main([str(before), str(after), "--min-difficulty", "4096", "--format", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["min_difficulty"] == 4096
    assert [item["name"] for item in output["comparisons"]] == ["high-diff"]


def test_format_text_outputs_automation_friendly_rows():
    result = compare.compare_reports(
        _report(
            _run(
                "cuda-a",
                100.0,
                timings={"input_ms": 10.0, "compute_ms": 5.0},
                timing_per_attempt={"input_ms": 0.010, "compute_ms": 0.005},
                stage_pct={"setup_ms": 20.0, "input_ms": 60.0},
                nested_stage_pct={"first_block_digest_cpu_ms": 300.0},
                analysis_metrics={"input_residual_pct": 4.0},
                first_block_worker_count=4,
                first_block_dynamic_chunk_size=64,
                first_block_dynamic_chunk_auto=True,
                first_block_chunk_size=16,
                first_block_dynamic_chunk_size_min=0,
                first_block_dynamic_chunk_size_max=64,
                first_block_chunk_size_min=16,
                first_block_chunk_size_max=256,
            )
        ),
        _report(
            _run(
                "cuda-a",
                105.0,
                timings={"input_ms": 7.0, "compute_ms": 6.0},
                timing_per_attempt={"input_ms": 0.007, "compute_ms": 0.006},
                stage_pct={"setup_ms": 10.0, "input_ms": 70.0},
                nested_stage_pct={"first_block_digest_cpu_ms": 240.0},
                analysis_metrics={"input_residual_pct": 1.0},
                first_block_worker_count=8,
                first_block_dynamic_chunk_size=64,
                first_block_dynamic_chunk_auto=True,
                first_block_chunk_size=8,
                first_block_dynamic_chunk_size_min=0,
                first_block_dynamic_chunk_size_max=64,
                first_block_chunk_size_min=8,
                first_block_chunk_size_max=256,
            )
        ),
    )

    text = compare.format_text(result)

    assert text.splitlines()[0].startswith("scenario,status,before_hashrate")
    assert "cuda-a,improved,100.000000,105.000000,5.000000,5.000" in text
    assert ",fixed,0," in text
    assert "2.000,2.000" in text
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    row = rows[1]
    assert row[header.index("first_block_workers")] == "0"
    assert row[header.index("first_block_dynamic_chunk_size")] == "64"
    assert row[header.index("first_block_dynamic_chunk_auto")] == "true"
    assert row[header.index("first_block_worker_count")] == "8"
    assert row[header.index("first_block_chunk_size")] == "8"
    assert row[header.index("batch_size_mode")] == "fixed"
    assert row[header.index("batch_size_min")] == "64"
    assert row[header.index("batch_size_max")] == "64"
    assert row[header.index("first_block_dynamic_chunk_size_min")] == "0"
    assert row[header.index("first_block_dynamic_chunk_size_max")] == "64"
    assert row[header.index("first_block_chunk_size_min")] == "8"
    assert row[header.index("first_block_chunk_size_max")] == "256"
    assert row[header.index("gpu_first_blocks")] == "false"
    assert "input_ms:-3.000ms" in text
    assert "input_ms:-0.003000ms/attempt" in text
    assert "input_ms:10.000pp" in text
    assert "first_block_digest_cpu_ms:-60.000pp" in text
    assert "input_residual_pct:-3.000000" in text


def test_format_text_escapes_csv_fields():
    result = compare.compare_reports(_report(_run("cuda,a", 100.0)), _report(_run("cuda,a", 105.0)))

    rows = list(csv.reader(io.StringIO(compare.format_text(result))))

    assert rows[1][0] == "cuda,a"
