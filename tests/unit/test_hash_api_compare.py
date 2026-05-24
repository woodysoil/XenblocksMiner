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
    spread_pct: float = 2.0,
    difficulty_sequence: list[int] | None = None,
    key_mode: str = "generated",
) -> dict:
    sequence = difficulty_sequence or []
    return {
        "scenario": {
            "name": name,
            "backend": "cuda",
            "difficulty": 1,
            "difficulty_sequence": sequence,
            "key_mode": key_mode,
            "batch_size": 64,
            "seconds": 3,
            "device": 0,
            "warmup": 1,
            "repeat": 3,
        },
        "summary": {
            "name": name,
            "backend": "cuda",
            "device_id": 0,
            "difficulty": 1,
            "batch_size": 64,
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
        },
    }


def _report(*runs: dict) -> dict:
    return {
        "schema": "xenblocks.hashapi.benchmark.v1",
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


def test_compare_reports_includes_difficulty_sequence_metadata():
    result = compare.compare_reports(
        _report(_run("cuda-seq", 100.0, difficulty_sequence=[1, 8, 1, 8])),
        _report(_run("cuda-seq", 120.0, difficulty_sequence=[1, 8, 1, 8])),
    )

    item = result["comparisons"][0]

    assert item["difficulty_mode"] == "sequence"
    assert item["difficulty_sequence"] == [1, 8, 1, 8]
    assert item["difficulty_changes"] == 3


def test_compare_reports_includes_key_mode_metadata():
    result = compare.compare_reports(
        _report(_run("cuda-fixed", 100.0, key_mode="fixed")),
        _report(_run("cuda-fixed", 120.0, key_mode="fixed")),
    )

    assert result["comparisons"][0]["key_mode"] == "fixed"


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


def test_format_text_outputs_automation_friendly_rows():
    result = compare.compare_reports(
        _report(
            _run(
                "cuda-a",
                100.0,
                timings={"input_ms": 10.0, "compute_ms": 5.0},
                timing_per_attempt={"input_ms": 0.010, "compute_ms": 0.005},
            )
        ),
        _report(
            _run(
                "cuda-a",
                105.0,
                timings={"input_ms": 7.0, "compute_ms": 6.0},
                timing_per_attempt={"input_ms": 0.007, "compute_ms": 0.006},
            )
        ),
    )

    text = compare.format_text(result)

    assert text.splitlines()[0].startswith("scenario,status,before_hashrate")
    assert "cuda-a,improved,100.000000,105.000000,5.000000,5.000" in text
    assert ",fixed,0," in text
    assert "2.000,2.000" in text
    assert "input_ms:-3.000ms" in text
    assert "input_ms:-0.003000ms/attempt" in text


def test_format_text_escapes_csv_fields():
    result = compare.compare_reports(_report(_run("cuda,a", 100.0)), _report(_run("cuda,a", 105.0)))

    rows = list(csv.reader(io.StringIO(compare.format_text(result))))

    assert rows[1][0] == "cuda,a"
