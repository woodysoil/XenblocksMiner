"""Tests for the Hash API benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.hash_api_benchmark as benchmark


def _summary(hashrate: float, attempts: int = 1, ok: bool = True, timings: dict | None = None) -> dict:
    return {
        "name": "cuda-test",
        "backend": "cuda",
        "device_id": 0,
        "difficulty": 1,
        "batch_size": 2,
        "attempts": attempts,
        "elapsed_ms": 1000.0,
        "hashrate": hashrate,
        "timings": timings or {},
        "matches": 0,
        "ok": ok,
        "error": "" if ok else "failed",
    }


def test_parse_scenario_inherits_warmup_and_repeat_defaults():
    scenario = benchmark.parse_scenario(
        "name=cuda-test,backend=cuda,difficulty=8,batch_size=64,seconds=3,device=1",
        default_warmup=2,
        default_repeat=5,
    )

    assert scenario.name == "cuda-test"
    assert scenario.backend == "cuda"
    assert scenario.difficulty == 8
    assert scenario.batch_size == 64
    assert scenario.seconds == 3
    assert scenario.device == 1
    assert scenario.warmup == 2
    assert scenario.repeat == 5


def test_parse_scenario_allows_scenario_specific_warmup_and_repeat():
    scenario = benchmark.parse_scenario(
        "backend=cuda,difficulty=1,batch_size=2,seconds=1,warmup=1,repeat=3",
        default_warmup=0,
        default_repeat=1,
    )

    assert scenario.name == "cuda-d1-b2"
    assert scenario.warmup == 1
    assert scenario.repeat == 3


def test_preset_scenarios_builds_warm_short_matrix():
    scenarios = benchmark.preset_scenarios(
        "warm-short",
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=5,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-warm-short-d1-b1",
        "cuda-warm-short-d1-b64",
        "cuda-warm-short-d8-b64",
    ]
    assert [scenario.difficulty for scenario in scenarios] == [1, 1, 8]
    assert [scenario.batch_size for scenario in scenarios] == [1, 64, 64]
    assert all(scenario.seconds == 3 for scenario in scenarios)
    assert all(scenario.device == 1 for scenario in scenarios)
    assert all(scenario.warmup == 2 for scenario in scenarios)
    assert all(scenario.repeat == 5 for scenario in scenarios)


def test_ensure_unique_scenario_names_rejects_duplicates():
    scenario = benchmark.BenchmarkScenario(
        name="duplicate",
        backend="cuda",
        difficulty=1,
        batch_size=2,
        seconds=1,
    )

    try:
        benchmark.ensure_unique_scenario_names([scenario, scenario])
    except ValueError as exc:
        assert "duplicate benchmark scenario name" in str(exc)
    else:
        raise AssertionError("expected duplicate scenario rejection")


def test_summarize_iterations_reports_median_min_max_and_totals():
    scenario = benchmark.BenchmarkScenario(
        name="cuda-test",
        backend="cuda",
        difficulty=1,
        batch_size=2,
        seconds=1,
        warmup=1,
        repeat=3,
    )

    aggregate = benchmark.summarize_iterations(
        scenario,
        [_summary(10.0, attempts=10), _summary(30.0, attempts=30), _summary(20.0, attempts=20)],
    )

    assert aggregate["hashrate"] == 20.0
    assert aggregate["median_hashrate"] == 20.0
    assert aggregate["min_hashrate"] == 10.0
    assert aggregate["max_hashrate"] == 30.0
    assert aggregate["attempts"] == 60
    assert aggregate["warmup"] == 1
    assert aggregate["repeat"] == 3
    assert aggregate["ok"] is True


def test_summarize_iterations_reports_median_timing_breakdown():
    scenario = benchmark.BenchmarkScenario(
        name="cuda-test",
        backend="cuda",
        difficulty=1,
        batch_size=2,
        seconds=1,
        repeat=3,
    )

    aggregate = benchmark.summarize_iterations(
        scenario,
        [
            _summary(10.0, timings={"compute_ms": 3.0, "input_ms": 1.0}),
            _summary(30.0, timings={"compute_ms": 5.0, "input_ms": 2.0}),
            _summary(20.0, timings={"compute_ms": 4.0, "input_ms": 9.0}),
        ],
    )

    assert aggregate["timings"]["compute_ms"] == 4.0
    assert aggregate["timings"]["input_ms"] == 2.0


def test_run_scenario_records_warmup_iterations_and_selects_best_result(monkeypatch):
    calls = {"count": 0}

    def fake_run(command, text, capture_output, check):
        calls["count"] += 1
        hashrate = 100.0 + calls["count"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "backend": "cuda",
                    "device_id": 0,
                    "batch_size": 2,
                    "attempts": 2,
                    "elapsed_ms": 1000.0,
                    "hashrate": hashrate,
                    "timings": {"compute_ms": hashrate},
                    "matches": [],
                    "error": "",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    scenario = benchmark.BenchmarkScenario(
        name="cuda-test",
        backend="cuda",
        difficulty=1,
        batch_size=2,
        seconds=1,
        warmup=1,
        repeat=2,
    )

    result = benchmark.run_scenario(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert len(result["warmup_runs"]) == 1
    assert len(result["iterations"]) == 2
    assert len(result["iteration_summaries"]) == 2
    assert result["result"]["hashrate"] == 103.0
    assert result["summary"]["min_hashrate"] == 102.0
    assert result["summary"]["max_hashrate"] == 103.0
    assert result["summary"]["timings"]["compute_ms"] == 102.5
    assert result["exit_code"] == 0


def test_main_writes_output_file(monkeypatch, tmp_path, capsys):
    def fake_metadata():
        return {"nvidia_smi": {"available": False}, "nvcc": {"available": False}}

    def fake_run_scenario(binary, salt, scenario):
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(42.0),
            "aggregate": _summary(42.0),
            "command": [str(binary)],
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    monkeypatch.setattr(benchmark, "collect_hardware_metadata", fake_metadata)
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)
    output = tmp_path / "report.json"

    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--backend",
            "cuda",
            "--seconds",
            "1",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "xenblocks.hashapi.benchmark.v1"
    assert report["runs"][0]["scenario"]["warmup"] == 1
    assert report["runs"][0]["scenario"]["repeat"] == 2
    assert json.loads(capsys.readouterr().out)["runs"][0]["summary"]["hashrate"] == 42.0


def test_main_combines_presets_and_manual_scenarios(monkeypatch, tmp_path):
    captured_names = []

    def fake_metadata():
        return {"nvidia_smi": {"available": False}, "nvcc": {"available": False}}

    def fake_run_scenario(binary, salt, scenario):
        captured_names.append(scenario.name)
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(42.0),
            "aggregate": _summary(42.0),
            "command": [str(binary)],
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    monkeypatch.setattr(benchmark, "collect_hardware_metadata", fake_metadata)
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)
    output = tmp_path / "report.json"

    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--backend",
            "cuda",
            "--seconds",
            "1",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--preset",
            "smoke",
            "--scenario",
            "name=manual,backend=cuda,difficulty=8,batch_size=16,seconds=1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured_names == ["cuda-smoke-b1-d1", "cuda-batch-b8-d1", "manual"]
    assert json.loads(output.read_text(encoding="utf-8"))["presets"] == ["smoke"]


def test_main_rejects_duplicate_scenario_names(capsys):
    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--preset",
            "smoke",
            "--scenario",
            "name=cpu-smoke-b1-d1,backend=cpu,difficulty=1,batch_size=1,seconds=1",
        ]
    )

    assert exit_code == 2
    assert "duplicate benchmark scenario name" in capsys.readouterr().err
