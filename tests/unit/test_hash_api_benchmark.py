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


def test_preset_scenarios_builds_batch_scan_matrix():
    scenarios = benchmark.preset_scenarios(
        "batch-scan",
        seconds=2,
        backend="cuda",
        device=0,
        warmup=1,
        repeat=2,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-batch-scan-d1-b64",
        "cuda-batch-scan-d1-b128",
        "cuda-batch-scan-d1-b256",
        "cuda-batch-scan-d1-b512",
        "cuda-batch-scan-d8-b64",
        "cuda-batch-scan-d8-b128",
        "cuda-batch-scan-d8-b256",
        "cuda-batch-scan-d8-b512",
    ]
    assert [scenario.difficulty for scenario in scenarios] == [1, 1, 1, 1, 8, 8, 8, 8]
    assert [scenario.batch_size for scenario in scenarios] == [64, 128, 256, 512, 64, 128, 256, 512]
    assert all(scenario.seconds == 2 for scenario in scenarios)
    assert all(scenario.warmup == 1 for scenario in scenarios)
    assert all(scenario.repeat == 2 for scenario in scenarios)


def test_scan_scenarios_builds_custom_matrix():
    scenarios = benchmark.scan_scenarios(
        difficulties=[1, 8],
        batch_sizes=[512, 1024],
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-scan-d1-b512",
        "cuda-scan-d1-b1024",
        "cuda-scan-d8-b512",
        "cuda-scan-d8-b1024",
    ]
    assert [scenario.difficulty for scenario in scenarios] == [1, 1, 8, 8]
    assert [scenario.batch_size for scenario in scenarios] == [512, 1024, 512, 1024]
    assert all(scenario.seconds == 3 for scenario in scenarios)
    assert all(scenario.device == 1 for scenario in scenarios)
    assert all(scenario.warmup == 2 for scenario in scenarios)
    assert all(scenario.repeat == 4 for scenario in scenarios)


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
    assert aggregate["hashrate_spread_pct"] == 100.0
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
    assert aggregate["timing_analysis"]["dominant_stage"] == "compute_ms"
    assert aggregate["timing_analysis"]["dominant_stage_ms"] == 4.0
    assert aggregate["timing_analysis"]["dominant_stage_pct"] == 0.0


def test_summarize_iterations_reports_timing_stage_percentages():
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
            _summary(10.0, timings={"compute_ms": 2.0, "input_ms": 6.0, "total_ms": 10.0}),
            _summary(30.0, timings={"compute_ms": 4.0, "input_ms": 8.0, "total_ms": 10.0}),
            _summary(20.0, timings={"compute_ms": 3.0, "input_ms": 7.0, "total_ms": 10.0}),
        ],
    )

    assert aggregate["timings"]["compute_ms"] == 3.0
    assert aggregate["timings"]["input_ms"] == 7.0
    assert aggregate["timing_analysis"]["dominant_stage"] == "input_ms"
    assert aggregate["timing_analysis"]["dominant_stage_ms"] == 7.0
    assert aggregate["timing_analysis"]["dominant_stage_pct"] == 70.0
    assert aggregate["timing_analysis"]["stage_pct"]["compute_ms"] == 30.0
    assert aggregate["timing_analysis"]["stage_pct"]["input_ms"] == 70.0
    assert "total_ms" not in aggregate["timing_analysis"]["stage_pct"]


def test_build_recommendations_selects_best_batch_per_difficulty():
    runs = [
        {"summary": {**_summary(100.0), "name": "d1-b64", "difficulty": 1, "batch_size": 64}},
        {
            "summary": {
                **_summary(150.0),
                "name": "d1-b128",
                "difficulty": 1,
                "batch_size": 128,
                "hashrate_spread_pct": 5.0,
                "timing_analysis": {"dominant_stage": "input_ms", "dominant_stage_pct": 75.0},
            }
        },
        {
            "summary": {
                **_summary(120.0),
                "name": "d8-b64",
                "difficulty": 8,
                "batch_size": 64,
                "hashrate_spread_pct": 15.0,
                "timing_analysis": {"dominant_stage": "compute_ms", "dominant_stage_pct": 55.0},
            }
        },
        {"summary": {**_summary(90.0, ok=False), "name": "d8-b128", "difficulty": 8, "batch_size": 128}},
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["stable_spread_pct"] == 10.0
    assert recommendations["batch_size_by_difficulty"] == [
        {
            "backend": "cuda",
            "device_id": 0,
            "difficulty": 1,
            "batch_size": 128,
            "median_hashrate": 150.0,
            "hashrate_spread_pct": 5.0,
            "stable": True,
            "selection_reason": "best_stable_median",
            "dominant_stage": "input_ms",
            "dominant_stage_pct": 75.0,
            "scenario": "d1-b128",
        },
        {
            "backend": "cuda",
            "device_id": 0,
            "difficulty": 8,
            "batch_size": 64,
            "median_hashrate": 120.0,
            "hashrate_spread_pct": 15.0,
            "stable": False,
            "selection_reason": "no_stable_candidate",
            "dominant_stage": "compute_ms",
            "dominant_stage_pct": 55.0,
            "scenario": "d8-b64",
        },
    ]


def test_build_recommendations_prefers_stable_candidate_over_noisy_higher_median():
    runs = [
        {
            "summary": {
                **_summary(400.0),
                "name": "d1-b2048-noisy",
                "difficulty": 1,
                "batch_size": 2048,
                "hashrate_spread_pct": 40.0,
            }
        },
        {
            "summary": {
                **_summary(300.0),
                "name": "d1-b512-stable",
                "difficulty": 1,
                "batch_size": 512,
                "hashrate_spread_pct": 5.0,
            }
        },
    ]

    recommendation = benchmark.build_recommendations(runs)["batch_size_by_difficulty"][0]

    assert recommendation["batch_size"] == 512
    assert recommendation["median_hashrate"] == 300.0
    assert recommendation["stable"] is True
    assert recommendation["selection_reason"] == "best_stable_median"


def test_build_sanitized_report_drops_private_fields():
    report = {
        "schema": "xenblocks.hashapi.benchmark.v1",
        "created_at_unix": 123.0,
        "host": {"system": "Windows", "machine": "private-host"},
        "hardware": {"nvidia_smi": {"stdout": "0, Private GPU, 999.99, 4096 MiB"}},
        "binary": r"D:\private\miner.exe",
        "salt": "private-salt",
        "presets": ["warm-short"],
        "recommendations": {"batch_size_by_difficulty": []},
        "runs": [
            {
                "scenario": {
                    "name": "cuda-test",
                    "backend": "cuda",
                    "difficulty": 1,
                    "batch_size": 64,
                    "seconds": 3,
                    "device": 0,
                    "warmup": 1,
                    "repeat": 2,
                    "prefix": "deadbeef",
                    "pattern": "XEN11",
                },
                "summary": _summary(42.0),
                "command": [r"D:\private\miner.exe", "--salt", "private-salt"],
                "warmup_runs": [{"result": {"matches": [{"key": "secret-key"}]}}],
                "iterations": [{"result": {"matches": [{"key": "secret-key"}]}}],
                "result": {"matches": [{"key": "secret-key"}]},
            }
        ],
    }

    sanitized = benchmark.build_sanitized_report(report)
    encoded = json.dumps(sanitized)

    assert sanitized["schema"] == "xenblocks.hashapi.benchmark-summary.v1"
    assert sanitized["source_schema"] == "xenblocks.hashapi.benchmark.v1"
    assert sanitized["privacy"]["sanitized"] is True
    assert sanitized["runs"][0]["scenario"]["prefix_length"] == 8
    assert "prefix" not in sanitized["runs"][0]["scenario"]
    assert sanitized["runs"][0]["summary"]["hashrate"] == 42.0
    assert "binary" not in sanitized
    assert "hardware" not in sanitized
    assert "host" not in sanitized
    assert "salt" not in sanitized
    assert "command" not in sanitized["runs"][0]
    assert "warmup_runs" not in sanitized["runs"][0]
    assert "iterations" not in sanitized["runs"][0]
    assert "result" not in sanitized["runs"][0]
    for token in [
        r"D:\private",
        "private-host",
        "Private GPU",
        "private-salt",
        "deadbeef",
        "secret-key",
    ]:
        assert token not in encoded


def test_run_scenario_records_warmup_iterations_and_selects_median_result(monkeypatch):
    calls = {"count": 0}
    hashrates = [101.0, 300.0, 102.0, 103.0]

    def fake_run(command, text, capture_output, check):
        hashrate = hashrates[calls["count"]]
        calls["count"] += 1
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
        repeat=3,
    )

    result = benchmark.run_scenario(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert len(result["warmup_runs"]) == 1
    assert len(result["iterations"]) == 3
    assert len(result["iteration_summaries"]) == 3
    assert result["result"]["hashrate"] == 103.0
    assert result["summary"]["min_hashrate"] == 102.0
    assert result["summary"]["max_hashrate"] == 300.0
    assert result["summary"]["median_hashrate"] == 103.0
    assert result["summary"]["timings"]["compute_ms"] == 103.0
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
    assert report["recommendations"]["stable_spread_pct"] == 10.0
    assert report["recommendations"]["batch_size_by_difficulty"][0]["batch_size"] == 2
    assert report["runs"][0]["scenario"]["warmup"] == 1
    assert report["runs"][0]["scenario"]["repeat"] == 2
    assert json.loads(capsys.readouterr().out)["runs"][0]["summary"]["hashrate"] == 42.0


def test_main_writes_sanitized_output_file(monkeypatch, tmp_path, capsys):
    def fake_metadata():
        return {"nvidia_smi": {"stdout": "private gpu"}}

    def fake_run_scenario(binary, salt, scenario):
        return {
            "scenario": {**benchmark.asdict(scenario), "prefix": "deadbeef"},
            "summary": _summary(42.0),
            "aggregate": _summary(42.0),
            "command": [str(binary), "--salt", salt],
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [{"exit_code": 0, "result": {"ok": True}}],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0, "matches": [{"key": "secret-key"}]},
        }

    monkeypatch.setattr(benchmark, "collect_hardware_metadata", fake_metadata)
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)
    sanitized_output = tmp_path / "summary.json"

    exit_code = benchmark.main(
        [
            "--binary",
            r"D:\private\miner.exe",
            "--salt",
            "private-salt",
            "--backend",
            "cuda",
            "--seconds",
            "1",
            "--sanitized-output",
            str(sanitized_output),
        ]
    )

    assert exit_code == 0
    sanitized = json.loads(sanitized_output.read_text(encoding="utf-8"))
    encoded = json.dumps(sanitized)
    assert sanitized["schema"] == "xenblocks.hashapi.benchmark-summary.v1"
    assert sanitized["runs"][0]["summary"]["hashrate"] == 42.0
    assert "binary" not in sanitized
    assert "hardware" not in sanitized
    assert r"D:\private" not in encoded
    assert "private gpu" not in encoded
    assert "private-salt" not in encoded
    assert "deadbeef" not in encoded
    assert "secret-key" not in encoded
    capsys.readouterr()


def test_main_can_print_recommendations_only(monkeypatch, tmp_path, capsys):
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
            "--output",
            str(output),
            "--recommendations-only",
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert list(stdout) == ["batch_size_by_difficulty", "stable_spread_pct"]
    assert stdout["batch_size_by_difficulty"][0]["batch_size"] == 2
    assert "runs" in json.loads(output.read_text(encoding="utf-8"))


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


def test_main_combines_custom_scan_scenarios(monkeypatch, tmp_path):
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
            "--scan-difficulty",
            "1",
            "--scan-difficulty",
            "8",
            "--scan-batch-size",
            "512",
            "--scan-batch-size",
            "1024",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured_names == ["cuda-scan-d1-b512", "cuda-scan-d1-b1024", "cuda-scan-d8-b512", "cuda-scan-d8-b1024"]


def test_main_rejects_partial_custom_scan(capsys):
    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--scan-difficulty",
            "1",
        ]
    )

    assert exit_code == 2
    assert "--scan-difficulty and --scan-batch-size must be used together" in capsys.readouterr().err


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
