"""Tests for the Hash API benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.hash_api_benchmark as benchmark


def _summary(hashrate: float, attempts: int = 1, ok: bool = True, timings: dict | None = None, **extra) -> dict:
    summary = {
        "name": "cuda-test",
        "backend": "cuda",
        "device_id": 0,
        "difficulty": 1,
        "batch_size": 2,
        "batch_size_min": 2,
        "batch_size_max": 2,
        "attempts": attempts,
        "first_block_workers": 0,
        "first_block_dynamic_chunk_size": 0,
        "first_block_dynamic_chunk_auto": False,
        "first_block_worker_count": 0,
        "first_block_chunk_size": 0,
        "first_block_dynamic_chunk_size_min": 0,
        "first_block_dynamic_chunk_size_max": 0,
        "first_block_chunk_size_min": 0,
        "first_block_chunk_size_max": 0,
        "gpu_first_blocks": False,
        "elapsed_ms": 1000.0,
        "hashrate": hashrate,
        "timings": timings or {},
        "matches": 0,
        "ok": ok,
        "error": "" if ok else "failed",
        "warmup": 1,
        "repeat": 2,
    }
    summary.update(extra)
    if "batch_size_min" not in extra:
        summary["batch_size_min"] = summary["batch_size"]
    if "batch_size_max" not in extra:
        summary["batch_size_max"] = summary["batch_size"]
    return summary


def _stub_metadata(monkeypatch):
    environment_calls = {"count": 0}

    def fake_environment_metadata():
        environment_calls["count"] += 1
        cpu_load_pct = 10.0 + environment_calls["count"]
        return {
            "available": True,
            "cpu_load_pct": cpu_load_pct,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        }

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(benchmark, "collect_environment_metadata", fake_environment_metadata)
    return environment_calls


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


def test_parse_scenario_can_disable_xuni_matching():
    scenario = benchmark.parse_scenario(
        "backend=cuda,difficulty=1,batch_size=2,seconds=1,allow_xuni=false",
    )

    assert scenario.allow_xuni is False


def test_parse_scenario_supports_fixed_key():
    fixed_key = "0" * 64
    scenario = benchmark.parse_scenario(
        f"name=cuda-fixed,backend=cuda,difficulty=8,batch_size=1,seconds=3,key={fixed_key}",
    )

    assert scenario.name == "cuda-fixed"
    assert scenario.key == fixed_key


def test_parse_scenario_supports_gpu_first_blocks_flag():
    scenario = benchmark.parse_scenario(
        "backend=cuda,difficulty=8,batch_size=64,seconds=3,gpu_first_blocks=true",
    )

    assert scenario.gpu_first_blocks is True


def test_parse_scenario_supports_auto_batch_size_flag():
    scenario = benchmark.parse_scenario(
        "backend=cuda,difficulty=8,batch_size=0,seconds=3,auto_batch_size=true",
    )

    assert scenario.auto_batch_size is True
    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)
    assert "--auto-batch-size" in command
    assert "--batch-size" not in command


def test_parse_scenario_supports_difficulty_sequence():
    scenario = benchmark.parse_scenario(
        "backend=cuda,difficulty_sequence=1|8|1|8,batch_size=512,seconds=2",
    )

    assert scenario.name == "cuda-seq-d1x8x1x8-b512"
    assert scenario.difficulty == 1
    assert scenario.difficulty_sequence == (1, 8, 1, 8)


def test_parse_scenario_supports_batch_size_sequence():
    scenario = benchmark.parse_scenario(
        "backend=cuda,difficulty_sequence=1|8|64,batch_size_sequence=2048|3072|3072,seconds=2",
    )

    assert scenario.name == "cuda-seq-d1x8x64-bseq-2048x3072x3072"
    assert scenario.difficulty == 1
    assert scenario.batch_size == 2048
    assert scenario.difficulty_sequence == (1, 8, 64)
    assert scenario.batch_size_sequence == (2048, 3072, 3072)


def test_parse_scenario_rejects_malformed_key_value_pairs():
    try:
        benchmark.parse_scenario("backend=cuda,difficulty_sequence=1,8,batch_size=512")
    except ValueError as exc:
        assert "use difficulty_sequence=1|8|1|8 inside --scenario" in str(exc)
    else:
        raise AssertionError("expected malformed scenario rejection")


def test_parse_difficulty_sequence_rejects_invalid_values():
    for text in ["", "1,,8", "1,zero", "1,0"]:
        try:
            benchmark.parse_difficulty_sequence(text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid difficulty sequence rejection for {text!r}")


def test_parse_batch_size_sequence_rejects_invalid_values():
    for text in ["", "512,,1024", "512,zero", "512,0"]:
        try:
            benchmark.parse_batch_size_sequence(text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid batch-size sequence rejection for {text!r}")


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


def test_preset_scenarios_builds_difficulty_sequence_matrix():
    scenarios = benchmark.preset_scenarios(
        "difficulty-sequence",
        seconds=2,
        backend="cuda",
        device=0,
        warmup=1,
        repeat=2,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-difficulty-sequence-d1x1x1x1-b512",
        "cuda-difficulty-sequence-d1x8x1x8-b512",
        "cuda-difficulty-sequence-d8x64x8x64-b512",
    ]
    assert [scenario.difficulty_sequence for scenario in scenarios] == [
        (1, 1, 1, 1),
        (1, 8, 1, 8),
        (8, 64, 8, 64),
    ]
    assert [scenario.difficulty for scenario in scenarios] == [1, 1, 8]
    assert all(scenario.batch_size == 512 for scenario in scenarios)
    assert all(scenario.warmup == 1 for scenario in scenarios)
    assert all(scenario.repeat == 2 for scenario in scenarios)


def test_preset_scenarios_builds_isolation_matrix():
    scenarios = benchmark.preset_scenarios(
        "isolation",
        seconds=4,
        backend="cuda",
        device=1,
        warmup=1,
        repeat=3,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-isolation-generated-d8-b2048",
        "cuda-isolation-fixed-d8-b1",
    ]
    assert [scenario.key for scenario in scenarios] == ["", "0" * 64]
    assert [scenario.batch_size for scenario in scenarios] == [2048, 1]
    assert all(scenario.difficulty == 8 for scenario in scenarios)
    assert all(scenario.seconds == 4 for scenario in scenarios)
    assert all(scenario.device == 1 for scenario in scenarios)
    assert all(scenario.warmup == 1 for scenario in scenarios)
    assert all(scenario.repeat == 3 for scenario in scenarios)


def test_scan_scenarios_builds_custom_matrix():
    scenarios = benchmark.scan_scenarios(
        difficulties=[1, 8],
        batch_sizes=[512, 1024],
        first_block_workers=[],
        first_block_dynamic_chunk_sizes=[],
        detailed_timings=False,
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


def test_scan_scenarios_treats_zero_batch_size_as_auto_batch():
    scenarios = benchmark.scan_scenarios(
        difficulties=[4096],
        batch_sizes=[0],
        first_block_workers=[],
        first_block_dynamic_chunk_sizes=[],
        detailed_timings=False,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=1,
        repeat=2,
        first_block_dynamic_chunk_auto=True,
        scan_gpu_first_blocks=True,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-scan-d4096-bauto-fbda",
        "cuda-scan-d4096-bauto-fbda-gfb",
    ]
    assert [scenario.batch_size for scenario in scenarios] == [0, 0]
    assert [scenario.auto_batch_size for scenario in scenarios] == [True, True]
    assert [scenario.gpu_first_blocks for scenario in scenarios] == [False, True]
    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenarios[1])
    assert "--auto-batch-size" in command
    assert "--batch-size" not in command
    assert "--gpu-first-blocks" in command


def test_scan_scenarios_can_scan_first_block_workers():
    scenarios = benchmark.scan_scenarios(
        difficulties=[8],
        batch_sizes=[1024],
        first_block_workers=[0, 4],
        first_block_dynamic_chunk_sizes=[],
        detailed_timings=False,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-scan-d8-b1024",
        "cuda-scan-d8-b1024-fbw4",
    ]
    assert [scenario.first_block_workers for scenario in scenarios] == [0, 4]


def test_scan_scenarios_can_scan_first_block_dynamic_chunks():
    scenarios = benchmark.scan_scenarios(
        difficulties=[8],
        batch_sizes=[1024],
        first_block_workers=[0],
        first_block_dynamic_chunk_sizes=[0, 64],
        detailed_timings=False,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-scan-d8-b1024",
        "cuda-scan-d8-b1024-fbd64",
    ]
    assert [scenario.first_block_dynamic_chunk_size for scenario in scenarios] == [0, 64]


def test_scan_scenarios_can_enable_first_block_dynamic_chunk_auto():
    scenarios = benchmark.scan_scenarios(
        difficulties=[8],
        batch_sizes=[1024],
        first_block_workers=[0],
        first_block_dynamic_chunk_sizes=[],
        detailed_timings=False,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
        first_block_dynamic_chunk_auto=True,
    )

    assert [scenario.name for scenario in scenarios] == ["cuda-scan-d8-b1024-fbda"]
    assert [scenario.first_block_dynamic_chunk_auto for scenario in scenarios] == [True]


def test_scan_scenarios_can_enable_detailed_timings():
    scenarios = benchmark.scan_scenarios(
        difficulties=[8],
        batch_sizes=[1024],
        first_block_workers=[0],
        first_block_dynamic_chunk_sizes=[],
        detailed_timings=True,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == ["cuda-scan-d8-b1024"]
    assert [scenario.detailed_timings for scenario in scenarios] == [True]


def test_difficulty_sequence_scenarios_build_custom_matrix():
    scenarios = benchmark.difficulty_sequence_scenarios(
        sequences=[(1, 1, 1, 1), (1, 8, 1, 8)],
        batch_sizes=[512, 1024],
        detailed_timings=False,
        first_block_dynamic_chunk_auto=False,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-difficulty-sequence-d1x1x1x1-b512",
        "cuda-difficulty-sequence-d1x1x1x1-b1024",
        "cuda-difficulty-sequence-d1x8x1x8-b512",
        "cuda-difficulty-sequence-d1x8x1x8-b1024",
    ]
    assert [scenario.difficulty_sequence for scenario in scenarios] == [
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 8, 1, 8),
        (1, 8, 1, 8),
    ]
    assert [scenario.difficulty for scenario in scenarios] == [1, 1, 1, 1]
    assert [scenario.batch_size for scenario in scenarios] == [512, 1024, 512, 1024]
    assert all(scenario.device == 1 for scenario in scenarios)
    assert all(scenario.warmup == 2 for scenario in scenarios)
    assert all(scenario.repeat == 4 for scenario in scenarios)


def test_difficulty_sequence_scenarios_can_enable_detailed_timings():
    scenarios = benchmark.difficulty_sequence_scenarios(
        sequences=[(8, 64, 8, 64)],
        batch_sizes=[512],
        detailed_timings=True,
        first_block_dynamic_chunk_auto=False,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == ["cuda-difficulty-sequence-d8x64x8x64-b512"]
    assert [scenario.detailed_timings for scenario in scenarios] == [True]


def test_automatic_batch_difficulty_sequence_scenarios_build_custom_matrix():
    scenarios = benchmark.automatic_batch_difficulty_sequence_scenarios(
        sequences=[(1, 8, 64)],
        detailed_timings=True,
        first_block_dynamic_chunk_auto=True,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == ["cuda-difficulty-sequence-d1x8x64-bauto"]
    assert scenarios[0].difficulty == 1
    assert scenarios[0].batch_size == 0
    assert scenarios[0].difficulty_sequence == (1, 8, 64)
    assert scenarios[0].auto_batch_size is True
    assert scenarios[0].detailed_timings is True
    assert scenarios[0].first_block_dynamic_chunk_auto is True


def test_paired_sequence_scenarios_build_variable_shape_matrix():
    scenarios = benchmark.paired_sequence_scenarios(
        difficulty_sequences=[(1, 8, 64)],
        batch_size_sequences=[(2048, 3072, 3072)],
        detailed_timings=True,
        first_block_dynamic_chunk_auto=True,
        seconds=3,
        backend="cuda",
        device=1,
        warmup=2,
        repeat=4,
    )

    assert [scenario.name for scenario in scenarios] == [
        "cuda-difficulty-sequence-d1x8x64-bseq-2048x3072x3072",
    ]
    assert scenarios[0].difficulty == 1
    assert scenarios[0].batch_size == 2048
    assert scenarios[0].difficulty_sequence == (1, 8, 64)
    assert scenarios[0].batch_size_sequence == (2048, 3072, 3072)
    assert scenarios[0].detailed_timings is True
    assert scenarios[0].first_block_dynamic_chunk_auto is True


def test_paired_sequence_scenarios_reject_mismatched_sequence_lengths():
    try:
        benchmark.paired_sequence_scenarios(
            difficulty_sequences=[(1, 8, 64)],
            batch_size_sequences=[(2048, 3072)],
            detailed_timings=False,
            first_block_dynamic_chunk_auto=False,
            seconds=3,
            backend="cuda",
            device=1,
            warmup=2,
            repeat=4,
        )
    except ValueError as exc:
        assert "difficulty sequence and batch-size sequence lengths must match" in str(exc)
    else:
        raise AssertionError("expected mismatched paired sequence rejection")


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
    assert aggregate["stable"] is False
    assert aggregate["stable_spread_pct"] == 10.0
    assert aggregate["attempts"] == 60
    assert aggregate["first_block_workers"] == 0
    assert aggregate["gpu_first_blocks"] is False
    assert aggregate["elapsed_ms"] == 3000.0
    assert aggregate["ms_per_attempt"] == 50.0
    assert aggregate["difficulty_mode"] == "fixed"
    assert aggregate["difficulty_sequence"] == []
    assert aggregate["difficulty_changes"] == 0
    assert aggregate["batch_size_mode"] == "fixed"
    assert aggregate["batch_size_sequence"] == []
    assert aggregate["batch_size_changes"] == 0
    assert aggregate["batch_size_min"] == 2
    assert aggregate["batch_size_max"] == 2
    assert aggregate["key_mode"] == "generated"
    assert aggregate["warmup"] == 1
    assert aggregate["repeat"] == 3
    assert aggregate["sample_count"] == 3
    assert aggregate["ok_sample_count"] == 3
    assert aggregate["ok"] is True


def test_summarize_iterations_reports_gpu_first_blocks_flag():
    scenario = benchmark.BenchmarkScenario(
        name="cuda-gpu-first-blocks",
        backend="cuda",
        difficulty=8,
        batch_size=64,
        seconds=1,
        gpu_first_blocks=True,
    )

    aggregate = benchmark.summarize_iterations(scenario, [_summary(100.0, attempts=100)])

    assert aggregate["gpu_first_blocks"] is True


def test_summarize_iterations_marks_stable_repeated_samples():
    scenario = benchmark.BenchmarkScenario(
        name="cuda-stable",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        repeat=3,
        first_block_workers=4,
    )

    aggregate = benchmark.summarize_iterations(
        scenario,
        [_summary(100.0, attempts=100), _summary(104.0, attempts=104), _summary(102.0, attempts=102)],
    )

    assert aggregate["hashrate_spread_pct"] < aggregate["stable_spread_pct"]
    assert aggregate["stable"] is True
    assert aggregate["first_block_workers"] == 4
    assert aggregate["sample_count"] == 3
    assert aggregate["ok_sample_count"] == 3


def test_summarize_iterations_reports_sequence_metadata():
    scenario = benchmark.BenchmarkScenario(
        name="cuda-sequence",
        backend="cuda",
        difficulty=1,
        difficulty_sequence=(1, 8, 1, 8),
        batch_size=512,
        seconds=1,
        repeat=2,
    )

    aggregate = benchmark.summarize_iterations(
        scenario,
        [
            _summary(
                10.0,
                attempts=10,
                batch_size=512,
                first_block_dynamic_chunk_size_min=0,
                first_block_dynamic_chunk_size_max=32,
                first_block_chunk_size_min=16,
                first_block_chunk_size_max=256,
            ),
            _summary(
                20.0,
                attempts=20,
                batch_size=512,
                first_block_dynamic_chunk_size_min=0,
                first_block_dynamic_chunk_size_max=32,
                first_block_chunk_size_min=16,
                first_block_chunk_size_max=256,
            ),
        ],
    )

    assert aggregate["difficulty"] == 1
    assert aggregate["difficulty_mode"] == "sequence"
    assert aggregate["difficulty_sequence"] == [1, 8, 1, 8]
    assert aggregate["difficulty_changes"] == 3
    assert aggregate["batch_size_mode"] == "fixed"
    assert aggregate["batch_size_min"] == 512
    assert aggregate["batch_size_max"] == 512
    assert aggregate["first_block_dynamic_chunk_size_min"] == 0
    assert aggregate["first_block_dynamic_chunk_size_max"] == 32
    assert aggregate["first_block_chunk_size_min"] == 16
    assert aggregate["first_block_chunk_size_max"] == 256


def test_summarize_iterations_marks_nonzero_process_exit_invalid():
    scenario = benchmark.BenchmarkScenario(
        name="cuda-crash",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        repeat=2,
    )
    ok_summary = _summary(100.0, attempts=100)
    crashed_summary = {
        **_summary(120.0, attempts=120),
        "ok": False,
        "error": "process exited with code 3221225477",
        "process_exit_code": 3221225477,
    }

    aggregate = benchmark.summarize_iterations(scenario, [ok_summary, crashed_summary])

    assert aggregate["ok"] is False
    assert aggregate["attempts"] == 100
    assert aggregate["hashrate"] == 100.0
    assert aggregate["stable"] is False
    assert aggregate["sample_count"] == 2
    assert aggregate["ok_sample_count"] == 1
    assert aggregate["error"] == "process exited with code 3221225477"


def test_summarize_iterations_reports_fixed_key_mode():
    scenario = benchmark.BenchmarkScenario(
        name="cuda-fixed",
        backend="cuda",
        difficulty=8,
        batch_size=1,
        seconds=1,
        key="0" * 64,
    )

    aggregate = benchmark.summarize_iterations(
        scenario,
        [_summary(10.0, attempts=1)],
    )

    assert aggregate["key_mode"] == "fixed"


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
    assert aggregate["timing_per_attempt"]["compute_ms"] == 4.0
    assert aggregate["timing_per_attempt"]["input_ms"] == 2.0
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


def test_timing_analysis_treats_sub_timings_as_nested_timing():
    analysis = benchmark.timing_analysis(
        {
            "input_ms": 6.0,
            "keygen_ms": 0.5,
            "setup_ms": 5.5,
            "setup_normalize_cpu_ms": 0.1,
            "setup_activate_cpu_ms": 0.2,
            "setup_device_info_cpu_ms": 0.3,
            "setup_params_cpu_ms": 0.4,
            "setup_backend_init_cpu_ms": 0.5,
            "first_block_ms": 5.0,
            "first_block_initial_hash_cpu_ms": 3.0,
            "first_block_digest_cpu_ms": 2.0,
            "first_block_max_worker_ms": 4.0,
            "first_block_thread_launch_ms": 0.5,
            "first_block_max_worker_start_ms": 0.4,
            "first_block_worker_start_span_ms": 0.3,
            "first_block_max_worker_finish_ms": 4.5,
            "first_block_worker_finish_span_ms": 4.4,
            "compute_ms": 4.0,
            "kernel_ms": 9.0,
            "host_to_device_ms": 11.0,
            "gpu_first_block_ms": 10.0,
            "device_to_host_ms": 12.0,
            "finalize_ms": 3.0,
            "finalize_hash_ms": 8.0,
            "argon2_finalize_ms": 6.0,
            "base64_ms": 5.0,
            "match_ms": 7.0,
            "total_ms": 10.0,
        }
    )

    assert analysis["dominant_stage"] == "input_ms"
    assert "first_block_initial_hash_cpu_ms" not in analysis["stage_pct"]
    assert "first_block_digest_cpu_ms" not in analysis["stage_pct"]
    assert "first_block_max_worker_ms" not in analysis["stage_pct"]
    assert "first_block_thread_launch_ms" not in analysis["stage_pct"]
    assert "first_block_max_worker_start_ms" not in analysis["stage_pct"]
    assert "first_block_worker_start_span_ms" not in analysis["stage_pct"]
    assert "first_block_max_worker_finish_ms" not in analysis["stage_pct"]
    assert "first_block_worker_finish_span_ms" not in analysis["stage_pct"]
    assert "setup_normalize_cpu_ms" not in analysis["stage_pct"]
    assert "setup_activate_cpu_ms" not in analysis["stage_pct"]
    assert "setup_device_info_cpu_ms" not in analysis["stage_pct"]
    assert "setup_params_cpu_ms" not in analysis["stage_pct"]
    assert "setup_backend_init_cpu_ms" not in analysis["stage_pct"]
    assert "kernel_ms" not in analysis["stage_pct"]
    assert "host_to_device_ms" not in analysis["stage_pct"]
    assert "gpu_first_block_ms" not in analysis["stage_pct"]
    assert "device_to_host_ms" not in analysis["stage_pct"]
    assert "finalize_hash_ms" not in analysis["stage_pct"]
    assert "argon2_finalize_ms" not in analysis["stage_pct"]
    assert "base64_ms" not in analysis["stage_pct"]
    assert "match_ms" not in analysis["stage_pct"]
    assert analysis["stage_pct"]["finalize_ms"] == 30.0
    assert analysis["nested_stage_pct"]["first_block_initial_hash_cpu_ms"] == 60.0
    assert analysis["nested_stage_pct"]["first_block_digest_cpu_ms"] == 40.0
    assert analysis["nested_stage_pct"]["first_block_max_worker_ms"] == 80.0
    assert analysis["nested_stage_pct"]["first_block_thread_launch_ms"] == 10.0
    assert analysis["nested_stage_pct"]["first_block_max_worker_start_ms"] == 8.0
    assert analysis["nested_stage_pct"]["first_block_worker_start_span_ms"] == 6.0
    assert analysis["nested_stage_pct"]["first_block_max_worker_finish_ms"] == 90.0
    assert round(analysis["nested_stage_pct"]["first_block_worker_finish_span_ms"], 6) == 88.0
    assert analysis["nested_stage_pct"]["setup_activate_cpu_ms"] == 0.2 / 5.5 * 100.0
    assert analysis["nested_stage_pct"]["kernel_ms"] == 9.0 / 4.0 * 100.0
    assert analysis["nested_stage_pct"]["gpu_first_block_ms"] == 10.0 / 4.0 * 100.0
    assert analysis["nested_stage_pct"]["finalize_hash_ms"] == 8.0 / 3.0 * 100.0
    assert analysis["nested_stage_pct"]["argon2_finalize_ms"] == 200.0
    assert analysis["input_explained_ms"] == 5.5
    assert analysis["input_residual_ms"] == 0.5
    assert analysis["input_explained_to_input"] == 5.5 / 6.0
    assert analysis["input_residual_pct"] == 0.5 / 6.0 * 100.0
    assert analysis["first_block_cpu_sum_ms"] == 5.0
    assert analysis["first_block_cpu_sum_to_wall"] == 1.0
    assert analysis["first_block_worker_wall_to_wall"] == 0.8
    assert analysis["first_block_scheduling_overhead_ms"] == 1.0
    assert analysis["first_block_finish_wall_to_wall"] == 0.9
    assert analysis["first_block_post_worker_overhead_ms"] == 0.5


def test_timing_analysis_omits_nested_percentages_without_parent_timing():
    analysis = benchmark.timing_analysis(
        {
            "compute_ms": 0.0,
            "kernel_ms": 9.0,
            "first_block_ms": 0.0,
            "first_block_digest_cpu_ms": 2.0,
            "first_block_max_worker_ms": 4.0,
            "total_ms": 10.0,
        }
    )

    assert analysis["nested_stage_pct"] == {}
    assert analysis["input_explained_ms"] == 0.0
    assert analysis["input_residual_ms"] == 0.0
    assert analysis["input_explained_to_input"] == 0.0
    assert analysis["input_residual_pct"] == 0.0
    assert analysis["first_block_cpu_sum_ms"] == 2.0
    assert analysis["first_block_cpu_sum_to_wall"] == 0.0
    assert analysis["first_block_worker_wall_to_wall"] == 0.0
    assert analysis["first_block_scheduling_overhead_ms"] == 0.0
    assert analysis["first_block_finish_wall_to_wall"] == 0.0
    assert analysis["first_block_post_worker_overhead_ms"] == 0.0


def test_collect_build_metadata_reads_public_safe_cmake_cache(tmp_path, monkeypatch):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        "\n".join(
            [
                "CMAKE_BUILD_TYPE:STRING=Release",
                "CMAKE_CUDA_ARCHITECTURES:STRING=75;80;86",
                "CMAKE_CUDA_COMPILER:FILEPATH=<private-cuda>/bin/nvcc.exe",
                "CMAKE_GENERATOR:INTERNAL=Ninja",
                "VCPKG_TARGET_TRIPLET:STRING=x64-windows",
            ]
        ),
        encoding="utf-8",
    )

    def fake_compiler_metadata(compiler):
        assert compiler.endswith("nvcc.exe")
        return {
            "basename": "nvcc.exe",
            "available": True,
            "release": "12.8",
            "version": "12.8.93",
        }

    monkeypatch.setattr(benchmark, "cuda_compiler_metadata", fake_compiler_metadata)

    metadata = benchmark.collect_build_metadata(cache)

    assert metadata["provided"] is True
    assert metadata["available"] is True
    assert metadata["generator"] == "Ninja"
    assert metadata["build_type"] == "Release"
    assert metadata["cuda_architectures"] == ["75", "80", "86"]
    assert metadata["vcpkg_target_triplet"] == "x64-windows"
    assert metadata["cuda_compiler"]["basename"] == "nvcc.exe"
    assert metadata["cuda_compiler"]["release"] == "12.8"


def test_collect_build_metadata_handles_missing_cache(tmp_path):
    metadata = benchmark.collect_build_metadata(tmp_path / "missing")

    assert metadata == {
        "provided": True,
        "available": False,
        "error": "CMakeCache.txt not found",
    }


def test_parse_scenario_supports_detailed_timings():
    scenario = benchmark.parse_scenario(
        "name=diag,backend=cuda,difficulty=8,batch_size=2048,seconds=1,detailed_timings=true"
    )

    assert scenario.detailed_timings is True


def test_parse_scenario_supports_first_block_workers():
    scenario = benchmark.parse_scenario(
        "name=diag,backend=cuda,difficulty=8,batch_size=2048,seconds=1,first_block_workers=4"
    )

    assert scenario.first_block_workers == 4


def test_parse_scenario_supports_first_block_dynamic_chunk_size():
    scenario = benchmark.parse_scenario(
        "name=diag,backend=cuda,difficulty=8,batch_size=2048,seconds=1,first_block_dynamic_chunk_size=64"
    )

    assert scenario.first_block_dynamic_chunk_size == 64


def test_parse_scenario_supports_first_block_dynamic_chunk_auto():
    scenario = benchmark.parse_scenario(
        "name=diag,backend=cuda,difficulty=8,batch_size=2048,seconds=1,first_block_dynamic_chunk_auto=true"
    )

    assert scenario.first_block_dynamic_chunk_auto is True


def test_build_hash_command_adds_detailed_timings_flag():
    scenario = benchmark.BenchmarkScenario(
        name="diag",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        detailed_timings=True,
    )

    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert "--detailed-timings" in command


def test_build_hash_command_adds_first_block_workers_when_set():
    scenario = benchmark.BenchmarkScenario(
        name="diag",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        first_block_workers=4,
    )

    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert "--first-block-workers" in command
    assert command[command.index("--first-block-workers") + 1] == "4"


def test_build_hash_command_adds_first_block_dynamic_chunk_size_when_set():
    scenario = benchmark.BenchmarkScenario(
        name="diag",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        first_block_dynamic_chunk_size=64,
    )

    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert "--first-block-dynamic-chunk-size" in command
    assert command[command.index("--first-block-dynamic-chunk-size") + 1] == "64"


def test_build_hash_command_adds_first_block_dynamic_chunk_auto_when_set():
    scenario = benchmark.BenchmarkScenario(
        name="diag",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        first_block_dynamic_chunk_auto=True,
    )

    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert "--first-block-dynamic-chunk-auto" in command


def test_build_hash_command_adds_batch_size_sequence_when_set():
    scenario = benchmark.BenchmarkScenario(
        name="diag",
        backend="cuda",
        difficulty=1,
        batch_size=2048,
        seconds=1,
        difficulty_sequence=(1, 8, 64),
        batch_size_sequence=(2048, 3072, 3072),
    )

    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert "--batch-size-sequence" in command
    assert command[command.index("--batch-size-sequence") + 1] == "2048,3072,3072"


def test_build_hash_command_adds_auto_batch_size_when_set():
    scenario = benchmark.BenchmarkScenario(
        name="diag",
        backend="cuda",
        difficulty=1,
        batch_size=0,
        seconds=1,
        difficulty_sequence=(1, 8, 64),
        auto_batch_size=True,
    )

    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert "--auto-batch-size" in command
    assert "--batch-size" not in command


def test_summarize_result_uses_selected_dynamic_chunk_size():
    scenario = benchmark.BenchmarkScenario(
        name="diag",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        first_block_dynamic_chunk_auto=True,
    )
    summary = benchmark.summarize_result(
        scenario,
        {
            "backend": "cuda",
            "device_id": 0,
            "batch_size": 2048,
            "batch_size_min": 1024,
            "batch_size_max": 3072,
            "attempts": 2048,
            "first_block_dynamic_chunk_size": 32,
            "first_block_dynamic_chunk_auto": True,
            "first_block_worker_count": 8,
            "first_block_chunk_size": 32,
            "first_block_dynamic_chunk_size_min": 16,
            "first_block_dynamic_chunk_size_max": 32,
            "first_block_chunk_size_min": 16,
            "first_block_chunk_size_max": 256,
            "elapsed_ms": 1000.0,
            "hashrate": 2048.0,
            "timings": {},
            "matches": [],
            "ok": True,
            "error": "",
        },
    )

    assert summary["first_block_dynamic_chunk_auto"] is True
    assert summary["batch_size_min"] == 1024
    assert summary["batch_size_max"] == 3072
    assert summary["first_block_dynamic_chunk_size"] == 32
    assert summary["first_block_chunk_size"] == 32
    assert summary["first_block_dynamic_chunk_size_min"] == 16
    assert summary["first_block_dynamic_chunk_size_max"] == 32
    assert summary["first_block_chunk_size_min"] == 16
    assert summary["first_block_chunk_size_max"] == 256


def test_summarize_iterations_reports_median_timing_per_attempt():
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
            _summary(10.0, attempts=10, timings={"input_ms": 100.0}),
            _summary(20.0, attempts=20, timings={"input_ms": 100.0}),
            _summary(30.0, attempts=50, timings={"input_ms": 100.0}),
        ],
    )

    assert aggregate["timing_per_attempt"]["input_ms"] == 5.0


def test_build_recommendations_selects_best_batch_per_difficulty():
    runs = [
        {
            "summary": {
                **_summary(100.0),
                "name": "d1-b64",
                "difficulty": 1,
                "batch_size": 64,
                "batch_size_min": 64,
                "batch_size_max": 64,
            }
        },
        {
            "summary": {
                **_summary(150.0),
                "name": "d1-b128",
                "difficulty": 1,
                "batch_size": 128,
                "batch_size_min": 128,
                "batch_size_max": 128,
                "first_block_workers": 4,
                "first_block_dynamic_chunk_size": 64,
                "first_block_dynamic_chunk_auto": True,
                "first_block_worker_count": 4,
                "first_block_chunk_size": 32,
                "first_block_dynamic_chunk_size_min": 64,
                "first_block_dynamic_chunk_size_max": 64,
                "first_block_chunk_size_min": 32,
                "first_block_chunk_size_max": 32,
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
                "batch_size_min": 64,
                "batch_size_max": 64,
                "hashrate_spread_pct": 15.0,
                "timing_analysis": {"dominant_stage": "compute_ms", "dominant_stage_pct": 55.0},
            }
        },
        {"summary": {**_summary(90.0, ok=False), "name": "d8-b128", "difficulty": 8, "batch_size": 128}},
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["report_ok"] is False
    assert recommendations["run_count"] == 4
    assert recommendations["valid_run_count"] == 3
    assert recommendations["invalid_run_count"] == 1
    assert recommendations["invalid_scenarios"] == ["d8-b128"]
    assert recommendations["stable_spread_pct"] == 10.0
    assert recommendations["batch_size_by_difficulty"] == [
        {
            "backend": "cuda",
            "device_id": 0,
            "difficulty": 1,
            "batch_size": 128,
            "batch_size_min": 128,
            "batch_size_max": 128,
            "first_block_workers": 4,
            "first_block_dynamic_chunk_size": 64,
            "first_block_dynamic_chunk_auto": True,
            "first_block_worker_count": 4,
            "first_block_chunk_size": 32,
            "first_block_dynamic_chunk_size_min": 64,
            "first_block_dynamic_chunk_size_max": 64,
            "first_block_chunk_size_min": 32,
            "first_block_chunk_size_max": 32,
            "gpu_first_blocks": False,
            "median_hashrate": 150.0,
            "min_hashrate": 150.0,
            "max_hashrate": 150.0,
            "hashrate_spread_pct": 5.0,
            "ms_per_attempt": 0.0,
            "stable": True,
            "warm_evidence": True,
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
            "batch_size_min": 64,
            "batch_size_max": 64,
            "first_block_workers": 0,
            "first_block_dynamic_chunk_size": 0,
            "first_block_dynamic_chunk_auto": False,
            "first_block_worker_count": 0,
            "first_block_chunk_size": 0,
            "first_block_dynamic_chunk_size_min": 0,
            "first_block_dynamic_chunk_size_max": 0,
            "first_block_chunk_size_min": 0,
            "first_block_chunk_size_max": 0,
            "gpu_first_blocks": False,
            "median_hashrate": 120.0,
            "min_hashrate": 120.0,
            "max_hashrate": 120.0,
            "hashrate_spread_pct": 15.0,
            "ms_per_attempt": 0.0,
            "stable": False,
            "warm_evidence": True,
            "selection_reason": "no_stable_candidate",
            "dominant_stage": "compute_ms",
            "dominant_stage_pct": 55.0,
            "scenario": "d8-b64",
        },
    ]
    assert recommendations["candidates_by_difficulty"][0]["difficulty"] == 1
    assert [item["batch_size"] for item in recommendations["candidates_by_difficulty"][0]["candidates"]] == [64, 128]
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["stable"] is True
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["first_block_workers"] == 4
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["first_block_dynamic_chunk_size"] == 64
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["first_block_dynamic_chunk_auto"] is True
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["first_block_worker_count"] == 4
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["first_block_chunk_size"] == 32
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["first_block_chunk_size_min"] == 32
    assert recommendations["candidates_by_difficulty"][0]["candidates"][1]["first_block_chunk_size_max"] == 32


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


def test_build_recommendations_ignores_sequence_runs():
    runs = [
        {"summary": {**_summary(100.0), "name": "d1-b512", "difficulty": 1, "batch_size": 512}},
        {
            "summary": {
                **_summary(200.0),
                "name": "d1x8-b512",
                "difficulty": 1,
                "difficulty_sequence": [1, 8, 1, 8],
                "batch_size": 512,
            }
        },
        {
            "summary": {
                **_summary(250.0),
                "name": "d1x8-bseq",
                "difficulty": 1,
                "difficulty_sequence": [1, 8, 64],
                "batch_size": 3072,
                "batch_size_sequence": [2048, 3072, 3072],
                "batch_size_min": 2048,
                "batch_size_max": 3072,
            }
        },
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["batch_size_by_difficulty"][0]["scenario"] == "d1-b512"
    assert len(recommendations["candidates_by_difficulty"][0]["candidates"]) == 1


def test_build_recommendations_ignores_fixed_key_runs():
    runs = [
        {"summary": {**_summary(100.0), "name": "d1-b512", "difficulty": 1, "batch_size": 512}},
        {
            "summary": {
                **_summary(1000.0),
                "name": "d1-fixed-b1",
                "difficulty": 1,
                "batch_size": 1,
                "key_mode": "fixed",
            }
        },
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["batch_size_by_difficulty"][0]["scenario"] == "d1-b512"
    assert len(recommendations["candidates_by_difficulty"][0]["candidates"]) == 1


def test_build_recommendations_ignores_process_exit_failures():
    runs = [
        {
            "summary": {
                **_summary(500.0, ok=False),
                "name": "d8-crashed",
                "difficulty": 8,
                "batch_size": 2048,
                "process_exit_code": 3221225477,
                "error": "process exited with code 3221225477",
            }
        }
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["report_ok"] is False
    assert recommendations["invalid_run_count"] == 1
    assert recommendations["invalid_scenarios"] == ["d8-crashed"]
    assert recommendations["batch_size_by_difficulty"] == []
    assert recommendations["candidates_by_difficulty"] == []


def test_build_recommendations_marks_clean_report():
    runs = [
        {"summary": {**_summary(100.0), "name": "d1-b512", "difficulty": 1, "batch_size": 512}},
        {
            "summary": {
                **_summary(120.0),
                "name": "d1-b1024",
                "difficulty": 1,
                "batch_size": 1024,
                "hashrate_spread_pct": 5.0,
            }
        },
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["report_ok"] is True
    assert recommendations["run_count"] == 2
    assert recommendations["valid_run_count"] == 2
    assert recommendations["invalid_run_count"] == 0
    assert recommendations["invalid_scenarios"] == []


def test_add_recommendation_quality_marks_low_trust_environment():
    recommendations = {"report_ok": True, "run_count": 1}
    environment = {
        "available": True,
        "benchmark_trust": "low",
        "high_cpu_load": True,
        "sample_count": 8,
    }

    annotated = benchmark.add_recommendation_quality(recommendations, environment)

    assert annotated["report_ok"] is True
    assert annotated["benchmark_trust"] == "low"
    assert annotated["high_cpu_load"] is True
    assert annotated["environment_available"] is True
    assert annotated["environment_sample_count"] == 8
    assert annotated["report_quality_ok"] is False
    assert annotated["report_quality_failure_reasons"] == [
        "low_benchmark_trust",
        "high_cpu_load",
    ]


def test_add_recommendation_quality_rejects_cold_report():
    recommendations = {
        "report_ok": True,
        "run_count": 1,
        "valid_run_count": 1,
        "warm_evidence_run_count": 0,
        "stable_run_count": 1,
        "cold_scenarios": ["cold-control"],
        "unstable_scenarios": [],
    }
    environment = {
        "available": True,
        "benchmark_trust": "normal",
        "high_cpu_load": False,
        "sample_count": 3,
    }

    annotated = benchmark.add_recommendation_quality(recommendations, environment)

    assert annotated["report_quality_ok"] is False
    assert annotated["report_quality_failure_reasons"] == ["missing_warm_evidence"]


def test_add_recommendation_quality_rejects_unstable_report():
    recommendations = {
        "report_ok": True,
        "run_count": 1,
        "valid_run_count": 1,
        "warm_evidence_run_count": 1,
        "stable_run_count": 0,
        "cold_scenarios": [],
        "unstable_scenarios": ["unstable-control"],
    }
    environment = {
        "available": True,
        "benchmark_trust": "normal",
        "high_cpu_load": False,
        "sample_count": 3,
    }

    annotated = benchmark.add_recommendation_quality(recommendations, environment)

    assert annotated["report_quality_ok"] is False
    assert annotated["report_quality_failure_reasons"] == ["unstable_runs"]


def test_add_recommendation_quality_reports_multiple_failure_reasons():
    recommendations = {
        "report_ok": False,
        "run_count": 2,
        "valid_run_count": 1,
        "warm_evidence_run_count": 1,
        "stable_run_count": 1,
        "invalid_run_count": 1,
        "cold_scenarios": [],
        "unstable_scenarios": ["unstable-control"],
    }
    environment = {
        "available": True,
        "benchmark_trust": "low",
        "high_cpu_load": True,
        "sample_count": 3,
    }

    annotated = benchmark.add_recommendation_quality(recommendations, environment)

    assert annotated["report_quality_ok"] is False
    assert annotated["report_quality_failure_reasons"] == [
        "invalid_runs",
        "low_benchmark_trust",
        "high_cpu_load",
        "missing_warm_evidence",
        "unstable_runs",
    ]


def test_build_recommendations_records_cold_scenarios():
    runs = [
        {
            "summary": {
                **_summary(100.0, warmup=0, repeat=1),
                "name": "cold-control",
                "difficulty": 4096,
                "batch_size": 797,
            }
        }
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["report_ok"] is True
    assert recommendations["valid_run_count"] == 1
    assert recommendations["warm_evidence_run_count"] == 0
    assert recommendations["stable_run_count"] == 1
    assert recommendations["cold_scenarios"] == ["cold-control"]
    assert recommendations["unstable_scenarios"] == []
    assert recommendations["batch_size_by_difficulty"][0]["warm_evidence"] is False


def test_build_recommendations_records_unstable_scenarios():
    runs = [
        {
            "summary": {
                **_summary(100.0, hashrate_spread_pct=29.0, stable=False),
                "name": "unstable-control",
                "difficulty": 4096,
                "batch_size": 797,
            }
        }
    ]

    recommendations = benchmark.build_recommendations(runs)

    assert recommendations["report_ok"] is True
    assert recommendations["valid_run_count"] == 1
    assert recommendations["warm_evidence_run_count"] == 1
    assert recommendations["stable_run_count"] == 0
    assert recommendations["cold_scenarios"] == []
    assert recommendations["unstable_scenarios"] == ["unstable-control"]


def test_build_sanitized_report_drops_private_fields():
    fixed_key = "0" * 64
    report = {
        "schema": "xenblocks.hashapi.benchmark.v1",
        "created_at_unix": 123.0,
        "host": {"system": "Windows", "machine": "private-host"},
        "hardware": {"nvidia_smi": {"stdout": "0, Private GPU, 999.99, 4096 MiB"}},
        "build": {
            "provided": True,
            "available": True,
            "build_type": "Release",
            "cuda_architectures": ["75", "86"],
            "cuda_compiler": {
                "path": "<private-cuda>/bin/nvcc.exe",
                "basename": "nvcc.exe",
                "release": "12.8",
                "version": "12.8.93",
            },
        },
        "binary": "<private-binary>",
        "salt": "private-salt",
        "environment": {
            "available": True,
            "cpu_load_pct": 95.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
        "presets": ["warm-short"],
        "recommendations": {"batch_size_by_difficulty": []},
        "runs": [
            {
                "scenario": {
                    "name": "cuda-test",
                    "backend": "cuda",
                    "difficulty": 1,
                    "difficulty_sequence": [1, 8, 1, 8],
                    "batch_size": 64,
                    "batch_size_sequence": [64, 128, 64, 128],
                    "seconds": 3,
                    "device": 0,
                    "warmup": 1,
                    "repeat": 2,
                    "prefix": "deadbeef",
                    "key": fixed_key,
                    "key_mode": "fixed",
                    "pattern": "XEN11",
                },
                "summary": _summary(42.0),
                "command": ["<private-binary>", "--salt", "private-salt"],
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
    assert sanitized["build"]["build_type"] == "Release"
    assert sanitized["build"]["cuda_architectures"] == ["75", "86"]
    assert sanitized["build"]["cuda_compiler"] == {
        "basename": "nvcc.exe",
        "release": "12.8",
        "version": "12.8.93",
    }
    assert sanitized["environment"] == {
        "available": True,
        "cpu_load_pct": 95.0,
        "high_cpu_load": True,
        "benchmark_trust": "low",
    }
    assert sanitized["privacy"]["sanitized"] is True
    assert sanitized["runs"][0]["scenario"]["prefix_length"] == 8
    assert sanitized["runs"][0]["scenario"]["difficulty_sequence"] == [1, 8, 1, 8]
    assert sanitized["runs"][0]["scenario"]["batch_size_sequence"] == [64, 128, 64, 128]
    assert sanitized["runs"][0]["scenario"]["key_mode"] == "fixed"
    assert "prefix" not in sanitized["runs"][0]["scenario"]
    assert "key" not in sanitized["runs"][0]["scenario"]
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
        "<private-binary>",
        "private-host",
        "Private GPU",
        "<private-cuda>",
        "private-salt",
        "deadbeef",
        fixed_key,
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
    environment_calls = _stub_metadata(monkeypatch)
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
    assert environment_calls["count"] == 8
    assert result["environment"] == {
        "available": True,
        "cpu_load_pct": 18.0,
        "start_cpu_load_pct": 11.0,
        "end_cpu_load_pct": 18.0,
        "sample_count": 8,
        "high_cpu_load": False,
        "benchmark_trust": "normal",
    }


def test_run_scenario_treats_nonzero_exit_with_valid_json_as_failure(monkeypatch):
    calls = {"count": 0}

    def fake_run(command, text, capture_output, check):
        calls["count"] += 1
        return SimpleNamespace(
            returncode=3221225477,
            stdout=json.dumps(
                {
                    "ok": True,
                    "backend": "cuda",
                    "device_id": 0,
                    "batch_size": 2048,
                    "attempts": 2048,
                    "elapsed_ms": 1000.0,
                    "hashrate": 2048.0,
                    "timings": {"compute_ms": 1.0},
                    "matches": [],
                    "error": "",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    _stub_metadata(monkeypatch)
    scenario = benchmark.BenchmarkScenario(
        name="cuda-crash",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        warmup=1,
        repeat=2,
    )

    result = benchmark.run_scenario(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert calls["count"] == 3
    assert result["exit_code"] == 2
    assert result["summary"]["ok"] is False
    assert result["summary"]["attempts"] == 0
    assert "process exited with code 3221225477" in result["summary"]["error"]
    assert result["iterations"][0]["result"]["ok"] is False
    assert result["iterations"][0]["result"]["process_exit_code"] == 3221225477


def test_run_scenario_treats_warmup_nonzero_exit_as_summary_failure(monkeypatch):
    calls = {"count": 0}

    def fake_run(command, text, capture_output, check):
        calls["count"] += 1
        returncode = 3221225477 if calls["count"] == 1 else 0
        return SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps(
                {
                    "ok": True,
                    "backend": "cuda",
                    "device_id": 0,
                    "batch_size": 2048,
                    "attempts": 2048,
                    "elapsed_ms": 1000.0,
                    "hashrate": 2048.0,
                    "timings": {"compute_ms": 1.0},
                    "matches": [],
                    "error": "",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    _stub_metadata(monkeypatch)
    scenario = benchmark.BenchmarkScenario(
        name="cuda-warmup-crash",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        warmup=1,
        repeat=1,
    )

    result = benchmark.run_scenario(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert result["exit_code"] == 2
    assert result["summary"]["ok"] is False
    assert result["summary"]["attempts"] == 2048
    assert result["summary"]["hashrate"] == 2048.0
    assert result["summary"]["process_exit_codes"] == [3221225477]
    assert "process exited with code 3221225477" in result["summary"]["error"]


def test_run_scenario_preflight_wait_can_skip_subprocess(monkeypatch):
    def fail_run(command, text, capture_output, check):
        raise AssertionError("low-trust preflight should skip subprocess launch")

    monkeypatch.setattr(benchmark.subprocess, "run", fail_run)
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 99.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )
    scenario = benchmark.BenchmarkScenario(
        name="cuda-low-trust",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        warmup=0,
        repeat=1,
    )

    result = benchmark.run_scenario(
        Path("miner"),
        benchmark.DEFAULT_SALT,
        scenario,
        preflight_wait_seconds=0.1,
        preflight_wait_interval=0.1,
    )

    assert result["exit_code"] == 2
    assert result["iterations"][0]["result"]["preflight_skipped"] is True
    assert result["summary"]["ok"] is False
    assert "benchmark report quality preflight failed" in result["summary"]["error"]


def test_run_scenario_preflight_start_sample_can_skip_subprocess(monkeypatch):
    samples = [
        {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
        {
            "available": True,
            "cpu_load_pct": 97.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    ]

    def fake_environment_metadata():
        if samples:
            return samples.pop(0)
        return {
            "available": True,
            "cpu_load_pct": 97.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        }

    def fail_run(command, text, capture_output, check):
        raise AssertionError("low-trust start sample should skip subprocess launch")

    monkeypatch.setattr(benchmark, "collect_environment_metadata", fake_environment_metadata)
    monkeypatch.setattr(benchmark.subprocess, "run", fail_run)
    scenario = benchmark.BenchmarkScenario(
        name="cuda-start-low-trust",
        backend="cuda",
        difficulty=8,
        batch_size=2048,
        seconds=1,
        warmup=0,
        repeat=1,
    )

    result = benchmark.run_scenario(
        Path("miner"),
        benchmark.DEFAULT_SALT,
        scenario,
        preflight_wait_seconds=0.1,
        preflight_wait_interval=0.1,
        preflight_stable_samples=1,
    )

    assert result["exit_code"] == 2
    assert result["iterations"][0]["result"]["preflight_skipped"] is True
    assert result["environment"]["benchmark_trust"] == "low"
    assert result["summary"]["ok"] is False


def test_run_hash_command_can_retry_preflight_skips(monkeypatch):
    calls = {"count": 0}

    def fake_run_hash_command(
        command,
        environment_samples=None,
        preflight_wait_seconds=0.0,
        preflight_wait_interval=5.0,
        preflight_stable_samples=1,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            if environment_samples is not None:
                environment_samples.append(
                    {
                        "available": True,
                        "cpu_load_pct": 99.0,
                        "high_cpu_load": True,
                        "benchmark_trust": "low",
                    }
                )
            return {
                "exit_code": 2,
                "wall_elapsed_ms": 0.0,
                "result": {
                    "ok": False,
                    "error": "benchmark report quality preflight failed",
                    "preflight_skipped": True,
                },
            }
        if environment_samples is not None:
            environment_samples.append(
                {
                    "available": True,
                    "cpu_load_pct": 12.0,
                    "high_cpu_load": False,
                    "benchmark_trust": "normal",
                }
            )
        return {
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "result": {"ok": True, "hashrate": 42.0},
        }

    monkeypatch.setattr(benchmark, "run_hash_command", fake_run_hash_command)
    environment_samples = []

    result = benchmark.run_hash_command_with_preflight_retries(
        ["miner", "hash-benchmark"],
        environment_samples,
        preflight_wait_seconds=10.0,
        preflight_wait_interval=1.0,
        preflight_stable_samples=2,
        preflight_skip_retries=1,
    )

    assert calls == {"count": 2}
    assert result["exit_code"] == 0
    assert result["preflight_skip_retries"] == 1
    assert result["result"]["preflight_skip_retries"] == 1
    assert environment_samples == [
        {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        }
    ]


def test_run_hash_command_retry_does_not_hide_real_failures(monkeypatch):
    calls = {"count": 0}

    def fake_run_hash_command(
        command,
        environment_samples=None,
        preflight_wait_seconds=0.0,
        preflight_wait_interval=5.0,
        preflight_stable_samples=1,
    ):
        calls["count"] += 1
        if environment_samples is not None:
            environment_samples.append(
                {
                    "available": True,
                    "cpu_load_pct": 12.0,
                    "high_cpu_load": False,
                    "benchmark_trust": "normal",
                }
            )
        return {
            "exit_code": 1,
            "wall_elapsed_ms": 1.0,
            "result": {"ok": False, "error": "hash-benchmark failed"},
        }

    monkeypatch.setattr(benchmark, "run_hash_command", fake_run_hash_command)
    environment_samples = []

    result = benchmark.run_hash_command_with_preflight_retries(
        ["miner", "hash-benchmark"],
        environment_samples,
        preflight_skip_retries=3,
    )

    assert calls == {"count": 1}
    assert result["exit_code"] == 1
    assert result["result"]["error"] == "hash-benchmark failed"
    assert environment_samples == [
        {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        }
    ]


def test_main_writes_output_file(monkeypatch, tmp_path, capsys):
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

    _stub_metadata(monkeypatch)
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
    assert report["environment"]["benchmark_trust"] == "normal"
    assert report["recommendations"]["batch_size_by_difficulty"][0]["batch_size"] == 2
    assert report["runs"][0]["scenario"]["warmup"] == 1
    assert report["runs"][0]["scenario"]["repeat"] == 2
    assert json.loads(capsys.readouterr().out)["runs"][0]["summary"]["hashrate"] == 42.0


def test_main_records_build_cache_metadata(monkeypatch, tmp_path, capsys):
    def fake_build_metadata(cache_path):
        assert cache_path == tmp_path / "build"
        return {
            "provided": True,
            "available": True,
            "build_type": "Release",
            "cuda_architectures": ["86"],
        }

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

    _stub_metadata(monkeypatch)
    monkeypatch.setattr(benchmark, "collect_build_metadata", fake_build_metadata)
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)
    output = tmp_path / "report.json"
    sanitized_output = tmp_path / "summary.json"

    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--build-cache",
            str(tmp_path / "build"),
            "--backend",
            "cuda",
            "--seconds",
            "1",
            "--output",
            str(output),
            "--sanitized-output",
            str(sanitized_output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    sanitized = json.loads(sanitized_output.read_text(encoding="utf-8"))
    assert report["build"]["build_type"] == "Release"
    assert report["build"]["cuda_architectures"] == ["86"]
    assert sanitized["build"]["build_type"] == "Release"
    assert sanitized["build"]["cuda_architectures"] == ["86"]
    capsys.readouterr()


def test_main_can_disable_xuni_for_all_scenarios(monkeypatch, tmp_path):
    captured = []

    def fake_run_scenario(binary, salt, scenario):
        captured.append(scenario)
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(42.0),
            "aggregate": _summary(42.0),
            "command": benchmark.build_hash_command(binary, salt, scenario),
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    _stub_metadata(monkeypatch)
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)

    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--preset",
            "warm-short",
            "--scenario",
            "name=manual,backend=cuda,difficulty=1,batch_size=2,seconds=1,allow_xuni=true",
            "--scan-difficulty",
            "1",
            "--scan-batch-size",
            "2",
            "--seconds",
            "1",
            "--no-xuni",
        ]
    )

    assert exit_code == 0
    assert captured
    assert all(scenario.allow_xuni is False for scenario in captured)
    assert all("--no-xuni" in benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario) for scenario in captured)


def test_build_hash_command_includes_fixed_key():
    fixed_key = "0" * 64
    scenario = benchmark.BenchmarkScenario(
        name="cuda-fixed",
        backend="cuda",
        difficulty=8,
        batch_size=1,
        seconds=1,
        key=fixed_key,
    )

    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario)

    assert "--key" in command
    assert command[command.index("--key") + 1] == fixed_key


def test_main_writes_sanitized_output_file(monkeypatch, tmp_path, capsys):
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

    monkeypatch.setattr(benchmark, "collect_hardware_metadata", lambda: {"nvidia_smi": {"stdout": "private gpu"}})
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 95.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)
    sanitized_output = tmp_path / "summary.json"

    exit_code = benchmark.main(
        [
            "--binary",
            "<private-binary>",
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
    assert sanitized["environment"]["benchmark_trust"] == "low"
    assert "<private-binary>" not in encoded
    assert "private gpu" not in encoded
    assert "private-salt" not in encoded
    assert "deadbeef" not in encoded
    assert "secret-key" not in encoded
    capsys.readouterr()


def test_collect_environment_metadata_marks_high_windows_cpu_load(monkeypatch):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        benchmark,
        "run_metadata_command",
        lambda command, timeout=10: {
            "available": True,
            "exit_code": 0,
            "stdout": "99\n",
            "stderr": "",
        },
    )

    metadata = benchmark.collect_environment_metadata()

    assert metadata == {
        "available": True,
        "cpu_load_pct": 99.0,
        "high_cpu_load": True,
        "benchmark_trust": "low",
    }


def test_collect_environment_metadata_handles_unavailable_cpu_load(monkeypatch):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        benchmark,
        "run_metadata_command",
        lambda command, timeout=10: {
            "available": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "failed",
        },
    )

    metadata = benchmark.collect_environment_metadata()

    assert metadata == {
        "available": False,
        "reason": "cpu_load_unavailable",
    }


def test_combine_environment_metadata_uses_max_cpu_load_for_trust():
    metadata = benchmark.combine_environment_metadata(
        {
            "available": True,
            "cpu_load_pct": 25.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
        {
            "available": True,
            "cpu_load_pct": 97.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )

    assert metadata == {
        "available": True,
        "cpu_load_pct": 97.0,
        "start_cpu_load_pct": 25.0,
        "end_cpu_load_pct": 97.0,
        "sample_count": 2,
        "high_cpu_load": True,
        "benchmark_trust": "low",
    }


def test_main_combines_per_run_environment_samples(monkeypatch, tmp_path, capsys):
    loads = iter([15.0, 20.0, 97.0, 30.0])

    def fake_environment_metadata():
        cpu_load_pct = next(loads)
        return {
            "available": True,
            "cpu_load_pct": cpu_load_pct,
            "high_cpu_load": cpu_load_pct >= 90.0,
            "benchmark_trust": "low" if cpu_load_pct >= 90.0 else "normal",
        }

    def fake_run(command, text, capture_output, check):
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
                    "hashrate": 2.0,
                    "timings": {"compute_ms": 1.0},
                    "matches": [],
                    "error": "",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark, "collect_environment_metadata", fake_environment_metadata)
    monkeypatch.setattr(benchmark, "collect_hardware_metadata", lambda: {"nvidia_smi": {"available": False}})
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    output = tmp_path / "report.json"

    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--backend",
            "cuda",
            "--seconds",
            "1",
            "--scenario",
            "name=manual,backend=cuda,difficulty=8,batch_size=2,seconds=1",
            "--repeat",
            "2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["environment"] == {
        "available": True,
        "cpu_load_pct": 97.0,
        "start_cpu_load_pct": 15.0,
        "end_cpu_load_pct": 30.0,
        "sample_count": 4,
        "high_cpu_load": True,
        "benchmark_trust": "low",
    }
    assert report["runs"][0]["environment"] == report["environment"]
    capsys.readouterr()


def test_main_can_print_recommendations_only(monkeypatch, tmp_path, capsys):
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

    _stub_metadata(monkeypatch)
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
    assert list(stdout) == [
        "batch_size_by_difficulty",
        "benchmark_trust",
        "candidates_by_difficulty",
        "cold_scenarios",
        "environment_available",
        "environment_sample_count",
        "high_cpu_load",
        "invalid_run_count",
        "invalid_scenarios",
        "report_ok",
        "report_quality_failure_reasons",
        "report_quality_ok",
        "run_count",
        "stable_run_count",
        "stable_spread_pct",
        "unstable_scenarios",
        "valid_run_count",
        "warm_evidence_run_count",
    ]
    assert stdout["report_ok"] is True
    assert stdout["report_quality_ok"] is True
    assert stdout["benchmark_trust"] == "normal"
    assert stdout["batch_size_by_difficulty"][0]["batch_size"] == 2
    assert "runs" in json.loads(output.read_text(encoding="utf-8"))


def test_main_can_fail_on_low_report_quality(monkeypatch, tmp_path, capsys):
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

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 96.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )
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
            "--fail-on-report-quality",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "benchmark report quality check failed" in captured.err
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["recommendations"]["report_ok"] is True
    assert report["recommendations"]["report_quality_ok"] is False


def test_recommendations_only_can_fail_on_low_report_quality(monkeypatch, tmp_path, capsys):
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

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 97.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)

    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--backend",
            "cuda",
            "--seconds",
            "1",
            "--recommendations-only",
            "--fail-on-report-quality",
        ]
    )

    captured = capsys.readouterr()
    stdout = json.loads(captured.out)
    assert exit_code == 2
    assert stdout["report_quality_ok"] is False
    assert stdout["benchmark_trust"] == "low"
    assert "benchmark report quality check failed" in captured.err


def test_preflight_report_quality_skips_low_trust_benchmarks(monkeypatch, tmp_path, capsys):
    def fail_run_scenario(binary, salt, scenario):
        raise AssertionError("preflight should skip benchmark scenarios")

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 98.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )
    monkeypatch.setattr(benchmark, "run_scenario", fail_run_scenario)
    output = tmp_path / "preflight.json"

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
            "--preflight-report-quality",
        ]
    )

    captured = capsys.readouterr()
    stdout = json.loads(captured.out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert stdout["run_count"] == 0
    assert stdout["report_quality_ok"] is False
    assert stdout["benchmark_trust"] == "low"
    assert report["runs"] == []
    assert report["recommendations"]["run_count"] == 0
    assert "benchmark report quality preflight failed" in captured.err


def test_preflight_report_quality_allows_normal_trust_benchmarks(monkeypatch, tmp_path, capsys):
    calls = {"count": 0}

    def fake_run_scenario(binary, salt, scenario):
        calls["count"] += 1
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

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
    )
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
            "--scenario",
            "name=manual,backend=cuda,difficulty=1,batch_size=2,seconds=1",
            "--output",
            str(output),
            "--preflight-report-quality",
        ]
    )

    assert exit_code == 0
    assert calls["count"] == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["runs"]) == 1
    assert report["recommendations"]["report_quality_ok"] is True
    capsys.readouterr()


def test_preflight_report_quality_can_wait_for_normal_trust(monkeypatch, tmp_path, capsys):
    calls = {"run": 0, "sleep": 0}
    environment_samples = [
        {
            "available": True,
            "cpu_load_pct": 98.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
        {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
        {
            "available": True,
            "cpu_load_pct": 13.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
    ]

    def fake_run_scenario(
        binary,
        salt,
        scenario,
        preflight_wait_seconds=0.0,
        preflight_wait_interval=5.0,
        preflight_stable_samples=1,
        preflight_skip_retries=0,
    ):
        calls["run"] += 1
        assert preflight_wait_seconds == 10.0
        assert preflight_wait_interval == 1.0
        assert preflight_stable_samples == 2
        assert preflight_skip_retries == 0
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

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    def fake_environment_metadata():
        if environment_samples:
            return environment_samples.pop(0)
        return {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        }

    monkeypatch.setattr(benchmark, "collect_environment_metadata", fake_environment_metadata)
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: calls.__setitem__("sleep", calls["sleep"] + 1))
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
            "--scenario",
            "name=manual,backend=cuda,difficulty=1,batch_size=2,seconds=1",
            "--output",
            str(output),
            "--preflight-report-quality",
            "--preflight-wait-seconds",
            "10",
            "--preflight-wait-interval",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == {"run": 1, "sleep": 2}
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["runs"]) == 1
    assert report["recommendations"]["report_quality_ok"] is True
    capsys.readouterr()


def test_preflight_report_quality_can_use_single_stable_sample(monkeypatch, tmp_path, capsys):
    calls = {"run": 0, "sleep": 0}
    environment_samples = [
        {
            "available": True,
            "cpu_load_pct": 98.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
        {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
    ]

    def fake_run_scenario(
        binary,
        salt,
        scenario,
        preflight_wait_seconds=0.0,
        preflight_wait_interval=5.0,
        preflight_stable_samples=1,
        preflight_skip_retries=0,
    ):
        calls["run"] += 1
        assert preflight_stable_samples == 1
        assert preflight_skip_retries == 0
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

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )

    def fake_environment_metadata():
        if environment_samples:
            return environment_samples.pop(0)
        return {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        }

    monkeypatch.setattr(benchmark, "collect_environment_metadata", fake_environment_metadata)
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: calls.__setitem__("sleep", calls["sleep"] + 1))
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
            "--scenario",
            "name=manual,backend=cuda,difficulty=1,batch_size=2,seconds=1",
            "--output",
            str(output),
            "--preflight-report-quality",
            "--preflight-wait-seconds",
            "10",
            "--preflight-wait-interval",
            "1",
            "--preflight-stable-samples",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == {"run": 1, "sleep": 1}
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["runs"]) == 1
    assert report["recommendations"]["report_quality_ok"] is True
    capsys.readouterr()


def test_preflight_report_quality_can_limit_subprocess_wait(monkeypatch, tmp_path, capsys):
    calls = {"run": 0}

    def fake_run_scenario(
        binary,
        salt,
        scenario,
        preflight_wait_seconds=0.0,
        preflight_wait_interval=5.0,
        preflight_stable_samples=1,
        preflight_skip_retries=0,
    ):
        calls["run"] += 1
        assert preflight_wait_seconds == 2.0
        assert preflight_wait_interval == 1.0
        assert preflight_stable_samples == 2
        assert preflight_skip_retries == 0
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

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
    )
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
            "--scenario",
            "name=manual,backend=cuda,difficulty=1,batch_size=2,seconds=1",
            "--output",
            str(output),
            "--preflight-report-quality",
            "--preflight-wait-seconds",
            "60",
            "--subprocess-preflight-wait-seconds",
            "2",
            "--preflight-wait-interval",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == {"run": 1}
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["recommendations"]["report_quality_ok"] is True
    capsys.readouterr()


def test_preflight_report_quality_can_retry_subprocess_preflight_skips(monkeypatch, tmp_path, capsys):
    calls = {"run": 0}

    def fake_run_scenario(
        binary,
        salt,
        scenario,
        preflight_wait_seconds=0.0,
        preflight_wait_interval=5.0,
        preflight_stable_samples=1,
        preflight_skip_retries=0,
    ):
        calls["run"] += 1
        assert preflight_wait_seconds == 2.0
        assert preflight_wait_interval == 1.0
        assert preflight_stable_samples == 2
        assert preflight_skip_retries == 3
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

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
    )
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
            "--scenario",
            "name=manual,backend=cuda,difficulty=1,batch_size=2,seconds=1",
            "--output",
            str(output),
            "--preflight-report-quality",
            "--preflight-wait-seconds",
            "60",
            "--subprocess-preflight-wait-seconds",
            "2",
            "--preflight-wait-interval",
            "1",
            "--preflight-skip-retries",
            "3",
        ]
    )

    assert exit_code == 0
    assert calls == {"run": 1}
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["recommendations"]["report_quality_ok"] is True
    capsys.readouterr()


def test_preflight_report_quality_wait_timeout_skips_benchmarks(monkeypatch, tmp_path, capsys):
    calls = {"run": 0, "sleep": 0}
    monotonic_values = iter([0.0, 0.0, 1.0, 1.0])

    def fake_run_scenario(binary, salt, scenario):
        calls["run"] += 1
        raise AssertionError("preflight timeout should skip benchmark scenarios")

    monkeypatch.setattr(
        benchmark,
        "collect_hardware_metadata",
        lambda: {"nvidia_smi": {"available": False}, "nvcc": {"available": False}},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 99.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: calls.__setitem__("sleep", calls["sleep"] + 1))
    monkeypatch.setattr(benchmark, "run_scenario", fake_run_scenario)
    output = tmp_path / "preflight-timeout.json"

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
            "--preflight-report-quality",
            "--preflight-wait-seconds",
            "1",
            "--preflight-wait-interval",
            "1",
        ]
    )

    captured = capsys.readouterr()
    stdout = json.loads(captured.out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert calls == {"run": 0, "sleep": 1}
    assert stdout["run_count"] == 0
    assert stdout["report_quality_ok"] is False
    assert stdout["environment_sample_count"] == 2
    assert stdout["preflight_stable_samples_required"] == 2
    assert stdout["preflight_stable_samples_observed"] == 0
    assert report["runs"] == []
    assert report["environment"]["sample_count"] == 2
    assert "benchmark report quality preflight failed" in captured.err


def test_preflight_only_emits_empty_report_without_hardware_probe(monkeypatch, tmp_path, capsys):
    def fail_run_scenario(binary, salt, scenario):
        raise AssertionError("preflight-only should not run benchmark scenarios")

    def fail_hardware_metadata():
        raise AssertionError("preflight-only should not collect hardware metadata")

    monkeypatch.setattr(benchmark, "run_scenario", fail_run_scenario)
    monkeypatch.setattr(benchmark, "collect_hardware_metadata", fail_hardware_metadata)
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 12.0,
            "high_cpu_load": False,
            "benchmark_trust": "normal",
        },
    )
    output = tmp_path / "preflight-only.json"

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
            "--preflight-only",
        ]
    )

    captured = capsys.readouterr()
    stdout = json.loads(captured.out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout["report_quality_ok"] is True
    assert stdout["run_count"] == 0
    assert report["runs"] == []
    assert report["hardware"] == {}


def test_preflight_only_fails_low_trust_without_running_benchmarks(monkeypatch, tmp_path, capsys):
    def fail_run_scenario(binary, salt, scenario):
        raise AssertionError("preflight-only should not run benchmark scenarios")

    monkeypatch.setattr(benchmark, "run_scenario", fail_run_scenario)
    monkeypatch.setattr(benchmark, "collect_hardware_metadata", lambda: {"nvidia_smi": {"available": False}})
    monkeypatch.setattr(
        benchmark,
        "collect_environment_metadata",
        lambda: {
            "available": True,
            "cpu_load_pct": 98.0,
            "high_cpu_load": True,
            "benchmark_trust": "low",
        },
    )
    output = tmp_path / "preflight-only-low.json"

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
            "--preflight-only",
        ]
    )

    captured = capsys.readouterr()
    stdout = json.loads(captured.out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert stdout["report_quality_ok"] is False
    assert stdout["run_count"] == 0
    assert report["runs"] == []
    assert report["hardware"] == {}
    assert "benchmark report quality preflight failed" in captured.err


def test_main_combines_presets_and_manual_scenarios(monkeypatch, tmp_path):
    captured_names = []

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

    _stub_metadata(monkeypatch)
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
    captured_detailed_timings = []
    captured_dynamic_auto = []

    def fake_run_scenario(binary, salt, scenario):
        captured_names.append(scenario.name)
        captured_detailed_timings.append(scenario.detailed_timings)
        captured_dynamic_auto.append(scenario.first_block_dynamic_chunk_auto)
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

    _stub_metadata(monkeypatch)
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
            "--scan-first-block-dynamic-chunk-size",
            "0",
            "--scan-first-block-dynamic-chunk-size",
            "64",
            "--scan-first-block-dynamic-chunk-auto",
            "--scan-detailed-timings",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured_names == [
        "cuda-scan-d1-b512-fbda",
        "cuda-scan-d1-b512-fbd64-fbda",
        "cuda-scan-d1-b1024-fbda",
        "cuda-scan-d1-b1024-fbd64-fbda",
        "cuda-scan-d8-b512-fbda",
        "cuda-scan-d8-b512-fbd64-fbda",
        "cuda-scan-d8-b1024-fbda",
        "cuda-scan-d8-b1024-fbd64-fbda",
    ]
    assert captured_detailed_timings == [True] * 8
    assert captured_dynamic_auto == [True] * 8


def test_main_scan_can_include_gpu_first_block_variants(monkeypatch, tmp_path):
    captured = []

    def fake_run_scenario(binary, salt, scenario):
        captured.append(scenario)
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(42.0),
            "aggregate": _summary(42.0),
            "command": benchmark.build_hash_command(binary, salt, scenario),
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    _stub_metadata(monkeypatch)
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
            "8",
            "--scan-batch-size",
            "2048",
            "--scan-gpu-first-blocks",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert [scenario.name for scenario in captured] == [
        "cuda-scan-d8-b2048",
        "cuda-scan-d8-b2048-gfb",
    ]
    assert [scenario.gpu_first_blocks for scenario in captured] == [False, True]
    assert "--gpu-first-blocks" not in benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, captured[0])
    assert "--gpu-first-blocks" in benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, captured[1])


def test_main_combines_difficulty_sequence_scenarios(monkeypatch, tmp_path):
    captured = []

    def fake_run_scenario(binary, salt, scenario):
        captured.append(scenario)
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(42.0),
            "aggregate": _summary(42.0),
            "command": benchmark.build_hash_command(binary, salt, scenario),
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    _stub_metadata(monkeypatch)
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
            "--difficulty-sequence",
            "1,1,1,1",
            "--difficulty-sequence",
            "1,8,1,8",
            "--sequence-batch-size",
            "512",
            "--sequence-detailed-timings",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert [scenario.name for scenario in captured] == [
        "cuda-difficulty-sequence-d1x1x1x1-b512",
        "cuda-difficulty-sequence-d1x8x1x8-b512",
    ]
    assert [scenario.difficulty_sequence for scenario in captured] == [(1, 1, 1, 1), (1, 8, 1, 8)]
    assert [scenario.detailed_timings for scenario in captured] == [True, True]
    assert all("--difficulty-sequence" in benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, scenario) for scenario in captured)


def test_main_global_gpu_first_blocks_updates_generated_sequence(monkeypatch, tmp_path):
    captured = []

    def fake_run_scenario(binary, salt, scenario):
        captured.append(scenario)
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(42.0, batch_size=2048, batch_size_min=2048, batch_size_max=2048),
            "aggregate": _summary(42.0),
            "command": benchmark.build_hash_command(binary, salt, scenario),
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    _stub_metadata(monkeypatch)
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
            "--difficulty-sequence",
            "1,8,64",
            "--sequence-auto-batch-size",
            "--sequence-first-block-dynamic-chunk-auto",
            "--gpu-first-blocks",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert [scenario.name for scenario in captured] == ["cuda-difficulty-sequence-d1x8x64-bauto-gfb"]
    assert captured[0].gpu_first_blocks is True
    assert "--gpu-first-blocks" in benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, captured[0])


def test_main_combines_paired_batch_size_sequence_scenarios(monkeypatch, tmp_path):
    captured = []

    def fake_run_scenario(binary, salt, scenario):
        captured.append(scenario)
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(
                42.0,
                batch_size=3072,
                batch_size_min=2048,
                batch_size_max=3072,
            ),
            "aggregate": _summary(42.0),
            "command": benchmark.build_hash_command(binary, salt, scenario),
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    _stub_metadata(monkeypatch)
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
            "--difficulty-sequence",
            "1,8,64",
            "--batch-size-sequence",
            "2048,3072,3072",
            "--sequence-detailed-timings",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert [scenario.name for scenario in captured] == [
        "cuda-difficulty-sequence-d1x8x64-bseq-2048x3072x3072",
    ]
    assert captured[0].difficulty_sequence == (1, 8, 64)
    assert captured[0].batch_size_sequence == (2048, 3072, 3072)
    assert captured[0].batch_size == 2048
    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, captured[0])
    assert "--difficulty-sequence" in command
    assert "--batch-size-sequence" in command


def test_main_combines_auto_batch_difficulty_sequence_scenarios(monkeypatch, tmp_path):
    captured = []

    def fake_run_scenario(binary, salt, scenario):
        captured.append(scenario)
        return {
            "scenario": benchmark.asdict(scenario),
            "summary": _summary(42.0, batch_size=2048, batch_size_min=2048, batch_size_max=2048),
            "aggregate": _summary(42.0),
            "command": benchmark.build_hash_command(binary, salt, scenario),
            "exit_code": 0,
            "wall_elapsed_ms": 1.0,
            "warmup_runs": [],
            "iterations": [{"exit_code": 0, "result": {"ok": True}}],
            "iteration_summaries": [_summary(42.0)],
            "result": {"ok": True, "hashrate": 42.0},
        }

    _stub_metadata(monkeypatch)
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
            "--difficulty-sequence",
            "1,8,64",
            "--sequence-auto-batch-size",
            "--sequence-detailed-timings",
            "--sequence-first-block-dynamic-chunk-auto",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert [scenario.name for scenario in captured] == ["cuda-difficulty-sequence-d1x8x64-bauto"]
    assert captured[0].difficulty_sequence == (1, 8, 64)
    assert captured[0].auto_batch_size is True
    assert captured[0].first_block_dynamic_chunk_auto is True
    command = benchmark.build_hash_command(Path("miner"), benchmark.DEFAULT_SALT, captured[0])
    assert "--difficulty-sequence" in command
    assert "--auto-batch-size" in command
    assert "--first-block-dynamic-chunk-auto" in command
    assert "--batch-size" not in command


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


def test_main_rejects_global_and_scan_gpu_first_blocks(capsys):
    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--scan-difficulty",
            "8",
            "--scan-batch-size",
            "2048",
            "--gpu-first-blocks",
            "--scan-gpu-first-blocks",
        ]
    )

    assert exit_code == 2
    assert "--gpu-first-blocks and --scan-gpu-first-blocks cannot be used together" in capsys.readouterr().err


def test_main_rejects_partial_difficulty_sequence(capsys):
    exit_code = benchmark.main(
        [
            "--binary",
            "miner",
            "--difficulty-sequence",
            "1,8,1,8",
        ]
    )

    assert exit_code == 2
    assert (
        "--difficulty-sequence requires --sequence-batch-size, --sequence-auto-batch-size, or --batch-size-sequence"
        in capsys.readouterr().err
    )


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
