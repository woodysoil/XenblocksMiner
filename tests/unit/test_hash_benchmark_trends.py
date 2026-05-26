"""Tests for public-safe Hash API benchmark trend rendering."""

from __future__ import annotations

import json

import scripts.hash_benchmark_trends as trends
from scripts.hash_benchmark_trends import load_points, main, trusted_trend_summaries


def _report(name: str, difficulty: int, median_hashrate: float, *, source_path: str) -> dict:
    return {
        "created_at_unix": 1000.0,
        "binary": source_path,
        "hardware": {"nvidia_smi": {"stdout": "private gpu model"}},
        "host": {"node": "private-host"},
        "salt": "private-salt",
        "recommendations": {"report_ok": True, "report_quality_ok": True},
        "runs": [
            {
                "command": [source_path, "--salt", "private-salt"],
                "scenario": {"name": name, "backend": "cuda", "difficulty": difficulty},
                "summary": {
                    "name": name,
                    "backend": "cuda",
                    "difficulty": difficulty,
                    "batch_size": 128,
                    "gpu_first_blocks": True,
                    "median_hashrate": median_hashrate,
                    "hashrate_spread_pct": 2.5,
                    "stable": True,
                    "warmup": 1,
                    "repeat": 2,
                    "timing_analysis": {
                        "stage_pct": {"compute_ms": 95.0},
                        "nested_stage_pct": {"kernel_ms": 97.0},
                    },
                    "ok": True,
                },
            }
        ],
    }


def test_load_points_filters_by_min_difficulty(tmp_path):
    (tmp_path / "low.json").write_text(
        json.dumps(_report("low", 8, 100.0, source_path="<private-binary>")),
        encoding="utf-8",
    )
    (tmp_path / "high.json").write_text(
        json.dumps(_report("high", 4096, 200.0, source_path="<private-binary>")),
        encoding="utf-8",
    )

    points = load_points(tmp_path, min_difficulty=4096)

    assert [point.name for point in points] == ["high"]
    assert points[0].median_hashrate == 200.0


def test_trusted_trend_summaries_report_best_and_gain(tmp_path):
    (tmp_path / "first.json").write_text(
        json.dumps(_report("first", 4096, 100.0, source_path="<private-binary>")),
        encoding="utf-8",
    )
    second = _report("second", 4096, 120.0, source_path="<private-binary>")
    second["created_at_unix"] = 2000.0
    second["runs"][0]["summary"]["hashrate_spread_pct"] = 3.0
    second["runs"][0]["summary"]["batch_size"] = 256
    (tmp_path / "second.json").write_text(json.dumps(second), encoding="utf-8")

    summaries = trusted_trend_summaries(load_points(tmp_path, min_difficulty=4096))

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.difficulty_label == "4096"
    assert summary.trusted_points == 2
    assert summary.first_median_hashrate == 100.0
    assert summary.latest_median_hashrate == 120.0
    assert summary.best_median_hashrate == 120.0
    assert summary.target_multiplier == 11.0
    assert summary.target_median_hashrate == 1100.0
    assert summary.latest_gain_pct == 20.0
    assert summary.best_gain_pct == 20.0
    assert summary.latest_target_progress_pct == 120.0 / 1100.0 * 100.0
    assert summary.best_target_progress_pct == 120.0 / 1100.0 * 100.0
    assert summary.latest_remaining_multiplier == 1100.0 / 120.0
    assert summary.best_remaining_multiplier == 1100.0 / 120.0
    assert summary.best_spread_pct == 3.0
    assert summary.best_batch_label == "256"
    assert summary.best_source == "second.json"
    assert summary.best_scenario == "second"


def test_load_points_ignores_empty_preflight_reports(tmp_path):
    (tmp_path / "preflight-only.json").write_text(
        json.dumps(
            {
                "created_at_unix": 1000.0,
                "recommendations": {
                    "report_ok": True,
                    "report_quality_ok": False,
                    "run_count": 0,
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )

    assert load_points(tmp_path, min_difficulty=4096) == []


def test_main_writes_public_safe_html(tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    input_dir.mkdir()
    (input_dir / "high.json").write_text(
        json.dumps(_report("high", 4096, 200.0, source_path="<private-binary>")),
        encoding="utf-8",
    )

    assert main(["--input-dir", str(input_dir), "--output", str(output), "--min-difficulty", "4096"]) == 0

    html = output.read_text(encoding="utf-8")
    assert "high" in html
    assert "median_ms_per_attempt" in html
    assert '<label>Metric <select id="metric">' in html
    assert '<option value="median_ms_per_attempt">Median ms/attempt</option>' in html
    assert "function metricInfo()" in html
    assert "function fmtMetric(value, info = metricInfo())" in html
    assert "Latest Trusted Gain" in html
    assert "Best Trusted Gain" in html
    assert "Trusted Points" in html
    assert "11x Target H/s" in html
    assert "Best Target" in html
    assert "Remaining" in html
    assert '<option value="stable" selected>Warm Stable + Quality OK</option>' in html
    assert '<option value="good">Warm Quality OK</option>' in html
    assert '<option value="all">Diagnostics</option>' in html
    assert "run_ok" in html
    assert "warm_evidence" in html
    assert "p.run_ok && p.warm_evidence && p.quality_ok && p.stable" in html
    assert "p.run_ok ? (p.warm_evidence ? (p.quality_ok ? (p.stable ? 'stable' : 'ok') : 'low') : 'cold') : 'invalid'" in html
    assert "function groupedByDifficulty(data)" in html
    assert "function trustedGainFor(data, referencePoint)" in html
    assert "function difficultySummaries(data)" in html
    assert "first.median_hashrate * 11" in html
    assert "bestTargetProgress" in html
    assert "remainingMultiplier" in html
    assert "p.difficulty_label === referencePoint.difficulty_label" in html
    assert "difficulty_label" in html
    assert "private gpu model" not in html
    assert "private-host" not in html
    assert "private-salt" not in html
    assert "<private-binary>" not in html


def test_main_can_write_public_safe_summary_json(tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    summary_output = tmp_path / "trend" / "summary.json"
    input_dir.mkdir()
    (input_dir / "high.json").write_text(
        json.dumps(_report("high", 4096, 200.0, source_path="<private-binary>")),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output",
                str(output),
                "--summary-output",
                str(summary_output),
                "--min-difficulty",
                "4096",
            ]
        )
        == 0
    )

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    summary_text = json.dumps(summary)
    assert summary["schema"] == "xenblocks.hashapi.trusted_trend_summary.v1"
    assert summary["summaries"][0]["difficulty_label"] == "4096"
    assert summary["summaries"][0]["best_median_hashrate"] == 200.0
    assert summary["summaries"][0]["target_multiplier"] == 11.0
    assert summary["summaries"][0]["target_median_hashrate"] == 2200.0
    assert summary["summaries"][0]["best_target_progress_pct"] == 100.0 / 11.0
    assert "private gpu model" not in summary_text
    assert "private-host" not in summary_text
    assert "private-salt" not in summary_text
    assert "<private-binary>" not in summary_text


def test_invalid_runs_are_not_trusted_points(tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    input_dir.mkdir()
    report = _report("invalid", 4096, 0.0, source_path="<private-binary>")
    report["recommendations"] = {"report_ok": False, "report_quality_ok": False}
    report["runs"][0]["summary"]["ok"] = False
    (input_dir / "invalid.json").write_text(json.dumps(report), encoding="utf-8")

    points = load_points(input_dir, min_difficulty=4096)

    assert len(points) == 1
    assert points[0].run_ok is False
    assert points[0].warm_evidence is True
    assert points[0].quality_ok is False
    assert points[0].median_ms_per_attempt == 0.0

    assert main(["--input-dir", str(input_dir), "--output", str(output), "--min-difficulty", "4096"]) == 0

    html = output.read_text(encoding="utf-8")
    assert '"run_ok": false' in html


def test_cold_single_repeat_runs_are_diagnostics_only(tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    input_dir.mkdir()
    report = _report("cold", 4096, 100.0, source_path="<private-binary>")
    report["runs"][0]["summary"]["warmup"] = 0
    report["runs"][0]["summary"]["repeat"] = 1
    (input_dir / "cold.json").write_text(json.dumps(report), encoding="utf-8")

    points = load_points(input_dir, min_difficulty=4096)

    assert len(points) == 1
    assert points[0].run_ok is True
    assert points[0].quality_ok is True
    assert points[0].stable is True
    assert points[0].warm_evidence is False

    assert main(["--input-dir", str(input_dir), "--output", str(output), "--min-difficulty", "4096"]) == 0

    html = output.read_text(encoding="utf-8")
    assert '"warm_evidence": false' in html
    assert "'cold'" in html


def test_trend_points_include_latency_metric(tmp_path):
    (tmp_path / "high.json").write_text(
        json.dumps(_report("high", 4096, 250.0, source_path="<private-binary>")),
        encoding="utf-8",
    )

    points = load_points(tmp_path, min_difficulty=4096)

    assert len(points) == 1
    assert points[0].median_hashrate == 250.0
    assert points[0].median_ms_per_attempt == 4.0


def test_main_can_write_auto_refresh_html(tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    input_dir.mkdir()
    (input_dir / "high.json").write_text(
        json.dumps(_report("high", 4096, 200.0, source_path="<private-binary>")),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output",
                str(output),
                "--min-difficulty",
                "4096",
                "--page-refresh-seconds",
                "3",
            ]
        )
        == 0
    )

    html = output.read_text(encoding="utf-8")
    assert '<meta http-equiv="refresh" content="3">' in html
    assert '<meta name="robots" content="noindex">' in html


def test_serve_mode_uses_local_refresh_defaults(monkeypatch, tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    captured = {}

    def fake_serve(
        input_path,
        output_path,
        min_difficulty,
        host,
        port,
        refresh_seconds,
        page_refresh_seconds,
        open_browser,
        summary_output,
    ):
        captured.update(
            {
                "input_path": input_path,
                "output_path": output_path,
                "min_difficulty": min_difficulty,
                "host": host,
                "port": port,
                "refresh_seconds": refresh_seconds,
                "page_refresh_seconds": page_refresh_seconds,
                "open_browser": open_browser,
                "summary_output": summary_output,
            }
        )
        return 0

    monkeypatch.setattr(trends, "serve_trends", fake_serve)

    assert main(["--serve", "--input-dir", str(input_dir), "--output", str(output)]) == 0

    assert captured == {
        "input_path": input_dir,
        "output_path": output,
        "min_difficulty": 4096,
        "host": "localhost",
        "port": 8766,
        "refresh_seconds": 5.0,
        "page_refresh_seconds": 10.0,
        "open_browser": False,
        "summary_output": None,
    }


def test_serve_mode_can_request_browser_open(monkeypatch, tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    captured = {}

    def fake_serve(
        input_path,
        output_path,
        min_difficulty,
        host,
        port,
        refresh_seconds,
        page_refresh_seconds,
        open_browser,
        summary_output,
    ):
        captured.update({"open_browser": open_browser})
        return 0

    monkeypatch.setattr(trends, "serve_trends", fake_serve)

    assert main(["--serve", "--open-browser", "--input-dir", str(input_dir), "--output", str(output)]) == 0

    assert captured == {"open_browser": True}


def test_serve_trends_opens_browser_when_requested(monkeypatch, tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    opened_urls = []

    class StopServer(Exception):
        pass

    class FakeServer:
        server_port = 12345

        def __init__(self, address, handler):
            self.address = address
            self.handler = handler

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return exc_type is StopServer

        def serve_forever(self):
            raise StopServer()

    monkeypatch.setattr(trends, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(trends, "write_trend_page", lambda *_args: 3)
    monkeypatch.setattr(trends.webbrowser, "open", lambda url: opened_urls.append(url))

    assert trends.serve_trends(input_dir, output, 4096, "localhost", 0, 5.0, 10.0, True) == 0

    assert opened_urls == ["http://localhost:12345/"]


def test_trend_page_cache_skips_unchanged_input_refresh(monkeypatch, tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    input_dir.mkdir()
    output.parent.mkdir()
    calls = {"write": 0}

    def fake_write(input_path, output_path, min_difficulty, page_refresh_seconds):
        calls["write"] += 1
        output_path.write_text("trend", encoding="utf-8")
        return 7

    monkeypatch.setattr(trends, "write_trend_page", fake_write)
    monkeypatch.setattr(trends, "input_signature", lambda path: (1, 2, 3))
    cache = trends.TrendPageCache(input_dir, output, 4096, 10.0, 5.0)

    assert cache.refresh(force=True, now=0.0) == 7
    assert cache.refresh(now=6.0) == 7

    assert calls == {"write": 1}


def test_trend_page_cache_refreshes_changed_input(monkeypatch, tmp_path):
    input_dir = tmp_path / "reports"
    output = tmp_path / "trend" / "index.html"
    input_dir.mkdir()
    output.parent.mkdir()
    calls = {"write": 0}
    signatures = [(1, 2, 3), (1, 2, 4)]

    def fake_write(input_path, output_path, min_difficulty, page_refresh_seconds):
        calls["write"] += 1
        output_path.write_text("trend", encoding="utf-8")
        return calls["write"]

    def fake_signature(input_path):
        if signatures:
            return signatures.pop(0)
        return (1, 2, 4)

    monkeypatch.setattr(trends, "write_trend_page", fake_write)
    monkeypatch.setattr(trends, "input_signature", fake_signature)
    cache = trends.TrendPageCache(input_dir, output, 4096, 10.0, 5.0)

    assert cache.refresh(force=True, now=0.0) == 1
    assert cache.refresh(now=6.0) == 2

    assert calls == {"write": 2}
