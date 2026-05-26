"""Tests for public-safe Hash API benchmark trend rendering."""

from __future__ import annotations

import json

import scripts.hash_benchmark_trends as trends
from scripts.hash_benchmark_trends import load_points, main


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
    assert "Latest Trusted Gain" in html
    assert "Best Trusted Gain" in html
    assert '<option value="stable" selected>Stable + Quality OK</option>' in html
    assert "function groupedByDifficulty(data)" in html
    assert "function trustedGainFor(data, referencePoint)" in html
    assert "p.difficulty_label === referencePoint.difficulty_label" in html
    assert "difficulty_label" in html
    assert "private gpu model" not in html
    assert "private-host" not in html
    assert "private-salt" not in html
    assert "<private-binary>" not in html


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

    def fake_serve(input_path, output_path, min_difficulty, host, port, refresh_seconds, page_refresh_seconds):
        captured.update(
            {
                "input_path": input_path,
                "output_path": output_path,
                "min_difficulty": min_difficulty,
                "host": host,
                "port": port,
                "refresh_seconds": refresh_seconds,
                "page_refresh_seconds": page_refresh_seconds,
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
    }


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
