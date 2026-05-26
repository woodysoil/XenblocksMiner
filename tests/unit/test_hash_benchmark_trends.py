"""Tests for public-safe Hash API benchmark trend rendering."""

from __future__ import annotations

import json

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
        json.dumps(_report("low", 8, 100.0, source_path=r"D:\private\miner.exe")),
        encoding="utf-8",
    )
    (tmp_path / "high.json").write_text(
        json.dumps(_report("high", 4096, 200.0, source_path=r"D:\private\miner.exe")),
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
        json.dumps(_report("high", 4096, 200.0, source_path=r"D:\private\miner.exe")),
        encoding="utf-8",
    )

    assert main(["--input-dir", str(input_dir), "--output", str(output), "--min-difficulty", "4096"]) == 0

    html = output.read_text(encoding="utf-8")
    assert "high" in html
    assert "Latest Trusted Gain" in html
    assert "Best Trusted Gain" in html
    assert "private gpu model" not in html
    assert "private-host" not in html
    assert "private-salt" not in html
    assert r"D:\private" not in html
