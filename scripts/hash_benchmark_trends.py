"""Generate a public-safe local HTML trend page for Hash API benchmark reports."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrendPoint:
    source: str
    created_at: float
    name: str
    backend: str
    difficulty_label: str
    difficulty_min: int
    difficulty_mode: str
    batch_label: str
    gpu_first_blocks: bool
    median_hashrate: float
    spread_pct: float
    compute_pct: float
    kernel_pct: float
    report_ok: bool
    quality_ok: bool
    stable: bool


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


def _summary_for(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("summary") or run.get("aggregate") or {}


def _scenario_for(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("scenario") or {}


def _difficulty_values(summary: dict[str, Any], scenario: dict[str, Any]) -> list[int]:
    values = summary.get("difficulty_sequence") or scenario.get("difficulty_sequence") or []
    if values:
        return [int(value) for value in values]
    return [_int_value(summary, "difficulty", _int_value(scenario, "difficulty", 0))]


def _difficulty_label(values: list[int]) -> str:
    if len(values) > 1:
        return "x".join(str(value) for value in values)
    return str(values[0] if values else 0)


def _batch_label(summary: dict[str, Any], scenario: dict[str, Any]) -> str:
    values = summary.get("batch_size_sequence") or scenario.get("batch_size_sequence") or []
    if values:
        return "x".join(str(value) for value in values)
    return str(_int_value(summary, "batch_size", _int_value(scenario, "batch_size", 0)))


def load_points(input_dir: Path, min_difficulty: int) -> list[TrendPoint]:
    points: list[TrendPoint] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        runs = report.get("runs")
        if not isinstance(runs, list):
            continue
        recommendations = report.get("recommendations") or {}
        report_ok = _bool_value(recommendations, "report_ok", True)
        quality_ok = _bool_value(recommendations, "report_quality_ok", report_ok)
        created_at = _float_value(report, "created_at_unix", 0.0)
        for run in runs:
            if not isinstance(run, dict):
                continue
            summary = _summary_for(run)
            scenario = _scenario_for(run)
            if not summary:
                continue
            difficulties = _difficulty_values(summary, scenario)
            if min(difficulties or [0]) < min_difficulty:
                continue
            timing_analysis = summary.get("timing_analysis") or {}
            stage_pct = timing_analysis.get("stage_pct") or {}
            nested_stage_pct = timing_analysis.get("nested_stage_pct") or {}
            points.append(
                TrendPoint(
                    source=path.name,
                    created_at=created_at,
                    name=str(summary.get("name") or scenario.get("name") or ""),
                    backend=str(summary.get("backend") or scenario.get("backend") or ""),
                    difficulty_label=_difficulty_label(difficulties),
                    difficulty_min=min(difficulties or [0]),
                    difficulty_mode=str(summary.get("difficulty_mode") or scenario.get("difficulty_mode") or "fixed"),
                    batch_label=_batch_label(summary, scenario),
                    gpu_first_blocks=_bool_value(summary, "gpu_first_blocks", _bool_value(scenario, "gpu_first_blocks", False)),
                    median_hashrate=_float_value(summary, "median_hashrate", _float_value(summary, "hashrate", 0.0)),
                    spread_pct=_float_value(summary, "hashrate_spread_pct", 0.0),
                    compute_pct=_float_value(stage_pct, "compute_ms", 0.0),
                    kernel_pct=_float_value(nested_stage_pct, "kernel_ms", 0.0),
                    report_ok=report_ok,
                    quality_ok=quality_ok,
                    stable=_bool_value(summary, "stable", False),
                )
            )
    points.sort(key=lambda point: (point.created_at, point.source, point.name))
    return points


def _json_for_html(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def _render_html(points: list[TrendPoint], min_difficulty: int) -> str:
    rows = [point.__dict__ for point in points]
    title = f"Hash Benchmark Trends d{min_difficulty}+"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #17202a;
  --muted: #5f6b7a;
  --line: #d8dee8;
  --accent: #0f766e;
  --warn: #b45309;
  --bad: #b91c1c;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 6px; font-size: 24px; font-weight: 650; }}
.sub {{ color: var(--muted); margin-bottom: 18px; }}
.toolbar {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}}
label {{ display: grid; gap: 4px; color: var(--muted); font-size: 12px; }}
select, input {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #fff;
  color: var(--text);
  font: inherit;
}}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-bottom: 14px;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 1px;
  overflow: hidden;
}}
.stat {{ padding: 14px; background: #fff; }}
.stat .k {{ color: var(--muted); font-size: 12px; }}
.stat .v {{ font-size: 20px; font-weight: 650; margin-top: 4px; }}
canvas {{ display: block; width: 100%; height: 360px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 9px 10px; border-top: 1px solid var(--line); text-align: left; white-space: nowrap; }}
th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
td.name {{ max-width: 360px; overflow: hidden; text-overflow: ellipsis; }}
.bad {{ color: var(--bad); }}
.warn {{ color: var(--warn); }}
@media (max-width: 760px) {{
  main {{ padding: 14px; }}
  .toolbar, .stats {{ grid-template-columns: 1fr 1fr; }}
  canvas {{ height: 300px; }}
  .table-wrap {{ overflow-x: auto; }}
}}
</style>
</head>
<body>
<main>
  <h1>Hash Benchmark Trends</h1>
  <div class="sub">Public-safe local view generated from ignored benchmark reports. Raw paths, hardware names, command lines, and salts are not embedded.</div>
  <section class="toolbar">
    <label>Difficulty <select id="difficulty"></select></label>
    <label>GPU First Blocks <select id="gfb"><option value="all">All</option><option value="true">true</option><option value="false">false</option></select></label>
    <label>Quality <select id="quality"><option value="all">All</option><option value="good">Quality OK</option><option value="stable">Stable + Quality OK</option></select></label>
    <label>Search <input id="search" placeholder="scenario or source"></label>
  </section>
  <section class="panel stats">
    <div class="stat"><div class="k">Visible Points</div><div class="v" id="visibleCount">0</div></div>
    <div class="stat"><div class="k">Best Median H/s</div><div class="v" id="bestRate">0</div></div>
    <div class="stat"><div class="k">Latest Median H/s</div><div class="v" id="latestRate">0</div></div>
    <div class="stat"><div class="k">Latest Spread</div><div class="v" id="latestSpread">0%</div></div>
    <div class="stat"><div class="k">Latest Trusted Gain</div><div class="v" id="latestTrustedGain">n/a</div></div>
    <div class="stat"><div class="k">Best Trusted Gain</div><div class="v" id="bestTrustedGain">n/a</div></div>
  </section>
  <section class="panel"><canvas id="chart" width="1100" height="360"></canvas></section>
  <section class="panel table-wrap">
    <table>
      <thead><tr><th>#</th><th>Difficulty</th><th>Median H/s</th><th>Spread</th><th>Compute</th><th>Kernel</th><th>Batch</th><th>GFB</th><th>Quality</th><th>Scenario</th><th>Source</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
</main>
<script>
const points = {_json_for_html(rows)};
const difficultySelect = document.getElementById('difficulty');
const gfbSelect = document.getElementById('gfb');
const qualitySelect = document.getElementById('quality');
const searchInput = document.getElementById('search');
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');

function fmt(value, digits = 2) {{
  if (!Number.isFinite(value)) return '0';
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, {{ maximumFractionDigits: digits }});
  return value.toFixed(digits);
}}

function setupFilters() {{
  const values = [...new Set(points.map(p => p.difficulty_label))].sort((a, b) => {{
    const na = Math.min(...a.split('x').map(Number));
    const nb = Math.min(...b.split('x').map(Number));
    return na - nb || a.localeCompare(b);
  }});
  difficultySelect.innerHTML = '<option value="all">All d{min_difficulty}+</option>' + values.map(v => `<option value="${{v}}">${{v}}</option>`).join('');
}}

function filtered() {{
  const difficulty = difficultySelect.value;
  const gfb = gfbSelect.value;
  const quality = qualitySelect.value;
  const search = searchInput.value.trim().toLowerCase();
  return points.filter(p => {{
    if (difficulty !== 'all' && p.difficulty_label !== difficulty) return false;
    if (gfb !== 'all' && String(p.gpu_first_blocks) !== gfb) return false;
    if (quality === 'good' && !p.quality_ok) return false;
    if (quality === 'stable' && (!p.quality_ok || !p.stable)) return false;
    if (search && !(p.name.toLowerCase().includes(search) || p.source.toLowerCase().includes(search))) return false;
    return true;
  }});
}}

function drawChart(data) {{
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(640, Math.floor(rect.width * scale));
  canvas.height = Math.floor(360 * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const width = canvas.width / scale;
  const height = canvas.height / scale;
  ctx.clearRect(0, 0, width, height);
  const pad = {{ left: 58, right: 18, top: 18, bottom: 42 }};
  ctx.strokeStyle = '#d8dee8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
  if (!data.length) return;
  const maxRate = Math.max(...data.map(p => p.median_hashrate), 1);
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.fillStyle = '#5f6b7a';
  ctx.font = '12px system-ui, sans-serif';
  for (let i = 0; i <= 4; i++) {{
    const y = pad.top + (plotH * i / 4);
    const rate = maxRate * (1 - i / 4);
    ctx.strokeStyle = '#edf0f5';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(fmt(rate, 0), 8, y + 4);
  }}
  ctx.strokeStyle = '#0f766e';
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((p, i) => {{
    const x = pad.left + (data.length === 1 ? plotW / 2 : plotW * i / (data.length - 1));
    const y = pad.top + plotH * (1 - p.median_hashrate / maxRate);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
  data.forEach((p, i) => {{
    const x = pad.left + (data.length === 1 ? plotW / 2 : plotW * i / (data.length - 1));
    const y = pad.top + plotH * (1 - p.median_hashrate / maxRate);
    ctx.fillStyle = p.quality_ok && p.stable ? '#0f766e' : (p.quality_ok ? '#b45309' : '#b91c1c');
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }});
}}

function render() {{
  const data = filtered();
  const best = data.reduce((acc, p) => Math.max(acc, p.median_hashrate), 0);
  const latest = data[data.length - 1];
  const trusted = data.filter(p => p.quality_ok && p.stable && p.median_hashrate > 0);
  const firstTrusted = trusted[0];
  const latestTrusted = trusted[trusted.length - 1];
  const bestTrusted = trusted.reduce((acc, p) => p.median_hashrate > acc.median_hashrate ? p : acc, {{ median_hashrate: 0 }});
  const gainPct = (point) => firstTrusted && point && firstTrusted.median_hashrate > 0
    ? ((point.median_hashrate - firstTrusted.median_hashrate) / firstTrusted.median_hashrate * 100)
    : null;
  const latestGain = gainPct(latestTrusted);
  const bestGain = gainPct(bestTrusted);
  document.getElementById('visibleCount').textContent = String(data.length);
  document.getElementById('bestRate').textContent = fmt(best, 2);
  document.getElementById('latestRate').textContent = latest ? fmt(latest.median_hashrate, 2) : '0';
  document.getElementById('latestSpread').textContent = latest ? `${{fmt(latest.spread_pct, 2)}}%` : '0%';
  document.getElementById('latestTrustedGain').textContent = latestGain === null ? 'n/a' : `${{fmt(latestGain, 2)}}%`;
  document.getElementById('bestTrustedGain').textContent = bestGain === null ? 'n/a' : `${{fmt(bestGain, 2)}}%`;
  drawChart(data);
  document.getElementById('rows').innerHTML = data.map((p, i) => `
    <tr>
      <td>${{i + 1}}</td>
      <td>d${{p.difficulty_label}}</td>
      <td>${{fmt(p.median_hashrate, 3)}}</td>
      <td class="${{p.spread_pct > 10 ? 'bad' : (p.spread_pct > 5 ? 'warn' : '')}}">${{fmt(p.spread_pct, 2)}}%</td>
      <td>${{fmt(p.compute_pct, 2)}}%</td>
      <td>${{fmt(p.kernel_pct, 2)}}%</td>
      <td>${{p.batch_label}}</td>
      <td>${{p.gpu_first_blocks}}</td>
      <td>${{p.quality_ok ? (p.stable ? 'stable' : 'ok') : 'low'}}</td>
      <td class="name" title="${{p.name}}">${{p.name}}</td>
      <td class="name" title="${{p.source}}">${{p.source}}</td>
    </tr>`).join('');
}}

[difficultySelect, gfbSelect, qualitySelect, searchInput].forEach(el => el.addEventListener('input', render));
window.addEventListener('resize', render);
setupFilters();
render();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(".benchmarks"), help="Directory containing benchmark JSON reports.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/hash-trends/index.html"),
        help="Generated HTML path. Keep this under ignored benchmark storage.",
    )
    parser.add_argument("--min-difficulty", type=int, default=4096, help="Minimum difficulty to include.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points = load_points(args.input_dir, args.min_difficulty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_html(points, args.min_difficulty), encoding="utf-8")
    print(f"wrote {args.output} with {len(points)} public-safe points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
