"""Generate a public-safe local HTML trend page for Hash API benchmark reports."""

from __future__ import annotations

import argparse
import html
import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
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
    median_ms_per_attempt: float
    spread_pct: float
    compute_pct: float
    kernel_pct: float
    run_ok: bool
    warm_evidence: bool
    report_ok: bool
    quality_ok: bool
    stable: bool


@dataclass(frozen=True)
class TrustedTrendSummary:
    difficulty_label: str
    trusted_points: int
    first_median_hashrate: float
    latest_median_hashrate: float
    best_median_hashrate: float
    target_multiplier: float
    target_median_hashrate: float
    latest_gain_pct: float
    best_gain_pct: float
    latest_target_progress_pct: float
    best_target_progress_pct: float
    latest_remaining_multiplier: float
    best_remaining_multiplier: float
    latest_spread_pct: float
    best_spread_pct: float
    best_batch_label: str
    best_source: str
    best_scenario: str


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
            median_hashrate = _float_value(summary, "median_hashrate", _float_value(summary, "hashrate", 0.0))
            warmup = _int_value(summary, "warmup", _int_value(scenario, "warmup", 0))
            repeat = _int_value(summary, "repeat", _int_value(scenario, "repeat", 1))
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
                    median_hashrate=median_hashrate,
                    median_ms_per_attempt=1000.0 / median_hashrate if median_hashrate > 0.0 else 0.0,
                    spread_pct=_float_value(summary, "hashrate_spread_pct", 0.0),
                    compute_pct=_float_value(stage_pct, "compute_ms", 0.0),
                    kernel_pct=_float_value(nested_stage_pct, "kernel_ms", 0.0),
                    run_ok=_bool_value(summary, "ok", report_ok) and median_hashrate > 0.0,
                    warm_evidence=warmup >= 1 and repeat >= 2,
                    report_ok=report_ok,
                    quality_ok=quality_ok,
                    stable=_bool_value(summary, "stable", False),
                )
            )
    points.sort(key=lambda point: (point.created_at, point.source, point.name))
    return points


def trusted_points(points: list[TrendPoint]) -> list[TrendPoint]:
    return [
        point
        for point in points
        if point.run_ok
        and point.warm_evidence
        and point.quality_ok
        and point.stable
        and point.median_hashrate > 0.0
    ]


def trusted_trend_summaries(points: list[TrendPoint], target_multiplier: float = 11.0) -> list[TrustedTrendSummary]:
    grouped: dict[str, list[TrendPoint]] = {}
    for point in trusted_points(points):
        grouped.setdefault(point.difficulty_label, []).append(point)

    summaries: list[TrustedTrendSummary] = []
    for difficulty_label, values in grouped.items():
        values = sorted(values, key=lambda point: (point.created_at, point.source, point.name))
        first = values[0]
        latest = values[-1]
        best = max(values, key=lambda point: point.median_hashrate)
        first_rate = first.median_hashrate
        target_rate = first_rate * target_multiplier
        latest_gain_pct = (
            (latest.median_hashrate - first_rate) / first_rate * 100.0
            if first_rate > 0.0
            else 0.0
        )
        best_gain_pct = (
            (best.median_hashrate - first_rate) / first_rate * 100.0
            if first_rate > 0.0
            else 0.0
        )
        latest_target_progress_pct = latest.median_hashrate / target_rate * 100.0 if target_rate > 0.0 else 0.0
        best_target_progress_pct = best.median_hashrate / target_rate * 100.0 if target_rate > 0.0 else 0.0
        latest_remaining_multiplier = target_rate / latest.median_hashrate if latest.median_hashrate > 0.0 else 0.0
        best_remaining_multiplier = target_rate / best.median_hashrate if best.median_hashrate > 0.0 else 0.0
        summaries.append(
            TrustedTrendSummary(
                difficulty_label=difficulty_label,
                trusted_points=len(values),
                first_median_hashrate=first.median_hashrate,
                latest_median_hashrate=latest.median_hashrate,
                best_median_hashrate=best.median_hashrate,
                target_multiplier=target_multiplier,
                target_median_hashrate=target_rate,
                latest_gain_pct=latest_gain_pct,
                best_gain_pct=best_gain_pct,
                latest_target_progress_pct=latest_target_progress_pct,
                best_target_progress_pct=best_target_progress_pct,
                latest_remaining_multiplier=latest_remaining_multiplier,
                best_remaining_multiplier=best_remaining_multiplier,
                latest_spread_pct=latest.spread_pct,
                best_spread_pct=best.spread_pct,
                best_batch_label=best.batch_label,
                best_source=best.source,
                best_scenario=best.name,
            )
        )

    summaries.sort(key=lambda item: (min(int(part) for part in item.difficulty_label.split("x")), item.difficulty_label))
    return summaries


def write_summary(points: list[TrendPoint], output: Path) -> int:
    summaries = trusted_trend_summaries(points)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "xenblocks.hashapi.trusted_trend_summary.v1",
                "summaries": [summary.__dict__ for summary in summaries],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(summaries)


def _json_for_html(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def _render_html(points: list[TrendPoint], min_difficulty: int, page_refresh_seconds: float = 0.0) -> str:
    rows = [point.__dict__ for point in points]
    title = f"Hash Benchmark Trends d{min_difficulty}+"
    refresh_meta = ""
    if page_refresh_seconds > 0.0:
        refresh_meta = f'<meta http-equiv="refresh" content="{max(1, int(page_refresh_seconds))}">\n'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_meta}<meta name="robots" content="noindex">
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
    <label>Metric <select id="metric"><option value="median_hashrate" selected>Median H/s</option><option value="median_ms_per_attempt">Median ms/attempt</option><option value="compute_pct">Compute %</option><option value="kernel_pct">Kernel %</option><option value="spread_pct">Spread %</option></select></label>
    <label>GPU First Blocks <select id="gfb"><option value="all">All</option><option value="true">true</option><option value="false">false</option></select></label>
    <label>Quality <select id="quality"><option value="stable" selected>Warm Stable + Quality OK</option><option value="good">Warm Quality OK</option><option value="all">Diagnostics</option></select></label>
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
      <thead><tr><th>Difficulty</th><th>Trusted Points</th><th>Latest H/s</th><th>Best H/s</th><th>Best Gain</th><th>11x Target H/s</th><th>Best Target</th><th>Remaining</th></tr></thead>
      <tbody id="difficultyRows"></tbody>
    </table>
  </section>
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
const metricSelect = document.getElementById('metric');
const gfbSelect = document.getElementById('gfb');
const qualitySelect = document.getElementById('quality');
const searchInput = document.getElementById('search');
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const seriesColors = ['#0f766e', '#2563eb', '#9333ea', '#c2410c', '#475569', '#be123c'];

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

function metricInfo() {{
  const metric = metricSelect.value;
  if (metric === 'median_ms_per_attempt') return {{ key: metric, label: 'Median ms/attempt', digits: 4, suffix: '', lowerIsBetter: true }};
  if (metric === 'compute_pct') return {{ key: metric, label: 'Compute %', digits: 2, suffix: '%', lowerIsBetter: false }};
  if (metric === 'kernel_pct') return {{ key: metric, label: 'Kernel %', digits: 2, suffix: '%', lowerIsBetter: false }};
  if (metric === 'spread_pct') return {{ key: metric, label: 'Spread %', digits: 2, suffix: '%', lowerIsBetter: true }};
  return {{ key: 'median_hashrate', label: 'Median H/s', digits: 2, suffix: '', lowerIsBetter: false }};
}}

function metricValue(point, info = metricInfo()) {{
  const value = Number(point[info.key]);
  return Number.isFinite(value) ? value : 0;
}}

function fmtMetric(value, info = metricInfo()) {{
  return `${{fmt(value, info.digits)}}${{info.suffix}}`;
}}

function filtered() {{
  const difficulty = difficultySelect.value;
  const gfb = gfbSelect.value;
  const quality = qualitySelect.value;
  const search = searchInput.value.trim().toLowerCase();
  return points.filter(p => {{
    if (difficulty !== 'all' && p.difficulty_label !== difficulty) return false;
    if (gfb !== 'all' && String(p.gpu_first_blocks) !== gfb) return false;
    if (quality === 'good' && (!p.run_ok || !p.warm_evidence || !p.quality_ok)) return false;
    if (quality === 'stable' && (!p.run_ok || !p.warm_evidence || !p.quality_ok || !p.stable)) return false;
    if (search && !(p.name.toLowerCase().includes(search) || p.source.toLowerCase().includes(search))) return false;
    return true;
  }});
}}

function groupedByDifficulty(data) {{
  const groups = new Map();
  data.forEach((point, index) => {{
    const key = point.difficulty_label;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({{ ...point, trend_index: index }});
  }});
  return [...groups.entries()].map(([label, values], index) => ({{
    label,
    values,
    color: seriesColors[index % seriesColors.length],
  }}));
}}

function trustedGainFor(data, referencePoint) {{
  if (!referencePoint) return {{ latest: null, best: null }};
  const trusted = data.filter(p =>
    p.quality_ok &&
    p.run_ok &&
    p.warm_evidence &&
    p.stable &&
    p.median_hashrate > 0 &&
    p.difficulty_label === referencePoint.difficulty_label
  );
  const firstTrusted = trusted[0];
  const latestTrusted = trusted[trusted.length - 1];
  const bestTrusted = trusted.reduce((acc, p) => p.median_hashrate > acc.median_hashrate ? p : acc, {{ median_hashrate: 0 }});
  const gainPct = (point) => firstTrusted && point && firstTrusted.median_hashrate > 0
    ? ((point.median_hashrate - firstTrusted.median_hashrate) / firstTrusted.median_hashrate * 100)
    : null;
  return {{
    latest: gainPct(latestTrusted),
    best: gainPct(bestTrusted),
  }};
}}

function difficultySummaries(data) {{
  return groupedByDifficulty(data).map(group => {{
    const trusted = group.values.filter(p => p.run_ok && p.warm_evidence && p.quality_ok && p.stable && p.median_hashrate > 0);
    const first = trusted[0];
    const latest = trusted[trusted.length - 1];
    const best = trusted.reduce((acc, p) => p.median_hashrate > acc.median_hashrate ? p : acc, {{ median_hashrate: 0 }});
    const targetRate = first ? first.median_hashrate * 11 : 0;
    const bestGain = first && best && first.median_hashrate > 0
      ? ((best.median_hashrate - first.median_hashrate) / first.median_hashrate * 100)
      : null;
    const bestTargetProgress = targetRate > 0 && best.median_hashrate > 0
      ? (best.median_hashrate / targetRate * 100)
      : null;
    const remainingMultiplier = targetRate > 0 && best.median_hashrate > 0
      ? (targetRate / best.median_hashrate)
      : null;
    return {{
      label: group.label,
      trustedCount: trusted.length,
      latestRate: latest ? latest.median_hashrate : 0,
      bestRate: best.median_hashrate || 0,
      bestGain,
      targetRate,
      bestTargetProgress,
      remainingMultiplier,
    }};
  }}).filter(row => row.trustedCount > 0);
}}

function drawChart(data) {{
  const info = metricInfo();
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
  const groups = groupedByDifficulty(data);
  const maxValue = Math.max(...data.map(p => metricValue(p, info)), 1);
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.fillStyle = '#5f6b7a';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText(info.label, pad.left, 12);
  for (let i = 0; i <= 4; i++) {{
    const y = pad.top + (plotH * i / 4);
    const value = maxValue * (1 - i / 4);
    ctx.strokeStyle = '#edf0f5';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(fmtMetric(value, info), 8, y + 4);
  }}
  groups.forEach(group => {{
    ctx.strokeStyle = group.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    group.values.forEach((p, i) => {{
      const x = pad.left + (data.length === 1 ? plotW / 2 : plotW * p.trend_index / (data.length - 1));
      const y = pad.top + plotH * (1 - metricValue(p, info) / maxValue);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
    group.values.forEach(p => {{
      const x = pad.left + (data.length === 1 ? plotW / 2 : plotW * p.trend_index / (data.length - 1));
      const y = pad.top + plotH * (1 - metricValue(p, info) / maxValue);
      ctx.fillStyle = p.run_ok && p.warm_evidence && p.quality_ok && p.stable ? group.color : (p.run_ok && p.quality_ok ? '#b45309' : '#b91c1c');
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }});
  }});
  ctx.font = '12px system-ui, sans-serif';
  groups.slice(0, 6).forEach((group, index) => {{
    const x = pad.left + 8 + (index % 3) * 150;
    const y = pad.top + 14 + Math.floor(index / 3) * 18;
    ctx.fillStyle = group.color;
    ctx.fillRect(x, y - 8, 10, 10);
    ctx.fillStyle = '#5f6b7a';
    ctx.fillText(`d${{group.label}}`, x + 16, y + 1);
  }});
}}

function render() {{
  const data = filtered();
  const best = data.reduce((acc, p) => Math.max(acc, p.median_hashrate), 0);
  const latest = data[data.length - 1];
  const gains = trustedGainFor(data, latest);
  document.getElementById('visibleCount').textContent = String(data.length);
  document.getElementById('bestRate').textContent = fmt(best, 2);
  document.getElementById('latestRate').textContent = latest ? fmt(latest.median_hashrate, 2) : '0';
  document.getElementById('latestSpread').textContent = latest ? `${{fmt(latest.spread_pct, 2)}}%` : '0%';
  document.getElementById('latestTrustedGain').textContent = gains.latest === null ? 'n/a' : `${{fmt(gains.latest, 2)}}%`;
  document.getElementById('bestTrustedGain').textContent = gains.best === null ? 'n/a' : `${{fmt(gains.best, 2)}}%`;
  drawChart(data);
  document.getElementById('difficultyRows').innerHTML = difficultySummaries(data).map(row => `
    <tr>
      <td>d${{row.label}}</td>
      <td>${{row.trustedCount}}</td>
      <td>${{fmt(row.latestRate, 3)}}</td>
      <td>${{fmt(row.bestRate, 3)}}</td>
      <td>${{row.bestGain === null ? 'n/a' : `${{fmt(row.bestGain, 2)}}%`}}</td>
      <td>${{fmt(row.targetRate, 3)}}</td>
      <td>${{row.bestTargetProgress === null ? 'n/a' : `${{fmt(row.bestTargetProgress, 2)}}%`}}</td>
      <td>${{row.remainingMultiplier === null ? 'n/a' : `${{fmt(row.remainingMultiplier, 2)}}x`}}</td>
    </tr>`).join('');
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
      <td>${{p.run_ok ? (p.warm_evidence ? (p.quality_ok ? (p.stable ? 'stable' : 'ok') : 'low') : 'cold') : 'invalid'}}</td>
      <td class="name" title="${{p.name}}">${{p.name}}</td>
      <td class="name" title="${{p.source}}">${{p.source}}</td>
    </tr>`).join('');
}}

[difficultySelect, metricSelect, gfbSelect, qualitySelect, searchInput].forEach(el => el.addEventListener('input', render));
window.addEventListener('resize', render);
setupFilters();
render();
</script>
</body>
</html>
"""


def write_trend_page(
    input_dir: Path,
    output: Path,
    min_difficulty: int,
    page_refresh_seconds: float = 0.0,
    summary_output: Path | None = None,
) -> int:
    points = load_points(input_dir, min_difficulty)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(points, min_difficulty, page_refresh_seconds), encoding="utf-8")
    if summary_output is not None:
        write_summary(points, summary_output)
    return len(points)


def input_signature(input_dir: Path) -> tuple[int, int, int]:
    count = 0
    latest_mtime_ns = 0
    total_size = 0
    for path in input_dir.glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
        total_size += stat.st_size
    return count, latest_mtime_ns, total_size


@dataclass
class TrendPageCache:
    input_dir: Path
    output: Path
    min_difficulty: int
    page_refresh_seconds: float
    refresh_seconds: float
    summary_output: Path | None = None
    last_refresh: float = 0.0
    points: int = 0
    cached_signature: tuple[int, int, int] | None = None

    def refresh(self, force: bool = False, now: float | None = None) -> int:
        current_time = time.monotonic() if now is None else now
        should_check_inputs = force or current_time - self.last_refresh >= self.refresh_seconds
        if should_check_inputs:
            signature = input_signature(self.input_dir)
            if force or signature != self.cached_signature or not self.output.exists():
                self.points = write_trend_page(
                    self.input_dir,
                    self.output,
                    self.min_difficulty,
                    self.page_refresh_seconds,
                )
                if self.summary_output is not None:
                    write_summary(load_points(self.input_dir, self.min_difficulty), self.summary_output)
                self.cached_signature = signature
            self.last_refresh = current_time
        return self.points


def serve_trends(
    input_dir: Path,
    output: Path,
    min_difficulty: int,
    host: str,
    port: int,
    refresh_seconds: float,
    page_refresh_seconds: float,
    open_browser: bool,
    summary_output: Path | None = None,
) -> int:
    root = output.parent.resolve()
    output_name = output.name
    lock = threading.Lock()
    cache = TrendPageCache(input_dir, output, min_difficulty, page_refresh_seconds, refresh_seconds, summary_output)

    def refresh(force: bool = False) -> int:
        with lock:
            return cache.refresh(force)

    class TrendRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self) -> None:
            request_path = urlsplit(self.path).path
            if request_path in {"/", f"/{output_name}"}:
                refresh()
            if request_path == "/":
                self.path = f"/{output_name}"
            super().do_GET()

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    initial_points = refresh(force=True)
    with ThreadingHTTPServer((host, port), TrendRequestHandler) as server:
        url = f"http://{host}:{server.server_port}/"
        print(
            f"serving {output_name} with {initial_points} public-safe points at "
            f"{url}"
        )
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped trend server")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(".benchmarks"), help="Directory containing benchmark JSON reports.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/hash-trends/index.html"),
        help="Generated HTML path. Keep this under ignored benchmark storage.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional public-safe trusted trend summary JSON path. Keep this under ignored benchmark storage.",
    )
    parser.add_argument("--min-difficulty", type=int, default=4096, help="Minimum difficulty to include.")
    parser.add_argument("--serve", action="store_true", help="Serve the generated trend page with automatic refresh.")
    parser.add_argument("--host", default="localhost", help="Host used by --serve.")
    parser.add_argument("--port", type=int, default=8766, help="Port used by --serve. Use 0 to choose a free port.")
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=5.0,
        help="Minimum seconds between benchmark directory rescans while serving.",
    )
    parser.add_argument(
        "--page-refresh-seconds",
        type=float,
        default=None,
        help="Browser auto-refresh interval. Defaults to 10 while serving and disabled for one-shot output.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local trend page in the default browser after the server starts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    page_refresh_seconds = args.page_refresh_seconds
    if page_refresh_seconds is None:
        page_refresh_seconds = 10.0 if args.serve else 0.0
    if args.serve:
        return serve_trends(
            args.input_dir,
            args.output,
            args.min_difficulty,
            args.host,
            args.port,
            max(0.0, args.refresh_seconds),
            max(0.0, page_refresh_seconds),
            args.open_browser,
            args.summary_output,
        )
    points = write_trend_page(
        args.input_dir,
        args.output,
        args.min_difficulty,
        max(0.0, page_refresh_seconds),
        args.summary_output,
    )
    print(f"wrote {args.output} with {points} public-safe points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
