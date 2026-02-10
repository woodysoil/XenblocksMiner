"""
dashboard.py - Minimal web dashboard for the hashpower marketplace.

Serves a single-page HTML dashboard at /dashboard that auto-refreshes via
JavaScript fetch() polling every 3 seconds. All data comes from existing
REST API endpoints.

Usage:
    Import and call register_dashboard(app) from server.py to add routes.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


def register_dashboard(app: FastAPI):
    """Register the dashboard route on the given FastAPI app."""

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return _DASHBOARD_HTML


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XenMiner Hashpower Marketplace</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text2: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
    background: var(--bg); color: var(--text); font-size: 14px;
    line-height: 1.5;
  }
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; align-items: center;
    justify-content: space-between;
  }
  header h1 { font-size: 16px; font-weight: 600; }
  header h1 span { color: var(--accent); }
  .status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); margin-right: 6px; animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  .header-info { font-size: 12px; color: var(--text2); }

  .container { max-width: 1200px; margin: 0 auto; padding: 16px; }

  /* Summary cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 14px 16px;
  }
  .card-label { font-size: 11px; text-transform: uppercase; color: var(--text2); letter-spacing: 0.5px; }
  .card-value { font-size: 28px; font-weight: 700; margin-top: 2px; }
  .card-value.green { color: var(--green); }
  .card-value.accent { color: var(--accent); }
  .card-value.yellow { color: var(--yellow); }

  /* Tabs */
  .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
  .tab {
    padding: 8px 16px; cursor: pointer; color: var(--text2);
    border-bottom: 2px solid transparent; font-size: 13px;
    transition: all 0.15s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* Tables */
  .panel { display: none; }
  .panel.active { display: block; }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left; font-size: 11px; text-transform: uppercase;
    color: var(--text2); padding: 8px 10px; border-bottom: 1px solid var(--border);
    letter-spacing: 0.5px;
  }
  td {
    padding: 8px 10px; border-bottom: 1px solid var(--border);
    font-size: 13px; font-family: monospace;
  }
  tr:hover td { background: rgba(88, 166, 255, 0.04); }

  /* State badges */
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
  }
  .badge-available, .badge-AVAILABLE { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge-mining, .badge-MINING { background: rgba(88,166,255,0.15); color: var(--accent); }
  .badge-leased, .badge-LEASED { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .badge-active { background: rgba(88,166,255,0.15); color: var(--accent); }
  .badge-completed, .badge-COMPLETED { background: rgba(139,148,158,0.15); color: var(--text2); }
  .badge-error, .badge-ERROR { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge-idle, .badge-IDLE, .badge-offline { background: rgba(139,148,158,0.1); color: var(--text2); }
  .badge-self { background: rgba(210,153,34,0.15); color: var(--yellow); }

  /* Rent form */
  .rent-form {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 20px; max-width: 480px;
  }
  .rent-form h3 { font-size: 14px; margin-bottom: 12px; }
  .form-row { margin-bottom: 10px; }
  .form-row label { display: block; font-size: 11px; color: var(--text2); margin-bottom: 3px; text-transform: uppercase; }
  .form-row input, .form-row select {
    width: 100%; padding: 6px 10px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); font-family: monospace; font-size: 13px;
  }
  .form-row input:focus, .form-row select:focus { outline: none; border-color: var(--accent); }
  .btn {
    padding: 8px 20px; background: var(--accent); color: #fff;
    border: none; border-radius: 4px; cursor: pointer; font-size: 13px;
    font-weight: 600;
  }
  .btn:hover { opacity: 0.9; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-danger { background: var(--red); }
  .result-msg { margin-top: 10px; font-size: 12px; padding: 8px; border-radius: 4px; }
  .result-msg.ok { background: rgba(63,185,80,0.1); color: var(--green); }
  .result-msg.err { background: rgba(248,81,73,0.1); color: var(--red); }

  .truncate { max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; vertical-align: bottom; }
  .text-muted { color: var(--text2); }
</style>
</head>
<body>

<header>
  <h1><span>XenMiner</span> Hashpower Marketplace</h1>
  <div class="header-info"><span class="status-dot"></span>Live &mdash; polling every 5s</div>
</header>

<div class="container">
  <!-- Summary Cards -->
  <div class="cards">
    <div class="card">
      <div class="card-label">Workers Online</div>
      <div class="card-value green" id="card-workers">-</div>
    </div>
    <div class="card">
      <div class="card-label">Active Leases</div>
      <div class="card-value accent" id="card-leases">-</div>
    </div>
    <div class="card">
      <div class="card-label">Total Blocks</div>
      <div class="card-value yellow" id="card-blocks">-</div>
    </div>
    <div class="card">
      <div class="card-label">Self-Mined</div>
      <div class="card-value yellow" id="card-self-mined">-</div>
    </div>
    <div class="card">
      <div class="card-label">Settlements</div>
      <div class="card-value" id="card-settlements">-</div>
    </div>
    <div class="card">
      <div class="card-label">MQTT Clients</div>
      <div class="card-value" id="card-mqtt">-</div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" data-panel="workers">Workers</div>
    <div class="tab" data-panel="leases">Leases</div>
    <div class="tab" data-panel="blocks">Blocks</div>
    <div class="tab" data-panel="accounts">Accounts</div>
    <div class="tab" data-panel="rent">Rent Hashpower</div>
    <div class="tab" data-panel="control">Control</div>
  </div>

  <!-- Workers Panel -->
  <div class="panel active" id="panel-workers">
    <table>
      <thead>
        <tr><th>Worker ID</th><th>State</th><th>Reputation</th><th>GPUs</th><th>Memory</th><th>Hashrate</th><th>Self Blocks</th><th>Config</th><th>$/min</th><th>Duration</th><th>Address</th><th>Last HB</th></tr>
      </thead>
      <tbody id="tbody-workers"></tbody>
    </table>
  </div>

  <!-- Leases Panel -->
  <div class="panel" id="panel-leases">
    <table>
      <thead>
        <tr><th>Lease ID</th><th>State</th><th>Worker</th><th>Consumer</th><th>Prefix</th><th>Duration</th><th>Blocks</th><th>Hashrate</th><th>Action</th></tr>
      </thead>
      <tbody id="tbody-leases"></tbody>
    </table>
  </div>

  <!-- Blocks Panel -->
  <div class="panel" id="panel-blocks">
    <table>
      <thead>
        <tr><th>Lease ID</th><th>Worker</th><th>Hash</th><th>Key Prefix</th><th>Prefix OK</th><th>Chain Verified</th><th>Attempts</th><th>Hashrate</th></tr>
      </thead>
      <tbody id="tbody-blocks"></tbody>
    </table>
  </div>

  <!-- Accounts Panel -->
  <div class="panel" id="panel-accounts">
    <table>
      <thead>
        <tr><th>Account ID</th><th>Role</th><th>Balance</th><th>Address</th></tr>
      </thead>
      <tbody id="tbody-accounts"></tbody>
    </table>
  </div>

  <!-- Rent Panel -->
  <div class="panel" id="panel-rent">
    <div class="rent-form">
      <h3>Rent Hashpower</h3>
      <div class="form-row">
        <label>Consumer ID</label>
        <input type="text" id="rent-consumer-id" value="consumer-1" />
      </div>
      <div class="form-row">
        <label>Consumer Address (0x...)</label>
        <input type="text" id="rent-consumer-addr" value="0xaabbccddee1234567890abcdef1234567890abcd" />
      </div>
      <div class="form-row">
        <label>Worker (optional, blank for any)</label>
        <select id="rent-worker"><option value="">(any available)</option></select>
      </div>
      <div class="form-row">
        <label>Duration (seconds)</label>
        <input type="number" id="rent-duration" value="3600" min="10" />
      </div>
      <button class="btn" id="btn-rent" onclick="doRent()">Rent</button>
      <div id="rent-result"></div>
    </div>
  </div>

  <!-- Control Panel -->
  <div class="panel" id="panel-control">
    <div class="rent-form" style="max-width:560px">
      <h3>Remote Control</h3>
      <div class="form-row">
        <label>Target Worker</label>
        <select id="ctrl-worker"></select>
      </div>
      <div class="form-row">
        <label>Difficulty</label>
        <div style="display:flex;gap:8px">
          <input type="number" id="ctrl-diff" min="1" placeholder="e.g. 1727" style="flex:1" />
          <button class="btn" onclick="applyConfig('difficulty',parseInt(document.getElementById('ctrl-diff').value))">Apply</button>
        </div>
      </div>
      <div class="form-row">
        <label>Mining Address (0x...)</label>
        <div style="display:flex;gap:8px">
          <input type="text" id="ctrl-addr" placeholder="0x..." style="flex:1" />
          <button class="btn" onclick="applyConfig('address',document.getElementById('ctrl-addr').value)">Apply</button>
        </div>
      </div>
      <div class="form-row">
        <label>Key Prefix (hex)</label>
        <div style="display:flex;gap:8px">
          <input type="text" id="ctrl-prefix" placeholder="e.g. DEADBEEF" style="flex:1" />
          <button class="btn" onclick="applyConfig('prefix','')">Clear</button>
          <button class="btn" onclick="applyConfig('prefix',document.getElementById('ctrl-prefix').value)">Apply</button>
        </div>
      </div>
      <div class="form-row">
        <label>Block Pattern</label>
        <div style="display:flex;gap:8px">
          <input type="text" id="ctrl-pattern" placeholder="e.g. XEN11" style="flex:1" />
          <button class="btn" onclick="applyConfig('block_pattern',document.getElementById('ctrl-pattern').value)">Apply</button>
        </div>
      </div>
      <div style="margin-top:14px;display:flex;gap:8px">
        <button class="btn" onclick="doControl(getCtrlWorker(),'pause',{})">Pause</button>
        <button class="btn" onclick="doControl(getCtrlWorker(),'resume',{})">Resume</button>
        <button class="btn btn-danger" onclick="if(confirm('Shutdown miner?'))doControl(getCtrlWorker(),'shutdown',{})">Shutdown</button>
      </div>
      <div id="ctrl-result"></div>
    </div>
  </div>
</div>

<script>
const API = '';  // same origin

// ── Tab switching ──
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.panel).classList.add('active');
  });
});

function badge(state) {
  const s = (state || 'unknown').toLowerCase();
  return `<span class="badge badge-${s}">${state}</span>`;
}

function truncate(s, n) {
  if (!s) return '<span class="text-muted">-</span>';
  if (s.length <= n) return s;
  return `<span class="truncate" title="${s}">${s}</span>`;
}

function renderStars(stars, score) {
  const full = Math.floor(stars);
  const half = stars - full >= 0.3 ? 1 : 0;
  const empty = 5 - full - half;
  let html = '';
  for (let i = 0; i < full; i++) html += '<span style="color:var(--yellow)">&#9733;</span>';
  for (let i = 0; i < half; i++) html += '<span style="color:var(--yellow)">&#9734;</span>';
  for (let i = 0; i < empty; i++) html += '<span style="color:var(--border)">&#9734;</span>';
  html += ` <span class="text-muted" style="font-size:11px">${score}</span>`;
  return html;
}

function timeAgo(ts) {
  if (!ts) return '-';
  const diff = Math.floor(Date.now()/1000 - ts);
  if (diff < 0) return 'just now';
  if (diff < 60) return diff + 's ago';
  if (diff < 3600) return Math.floor(diff/60) + 'm ago';
  return Math.floor(diff/3600) + 'h ago';
}

// ── Diffing helper: only replace innerHTML when content changed ──
function updateHTML(el, html) {
  if (el._prevHTML !== html) { el.innerHTML = html; el._prevHTML = html; }
}
function updateText(el, text) {
  if (el.textContent !== String(text)) el.textContent = text;
}

// ── Data fetchers ──
async function fetchJSON(url) {
  try { const r = await fetch(API + url); return await r.json(); }
  catch(e) { return null; }
}

async function refresh() {
  // Status cards
  const st = await fetchJSON('/api/status');
  if (st) {
    updateText(document.getElementById('card-workers'), st.workers ?? 0);
    updateText(document.getElementById('card-leases'), st.active_leases ?? 0);
    updateText(document.getElementById('card-blocks'), st.total_blocks ?? 0);
    updateText(document.getElementById('card-self-mined'), st.self_mined_blocks ?? 0);
    updateText(document.getElementById('card-settlements'), st.total_settlements ?? 0);
    updateText(document.getElementById('card-mqtt'), (st.mqtt_clients || []).length);
  }

  // Only fetch data for the active panel + always refresh workers (for rent dropdown)
  const activePanel = document.querySelector('.panel.active')?.id || 'panel-workers';

  // Workers (always needed for rent dropdown)
  const workers = await fetchJSON('/api/workers');
  if (workers) {
    const sel = document.getElementById('rent-worker');
    const prev = sel.value;
    const tbody = document.getElementById('tbody-workers');

    // Fetch reputation for all workers in parallel
    const workerList = Array.isArray(workers) ? workers : [];
    const repPromises = workerList.map(w => fetchJSON('/api/workers/' + w.worker_id + '/reputation'));
    const reps = await Promise.all(repPromises);
    const repMap = {};
    reps.forEach((r, i) => { if (r) repMap[workerList[i].worker_id] = r; });

    let rowsHTML = '';
    let optHTML = '<option value="">(any available)</option>';
    workerList.forEach(w => {
      const price = typeof w.price_per_min === 'number' ? '$' + w.price_per_min.toFixed(2) : '-';
      const minD = w.min_duration_sec || 60;
      const maxD = w.max_duration_sec || 86400;
      const durRange = minD + '-' + maxD + 's';
      const rep = repMap[w.worker_id];
      const stars = rep ? renderStars(rep.stars, rep.score) : '<span class="text-muted">-</span>';
      rowsHTML += `<tr>
        <td>${w.worker_id}</td>
        <td>${badge(w.state)}</td>
        <td>${stars}</td>
        <td>${w.gpu_count}</td>
        <td>${w.total_memory_gb} GB</td>
        <td>${typeof w.hashrate === 'number' ? w.hashrate.toFixed(1) : w.hashrate || '-'} H/s</td>
        <td>${w.self_blocks_found || 0}</td>
        <td style="font-size:11px">${configSummary(w)}</td>
        <td>${price}</td>
        <td>${durRange}</td>
        <td>${truncate(w.eth_address, 14)}</td>
        <td>${timeAgo(w.last_heartbeat)}</td>
      </tr>`;
      if (w.state === 'AVAILABLE') {
        const wPrice = typeof w.price_per_min === 'number' ? ' $' + w.price_per_min.toFixed(2) + '/min' : '';
        optHTML += `<option value="${w.worker_id}">${w.worker_id} (${w.gpu_count} GPUs${wPrice})</option>`;
      }
    });
    updateHTML(tbody, rowsHTML);
    updateHTML(sel, optHTML);
    sel.value = prev;

    // Populate control panel worker dropdown
    const ctrlSel = document.getElementById('ctrl-worker');
    const ctrlPrev = ctrlSel.value;
    let ctrlOptHTML = '<option value="__all__">All Workers</option>';
    workerList.forEach(w => {
      ctrlOptHTML += `<option value="${w.worker_id}">${w.worker_id}</option>`;
    });
    updateHTML(ctrlSel, ctrlOptHTML);
    ctrlSel.value = ctrlPrev || '__all__';
  }

  // Leases - only fetch when visible
  if (activePanel === 'panel-leases') {
    const leases = await fetchJSON('/api/leases');
    if (leases) {
      const tbody = document.getElementById('tbody-leases');
      let html = '';
      (Array.isArray(leases) ? leases : []).forEach(l => {
        const action = l.state === 'active'
          ? `<button class="btn btn-danger" style="padding:2px 8px;font-size:11px" onclick="doStop('${l.lease_id}')">Stop</button>`
          : '';
        html += `<tr>
          <td>${truncate(l.lease_id, 20)}</td>
          <td>${badge(l.state)}</td>
          <td>${l.worker_id}</td>
          <td>${truncate(l.consumer_id, 14)}</td>
          <td><code>${l.prefix}</code></td>
          <td>${l.elapsed_sec || 0}s / ${l.duration_sec}s</td>
          <td>${l.blocks_found}</td>
          <td>${(l.avg_hashrate || 0).toFixed(1)} H/s</td>
          <td>${action}</td>
        </tr>`;
      });
      updateHTML(tbody, html);
    }
  }

  // Blocks - only fetch when visible
  if (activePanel === 'panel-blocks') {
    const blocks = await fetchJSON('/api/blocks');
    if (blocks) {
      const tbody = document.getElementById('tbody-blocks');
      let html = '';
      (Array.isArray(blocks) ? blocks : []).forEach(b => {
        const pfx = b.prefix_valid !== undefined
          ? (b.prefix_valid ? '<span style="color:var(--green)">Yes</span>' : '<span style="color:var(--red)">No</span>')
          : '-';
        const cv = b.chain_verified !== undefined
          ? (b.chain_verified ? '<span style="color:var(--green)">Yes</span>' : '<span style="color:var(--red)">No</span>')
          : '-';
        html += `<tr>
          <td>${b.lease_id ? truncate(b.lease_id, 16) : '<span class="badge badge-self">self</span>'}</td>
          <td>${b.worker_id}</td>
          <td>${truncate(b.block_hash || b.hash, 18)}</td>
          <td><code>${(b.key || '').substring(0, 16)}</code></td>
          <td>${pfx}</td>
          <td>${cv}</td>
          <td>${(b.attempts || 0).toLocaleString()}</td>
          <td>${b.hashrate || '-'}</td>
        </tr>`;
      });
      updateHTML(tbody, html);
    }
  }

  // Accounts - only fetch when visible
  if (activePanel === 'panel-accounts') {
    const accounts = await fetchJSON('/api/accounts');
    if (accounts) {
      const tbody = document.getElementById('tbody-accounts');
      let html = '';
      const list = Array.isArray(accounts) ? accounts : Object.values(accounts);
      list.forEach(a => {
        html += `<tr>
          <td>${a.account_id}</td>
          <td>${badge(a.role)}</td>
          <td>${(a.balance || 0).toFixed(4)}</td>
          <td>${truncate(a.eth_address, 18)}</td>
        </tr>`;
      });
      updateHTML(tbody, html);
    }
  }
}

// ── Actions ──

function configSummary(w) {
  const parts = [];
  if (w.current_prefix) parts.push('pfx:' + w.current_prefix.substring(0,8));
  if (w.current_block_pattern) parts.push('pat:' + w.current_block_pattern);
  if (parts.length === 0) return '<span class="text-muted">default</span>';
  return parts.join(' ');
}

function getCtrlWorker() {
  return document.getElementById('ctrl-worker').value;
}

async function doControl(workerId, action, config) {
  const url = workerId === '__all__'
    ? API + '/api/control/broadcast'
    : API + '/api/workers/' + workerId + '/control';
  const result = document.getElementById('ctrl-result');
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action, config}),
    });
    const data = await r.json();
    if (r.ok) {
      result.className = 'result-msg ok';
      result.textContent = 'Sent: ' + action + ' → ' + (data.worker_id || (data.workers||[]).join(', '));
    } else {
      result.className = 'result-msg err';
      result.textContent = 'Error: ' + (data.detail || JSON.stringify(data));
    }
  } catch(e) {
    result.className = 'result-msg err';
    result.textContent = 'Request failed: ' + e.message;
  }
}

function applyConfig(key, value) {
  const wid = getCtrlWorker();
  if (!wid) { alert('Select a worker first'); return; }
  const config = {};
  config[key] = value;
  doControl(wid, 'set_config', config);
}

async function doRent() {
  const btn = document.getElementById('btn-rent');
  const result = document.getElementById('rent-result');
  btn.disabled = true;

  const body = {
    consumer_id: document.getElementById('rent-consumer-id').value,
    consumer_address: document.getElementById('rent-consumer-addr').value,
    duration_sec: parseInt(document.getElementById('rent-duration').value) || 3600,
  };
  const wid = document.getElementById('rent-worker').value;
  if (wid) body.worker_id = wid;

  try {
    const r = await fetch(API + '/api/rent', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (r.ok) {
      result.className = 'result-msg ok';
      result.textContent = 'Lease created: ' + data.lease_id + ' (prefix: ' + data.prefix + ')';
    } else {
      result.className = 'result-msg err';
      result.textContent = 'Error: ' + (data.detail || JSON.stringify(data));
    }
  } catch(e) {
    result.className = 'result-msg err';
    result.textContent = 'Request failed: ' + e.message;
  }
  btn.disabled = false;
}

async function doStop(leaseId) {
  try {
    const r = await fetch(API + '/api/stop', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({lease_id: leaseId}),
    });
    if (!r.ok) {
      const data = await r.json();
      alert('Stop failed: ' + (data.detail || 'unknown error'));
    }
  } catch(e) {
    alert('Stop failed: ' + e.message);
  }
}

// ── Start polling ──
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
