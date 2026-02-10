# XenBlocks Mining Platform -- Deployment & Operations Guide

## 1. Prerequisites

| Component  | Minimum Version | Purpose                          |
|------------|-----------------|----------------------------------|
| Python     | 3.11+           | Backend server, simulator        |
| Node.js    | 20+             | Frontend build toolchain         |
| npm        | 10+             | Package management               |
| SQLite     | 3.35+           | Bundled via `aiosqlite` (no separate install) |

Verify your environment:

```bash
python3 --version   # >= 3.11
node --version      # >= 20.0
npm --version       # >= 10.0
```

---

## 2. Backend Setup

### 2.1 Virtual Environment

```bash
cd /path/to/XenblocksMiner
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

Dependencies (`server/requirements.txt`):

```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
aiosqlite>=0.19.0
eth-account>=0.13.0
PyJWT>=2.9.0
```

### 2.2 Starting the Server

```bash
python3 -m server.server
```

Or via the helper script:

```bash
./scripts/run_mock_server.sh
```

### 2.3 CLI Options

| Flag              | Default                  | Description                                           |
|-------------------|--------------------------|-------------------------------------------------------|
| `--api-port`      | `8080`                   | REST API / WebSocket listen port                      |
| `--mqtt-port`     | `1883`                   | Embedded MQTT broker listen port                      |
| `--db-path`       | `data/marketplace.db`    | SQLite database file path                             |
| `--jwt-secret`    | *(auto-generated)*       | HMAC-SHA256 key for JWT signing                       |
| `--block-marker`  | `""` (uses default XEN11)| Override block detection marker for testing            |
| `--no-chain`      | *(off)*                  | Disable the embedded chain simulator                  |

Example -- production-like start:

```bash
python3 -m server.server \
  --api-port 8080 \
  --mqtt-port 1883 \
  --db-path /var/lib/xenblocks/marketplace.db \
  --jwt-secret "$(cat /etc/xenblocks/jwt-secret)"
```

---

## 3. Frontend Build

### 3.1 Install Dependencies

```bash
cd web
npm install
```

### 3.2 Development Server

```bash
npm run dev
```

Vite dev server starts on `http://localhost:5173` and proxies:

- `/api/*` to `http://localhost:8080`
- `/ws/*` to `ws://localhost:8080` (WebSocket)

### 3.3 Production Build

```bash
npm run build
```

Output lands in `web/dist/`. Serve these static files via nginx or any static file server.

### 3.4 Preview Build Locally

```bash
npm run preview
```

---

## 4. Production Configuration

### 4.1 Environment Variables

The server reads all configuration from CLI arguments. For systemd or container deployments, pass them directly:

```ini
# /etc/systemd/system/xenblocks.service
[Unit]
Description=XenBlocks Mining Platform
After=network.target

[Service]
Type=simple
User=xenblocks
WorkingDirectory=/opt/xenblocks
ExecStart=/opt/xenblocks/.venv/bin/python -m server.server \
  --api-port 8080 \
  --mqtt-port 1883 \
  --db-path /var/lib/xenblocks/marketplace.db \
  --jwt-secret-file /etc/xenblocks/jwt-secret
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now xenblocks
```

### 4.2 JWT Secret Management

Without `--jwt-secret`, the server generates an ephemeral random secret on every start. All previously issued JWTs become invalid after a restart.

For production, generate a persistent secret:

```bash
sudo mkdir -p /etc/xenblocks
python3 -c "import secrets; print(secrets.token_hex(32))" | sudo tee /etc/xenblocks/jwt-secret
sudo chmod 600 /etc/xenblocks/jwt-secret
```

Pass it at startup:

```bash
python3 -m server.server --jwt-secret "$(cat /etc/xenblocks/jwt-secret)"
```

### 4.3 Default Admin Key

The default admin API key is `admin-test-key-do-not-use-in-production` (hardcoded in `server/auth.py`). This grants full admin access to all endpoints. For any non-local deployment, review and change this in `server/auth.py`:

```python
DEFAULT_ADMIN_KEY = "admin-test-key-do-not-use-in-production"
```

### 4.4 Database File Location

Default: `data/marketplace.db` (relative to working directory).

For production, use an absolute path:

```bash
sudo mkdir -p /var/lib/xenblocks
sudo chown xenblocks:xenblocks /var/lib/xenblocks
python3 -m server.server --db-path /var/lib/xenblocks/marketplace.db
```

SQLite creates companion files alongside the database (`-wal`, `-shm`) for WAL journaling.

---

## 5. Reverse Proxy (nginx)

Serve the frontend static build and proxy API/WebSocket to the backend.

```nginx
upstream xenblocks_api {
    server 127.0.0.1:8080;
}

server {
    listen 80;
    server_name xenblocks.example.com;

    # Frontend static files
    root /opt/xenblocks/web/dist;
    index index.html;

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # REST API proxy
    location /api/ {
        proxy_pass http://xenblocks_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket proxy
    location /ws/ {
        proxy_pass http://xenblocks_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
    }

    # Built-in dashboard (served by FastAPI)
    location /dashboard {
        proxy_pass http://xenblocks_api;
        proxy_set_header Host $host;
    }
}
```

For TLS, add a `listen 443 ssl` block with your certificate and redirect port 80.

---

## 6. MQTT Setup

### 6.1 Embedded Broker

The server includes a built-in pure-Python MQTT 3.1.1 broker (`server/broker.py`). It starts automatically on the configured `--mqtt-port` (default `1883`). No external broker (Mosquitto, etc.) is required.

Capabilities:
- MQTT 3.1.1 protocol
- QoS 0 and 1
- Topic wildcard matching (`+`, `#`)
- No TLS, no authentication, no persistence (designed for local/trusted networks)

### 6.2 Topic Structure

All miner communication uses the pattern:

```
xenminer/{worker_id}/{message_type}
```

| Message Type  | Direction         | Purpose                                |
|---------------|-------------------|----------------------------------------|
| `register`    | Miner -> Server   | Worker registration with GPU info      |
| `heartbeat`   | Miner -> Server   | Periodic hashrate/status update        |
| `status`      | Miner -> Server   | State change notification              |
| `block`       | Miner -> Server   | Block discovery report                 |
| `control`     | Server -> Miner   | Remote commands (pause, resume, config)|

### 6.3 Remote Control via MQTT

The REST API publishes control messages to miners via MQTT:

```bash
# Send control to a specific worker
curl -X POST http://localhost:8080/api/workers/sim-worker-001/control \
  -H "Content-Type: application/json" \
  -d '{"action": "set_config", "config": {"difficulty": 1727}}'

# Broadcast to all available workers
curl -X POST http://localhost:8080/api/control/broadcast \
  -H "Content-Type: application/json" \
  -d '{"action": "pause", "config": {}}'
```

Available actions: `set_config`, `pause`, `resume`, `shutdown`.

Config fields: `difficulty`, `address`, `prefix`, `block_pattern`.

### 6.4 External Broker (Optional)

If you need TLS or multi-node clustering, run an external MQTT broker (e.g., Mosquitto) on port 1883 and disable the embedded one. The embedded broker currently cannot be disabled independently -- you would need to modify `server/server.py` to skip `broker.start()`.

---

## 7. Database

### 7.1 Schema

SQLite with WAL mode and foreign keys enabled. Current schema version: **10**.

Tables:

| Table                | Purpose                                    |
|----------------------|--------------------------------------------|
| `schema_version`     | Migration tracking                         |
| `accounts`           | Provider/consumer accounts with balances   |
| `workers`            | Registered mining workers and GPU metadata |
| `leases`             | Hashpower rental agreements                |
| `blocks`             | Mined block records                        |
| `settlements`        | Completed lease financial settlements      |
| `transactions`       | Balance change audit trail                 |
| `hashrate_snapshots` | Per-worker hashrate time series             |
| `wallet_snapshots`   | Per-wallet hourly/daily aggregated stats   |

### 7.2 Automatic Migrations

On startup, `StorageManager._migrate()` compares the stored `schema_version` against the code's `SCHEMA_VERSION` constant. If behind, it runs the full schema DDL plus incremental `ALTER TABLE` migrations (V2 through V10). Fresh databases get the complete schema in one pass.

No manual migration steps are required.

### 7.3 Backup Strategy

SQLite with WAL mode. Safe backup approaches:

```bash
# Option 1: sqlite3 .backup (online, consistent)
sqlite3 /var/lib/xenblocks/marketplace.db ".backup /backups/xenblocks-$(date +%Y%m%d-%H%M%S).db"

# Option 2: Copy all three files while server is running (WAL mode safe)
cp /var/lib/xenblocks/marketplace.db    /backups/
cp /var/lib/xenblocks/marketplace.db-wal /backups/ 2>/dev/null
cp /var/lib/xenblocks/marketplace.db-shm /backups/ 2>/dev/null
```

Automated daily backup via cron:

```cron
0 2 * * * sqlite3 /var/lib/xenblocks/marketplace.db ".backup /backups/xenblocks-$(date +\%Y\%m\%d).db"
```

### 7.4 Data Retention

The monitoring loop automatically cleans up:
- **Hashrate snapshots**: Deleted after 24 hours (`cleanup_old_snapshots`)
- **Wallet snapshots**: Hourly retained 7 days, daily retained 90 days

### 7.5 Generating Test Data

Populate the database with large datasets for pagination and load testing:

```bash
python3 scripts/generate_test_data.py \
  --workers 1000 \
  --blocks 50000 \
  --leases 5000 \
  --addresses 100 \
  --db data/marketplace.db
```

---

## 8. Monitoring

### 8.1 Health Check Endpoints

| Endpoint                        | Auth   | Purpose                                        |
|---------------------------------|--------|------------------------------------------------|
| `GET /`                         | None   | Service info (MQTT port, connected workers)    |
| `GET /api/status`               | None   | Worker count, active leases, block totals      |
| `GET /api/monitoring/stats`     | None   | Fleet hashrate, online/offline, GPU counts     |
| `GET /api/monitoring/fleet`     | None   | Per-worker status with pagination              |
| `GET /api/monitoring/hashrate-history` | None | Time-series hashrate data (up to 24h)   |
| `GET /api/monitoring/blocks/recent`    | None | Latest mined blocks                      |
| `GET /api/overview/stats`       | None   | Platform-wide aggregated statistics            |
| `GET /api/overview/network`     | None   | Difficulty, chain stats                        |

Quick health check for external monitoring tools:

```bash
# Basic liveness
curl -sf http://localhost:8080/ | jq .service

# Detailed status
curl -sf http://localhost:8080/api/status | jq .

# Fleet overview
curl -sf http://localhost:8080/api/monitoring/stats | jq .
```

### 8.2 WebSocket Real-Time Feed

Connect to `ws://localhost:8080/ws/dashboard` for live events:

- `heartbeat` -- worker hashrate updates
- `block` -- new block discoveries
- `health` -- unhealthy worker alerts (checked every 60s)

### 8.3 Built-In Dashboard

Access `http://localhost:8080/dashboard` for a self-contained HTML dashboard with auto-refresh polling (5-second interval). No frontend build required.

### 8.4 Logging

The server logs to stdout with the format:

```
HH:MM:SS [module    ] LEVEL message
```

Key loggers: `server`, `broker`, `storage`, `auth`, `monitoring`.

For production, redirect to a file or journald:

```bash
# journald (systemd)
journalctl -u xenblocks -f

# File-based
python3 -m server.server 2>&1 | tee -a /var/log/xenblocks/server.log
```

---

## 9. Development Mode

### 9.1 Running Both Servers

Terminal 1 -- Backend:

```bash
source .venv/bin/activate
python3 -m server.server --api-port 8080 --mqtt-port 1883
```

Terminal 2 -- Frontend (with hot reload):

```bash
cd web
npm run dev
```

The Vite dev server (`http://localhost:5173`) proxies `/api` and `/ws` to the backend at `localhost:8080` (configured in `web/vite.config.ts`).

### 9.2 Mock Fleet Simulator

Launch simulated mining workers that generate MQTT traffic:

```bash
# 5 workers, blocks every ~60 seconds (default)
python3 scripts/mock_fleet.py --workers 5 --broker localhost --port 1883

# High-frequency block discovery for testing
python3 scripts/mock_fleet.py --workers 10 --block-interval 5

# All workers under one wallet address
python3 scripts/mock_fleet.py --workers 3 --owner 0xYourAddress
```

The fleet simulator requires `paho-mqtt`:

```bash
pip install paho-mqtt
```

### 9.3 Built-In Simulator Module

The `server.simulator` module provides a more integrated simulator with lease-aware behavior:

```bash
python3 -m server.simulator \
  --workers 2 \
  --mqtt-port 1883 \
  --block-interval 3 \
  --gpu-count 2 \
  --api-port 8080
```

### 9.4 End-to-End Demo

Run the full marketplace lifecycle (server start, worker simulation, rent, mine, settle):

```bash
./scripts/demo.sh
```

This uses non-standard ports (`11883`/`18080`) and a temp database (`/tmp/demo-marketplace.db`) to avoid conflicts with a running instance.

---

## 10. Troubleshooting

### Server fails to start: "Address already in use"

Another process is using port 8080 or 1883.

```bash
# Find what's using the port
lsof -i :8080
lsof -i :1883

# Use alternative ports
python3 -m server.server --api-port 9090 --mqtt-port 11883
```

### "ERROR: FastAPI and uvicorn are required"

Python dependencies not installed.

```bash
source .venv/bin/activate
pip install -r server/requirements.txt
```

### "JWTs will invalidate on restart"

Expected if `--jwt-secret` is not provided. The server generates an ephemeral secret. See section 4.2 for persistent secret setup.

### Workers not appearing in the dashboard

1. Confirm the MQTT broker is accepting connections:
   ```bash
   curl -s http://localhost:8080/api/status | jq .mqtt_clients
   ```
2. Verify workers can reach the broker port (default `1883`).
3. Check that workers publish to the correct topic pattern: `xenminer/{worker_id}/register`.

### WebSocket connection drops immediately

The `/ws/dashboard` endpoint returns code `1013` (Service unavailable) if the `WSManager` is not initialized. Confirm the server started fully by checking `GET /api/status`.

### Database locked errors

SQLite WAL mode handles concurrent reads well, but only one writer at a time. If you run scripts like `generate_test_data.py` while the server is running, they will share the same database file. Stop the server first, or use a separate database path.

### Frontend build fails: TypeScript errors

```bash
cd web
rm -rf node_modules
npm install
npm run build
```

### MQTT broker: "No subscribers for topic"

This warning appears when the server publishes a control message but no worker is subscribed to that topic. Verify the target worker is connected and has subscribed to `xenminer/{worker_id}/control`.

### High memory usage from hashrate snapshots

Snapshots are recorded every 30 seconds and cleaned up after 24 hours. For very large fleets (1000+ workers), the snapshots table can grow. The cleanup runs every hour automatically. To force immediate cleanup:

```bash
sqlite3 /var/lib/xenblocks/marketplace.db "DELETE FROM hashrate_snapshots WHERE timestamp < (strftime('%s','now') - 86400)"
```
