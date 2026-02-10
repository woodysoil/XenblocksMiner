# Mock Platform Server

The `server/` package is a self-contained mock platform server for offline
development and testing of the XenblocksMiner hashpower marketplace. It runs
an embedded MQTT broker, a SQLite storage layer, and a REST API in a single
Python process.

## Quick Start

### Install dependencies

```bash
pip install -r server/requirements.txt
```

Required packages: `fastapi`, `uvicorn`, `pydantic`, `aiosqlite`, `gmqtt`.

### Start the server

```bash
# From the project root:
python3 -m server.server

# Or use the convenience script:
./scripts/run_mock_server.sh

# With custom ports:
python3 -m server.server --mqtt-port 11883 --api-port 18080

# With a custom database path:
python3 -m server.server --db-path /tmp/marketplace.db

# Disable the embedded chain simulator:
python3 -m server.server --no-chain
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mqtt-port` | 1883 | MQTT broker listen port |
| `--api-port` | 8080 | REST API / dashboard listen port |
| `--db-path` | `data/marketplace.db` | SQLite database file path |
| `--no-chain` | (off) | Disable the embedded chain simulator |

Once running, the following are available:

- **MQTT broker** at `tcp://localhost:1883`
- **REST API** at `http://localhost:8080`
- **Dashboard** at `http://localhost:8080/dashboard`

## Running the Demo

The demo script exercises the full lifecycle without requiring a GPU or the
C++ binary. It uses simulated Python workers.

```bash
./scripts/demo.sh
```

This starts the server, launches 2 simulated workers, rents hashpower, waits
for blocks, stops the lease, and prints the settlement summary.

## Running Tests

### Unit tests (no server needed)

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

### Integration tests (starts server automatically via fixtures)

```bash
python3 -m pytest tests/integration/ -v --tb=short
```

### C++ worker integration test (requires CUDA GPU and built binary)

```bash
# Bash script version
./scripts/test_cpp_integration.sh

# Pytest version (auto-skips if no GPU/binary)
python3 -m pytest tests/integration/test_cpp_worker.py -v --tb=short
```

## API Endpoint Reference

### Public Endpoints (no authentication required)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Server info (service name, ports, connected workers, uptime) |
| `GET` | `/api/workers` | List all registered workers with state, GPU info, and pricing |
| `GET` | `/api/marketplace` | Browse available hashpower with filters (sort_by, gpu_type, min_hashrate, max_price, min_gpus, available_only) |
| `GET` | `/api/marketplace/estimate` | Estimate rental cost (duration_sec, worker_id, min_hashrate) |
| `GET` | `/api/workers/{worker_id}/pricing` | Get pricing info for a specific worker |
| `GET` | `/api/workers/{worker_id}/pricing/suggest` | Get suggested pricing based on GPU specs |
| `GET` | `/api/workers/{worker_id}/reputation` | Get reputation score and breakdown for a worker |
| `GET` | `/api/leases` | List all leases (optional filter: ?state=active\|completed\|cancelled) |
| `GET` | `/api/leases/{lease_id}` | Get lease detail including blocks and settlement |
| `GET` | `/api/blocks` | List blocks (optional filter: ?lease_id=...) |
| `GET` | `/api/status` | Server status (MQTT clients, worker count, active leases, total blocks, total settlements) |
| `GET` | `/dashboard` | HTML dashboard with live-updating UI |

### Authentication Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/auth/register` | Register a new account (provider or consumer). Returns an API key. |
| `POST` | `/api/auth/login` | Get or regenerate API key for an existing account |
| `GET` | `/api/auth/me` | Get current account info (requires `X-API-Key` header) |

### Consumer Endpoints (auth optional for backward compatibility)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/rent` | Rent hashpower. Body: `{consumer_id, consumer_address, duration_sec, worker_id?}` |
| `POST` | `/api/stop` | Stop an active lease and trigger settlement. Body: `{lease_id}` |

### Provider Endpoints (auth optional for backward compatibility)

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/api/workers/{worker_id}/pricing` | Set worker pricing. Body: `{price_per_min, min_duration_sec?, max_duration_sec?}` |

### Account Endpoints (auth optional for backward compatibility)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/accounts/{account_id}/balance` | Get account balance and details |
| `POST` | `/api/accounts/{account_id}/deposit` | Deposit funds. Body: `{amount}` |
| `GET` | `/api/accounts` | List all accounts (admin only when authenticated) |
| `GET` | `/api/settlements` | List all settlements (admin only when authenticated) |

### Chain Simulator Endpoints (enabled by default)

These endpoints mimic the real XenBlocks RPC server for fully offline testing.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/difficulty` | Current network difficulty |
| `POST` | `/verify` | Submit and validate a block. Body: `{hash_to_verify, key, account, attempts, hashes_per_second, worker}` |
| `GET` | `/getblocks/lastblock` | Last 100 blocks on the simulated chain |
| `POST` | `/send_pow` | PoW submission (always accepted in mock mode) |
| `GET` | `/balance/{address}` | XNM balance for an address |
| `GET` | `/chain/stats` | Chain statistics (total blocks by type, difficulty, unique miners) |
| `GET` | `/chain/blocks` | Query chain blocks (optional filters: ?account=..., ?prefix=..., ?limit=100) |

### Authentication

Protected endpoints accept an `X-API-Key` header. Authentication is optional
for backward compatibility -- endpoints work without a key but enforce access
control when one is provided. The admin API key for testing defaults to
`admin-test-key-do-not-use-in-production`.

## Architecture Overview

The server is organized into the following modules:

```
server/
  __init__.py          Package metadata (version 0.2.0)
  server.py            Entry point. Wires all components together, defines
                       FastAPI routes, runs the MQTT broker + REST API + lease
                       watchdog in a single asyncio event loop.
  broker.py            Embedded async MQTT 3.1.1 broker. Pure-Python TCP
                       server with minimal packet parsing. Supports CONNECT,
                       PUBLISH (QoS 0/1), SUBSCRIBE, UNSUBSCRIBE, PING, and
                       DISCONNECT. Routes messages between clients and invokes
                       platform message handlers.
  storage.py           SQLite persistence layer using aiosqlite. Repository
                       pattern with one class per table (AccountRepo,
                       WorkerRepo, LeaseRepo, BlockRepo, SettlementRepo,
                       TransactionRepo). Auto-migrates schema on startup
                       (current version: 5).
  account.py           Account service. Manages consumer and provider accounts
                       (create, deposit, transfer, balance queries). Backed by
                       AccountRepo.
  matcher.py           Matching engine. Handles worker registration, heartbeat
                       updates, lease creation (matching consumers to available
                       workers), lease stop, and expired lease detection.
                       Generates 16-char hex prefixes for key validation.
  watcher.py           Block watcher. Processes block_found MQTT messages,
                       validates key prefix against the assigned lease prefix,
                       cross-checks against the chain simulator, and records
                       blocks in SQLite.
  settlement.py        Settlement engine. Calculates payment when a lease ends:
                       95% to provider, 5% platform fee. Uses per-second
                       pricing from the lease. Debits consumer, credits
                       provider and platform treasury.
  chain_simulator.py   Mock XenBlocks blockchain. Validates block submissions
                       (key format, XEN11/XUNI markers, superblock detection),
                       maintains balances, and auto-adjusts difficulty. Can run
                       standalone or embedded in the platform server.
  pricing.py           Provider pricing engine. Lets providers set price per
                       minute and duration limits. Supports marketplace
                       browsing with filters (GPU type, hashrate, price) and
                       cost estimation. Includes GPU-based pricing suggestions.
  reputation.py        Provider reputation engine. Calculates a 0-100 score
                       based on block history (40%), lease completion rate
                       (30%), uptime (20%), and account age (10%). Maps to
                       0-5 stars on the dashboard.
  auth.py              API key authentication and role-based access control.
                       Accounts get a random 32-char hex API key. Supports
                       consumer, provider, and admin roles.
  dashboard.py         Single-page HTML dashboard served at /dashboard.
                       Auto-refreshes via JavaScript polling every 3 seconds.
                       Shows workers, leases, blocks, accounts, and a rent
                       form.
  simulator.py         Python worker simulator. Mimics the C++ miner's MQTT
                       behavior without requiring a GPU. Connects to the
                       broker, registers with fake GPU info, handles lease
                       assignments, simulates block discovery, and sends
                       heartbeats.
  requirements.txt     Python package dependencies.
```

### Data Flow

1. Workers (real C++ or simulated Python) connect to the **MQTT broker** and
   publish `register` messages.
2. The **matching engine** processes registration, creates a provider account,
   and sends `register_ack`.
3. A consumer calls `POST /api/rent`. The matching engine finds an available
   worker, generates a prefix, creates a lease, and publishes `assign_task`
   via MQTT.
4. The worker transitions to MINING and publishes `heartbeat` and
   `block_found` messages.
5. The **block watcher** validates each block (prefix match + chain
   verification) and records it.
6. When the lease ends (via `POST /api/stop`, expiry, or the watchdog), the
   **settlement engine** calculates payment and updates account balances.
7. The **dashboard** polls the REST API every 3 seconds to display live state.
