# XenBlocks Mining Platform — Architecture

## 1. System Overview

The XenBlocks Mining Platform is a hashpower marketplace that connects GPU mining workers to consumers who rent compute capacity. The system runs as a single-process server combining an embedded MQTT broker, a FastAPI REST API, a WebSocket push layer, and a React SPA dashboard.

```
                                      +---------------------+
                                      |   React SPA (:5173) |
                                      |   (Vite dev server) |
                                      +----+--------+-------+
                                           |        |
                                    REST/  |        | WebSocket
                                    fetch  |        | /ws/dashboard
                                           |        |
+------------------+               +-------v--------v--------+
|  Mining Workers  |  MQTT 3.1.1   |  PlatformServer (:8080) |
|  (GPU nodes)     +<------------->+                         |
|                  |  port 1883    |  +-- MQTTBroker          |
+------------------+               |  +-- FastAPI (uvicorn)   |
                                   |  +-- WSManager           |
                                   |  +-- StorageManager      |
                                   |  |     (SQLite + WAL)    |
                                   |  +-- Service layer:      |
                                   |       AccountService     |
                                   |       MatchingEngine     |
                                   |       BlockWatcher       |
                                   |       SettlementEngine   |
                                   |       PricingEngine      |
                                   |       ReputationEngine   |
                                   |       MonitoringService  |
                                   |       WalletSnapshotSvc  |
                                   |       AuthService        |
                                   |       ChainSimulator     |
                                   +--------------------------+
```

All components run in a single asyncio event loop. The MQTT broker, HTTP server, and background tasks (lease watchdog, monitoring loop, wallet snapshots) share the same process.

---

## 2. Backend Architecture

### 2.1 Server Startup

Entry point: `/home/woody/XenblocksMiner/server/server.py`

`PlatformServer.__init__()` runs synchronously:
1. Creates `MQTTBroker` instance (no I/O yet).
2. Creates `FastAPI` app, registers all routers via `register_all_routers()`.
3. Registers the legacy HTML dashboard via `register_dashboard()`.
4. Optionally embeds `ChainSimulator` routes.
5. Binds `self._on_mqtt_message` as the broker's message handler.

`PlatformServer.start()` runs async:
1. `_init_services()` — opens SQLite via `StorageManager.initialize()`, instantiates all service objects, wires repos into services.
2. `broker.start()` — starts the TCP server on port 1883.
3. Spawns `_lease_watchdog` (5s interval) and `_monitoring_loop` (1s tick) as `asyncio.Task`.
4. Starts `WalletSnapshotService` background task (3600s interval).
5. Launches `uvicorn.Server.serve()` — blocks until shutdown.

### 2.2 Middleware and Dependency Injection

There is no traditional middleware stack. The server instance is injected via `app.state.server` (set in `_register_routes()`). Router modules retrieve it through a helper:

```python
# /home/woody/XenblocksMiner/server/deps.py
def get_server(request: Request):
    return request.app.state.server
```

Authentication is handled per-route through explicit `Header` parameters (`x_api_key`, `authorization`) passed to `AuthService` methods. There is no global auth middleware — endpoints opt in.

### 2.3 Router Organization

All routers registered in `/home/woody/XenblocksMiner/server/routers/__init__.py`:

| Module | Prefix | Purpose |
|--------|--------|---------|
| `overview` | `/api/overview/*` | Platform-wide stats, activity feed, network info |
| `monitoring` | `/api/monitoring/*` | Fleet overview, aggregated stats, hashrate history, recent blocks |
| `marketplace` | `/api/marketplace/*`, `/api/workers/*` | Browse workers, cost estimates, pricing, reputation |
| `rental` | `/api/rent`, `/api/stop`, `/api/leases/*`, `/api/blocks/*` | Lease lifecycle, block queries |
| `provider` | `/api/provider/*` | Provider dashboard, earnings, worker management |
| `account` | `/api/auth/*`, `/api/accounts/*` | Auth flows (nonce/verify/register/login/me), balance, deposit |
| `wallet` | `/api/wallet/*` | Per-wallet history, achievements, stats, worker commands, share data |
| `admin` | `/`, `/api/status`, `/api/accounts`, `/api/settlements`, `/api/workers/{id}/control` | Admin endpoints, MQTT control broadcast |
| `ws` | `/ws/dashboard` | WebSocket endpoint (registered directly on `app`, not as `APIRouter`) |

### 2.4 MQTT Broker

File: `/home/woody/XenblocksMiner/server/broker.py`

A minimal, pure-Python MQTT 3.1.1 broker built on `asyncio.start_server`. Supports:
- CONNECT/CONNACK handshake with client ID extraction.
- PUBLISH with QoS 0 and QoS 1 (PUBACK).
- SUBSCRIBE/SUBACK with `+` and `#` wildcard matching.
- UNSUBSCRIBE/UNSUBACK.
- PINGREQ/PINGRESP keepalive.
- DISCONNECT clean close.

Topic convention: `xenminer/{worker_id}/{message_type}`

The server registers a single `on_message` handler (`PlatformServer._on_mqtt_message`) that dispatches by `message_type`:

| Topic suffix | Handler | Side effects |
|---|---|---|
| `register` | `MatchingEngine.register_worker()` | Upserts worker in DB |
| `heartbeat` | `MatchingEngine.update_heartbeat()` | Updates hashrate/GPUs; broadcasts WS `heartbeat` |
| `status` | `MatchingEngine.update_worker_state()` | Updates worker state |
| `block` | `BlockWatcher.handle_block_found()` | Records block; broadcasts WS `block` |

Server-to-worker publishes use `broker.publish(topic, payload)` to deliver control commands (`pause`, `resume`, `shutdown`, `set_config`) to subscribed workers.

### 2.5 Background Tasks

| Task | Interval | Logic |
|---|---|---|
| `_lease_watchdog` | 5s | Finds expired leases via `MatchingEngine.check_expired_leases()`, settles each via `SettlementEngine` |
| `_monitoring_loop` | 1s tick | At 30s: records hashrate snapshots. At 60s: checks worker health, broadcasts WS `health` events. At 3600s: cleans old snapshots |
| `WalletSnapshotService._run` | 3600s | Aggregates per-wallet stats (hashrate, blocks, earnings) into `wallet_snapshots` table, cleans old entries |

---

## 3. Authentication Flow

File: `/home/woody/XenblocksMiner/server/auth.py`

Two auth mechanisms, resolved in priority order by `resolve_account()`:

### 3.1 Wallet Auth (EIP-191 Signature)

```
Client                              Server
  |                                    |
  |  GET /api/auth/nonce?address=0x..  |
  |  --------------------------------> |  generate_nonce(address)
  |  { nonce, message }                |  store nonce in memory (TTL 300s)
  |  <-------------------------------- |
  |                                    |
  |  MetaMask: signer.signMessage(msg) |
  |                                    |
  |  POST /api/auth/verify             |
  |  { address, signature, nonce }     |
  |  --------------------------------> |  verify_signature():
  |                                    |    1. Check nonce exists & not expired
  |                                    |    2. Reconstruct EIP-191 message
  |                                    |    3. Recover signer via eth_account
  |                                    |    4. Compare recovered address
  |                                    |    5. Consume nonce (one-time use)
  |                                    |  get_or_create_by_eth_address()
  |                                    |  issue_jwt(address, role, account_id)
  |  { token, address, account_id }    |
  |  <-------------------------------- |
  |                                    |
  |  Authorization: Bearer <jwt>       |  (all subsequent requests)
  |  --------------------------------> |  decode_jwt() -> claims
```

JWT payload: `{ sub: address, role, account_id, iat, exp }`. Signed HS256. TTL: 86400s (24h). Secret: CLI `--jwt-secret` or ephemeral `secrets.token_hex(32)`.

Nonce storage is in-memory (`dict[str, tuple[str, float]]`). Nonces are address-scoped and single-use.

### 3.2 API Key Fallback

Legacy flow for programmatic access:

1. `POST /api/auth/register` — creates account, generates `secrets.token_hex(16)` API key.
2. `POST /api/auth/login` — returns existing API key (or generates one).
3. Subsequent requests: `X-API-Key: <key>` header.

A hardcoded admin key (`admin-test-key-do-not-use-in-production`) grants `admin` role.

### 3.3 Resolution Order

`resolve_account()` checks:
1. `Authorization: Bearer <jwt>` — decode, lookup account by `claims.account_id`.
2. `X-API-Key` — check admin key, then `AccountRepo.get_by_api_key()`.
3. Returns `None` if no valid credentials.

Role-based guards: `require_consumer()`, `require_provider()`, `require_admin()` — each calls `get_current_account()` (which raises 401 if `None`), then checks role.

---

## 4. Real-time Data Flow

### 4.1 WebSocket Protocol

Endpoint: `ws://{host}/ws/dashboard`
Manager: `/home/woody/XenblocksMiner/server/ws.py`

Connection lifecycle:
1. Client connects -> `WSManager.connect()` accepts, enforces `MAX_WS_CLIENTS` (200) cap.
2. Server immediately sends a `snapshot` message with full state.
3. Connection enters receive loop (keeps connection alive; server ignores client messages).
4. On disconnect or timeout, client is removed.

### 4.2 Message Types

All messages follow the envelope `{ type: string, data: any, ts: number }`.

| Type | Source | Data Shape | Trigger |
|---|---|---|---|
| `snapshot` | Server -> Client (on connect) | `{ workers: Worker[], recent_blocks: Block[] }` | New WebSocket connection |
| `heartbeat` | Server -> All clients | `{ worker_id, hashrate, active_gpus }` | MQTT heartbeat from worker |
| `block` | Server -> All clients | `{ worker_id, hash, lease_id }` | MQTT block report from worker |
| `health` | Server -> All clients | `[{ worker_id, last_heartbeat, offline_sec }]` | Monitoring loop (60s) detects offline workers |

Broadcast implementation uses `asyncio.gather` with a 2s per-send timeout. Stale connections (send failures) are pruned after each broadcast.

### 4.3 Client Reconnection Strategy

File: `/home/woody/XenblocksMiner/web/src/hooks/useWebSocket.ts`

Exponential backoff on `ws.onclose`:
- Initial delay: 1000ms
- Multiplier: 2x on each failure
- Cap: 30000ms
- Reset to 1000ms on successful `onopen`

On initial mount, the hook also fetches `/api/monitoring/stats` via REST to seed `total_blocks` and `blocks_last_hour` (data not included in the WebSocket snapshot).

### 4.4 Client-side State Derivation

The `useWebSocket` hook derives `online` status client-side by comparing `Date.now()/1000 - worker.last_heartbeat` against `OFFLINE_THRESHOLD` (90s). Stats (hashrate, GPU counts, worker counts) are recomputed locally from the workers array on every message, avoiding redundant server round-trips.

---

## 5. Data Layer

### 5.1 SQLite Configuration

File: `/home/woody/XenblocksMiner/server/storage.py`

- Backend: `aiosqlite` (async wrapper around `sqlite3`).
- Journal mode: WAL (Write-Ahead Logging) for concurrent read/write.
- Foreign keys: enabled.
- Default path: `data/marketplace.db`.

### 5.2 Schema (v10)

| Table | Primary Key | Purpose |
|---|---|---|
| `schema_version` | — | Tracks applied migration version |
| `accounts` | `account_id TEXT` | Consumer and provider accounts. Fields: role, eth_address, balance, api_key |
| `workers` | `worker_id TEXT` | Registered mining workers. Fields: GPU info (gpus_json), state, hashrate, pricing (price_per_min, min/max duration), uptime tracking (total_online_sec, last_online_at), self_blocks_found |
| `leases` | `lease_id TEXT` | Hashpower rental agreements. Fields: worker_id, consumer_id, consumer_address, prefix, duration_sec, price_per_sec, state (active/completed/cancelled), hashrate stats |
| `blocks` | `id INTEGER AUTOINCREMENT` | Mined block records. Fields: lease_id (empty string = self-mined), worker_id, block_hash, key, chain_verified, chain_block_id |
| `settlements` | `id INTEGER AUTOINCREMENT` | Completed lease financial records. Fields: total_cost, provider_payout, platform_fee |
| `transactions` | `id INTEGER AUTOINCREMENT` | Audit trail for balance changes. Types: deposit, withdraw, lease_charge, provider_payout, platform_fee, transfer_in, transfer_out |
| `hashrate_snapshots` | `id INTEGER AUTOINCREMENT` | Time-series per-worker hashrate (30s interval). Cleaned after 24h |
| `wallet_snapshots` | `id INTEGER AUTOINCREMENT` | Per-address hourly aggregates: hashrate, online/total workers, blocks, earnings. Hourly retained 7 days, daily retained 90 days |

### 5.3 Indexes

```
idx_leases_state           leases(state)
idx_leases_worker          leases(worker_id)
idx_leases_consumer        leases(consumer_id)
idx_blocks_lease           blocks(lease_id)
idx_blocks_worker          blocks(worker_id)
idx_blocks_created         blocks(created_at)
idx_blocks_self            blocks(lease_id) WHERE lease_id = ''
idx_transactions_account   transactions(account_id)
idx_settlements_lease      settlements(lease_id)
idx_accounts_api_key       accounts(api_key)
idx_accounts_eth_address   accounts(eth_address COLLATE NOCASE)
idx_snapshots_worker_ts    hashrate_snapshots(worker_id, timestamp)
idx_snapshots_ts           hashrate_snapshots(timestamp)
idx_wallet_snapshots_addr_time  wallet_snapshots(eth_address, timestamp)
idx_wallet_snapshots_addr_type  wallet_snapshots(eth_address, interval_type)
```

### 5.4 Repository Pattern

`StorageManager` exposes one repo per table:

| Repo Class | Property | Table |
|---|---|---|
| `AccountRepo` | `storage.accounts` | `accounts` + `transactions` |
| `WorkerRepo` | `storage.workers` | `workers` |
| `LeaseRepo` | `storage.leases` | `leases` |
| `BlockRepo` | `storage.blocks` | `blocks` |
| `SettlementRepo` | `storage.settlements` | `settlements` |
| `TransactionRepo` | `storage.transactions` | `transactions` |
| `SnapshotRepo` | `storage.snapshots` | `hashrate_snapshots` |
| `WalletSnapshotRepo` | `storage.wallet_snapshots` | `wallet_snapshots` |

All repos share the same `aiosqlite.Connection` instance. Each method commits immediately after writes.

### 5.5 Migration System

Migrations run in `StorageManager._migrate()`:
1. Read current version from `schema_version` table (0 if table does not exist).
2. If behind `SCHEMA_VERSION` (currently 10), execute `SCHEMA_SQL` (full CREATE TABLE IF NOT EXISTS), then run incremental ALTER TABLE migrations for each version gap (V2 through V9).
3. Record new version in `schema_version`.

Migrations are idempotent — `ALTER TABLE ADD COLUMN` failures are silently caught since the column already exists in the CREATE TABLE statement for fresh databases.

---

## 6. Frontend Architecture

### 6.1 Tech Stack

- React 18 with TypeScript
- Vite (dev server and bundler)
- React Router v6 (BrowserRouter)
- TanStack React Query v5
- Tailwind CSS (utility classes)
- Sonner (toast notifications)
- ethers.js v6 (wallet signing)

### 6.2 Component Tree

```
App (/home/woody/XenblocksMiner/web/src/App.tsx)
  QueryClientProvider
    BrowserRouter
      WalletProvider (Context)
        DashboardProvider (Context + WebSocket)
          Suspense (lazy loading boundary)
            Routes
              Layout (/web/src/components/Layout.tsx)
                Sidebar (inline) — public nav + wallet nav + connection status
                Header — page title, search, wallet connect/disconnect button
                Outlet ->
                  Overview    (lazy, /web/src/pages/Overview.tsx)
                  Monitoring  (lazy, /web/src/pages/Monitoring.tsx)
                  Marketplace (lazy, /web/src/pages/Marketplace.tsx)
                  Provider    (lazy, /web/src/pages/Provider.tsx)
                  Renter      (lazy, /web/src/pages/Renter.tsx)
                  Account     (lazy, /web/src/pages/Account.tsx)
                  NotFound    (lazy, /web/src/pages/NotFound.tsx)
      Toaster (dark theme, bottom-right)
```

### 6.3 Route Map

| Path | Component | Auth Required |
|---|---|---|
| `/` | Overview | No |
| `/monitoring` | Monitoring | No |
| `/marketplace` | Marketplace | No |
| `/provider` | Provider | Wallet (JWT) |
| `/renter` | Renter | Wallet (JWT) |
| `/account` | Account | Wallet (JWT) |

Navigation is split into two sections in the sidebar:
- **Public**: Overview, Monitoring, Marketplace
- **Wallet**: Provider, Renter, Account (separated by a labeled divider)

### 6.4 Lazy Loading

All page components are `React.lazy()` imports wrapped in a single `<Suspense>` boundary with a pulsing text fallback. This code-splits each page into its own chunk.

---

## 7. Design System

### 7.1 Token Architecture

File: `/home/woody/XenblocksMiner/web/src/design/tokens.ts`

All visual constants are centralized in a single tokens file. Components import from `tokens.ts` — no hardcoded color values.

**Color system** (dark-only):
- Backgrounds: 5-level depth scale (`#0b0e11` base -> `#252d3a` hover)
- Borders: 3-level (`#2a3441` default -> `#4a6078` active)
- Text: 4-level hierarchy (`#eaecef` primary -> `#0b0e11` inverse)
- Accent: cyan/teal (`#22d1ee`) with hover, muted (12% opacity), and glow (25% opacity) variants
- Semantic: success (`#0ecb81`), danger (`#f6465d`), warning (`#f0b90b`), info (`#3b82f6`) — each with a muted background variant
- Chart palette: 6-color ordered series

**Typography**: Inter (sans) + JetBrains Mono (monospace).

**Spacing**: 4px grid system (0-64px).

**Radii**: sm(4px), md(6px), lg(10px), xl(14px), full.

### 7.2 Tailwind Presets

The `tw` object in tokens provides pre-composed Tailwind class strings for consistent usage:

| Category | Keys |
|---|---|
| Cards | `card`, `cardHover`, `cardInteractive` |
| Surfaces | `surface1`, `surface2`, `surface3` |
| Text | `textPrimary`, `textSecondary`, `textTertiary` |
| Badges | `badgeSuccess`, `badgeDanger`, `badgeInfo`, `badgeWarning`, `badgeAccent` |
| Status dots | `dotOnline`, `dotOffline`, `dotActive`, `dotIdle` (with glow shadows) |
| Buttons | `btnPrimary`, `btnSecondary`, `btnDanger` |
| Table | `tableHeader`, `tableRow`, `tableCell` |
| Inputs | `input` |

### 7.3 Component Library

Exported from `/home/woody/XenblocksMiner/web/src/design/index.ts`:

| Component | Purpose |
|---|---|
| `MetricCard` | Stat display with label, value, delta, icon, semantic variant coloring, and top-border glow |
| `StatusBadge` | Colored pill for worker/lease states |
| `GpuBadge` | GPU model display |
| `HashText` | Truncated hash display with monospace font |
| `ChartCard` | Container for chart content with title/subtitle/action header |
| `EmptyState` | Placeholder for empty data states |
| `Skeleton` | Loading placeholder with animation |
| `DataTable` | Sortable data table |
| `Pill` | Generic colored pill |
| `ConfirmDialog` | Confirmation modal |
| `LWChart` | Lightweight chart wrapper |
| `ViewToggle` | Grid/list view switcher |

### 7.4 Chart Theme

The `chartTheme` object provides Recharts-compatible styling: grid lines (`#1f2835`, dashed), axis labels (`#5e6673`, 11px), tooltip with dark surface background, and a cyan area gradient.

---

## 8. State Management

The frontend uses three distinct state categories, each with a clear owner:

### 8.1 Server State (React Query)

Configured in `/home/woody/XenblocksMiner/web/src/lib/queryClient.ts`:
- `staleTime: 10_000` (10s before refetch)
- `retry: 2`
- `refetchOnWindowFocus: true`
- Global error handlers on both `QueryCache` and `MutationCache` that surface errors via Sonner toasts

Used for: REST API data that doesn't need sub-second freshness — marketplace listings, provider dashboard, wallet history/achievements, account balances, lease details.

### 8.2 Real-time State (WebSocket + Context)

Two context providers manage live data:

**DashboardContext** (`/home/woody/XenblocksMiner/web/src/context/DashboardContext.tsx`):
- Wraps the `useWebSocket` hook.
- Provides: `workers: Worker[]`, `stats: Stats`, `recentBlocks: Block[]`, `connected: boolean`.
- State updates arrive via WebSocket messages; no polling.
- Stats are derived client-side from the workers array on every incoming message.
- Available to all components via `useDashboard()`.

**WalletContext** (`/home/woody/XenblocksMiner/web/src/context/WalletContext.tsx`):
- Manages MetaMask connection state.
- Provides: `address`, `connecting`, `connect()`, `disconnect()`.
- On mount, attempts session restoration via `GET /api/auth/me` using stored JWT from `localStorage` (`xb_jwt` key).
- Listens for MetaMask `accountsChanged` and `chainChanged` events — clears session on change.
- `connect()` executes the full nonce-sign-verify flow and stores the returned JWT.

### 8.3 API Utility

File: `/home/woody/XenblocksMiner/web/src/api.ts`

`apiFetch<T>()` is a thin wrapper around `fetch()` that:
1. Reads JWT from `localStorage`.
2. Attaches `Authorization: Bearer <jwt>` if present.
3. Throws on non-2xx responses.
4. Returns parsed JSON typed as `T`.

Token management via `getToken()`, `setToken()`, `clearToken()`.

### 8.4 State Ownership Summary

| Data | Owner | Update Mechanism | Scope |
|---|---|---|---|
| Worker list, stats, blocks | DashboardContext | WebSocket push | Global |
| Wallet address, auth token | WalletContext | User action + localStorage | Global |
| Marketplace listings | React Query | REST polling (10s stale) | Page-local |
| Provider dashboard | React Query | REST polling | Page-local |
| Wallet history/achievements | React Query | REST on-demand | Page-local |
| Form inputs, modals, pagination | `useState` | User interaction | Component-local |

### 8.5 Data Flow Diagram

```
  MQTT Workers
       |
       v
  MQTTBroker (1883)
       |
       v  (on_message dispatch)
  +---------+    +-----------+    +-------------+
  | Matcher  |    | Watcher   |    | WSManager   |
  | (state)  |    | (blocks)  |    | (broadcast) |
  +----+-----+    +-----+-----+    +------+------+
       |                |                  |
       v                v                  v
    SQLite           SQLite          WebSocket push
  (workers,        (blocks)        to all dashboard
   leases)                         clients
       |                |                  |
       +--------+-------+                  |
                |                          |
                v                          v
          REST API (:8080)          Browser WS client
          /api/monitoring/*         useWebSocket hook
          /api/provider/*               |
          /api/wallet/*                 v
                |               DashboardContext
                v                 (workers, stats,
          React Query              recentBlocks)
          (page data)
```
