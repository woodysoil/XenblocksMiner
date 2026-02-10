# WebSocket Real-Time Protocol

## 1. Connection

### Endpoint

```
ws://<host>/ws/dashboard
wss://<host>/ws/dashboard   (when served over HTTPS)
```

The client auto-selects the scheme based on `location.protocol`:

```ts
const proto = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${proto}//${location.host}/ws/dashboard`);
```

### Authentication

None. The `/ws/dashboard` endpoint accepts unauthenticated connections. Access control, if needed, is handled at the reverse-proxy layer.

### Connection Limits

The server enforces a hard cap of **200 concurrent WebSocket clients** (`MAX_WS_CLIENTS`). When the limit is reached, the server accepts the connection then immediately closes it:

```
close code: 1013 (Try Again Later)
reason:     "Server overloaded"
```

If `WSManager` has not been initialized (services not yet ready), the server closes with:

```
close code: 1013
reason:     "Service unavailable"
```

### Reconnection Strategy (Exponential Backoff)

| Parameter       | Value   |
|-----------------|---------|
| Initial delay   | 1000 ms |
| Multiplier      | 2x      |
| Maximum delay   | 30000 ms |

On `onclose`, the client schedules a reconnect after the current delay, then doubles the delay (capped at 30 s). On successful `onopen`, the delay resets to 1000 ms.

```
Attempt 1:  1 s
Attempt 2:  2 s
Attempt 3:  4 s
Attempt 4:  8 s
Attempt 5: 16 s
Attempt 6+: 30 s (capped)
```

On `onerror`, the client forces `ws.close()`, which triggers the `onclose` backoff path.

### Initial Handshake Sequence

1. Server accepts the WebSocket upgrade.
2. Server immediately sends a `snapshot` message containing the full fleet state.
3. The connection then enters a steady-state loop where the server pushes events and the client sends no application-level messages (the read loop exists only to detect disconnection).

---

## 2. Message Envelope

Every WebSocket message is a JSON text frame with a uniform envelope:

```json
{
  "type": "<message_type>",
  "data": <payload>,
  "ts": <unix_timestamp_float>
}
```

| Field  | Type     | Description |
|--------|----------|-------------|
| `type` | `string` | One of: `"snapshot"`, `"heartbeat"`, `"block"`, `"health"` |
| `data` | `object \| array` | Type-specific payload (see below) |
| `ts`   | `number` | Server-side Unix timestamp (seconds, floating point) generated via `time.time()` |

TypeScript definition:

```ts
interface WSMessage {
  type: "snapshot" | "heartbeat" | "block" | "health";
  data: any;
  ts: number;
}
```

---

## 3. Message Types

### 3.1 `snapshot` -- Full Fleet State

**Direction:** Server -> Client (sent once on connection, not repeated)

Delivered immediately after the WebSocket handshake. Contains the complete worker roster and recent block history. The client uses this to bootstrap its local state.

**Payload schema:**

```json
{
  "type": "snapshot",
  "data": {
    "workers": [ <Worker>, ... ],
    "recent_blocks": [ <Block>, ... ]
  },
  "ts": 1707600000.123
}
```

**`data.workers[]` fields** (from `WorkerRepo.list_all()`):

| Field              | Type       | Description |
|--------------------|------------|-------------|
| `worker_id`        | `string`   | Unique worker identifier |
| `eth_address`      | `string`   | Worker's Ethereum address |
| `gpu_count`        | `integer`  | Total GPU count on this worker |
| `total_memory_gb`  | `integer`  | Total GPU memory in GB |
| `gpus`             | `array`    | Per-GPU info: `[{"name": "RTX 4090", "memory_gb": 24}, ...]` |
| `version`          | `string`   | Miner software version |
| `state`            | `string`   | Worker state: `"AVAILABLE"`, `"LEASED"`, etc. |
| `hashrate`         | `number`   | Current hashrate (H/s) |
| `active_gpus`      | `integer`  | GPUs currently mining |
| `last_heartbeat`   | `number`   | Unix timestamp of last heartbeat |
| `registered_at`    | `number`   | Unix timestamp of registration |
| `price_per_min`    | `number`   | Rental price per minute |
| `min_duration_sec` | `integer`  | Minimum lease duration |
| `max_duration_sec` | `integer`  | Maximum lease duration |
| `total_online_sec` | `number`   | Cumulative uptime in seconds |
| `last_online_at`   | `number \| null` | Timestamp of last online transition |
| `self_blocks_found`| `integer`  | Blocks found outside any lease |

**`data.recent_blocks[]` fields** (from `BlockRepo.get_all(limit=20)`):

| Field            | Type            | Description |
|------------------|-----------------|-------------|
| `lease_id`       | `string`        | Associated lease ID (`""` for self-mined) |
| `worker_id`      | `string`        | Worker that found the block |
| `hash`           | `string`        | Block hash |
| `key`            | `string`        | Mining key used |
| `account`        | `string`        | Target account address |
| `attempts`       | `integer`       | Hash attempts before finding block |
| `hashrate`       | `string`        | Hashrate at time of discovery (string) |
| `prefix_valid`   | `boolean`       | Whether the hash prefix matches requirements |
| `chain_verified` | `boolean`       | Whether the block was accepted on-chain |
| `chain_block_id` | `integer \| null` | On-chain block ID if verified |
| `timestamp`      | `number`        | Unix timestamp of block creation |

**Example:**

```json
{
  "type": "snapshot",
  "data": {
    "workers": [
      {
        "worker_id": "worker-abc-123",
        "eth_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18",
        "gpu_count": 2,
        "total_memory_gb": 48,
        "gpus": [
          {"name": "RTX 4090", "memory_gb": 24},
          {"name": "RTX 4090", "memory_gb": 24}
        ],
        "version": "1.2.0",
        "state": "AVAILABLE",
        "hashrate": 1250.5,
        "active_gpus": 2,
        "last_heartbeat": 1707599950.0,
        "registered_at": 1707500000.0,
        "price_per_min": 0.60,
        "min_duration_sec": 60,
        "max_duration_sec": 86400,
        "total_online_sec": 86400.0,
        "last_online_at": 1707599950.0,
        "self_blocks_found": 14
      }
    ],
    "recent_blocks": [
      {
        "lease_id": "",
        "worker_id": "worker-abc-123",
        "hash": "00000a3f...deadbeef",
        "key": "Wk9x...B4mQ",
        "account": "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18",
        "attempts": 48210,
        "hashrate": "1250.5",
        "prefix_valid": true,
        "chain_verified": true,
        "chain_block_id": 98712,
        "timestamp": 1707599800.0
      }
    ]
  },
  "ts": 1707600000.123
}
```

---

### 3.2 `heartbeat` -- Worker Heartbeat

**Direction:** Server -> Client (pushed on every MQTT heartbeat from any worker)

**Origin:** When a worker publishes to MQTT topic `xenminer/<worker_id>/heartbeat`, the server extracts key metrics and broadcasts to all WebSocket clients.

**Payload schema:**

```json
{
  "type": "heartbeat",
  "data": {
    "worker_id": "<string>",
    "hashrate": <number>,
    "active_gpus": <integer>
  },
  "ts": 1707600030.456
}
```

| Field         | Type      | Description |
|---------------|-----------|-------------|
| `worker_id`   | `string`  | Worker that sent the heartbeat |
| `hashrate`    | `number`  | Current hashrate (H/s) |
| `active_gpus` | `integer` | GPUs currently mining |

**Server-side broadcast** (from `server.py`):

```python
await self.ws_manager.broadcast("heartbeat", {
    "worker_id": payload.get("worker_id", ""),
    "hashrate": payload.get("hashrate", 0.0),
    "active_gpus": payload.get("active_gpus", 0),
})
```

**Example:**

```json
{
  "type": "heartbeat",
  "data": {
    "worker_id": "worker-abc-123",
    "hashrate": 1305.2,
    "active_gpus": 2
  },
  "ts": 1707600030.456
}
```

---

### 3.3 `block` -- New Block Mined

**Direction:** Server -> Client (pushed when any worker finds a block)

**Origin:** When a worker publishes to MQTT topic `xenminer/<worker_id>/block`, the server processes the block via `BlockWatcher` and then broadcasts a summary to WebSocket clients.

**Payload schema:**

```json
{
  "type": "block",
  "data": {
    "worker_id": "<string>",
    "hash": "<string>",
    "lease_id": "<string>"
  },
  "ts": 1707600120.789
}
```

| Field       | Type     | Description |
|-------------|----------|-------------|
| `worker_id` | `string` | Worker that found the block |
| `hash`      | `string` | Block hash |
| `lease_id`  | `string` | Associated lease (`""` for self-mined blocks) |

**Server-side broadcast** (from `server.py`):

```python
await self.ws_manager.broadcast("block", {
    "worker_id": worker_id,
    "hash": payload.get("hash", ""),
    "lease_id": payload.get("lease_id", ""),
})
```

**Client-side enrichment:** The frontend constructs a full `Block` object by applying defaults for fields not present in the broadcast payload (`key`, `account`, `attempts`, `hashrate`, `prefix_valid`, `chain_verified`, `chain_block_id`). The `timestamp` is taken from the envelope `ts`.

**Example:**

```json
{
  "type": "block",
  "data": {
    "worker_id": "worker-abc-123",
    "hash": "00000b7e...cafebabe",
    "lease_id": "lease-xyz-789"
  },
  "ts": 1707600120.789
}
```

---

### 3.4 `health` -- Offline Worker Detection

**Direction:** Server -> Client (pushed every ~60 seconds by the monitoring loop, only when unhealthy workers exist)

**Origin:** The `_monitoring_loop` in `server.py` runs `MonitoringService.check_worker_health()` every 60 seconds. If any workers have not sent a heartbeat within the offline threshold (90 seconds), the list is broadcast.

**Payload schema:**

```json
{
  "type": "health",
  "data": [
    {
      "worker_id": "<string>",
      "last_heartbeat": <number>,
      "offline_sec": <number>
    }
  ],
  "ts": 1707600180.000
}
```

| Field            | Type     | Description |
|------------------|----------|-------------|
| `worker_id`      | `string` | Worker that is offline |
| `last_heartbeat` | `number` | Unix timestamp of last known heartbeat |
| `offline_sec`    | `number` | Seconds since last heartbeat (rounded to 1 decimal) |

Note: The client only reads the `worker_id` field from each element. The additional fields are available but unused by the current frontend.

**Example:**

```json
{
  "type": "health",
  "data": [
    {
      "worker_id": "worker-def-456",
      "last_heartbeat": 1707599900.0,
      "offline_sec": 280.0
    }
  ],
  "ts": 1707600180.000
}
```

---

## 4. Client State Derivation

The `useWebSocket` hook maintains four pieces of state: `workers`, `stats`, `recentBlocks`, and `connected`. Fleet-level statistics are **computed client-side** from the worker array, not received from the server.

### 4.1 Online/Offline Classification

On `snapshot`, each worker's `online` status is derived from `last_heartbeat`:

```ts
const OFFLINE_THRESHOLD = 90; // seconds
const now = Date.now() / 1000;
worker.online = (now - worker.last_heartbeat) < OFFLINE_THRESHOLD;
```

On `heartbeat`, the matching worker is set to `online: true` and its `last_heartbeat` is updated to the message `ts`.

On `health`, every worker whose `worker_id` appears in the health array is set to `online: false`.

### 4.2 Stats Recomputation (`recomputeStats`)

Called after every `snapshot`, `heartbeat`, and `health` message. Iterates the full worker array:

```ts
for (const w of workers) {
  totalGpus += w.gpu_count;        // always counted
  if (w.online) {
    onlineCount++;
    totalHashrate += w.hashrate;   // only online workers contribute
    activeGpus += w.active_gpus;   // only online workers contribute
  } else {
    offlineCount++;
  }
}
```

Resulting `Stats` object:

| Field            | Derivation |
|------------------|------------|
| `total_workers`  | `workers.length` |
| `online`         | Count of workers where `online === true` |
| `offline`        | Count of workers where `online === false` |
| `total_hashrate` | Sum of `hashrate` for online workers only |
| `total_gpus`     | Sum of `gpu_count` for all workers |
| `active_gpus`    | Sum of `active_gpus` for online workers only |
| `total_blocks`   | Seeded from REST `GET /api/monitoring/stats`, incremented by `block` messages |
| `blocks_last_hour` | Seeded from REST `GET /api/monitoring/stats`, incremented by `block` messages |

### 4.3 Block Counting

`total_blocks` and `blocks_last_hour` are **not** recomputed from the worker array. They are:
1. Initialized via a REST fetch to `GET /api/monitoring/stats` on mount.
2. Incremented by +1 on each incoming `block` message.

This means `blocks_last_hour` is an approximation that drifts upward over time (blocks older than one hour are never subtracted). A page refresh corrects this via the REST re-fetch.

### 4.4 Recent Blocks List

Maintained as a FIFO list capped at 20 entries. On `snapshot`, the list is replaced wholesale. On each `block` message, the new block is prepended and the list is truncated to 20:

```ts
setRecentBlocks(prev => [block, ...prev].slice(0, 20));
```

---

## 5. REST Bootstrap

On component mount, the `useWebSocket` hook issues a single REST call **in parallel** with the WebSocket connection:

```
GET /api/monitoring/stats
```

Response schema (identical to the `Stats` TypeScript interface):

```json
{
  "total_workers": 5,
  "online": 4,
  "offline": 1,
  "total_hashrate": 5200.75,
  "total_gpus": 10,
  "active_gpus": 8,
  "total_blocks": 142,
  "blocks_last_hour": 3
}
```

Only `total_blocks` and `blocks_last_hour` are extracted from this response and merged into local stats. All other stats fields are derived from the WebSocket `snapshot` data.

This architecture avoids a dependency on the REST response for real-time data while ensuring accurate cumulative block counts that the WebSocket protocol does not provide.

---

## 6. React Context Integration

The `DashboardProvider` wraps the application tree and exposes the `useWebSocket` return value via React context:

```
DashboardProvider
  -> useWebSocket()           // owns the WebSocket lifecycle
     -> returns { workers, stats, recentBlocks, connected }
  -> <Ctx.Provider value={...}>
       {children}             // all dashboard components
     </Ctx.Provider>
```

Any child component calls `useDashboard()` to access the live state. There is no React Query (`queryClient`) integration for WebSocket data; the hook manages its own state independently.

---

## 7. Error Handling

### 7.1 Malformed Messages

Client-side: `JSON.parse` failures in `onmessage` are silently caught and the message is discarded.

### 7.2 Server-Side Send Failures

Each broadcast to a client has a **2-second timeout** (`SEND_TIMEOUT`). If a `send_text` call fails or times out, the client's WebSocket is marked stale and removed from the client list. No close frame is sent; the client will detect the broken connection via its own `onclose` handler.

### 7.3 Stale Data Detection

There is no explicit staleness protocol. The client relies on:

- **`connected` boolean**: Set to `true` on `onopen`, `false` on `onclose`. Components can render a disconnection indicator.
- **Offline threshold**: Workers are considered offline after 90 seconds without a heartbeat. The `health` message proactively marks workers as offline before the client's local threshold would expire.
- **Page refresh**: Corrects any accumulated drift in `blocks_last_hour` by re-fetching from REST.

### 7.4 Connection Lifecycle Summary

```
[Client]                                    [Server]
   |                                           |
   |--- WebSocket UPGRADE /ws/dashboard ------>|
   |                                           |-- accept()
   |                                           |-- append to client list
   |<--- snapshot (full fleet state) ----------|
   |                                           |
   |<--- heartbeat (per worker) ---------------|  (on MQTT heartbeat)
   |<--- block (per discovery) ----------------|  (on MQTT block)
   |<--- health (periodic, if unhealthy) ------|  (every ~60s)
   |                                           |
   |--- [connection drops] ------------------->|
   |                                           |-- remove from client list
   |                                           |
   |--- [reconnect after backoff] ------------>|
   |                                           |-- accept()
   |<--- snapshot (fresh state) ---------------|
   |                                           |
```
