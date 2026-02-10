# XenBlocks Mining Platform API Reference

Base URL: `http://<host>:<port>`

---

## Authentication

The platform supports two authentication flows:

### Flow 1 -- Wallet Signature (EIP-191)

1. `GET /api/auth/nonce?address=0x...` to obtain a nonce.
2. Sign the message `Sign this message to authenticate with XenBlocks.\n\nNonce: {nonce}` with your wallet.
3. `POST /api/auth/verify` with address, signature, nonce to receive a JWT.
4. Pass `Authorization: Bearer <jwt>` on subsequent requests.

JWT expires after **24 hours**. Nonce expires after **5 minutes**.

### Flow 2 -- API Key (Legacy)

Pass `X-API-Key: <key>` header. Obtain the key via `/api/auth/register` or `/api/auth/login`.

### Role-Based Access

| Role | Description |
|------|-------------|
| `admin` | Full access. Authenticated via the admin API key. |
| `provider` | Owns workers. Can manage pricing and view earnings. |
| `consumer` | Rents hashpower. Can manage leases and deposits. |

---

## 1. Auth

### `GET /api/auth/nonce`

Request a challenge nonce for wallet-based authentication.

**Auth:** None

**Query Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `address` | `string` | Yes | Ethereum address (`0x`-prefixed, 42 chars) |

**Response `200`:**

```json
{
  "nonce": "a1b2c3d4e5f6...",
  "message": "Sign this message to authenticate with XenBlocks.\n\nNonce: a1b2c3d4e5f6..."
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Address does not start with `0x` or length is not 42. |

---

### `POST /api/auth/verify`

Verify a signed nonce and receive a JWT.

**Auth:** None

**Request Body:**

```json
{
  "address": "0x1234...abcd",
  "signature": "0x...",
  "nonce": "a1b2c3d4e5f6..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `address` | `string` | Yes | Ethereum address that signed the message. |
| `signature` | `string` | Yes | EIP-191 signature hex. |
| `nonce` | `string` | Yes | Nonce obtained from `/api/auth/nonce`. |

**Response `200`:**

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "address": "0x1234...abcd",
  "account_id": "wallet-0x1234abcd",
  "role": "provider"
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `401` | Invalid signature or expired/mismatched nonce. |
| `500` | Account creation failure on server side. |

---

### `POST /api/auth/register`

Register a new account (legacy API-key flow).

**Auth:** None

**Request Body:**

```json
{
  "account_id": "my-provider-1",
  "role": "provider",
  "eth_address": "0x...",
  "balance": 0.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `account_id` | `string` | Yes | -- | Unique account identifier. |
| `role` | `string` | Yes | -- | `"provider"` or `"consumer"`. |
| `eth_address` | `string` | No | `""` | Ethereum address to link. |
| `balance` | `float` | No | `0.0` | Initial balance. |

**Response `200`:**

```json
{
  "account_id": "my-provider-1",
  "role": "provider",
  "eth_address": "0x...",
  "balance": 0.0,
  "api_key": "3f8a...c9d1"
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Invalid role or account already exists. |
| `500` | Server failed to create the account. |

---

### `POST /api/auth/login`

Login to an existing account and retrieve its API key.

**Auth:** None

**Request Body:**

```json
{
  "account_id": "my-provider-1"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `account_id` | `string` | Yes | Existing account identifier. |

**Response `200`:**

```json
{
  "account_id": "my-provider-1",
  "role": "provider",
  "api_key": "3f8a...c9d1"
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `404` | Account not found. |

---

### `GET /api/auth/me`

Return the currently authenticated account profile.

**Auth:** JWT or API Key

**Headers:**

| Header | Description |
|--------|-------------|
| `Authorization` | `Bearer <jwt>` |
| `X-API-Key` | Legacy API key |

**Response `200`:**

```json
{
  "account_id": "my-provider-1",
  "role": "provider",
  "eth_address": "0x1234...abcd",
  "balance": 12.5
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `401` | No valid credentials provided. |

---

## 2. Accounts

### `GET /api/accounts/{account_id}/balance`

Retrieve the balance for an account.

**Auth:** JWT or API Key (optional, but enforced for non-owner access)

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `account_id` | `string` | Target account identifier. |

**Response `200`:**

```json
{
  "account_id": "my-provider-1",
  "role": "provider",
  "balance": 12.5,
  "eth_address": "0x..."
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `403` | Authenticated as a non-admin viewing another user's balance. |
| `404` | Account not found. |

---

### `POST /api/accounts/{account_id}/deposit`

Add funds to an account balance.

**Auth:** JWT or API Key (optional, but enforced for non-owner access)

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `account_id` | `string` | Target account identifier. |

**Request Body:**

```json
{
  "amount": 10.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `amount` | `float` | Yes | Amount to deposit. |

**Response `200`:**

```json
{
  "account_id": "my-provider-1",
  "balance": 22.5
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Invalid amount (e.g., negative). |
| `403` | Non-admin depositing to another user's account. |
| `404` | Account not found. |

---

## 3. Overview

### `GET /api/overview/stats`

Platform-wide aggregate statistics.

**Auth:** None

**Response `200`:**

```json
{
  "total_users": 42,
  "total_providers": 30,
  "total_consumers": 12,
  "total_workers": 85,
  "online_workers": 61,
  "total_hashrate": 15230.50,
  "total_blocks": 4200,
  "blocks_24h": 128,
  "total_leases": 310,
  "active_leases": 15,
  "total_settled": 295,
  "platform_revenue": 42.1234
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_users` | `int` | Total registered accounts. |
| `total_providers` | `int` | Accounts with role `provider`. |
| `total_consumers` | `int` | Accounts with role `consumer`. |
| `total_workers` | `int` | Total registered workers. |
| `online_workers` | `int` | Workers with heartbeat within 90 seconds. |
| `total_hashrate` | `float` | Sum of hashrates of online workers. |
| `total_blocks` | `int` | All-time mined blocks. |
| `blocks_24h` | `int` | Blocks mined in the last 24 hours. |
| `total_leases` | `int` | All-time lease count. |
| `active_leases` | `int` | Currently active leases. |
| `total_settled` | `int` | Number of completed settlements. |
| `platform_revenue` | `float` | Sum of platform fees from all settlements. |

---

### `GET /api/overview/activity`

Paginated feed of recent platform activity (blocks mined, leases started/completed).

**Auth:** None

**Query Parameters:**

| Name | Type | Default | Constraints | Description |
|------|------|---------|-------------|-------------|
| `page` | `int` | `1` | >= 1 | Page number. |
| `limit` | `int` | `50` | 1 -- 200 | Items per page. |

**Response `200`:**

```json
{
  "items": [
    {
      "type": "block",
      "timestamp": 1700000000,
      "details": {
        "worker_id": "worker-abc",
        "hash": "00000a3f...",
        "lease_id": "lease-xyz"
      }
    },
    {
      "type": "lease_started",
      "timestamp": 1699999800,
      "details": {
        "lease_id": "lease-xyz",
        "worker_id": "worker-abc",
        "consumer_id": "consumer-1",
        "duration_sec": 3600
      }
    },
    {
      "type": "lease_completed",
      "timestamp": 1700003400,
      "details": {
        "lease_id": "lease-xyz",
        "worker_id": "worker-abc",
        "consumer_id": "consumer-1",
        "blocks_found": 3
      }
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 50,
  "total_pages": 3
}
```

Activity `type` values: `"block"`, `"lease_started"`, `"lease_completed"`.

---

### `GET /api/overview/network`

Current network-level statistics.

**Auth:** None

**Response `200`:**

```json
{
  "difficulty": 80000,
  "total_workers": 85,
  "total_blocks": 4200,
  "chain_blocks": 120000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `difficulty` | `int` | Current mining difficulty (from chain, 0 if chain unavailable). |
| `total_workers` | `int` | Total registered workers. |
| `total_blocks` | `int` | Blocks recorded by the platform. |
| `chain_blocks` | `int` | Total blocks on-chain (0 if chain unavailable). |

---

## 4. Monitoring

### `GET /api/monitoring/fleet`

Paginated fleet overview of all workers with filtering and sorting.

**Auth:** None

**Query Parameters:**

| Name | Type | Default | Constraints | Description |
|------|------|---------|-------------|-------------|
| `page` | `int` | `1` | >= 1 | Page number. |
| `limit` | `int` | `20` | 1 -- 100 | Items per page. |
| `sort` | `string` | `"desc"` | `asc` or `desc` | Sort order by hashrate. |
| `eth_address` | `string` | `null` | max 128 chars | Filter by Ethereum address. |

**Response `200`:**

```json
{
  "items": [
    {
      "worker_id": "worker-abc",
      "state": "AVAILABLE",
      "hashrate": 520.0,
      "gpu_count": 2,
      "eth_address": "0x...",
      "last_heartbeat": 1700000000
    }
  ],
  "total": 85,
  "page": 1,
  "limit": 20,
  "total_pages": 5
}
```

Items are sorted by `hashrate` in the specified `sort` order. When `eth_address` is provided, only workers belonging to that address are returned.

---

### `GET /api/monitoring/stats`

Aggregated monitoring statistics across all workers.

**Auth:** None

**Response `200`:**

```json
{
  "total_workers": 85,
  "online_workers": 61,
  "total_hashrate": 15230.50,
  "avg_hashrate": 249.68,
  "total_blocks": 4200
}
```

The exact fields depend on the `MonitoringService.get_aggregated_stats()` implementation.

---

### `GET /api/monitoring/hashrate-history`

Time-series hashrate history. Can be scoped to a single worker or across the entire fleet.

**Auth:** None

**Query Parameters:**

| Name | Type | Default | Constraints | Description |
|------|------|---------|-------------|-------------|
| `worker_id` | `string` | `null` | max 128 chars | Filter to a specific worker. Omit for fleet-wide. |
| `hours` | `float` | `1.0` | 0.0167 -- 24.0 | Lookback window in hours (min ~1 minute). |

**Response `200`:**

```json
[
  {
    "timestamp": 1700000000,
    "hashrate": 520.0,
    "worker_id": "worker-abc"
  },
  {
    "timestamp": 1700000060,
    "hashrate": 518.3,
    "worker_id": "worker-abc"
  }
]
```

Returns an array of time-series data points.

---

### `GET /api/monitoring/blocks/recent`

List recently mined blocks.

**Auth:** None

**Query Parameters:**

| Name | Type | Default | Constraints | Description |
|------|------|---------|-------------|-------------|
| `limit` | `int` | `20` | 1 -- 200 | Maximum number of blocks to return. |

**Response `200`:**

```json
[
  {
    "hash": "00000a3f...",
    "worker_id": "worker-abc",
    "timestamp": 1700000000,
    "lease_id": "lease-xyz"
  }
]
```

Returns an array of recent block records, ordered by most recent first.

---

## 5. Marketplace

### `GET /api/marketplace`

Browse available workers for lease with filtering, sorting, and pagination.

**Auth:** None

**Query Parameters:**

| Name | Type | Default | Constraints | Description |
|------|------|---------|-------------|-------------|
| `page` | `int` | `1` | >= 1 | Page number. |
| `limit` | `int` | `18` | 1 -- 100 | Items per page. |
| `sort_by` | `string` | `"price"` | -- | Sort field (e.g., `"price"`, `"hashrate"`). |
| `gpu_type` | `string` | `null` | -- | Filter by GPU model substring. |
| `min_hashrate` | `float` | `null` | -- | Minimum hashrate threshold. |
| `max_price` | `float` | `null` | -- | Maximum price per minute. |
| `min_gpus` | `int` | `null` | -- | Minimum GPU count. |
| `available_only` | `bool` | `true` | -- | Show only available (non-leased) workers. |

**Response `200`:**

```json
{
  "items": [
    {
      "worker_id": "worker-abc",
      "hashrate": 520.0,
      "gpu_count": 2,
      "gpu_type": "RTX 4090",
      "price_per_min": 0.05,
      "state": "AVAILABLE"
    }
  ],
  "total": 30,
  "page": 1,
  "limit": 18,
  "total_pages": 2
}
```

---

### `GET /api/marketplace/estimate`

Estimate the cost for renting hashpower.

**Auth:** None

**Query Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `duration_sec` | `int` | Yes | Rental duration in seconds. |
| `worker_id` | `string` | No | Specific worker to estimate. |
| `min_hashrate` | `float` | No | Minimum hashrate for auto-matching. |

**Response `200`:**

```json
{
  "worker_id": "worker-abc",
  "duration_sec": 3600,
  "price_per_min": 0.05,
  "estimated_cost": 3.0
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `404` | No matching worker found. |

---

### `GET /api/workers`

List all currently available workers with reputation scores.

**Auth:** None

**Response `200`:**

```json
[
  {
    "worker_id": "worker-abc",
    "hashrate": 520.0,
    "gpu_count": 2,
    "state": "AVAILABLE",
    "price_per_min": 0.05,
    "reputation": {
      "score": 0.95,
      "uptime_pct": 99.2,
      "blocks_found": 120
    }
  }
]
```

Returns an array of available workers, each enriched with a `reputation` object.

---

### `GET /api/workers/{worker_id}/pricing`

Get the pricing configuration for a specific worker.

**Auth:** None

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `worker_id` | `string` | Worker identifier. |

**Response `200`:**

```json
{
  "worker_id": "worker-abc",
  "price_per_min": 0.05,
  "min_duration_sec": 60,
  "max_duration_sec": 86400
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `404` | Worker not found. |

---

### `GET /api/workers/{worker_id}/pricing/suggest`

Get a suggested price for a worker based on market conditions and hardware.

**Auth:** None

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `worker_id` | `string` | Worker identifier. |

**Response `200`:**

```json
{
  "worker_id": "worker-abc",
  "suggested_price_per_min": 0.045,
  "market_avg": 0.05,
  "hashrate": 520.0
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `404` | Worker not found. |

---

### `PUT /api/workers/{worker_id}/pricing`

Set pricing for a worker (marketplace router variant).

**Auth:** API Key

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `worker_id` | `string` | Worker identifier. |

**Request Body:**

```json
{
  "price_per_min": 0.05,
  "min_duration_sec": 60,
  "max_duration_sec": 86400
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `price_per_min` | `float` | Yes | -- | Price per minute in platform units. |
| `min_duration_sec` | `int` | No | `60` | Minimum lease duration (seconds). |
| `max_duration_sec` | `int` | No | `86400` | Maximum lease duration (seconds). |

**Response `200`:**

```json
{
  "worker_id": "worker-abc",
  "price_per_min": 0.05,
  "min_duration_sec": 60,
  "max_duration_sec": 86400
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Invalid pricing values. |
| `403` | Not a provider, or setting pricing for another provider's worker. |
| `404` | Worker not found. |

---

### `GET /api/workers/{worker_id}/reputation`

Get the reputation score for a specific worker.

**Auth:** None

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `worker_id` | `string` | Worker identifier. |

**Response `200`:**

```json
{
  "score": 0.95,
  "uptime_pct": 99.2,
  "blocks_found": 120,
  "total_leases": 45
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `404` | Worker not found. |

---

## 6. Provider

### `GET /api/provider/dashboard`

Aggregate dashboard for a provider: worker count, earnings, active leases, blocks mined.

**Auth:** JWT or `provider_id` query param

**Query Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `provider_id` | `string` | `""` | Ethereum address, worker ID, or account ID. If empty, resolved from JWT `sub` claim. |

**Headers:**

| Header | Description |
|--------|-------------|
| `Authorization` | `Bearer <jwt>` (used when `provider_id` is empty) |

**Response `200`:**

```json
{
  "provider_id": "0x1234...abcd",
  "worker_count": 5,
  "total_earned": 42.1234,
  "active_leases": 2,
  "total_blocks_mined": 320,
  "avg_hashrate": 485.60
}
```

Returns zeroed fields when the provider has no workers.

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Neither `provider_id` nor a valid JWT was provided. |

---

### `GET /api/provider/earnings`

List all settlement records (earnings) for a provider's workers.

**Auth:** JWT or `provider_id` query param

**Query Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `provider_id` | `string` | `""` | Ethereum address, worker ID, or account ID. Resolved from JWT if empty. |

**Response `200`:**

```json
{
  "provider_id": "0x1234...abcd",
  "earnings": [
    {
      "settlement_id": "settle-001",
      "worker_id": "worker-abc",
      "provider_payment": 1.25,
      "platform_fee": 0.125,
      "settled_at": 1700000000
    }
  ]
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Neither `provider_id` nor a valid JWT was provided. |

---

### `GET /api/provider/workers`

Paginated list of a provider's workers with online status and hardware details.

**Auth:** JWT or `provider_id` query param

**Query Parameters:**

| Name | Type | Default | Constraints | Description |
|------|------|---------|-------------|-------------|
| `page` | `int` | `1` | >= 1 | Page number. |
| `limit` | `int` | `20` | 1 -- 100 | Items per page. |
| `provider_id` | `string` | `""` | -- | Ethereum address, worker ID, or account ID. Resolved from JWT if empty. |

**Response `200`:**

```json
{
  "provider_id": "0x1234...abcd",
  "items": [
    {
      "worker_id": "worker-abc",
      "state": "AVAILABLE",
      "online": true,
      "hashrate": 520.0,
      "gpu_count": 2,
      "active_gpus": 2,
      "gpu_name": "RTX 4090",
      "memory_gb": 48,
      "price_per_min": 0.05,
      "self_blocks_found": 120,
      "total_online_sec": 864000
    }
  ],
  "total": 5,
  "page": 1,
  "limit": 20,
  "total_pages": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `online` | `bool` | `true` if last heartbeat was within 90 seconds. |
| `gpu_name` | `string` | Formatted GPU label (e.g., `"2x RTX 4090"`, `"RTX 3080 + RTX 3070"`). |
| `memory_gb` | `float` | Total GPU memory in GB. |

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Neither `provider_id` nor a valid JWT was provided. |

---

### `PUT /api/provider/workers/{worker_id}/pricing`

Set pricing for a worker (provider router variant, with JWT + API Key support).

**Auth:** JWT or API Key (provider or admin role)

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `worker_id` | `string` | Worker identifier. |

**Request Body:**

```json
{
  "price_per_min": 0.05,
  "min_duration_sec": 60,
  "max_duration_sec": 86400
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `price_per_min` | `float` | Yes | -- | Price per minute. |
| `min_duration_sec` | `int` | No | `60` | Minimum lease duration (seconds). |
| `max_duration_sec` | `int` | No | `86400` | Maximum lease duration (seconds). |

**Response `200`:**

```json
{
  "worker_id": "worker-abc",
  "price_per_min": 0.05,
  "min_duration_sec": 60,
  "max_duration_sec": 86400
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Invalid pricing values. |
| `403` | Not a provider/admin, or provider setting pricing for another's worker. |
| `404` | Worker not found. |

---

## 7. Wallet

All wallet endpoints require JWT authentication (`Authorization: Bearer <jwt>`). The wallet address is extracted from the JWT `sub` claim.

### `GET /api/wallet/history`

Historical snapshots for the connected wallet.

**Auth:** JWT (required)

**Query Parameters:**

| Name | Type | Default | Constraints | Description |
|------|------|---------|-------------|-------------|
| `period` | `string` | `"30d"` | `24h`, `7d`, `30d` | Lookback period. |

**Response `200`:**

```json
{
  "address": "0x1234...abcd",
  "period": "7d",
  "count": 168,
  "data": [
    {
      "timestamp": 1700000000,
      "hashrate": 520.0,
      "workers_online": 3,
      "blocks": 2
    }
  ]
}
```

Data points are always at hourly intervals. `24h` returns up to 24 points, `7d` up to 168, `30d` up to 720.

**Errors:**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid JWT. |

---

### `GET /api/wallet/achievements`

Achievement and milestone stats for the connected wallet.

**Auth:** JWT (required)

**Response `200`:**

```json
{
  "address": "0x1234...abcd",
  "current_hashrate": 520.0,
  "online_workers": 3,
  "total_workers": 5,
  "total_blocks": 320,
  "peak_hashrate": 610.0,
  "total_earnings": 42.5,
  "mining_days": 45.2,
  "first_seen": 1695000000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `current_hashrate` | `float` | Sum of hashrates of currently online workers. |
| `peak_hashrate` | `float` | All-time peak hashrate from snapshot history. |
| `total_earnings` | `float` | Cumulative earnings from snapshot history. |
| `mining_days` | `float` | Total uptime across all workers, in days. |
| `first_seen` | `float\|null` | Unix timestamp of the first snapshot, or `null`. |

**Errors:**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid JWT. |

---

### `GET /api/wallet/stats`

Real-time stats for the connected wallet.

**Auth:** JWT (required)

**Response `200`:**

```json
{
  "address": "0x1234...abcd",
  "online_workers": 3,
  "total_workers": 5,
  "total_hashrate": 520.0,
  "total_gpus": 10,
  "total_blocks": 320,
  "total_earnings": 42.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `online_workers` | `int` | Workers with heartbeat within 90 seconds. |
| `total_hashrate` | `float` | Sum of hashrates of online workers only. |
| `total_gpus` | `int` | Sum of GPU counts across all workers. |
| `total_earnings` | `float` | Sum of `provider_payout` from settlements. |

**Errors:**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid JWT. |

---

### `POST /api/wallet/workers/{worker_id}/command`

Send a remote control command to a worker owned by the connected wallet.

**Auth:** JWT (required)

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `worker_id` | `string` | Target worker identifier. |

**Request Body:**

```json
{
  "command": "restart",
  "params": {}
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `command` | `string` | Yes | -- | One of: `restart`, `stop`, `start`, `update_config`, `list`, `unlist`. |
| `params` | `object` | No | `{}` | Additional parameters for the command. |

**Command behavior:**

| Command | Effect |
|---------|--------|
| `restart` | Sends restart signal via MQTT. |
| `stop` | Sends stop signal via MQTT. |
| `start` | Sends start signal via MQTT. |
| `update_config` | Sends config update via MQTT. Use `params.state` to set worker state. |
| `list` | Sets worker state to `AVAILABLE` and sends `update_config` via MQTT. |
| `unlist` | Sets worker state to `SELF_MINING` and sends `update_config` via MQTT. |

Valid states for `params.state`: `SELF_MINING`, `AVAILABLE`, `LEASED`.

**Response `200`:**

```json
{
  "status": "sent",
  "worker_id": "worker-abc",
  "command": "list"
}
```

**Errors:**

| Code | Condition |
|------|-----------|
| `400` | Invalid command or invalid state value. |
| `401` | Missing or invalid JWT. |
| `403` | Worker does not belong to the authenticated wallet. |
| `404` | Worker not found. |
| `503` | MQTT broker (control service) is unavailable. |

---

### `GET /api/wallet/share`

Get data for generating a shareable achievement card.

**Auth:** JWT (required)

**Response `200`:**

```json
{
  "address": "0x1234...abcd",
  "full_address": "0x1234567890abcdef1234567890abcdef12345678",
  "total_blocks": 320,
  "peak_hashrate": 610.0,
  "total_earnings": 42.5,
  "mining_hours": 1084.8,
  "worker_count": 5,
  "generated_at": 1700000000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `address` | `string` | Truncated address for display (e.g., `0x1234...5678`). |
| `full_address` | `string` | Complete Ethereum address. |
| `mining_hours` | `float` | Total uptime across all workers, in hours. |
| `generated_at` | `float` | Unix timestamp when the data was generated. |

**Errors:**

| Code | Condition |
|------|-----------|
| `401` | Missing or invalid JWT. |

---

## Common Error Response Format

All error responses follow this structure:

```json
{
  "detail": "Human-readable error message"
}
```

## Pagination Response Format

All paginated endpoints return:

```json
{
  "items": [],
  "total": 0,
  "page": 1,
  "limit": 20,
  "total_pages": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `items` | `array` | Page of results. |
| `total` | `int` | Total matching items across all pages. |
| `page` | `int` | Current page number (1-indexed). |
| `limit` | `int` | Items per page. |
| `total_pages` | `int` | Total number of pages. |
