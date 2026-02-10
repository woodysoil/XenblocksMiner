# Authentication System

## 1. Overview

The XenBlocks Mining Platform implements a dual-mode authentication architecture:

1. **Wallet-based (SIWE-style)** -- Ethereum wallet authentication using EIP-191 signed messages, producing a JWT for session management. This is the primary flow for the web dashboard.
2. **Legacy API Key** -- A static `X-API-Key` header for programmatic/miner access. Retained for backward compatibility with existing integrations.

Both flows converge at `resolve_account()`, which normalizes the caller identity into a unified account dict used by all downstream authorization checks.

Source files:

| Component | Path |
|-----------|------|
| Auth service | `server/auth.py` |
| Account service | `server/account.py` |
| Account router | `server/routers/account.py` |
| Storage layer | `server/storage.py` |
| Frontend context | `web/src/context/WalletContext.tsx` |
| API client | `web/src/api.ts` |

---

## 2. Authentication Flow

### 2.1 Wallet-based (JWT)

```
  Browser (MetaMask)                     Server
  ─────────────────                     ──────
        │                                  │
        │  GET /api/auth/nonce?address=0x… │
        │─────────────────────────────────>│
        │                                  │  generate_nonce(address)
        │                                  │  store (nonce, expiry) in memory
        │  { nonce, message }              │
        │<─────────────────────────────────│
        │                                  │
        │  eth.personal_sign(message)      │
        │  (MetaMask popup)                │
        │                                  │
        │  POST /api/auth/verify           │
        │  { address, signature, nonce }   │
        │─────────────────────────────────>│
        │                                  │  verify_signature()
        │                                  │    1. lookup stored nonce by address
        │                                  │    2. check TTL (5 min)
        │                                  │    3. reconstruct EIP-191 message
        │                                  │    4. ecrecover → compare address
        │                                  │    5. consume nonce (one-time use)
        │                                  │  get_or_create_by_eth_address()
        │                                  │  issue_jwt(address, role, account_id)
        │  { token, address, account_id }  │
        │<─────────────────────────────────│
        │                                  │
        │  localStorage.setItem("xb_jwt") │
        │                                  │
        │  GET /api/auth/me                │
        │  Authorization: Bearer <jwt>     │
        │─────────────────────────────────>│
        │                                  │  decode_jwt() → resolve_account()
        │  { account_id, role, balance }   │
        │<─────────────────────────────────│
```

### 2.2 Legacy API Key

```
  Miner / Script                         Server
  ──────────────                         ──────
        │                                  │
        │  POST /api/auth/register         │
        │  { account_id, role }            │
        │─────────────────────────────────>│
        │                                  │  create account + generate api_key
        │  { account_id, api_key }         │
        │<─────────────────────────────────│
        │                                  │
        │  ANY /api/*                      │
        │  X-API-Key: <api_key>            │
        │─────────────────────────────────>│
        │                                  │  resolve_account() → get_by_api_key()
```

---

## 3. Signed Message Format

The plaintext message presented to the user for signing is defined by `SIGN_MESSAGE_TEMPLATE`:

```
Sign this message to authenticate with XenBlocks.

Nonce: <32-char hex>
```

Concrete example:

```
Sign this message to authenticate with XenBlocks.

Nonce: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4
```

The message is encoded with `eth_account.messages.encode_defunct(text=...)` (EIP-191 personal message prefix `\x19Ethereum Signed Message:\n<len>`), then signed via MetaMask's `personal_sign` RPC. The server recovers the signer address with `EthAccount.recover_message()`.

---

## 4. JWT Token

### Claims

| Claim | Value | Description |
|-------|-------|-------------|
| `sub` | `address.lower()` | Ethereum address, checksumless |
| `role` | `"provider"` / `"consumer"` | Account role |
| `account_id` | e.g. `"wallet-0xa1b2c3d4"` | Internal account identifier |
| `iat` | Unix timestamp | Issued-at |
| `exp` | `iat + 86400` | Expiry (24 hours) |

### Signing

- Algorithm: **HS256** (HMAC-SHA256)
- Library: PyJWT
- Secret: Provided via `--jwt-secret` CLI argument, or auto-generated with `secrets.token_hex(32)` (64-char hex, 256 bits) if omitted.

### Lifecycle

- Issued on successful `POST /api/auth/verify`.
- Validated on every request by `decode_jwt()` -- rejects `ExpiredSignatureError` and `InvalidTokenError` silently (returns `None`).
- Stored client-side under `localStorage` key `"xb_jwt"`.

### Ephemeral Secret Warning

If `--jwt-secret` is not provided, the secret is generated in-memory. All issued JWTs become invalid on server restart. The server logs a warning:

```
No --jwt-secret provided; generated ephemeral secret (JWTs will invalidate on restart)
```

---

## 5. API Key Authentication

### Header

```
X-API-Key: <32-char hex token>
```

### Generation

`secrets.token_hex(16)` produces a 32-character hexadecimal key. Keys are generated during:
- `POST /api/auth/register` -- new account creation.
- `POST /api/auth/login` -- if the existing account has no key, one is generated on the fly.
- Server startup -- backfill for default accounts (`consumer-1`, `consumer-2`) via `ensure_api_keys_for_defaults()`.

### Admin Key

A special admin key (default: `admin-test-key-do-not-use-in-production`) bypasses database lookup and returns a synthetic admin account:

```python
{"account_id": "_admin", "role": "admin", "eth_address": "", "balance": 0.0}
```

### Coexistence with JWT

`resolve_account()` checks credentials in strict order:

1. `Authorization: Bearer <jwt>` -- decoded and validated first.
2. `X-API-Key` header -- checked only if JWT is absent or invalid.
3. Admin key -- matched against the configured admin key literal.
4. Database lookup -- `get_by_api_key()` against the `accounts.api_key` column.

If both headers are present, JWT takes precedence. If JWT decoding fails, the system falls through to the API key path.

---

## 6. Account Resolution

### `resolve_account()` Middleware

This is a FastAPI dependency injected into route handlers. It extracts identity from either authentication method and returns a unified account dict:

```python
{
    "account_id": str,
    "role": str,        # "provider" | "consumer" | "admin"
    "eth_address": str,
    "balance": float,
    "api_key": str,
}
```

Returns `None` when no credentials are provided (permissive -- used for optional auth).

### `get_current_account()`

Wraps `resolve_account()` with a mandatory check: raises `HTTP 401` if no valid credentials are found.

### Role Guards

| Dependency | Allowed Roles | HTTP Status on Denial |
|------------|---------------|----------------------|
| `require_consumer()` | `consumer`, `admin` | 403 |
| `require_provider()` | `provider`, `admin` | 403 |
| `require_admin()` | `admin` | 403 |

### Auto-creation on First Wallet Connect

When `POST /api/auth/verify` succeeds for an address with no existing account, `AccountService.get_or_create_by_eth_address()` creates one:

- `account_id`: `"wallet-{address[:10].lower()}"` (e.g. `"wallet-0xa1b2c3d4"`)
- `role`: `"provider"` (default for wallet-created accounts)
- `balance`: `0.0`

The lookup uses case-insensitive comparison (`COLLATE NOCASE`) on the `eth_address` column.

---

## 7. Frontend Implementation

### WalletContext Lifecycle (`WalletContext.tsx`)

**State:**
- `address: string | null` -- connected wallet address.
- `connecting: boolean` -- loading flag during the sign flow.

**Mount (session restore):**
1. Read `"xb_jwt"` from `localStorage`.
2. If present, call `GET /api/auth/me` with the stored JWT.
3. On success, set `address` from the response's `eth_address`.
4. On failure (expired/invalid), call `clearToken()` and reset state.

**Connect:**
1. Detect `window.ethereum` (MetaMask). Alert and abort if missing.
2. Create `ethers.BrowserProvider`, obtain signer, read address.
3. `GET /api/auth/nonce?address=<addr>` -- receive `{ nonce, message }`.
4. `signer.signMessage(message)` -- triggers MetaMask popup.
5. `POST /api/auth/verify` with `{ address, signature, nonce }`.
6. Store returned `token` in `localStorage`, set `address` in state.
7. User rejection (MetaMask code `4001`) is silently ignored.

**Disconnect:**
1. Remove `"xb_jwt"` from `localStorage`.
2. Clear `address` state.

**MetaMask Event Listeners:**
- `accountsChanged` -- fires when user switches accounts in MetaMask.
- `chainChanged` -- fires when user switches networks.
- Both trigger `clearToken()` + `setAddress(null)`, forcing re-authentication.

### API Client (`api.ts`)

`apiFetch<T>()` automatically attaches the JWT to every request:

```typescript
if (token) {
    headers.set("Authorization", `Bearer ${token}`);
}
```

The token is read from `localStorage` on each call, so token rotation or clearance takes effect immediately.

---

## 8. Security Considerations

### Nonce Management

| Property | Implementation | Risk |
|----------|---------------|------|
| Generation | `secrets.token_hex(16)` -- 128-bit random | Sufficient entropy |
| Storage | In-memory dict keyed by `address.lower()` | Lost on restart; no horizontal scaling |
| TTL | 5 minutes (`NONCE_TTL = 300`) | Limits replay window |
| One-time use | Consumed (deleted) after successful verification | Prevents signature replay |
| Overwrite | New nonce request for same address replaces the previous one | At most one valid nonce per address at any time |

### Replay Protection

- Nonces are single-use: `self._nonces.pop(addr_lower, None)` on success.
- Expired nonces are rejected and cleaned: TTL check precedes signature verification.
- Address-keyed storage means a new nonce request invalidates any prior pending nonce for the same address.

### JWT Secret Management

| Scenario | Behavior |
|----------|----------|
| `--jwt-secret` provided | Stable across restarts; sessions persist |
| `--jwt-secret` omitted | `secrets.token_hex(32)` generated at startup; all JWTs invalid on restart |
| Secret compromise | All issued tokens become forgeable; rotate immediately and restart |

**Recommendation:** Always provide `--jwt-secret` in production, sourced from environment variables or a secrets manager. The 256-bit (64 hex char) auto-generated secret is cryptographically strong but ephemeral.

### Admin Key

The default admin key (`admin-test-key-do-not-use-in-production`) is a hardcoded test credential. In production deployments, this must be overridden via the `admin_key` constructor parameter. The admin key grants unrestricted access to all role-gated endpoints.

### Token Storage

JWTs are stored in `localStorage` under key `"xb_jwt"`. This is accessible to any JavaScript running on the same origin, making it vulnerable to XSS. Mitigations:

- Content Security Policy (CSP) headers to limit script sources.
- Input sanitization to prevent stored/reflected XSS.

### Address Handling

- All address comparisons are case-insensitive (`.lower()` in Python, `COLLATE NOCASE` in SQLite).
- The nonce endpoint validates format: must start with `0x` and be exactly 42 characters.
- `EthAccount.recover_message()` is wrapped in a try/except to reject malformed signatures gracefully.

### CORS

No CORS middleware configuration was found in the server setup. When the web frontend and API are served from the same origin (same host/port), CORS is not required. If they are served from different origins, `CORSMiddleware` must be added to the FastAPI app with an explicit `allow_origins` list. Wildcard (`*`) origins should be avoided in production.

### Dependency Availability

`eth-account`, `PyJWT`, and `ethers` are wrapped in try/except imports. If `eth-account` or `PyJWT` are not installed, wallet authentication raises a `RuntimeError` at call time rather than failing at import. This is intentional for environments that only use API key authentication.
