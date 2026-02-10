"""
storage.py - SQLite persistent storage layer.

Async SQLite backend using aiosqlite, with repository pattern (one class per
table) and auto-migration on first run.  Designed as a drop-in replacement for
the in-memory dicts in account.py, matcher.py, watcher.py, settlement.py.

Usage:
    db = StorageManager("platform.db")
    await db.initialize()     # creates tables, runs migrations
    acct = await db.accounts.create("consumer-1", "consumer", 1000.0, "0x...")
    await db.close()
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

try:
    import aiosqlite
except ImportError:
    raise ImportError(
        "aiosqlite is required for the storage layer. "
        "Install with: pip install aiosqlite"
    )

logger = logging.getLogger("storage")

# ---------------------------------------------------------------------------
# Schema version & migrations
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 10

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version   INTEGER NOT NULL,
    applied_at REAL NOT NULL
);

-- Accounts: consumer + provider
CREATE TABLE IF NOT EXISTS accounts (
    account_id  TEXT PRIMARY KEY,
    role        TEXT NOT NULL CHECK (role IN ('provider', 'consumer')),
    eth_address TEXT NOT NULL DEFAULT '',
    balance     REAL NOT NULL DEFAULT 0.0,
    api_key     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

-- Workers: registered mining workers
CREATE TABLE IF NOT EXISTS workers (
    worker_id       TEXT PRIMARY KEY,
    eth_address     TEXT NOT NULL DEFAULT '',
    gpu_count       INTEGER NOT NULL DEFAULT 0,
    total_memory_gb INTEGER NOT NULL DEFAULT 0,
    gpus_json       TEXT NOT NULL DEFAULT '[]',
    version         TEXT NOT NULL DEFAULT '',
    state           TEXT NOT NULL DEFAULT 'AVAILABLE',
    hashrate        REAL NOT NULL DEFAULT 0.0,
    active_gpus     INTEGER NOT NULL DEFAULT 0,
    last_heartbeat  REAL NOT NULL,
    registered_at   REAL NOT NULL,
    price_per_min   REAL NOT NULL DEFAULT 0.60,
    min_duration_sec INTEGER NOT NULL DEFAULT 60,
    max_duration_sec INTEGER NOT NULL DEFAULT 86400,
    total_online_sec REAL NOT NULL DEFAULT 0.0,
    last_online_at   REAL,
    self_blocks_found INTEGER NOT NULL DEFAULT 0
);

-- Leases: hashpower rental agreements
CREATE TABLE IF NOT EXISTS leases (
    lease_id         TEXT PRIMARY KEY,
    worker_id        TEXT NOT NULL,
    consumer_id      TEXT NOT NULL,
    consumer_address TEXT NOT NULL,
    prefix           TEXT NOT NULL DEFAULT '',
    duration_sec     INTEGER NOT NULL,
    price_per_sec    REAL NOT NULL DEFAULT 0.01,
    state            TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'completed', 'cancelled')),
    created_at       REAL NOT NULL,
    ended_at         REAL,
    blocks_found     INTEGER NOT NULL DEFAULT 0,
    total_hashrate_samples REAL NOT NULL DEFAULT 0.0,
    hashrate_count   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (worker_id) REFERENCES workers(worker_id),
    FOREIGN KEY (consumer_id) REFERENCES accounts(account_id)
);

-- Blocks: mined block records
CREATE TABLE IF NOT EXISTS blocks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    lease_id     TEXT NOT NULL DEFAULT '',
    worker_id    TEXT NOT NULL,
    block_hash   TEXT NOT NULL,
    key          TEXT NOT NULL,
    account      TEXT NOT NULL DEFAULT '',
    attempts     INTEGER NOT NULL DEFAULT 0,
    hashrate     TEXT NOT NULL DEFAULT '0.0',
    prefix_valid INTEGER NOT NULL DEFAULT 1,
    chain_verified INTEGER NOT NULL DEFAULT 0,
    chain_block_id INTEGER,
    created_at   REAL NOT NULL
);

-- Settlements: completed lease settlements
CREATE TABLE IF NOT EXISTS settlements (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    lease_id         TEXT NOT NULL UNIQUE,
    consumer_id      TEXT NOT NULL,
    worker_id        TEXT NOT NULL,
    duration_sec     REAL NOT NULL,
    blocks_found     INTEGER NOT NULL DEFAULT 0,
    total_cost       REAL NOT NULL,
    provider_payout  REAL NOT NULL,
    platform_fee     REAL NOT NULL,
    settled_at       REAL NOT NULL,
    FOREIGN KEY (lease_id) REFERENCES leases(lease_id)
);

-- Transactions: audit trail for all balance changes
CREATE TABLE IF NOT EXISTS transactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT NOT NULL,
    type         TEXT NOT NULL CHECK (type IN ('deposit', 'withdraw', 'lease_charge', 'provider_payout', 'platform_fee', 'transfer_in', 'transfer_out')),
    amount       REAL NOT NULL,
    reference_id TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Hashrate time-series snapshots (per worker)
CREATE TABLE IF NOT EXISTS hashrate_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    hashrate REAL NOT NULL,
    active_gpus INTEGER NOT NULL DEFAULT 0,
    timestamp REAL NOT NULL
);

-- Wallet snapshots: hourly/daily aggregates per wallet address
CREATE TABLE IF NOT EXISTS wallet_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    eth_address TEXT NOT NULL COLLATE NOCASE,
    timestamp REAL NOT NULL,
    interval_type TEXT NOT NULL DEFAULT 'hourly',
    total_hashrate REAL NOT NULL DEFAULT 0,
    online_workers INTEGER NOT NULL DEFAULT 0,
    total_workers INTEGER NOT NULL DEFAULT 0,
    blocks_found INTEGER NOT NULL DEFAULT 0,
    cumulative_blocks INTEGER NOT NULL DEFAULT 0,
    earnings REAL NOT NULL DEFAULT 0
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_leases_state ON leases(state);
CREATE INDEX IF NOT EXISTS idx_leases_worker ON leases(worker_id);
CREATE INDEX IF NOT EXISTS idx_leases_consumer ON leases(consumer_id);
CREATE INDEX IF NOT EXISTS idx_blocks_lease ON blocks(lease_id);
CREATE INDEX IF NOT EXISTS idx_blocks_worker ON blocks(worker_id);
CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_settlements_lease ON settlements(lease_id);
CREATE INDEX IF NOT EXISTS idx_accounts_api_key ON accounts(api_key);
CREATE INDEX IF NOT EXISTS idx_accounts_eth_address ON accounts(eth_address COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_blocks_created ON blocks(created_at);
CREATE INDEX IF NOT EXISTS idx_blocks_self ON blocks(lease_id) WHERE lease_id = '';
CREATE INDEX IF NOT EXISTS idx_snapshots_worker_ts ON hashrate_snapshots(worker_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON hashrate_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_wallet_snapshots_addr_time ON wallet_snapshots(eth_address, timestamp);
CREATE INDEX IF NOT EXISTS idx_wallet_snapshots_addr_type ON wallet_snapshots(eth_address, interval_type);
"""


# ---------------------------------------------------------------------------
# Repository: Accounts
# ---------------------------------------------------------------------------

class AccountRepo:
    """CRUD operations for the accounts table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(
        self, account_id: str, role: str, balance: float = 0.0, eth_address: str = ""
    ) -> Optional[dict]:
        now = time.time()
        try:
            await self._db.execute(
                "INSERT OR IGNORE INTO accounts (account_id, role, eth_address, balance, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (account_id, role, eth_address, balance, now, now),
            )
            await self._db.commit()
        except Exception:
            logger.exception("Failed to create account %s", account_id)
            return None
        return await self.get(account_id)

    async def get(self, account_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT account_id, role, eth_address, balance, api_key, created_at, updated_at "
            "FROM accounts WHERE account_id = ?",
            (account_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "account_id": row[0],
            "role": row[1],
            "eth_address": row[2],
            "balance": row[3],
            "api_key": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    async def get_by_api_key(self, api_key: str) -> Optional[dict]:
        """Look up an account by API key."""
        if not api_key:
            return None
        async with self._db.execute(
            "SELECT account_id, role, eth_address, balance, api_key, created_at, updated_at "
            "FROM accounts WHERE api_key = ? AND api_key != ''",
            (api_key,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "account_id": row[0],
            "role": row[1],
            "eth_address": row[2],
            "balance": row[3],
            "api_key": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    async def get_by_eth_address(self, address: str) -> Optional[dict]:
        """Look up an account by Ethereum address (case-insensitive)."""
        if not address:
            return None
        async with self._db.execute(
            "SELECT account_id, role, eth_address, balance, api_key, created_at, updated_at "
            "FROM accounts WHERE eth_address = ? COLLATE NOCASE",
            (address,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "account_id": row[0],
            "role": row[1],
            "eth_address": row[2],
            "balance": row[3],
            "api_key": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    async def set_api_key(self, account_id: str, api_key: str):
        """Set the API key for an account."""
        now = time.time()
        await self._db.execute(
            "UPDATE accounts SET api_key = ?, updated_at = ? WHERE account_id = ?",
            (api_key, now, account_id),
        )
        await self._db.commit()

    async def get_or_create_provider(self, worker_id: str, eth_address: str) -> dict:
        acct = await self.get(worker_id)
        if acct is None:
            acct = await self.create(worker_id, "provider", eth_address=eth_address)
        return acct

    async def update_balance(self, account_id: str, new_balance: float):
        now = time.time()
        await self._db.execute(
            "UPDATE accounts SET balance = ?, updated_at = ? WHERE account_id = ?",
            (new_balance, now, account_id),
        )
        await self._db.commit()

    async def deposit(self, account_id: str, amount: float) -> dict:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        acct = await self.get(account_id)
        if acct is None:
            raise KeyError(f"Account {account_id} not found")
        new_balance = acct["balance"] + amount
        await self.update_balance(account_id, new_balance)
        await self._record_tx(account_id, "deposit", amount)
        acct["balance"] = new_balance
        return acct

    async def withdraw(self, account_id: str, amount: float) -> dict:
        if amount <= 0:
            raise ValueError("Withdraw amount must be positive")
        acct = await self.get(account_id)
        if acct is None:
            raise KeyError(f"Account {account_id} not found")
        if acct["balance"] < amount:
            raise ValueError(f"Insufficient balance: have {acct['balance']:.4f}, need {amount:.4f}")
        new_balance = acct["balance"] - amount
        await self.update_balance(account_id, new_balance)
        await self._record_tx(account_id, "withdraw", amount)
        acct["balance"] = new_balance
        return acct

    async def transfer(self, from_id: str, to_id: str, amount: float, ref: str = ""):
        """Atomic transfer between two accounts within a transaction."""
        from_acct = await self.get(from_id)
        to_acct = await self.get(to_id)
        if from_acct is None:
            raise KeyError(f"Account {from_id} not found")
        if to_acct is None:
            raise KeyError(f"Account {to_id} not found")
        if from_acct["balance"] < amount:
            raise ValueError(f"Insufficient balance in {from_id}")
        now = time.time()
        await self._db.execute(
            "UPDATE accounts SET balance = balance - ?, updated_at = ? WHERE account_id = ?",
            (amount, now, from_id),
        )
        await self._db.execute(
            "UPDATE accounts SET balance = balance + ?, updated_at = ? WHERE account_id = ?",
            (amount, now, to_id),
        )
        await self._db.execute(
            "INSERT INTO transactions (account_id, type, amount, reference_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (from_id, "transfer_out", amount, ref, now),
        )
        await self._db.execute(
            "INSERT INTO transactions (account_id, type, amount, reference_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (to_id, "transfer_in", amount, ref, now),
        )
        await self._db.commit()

    async def list_all(self) -> Dict[str, dict]:
        result = {}
        async with self._db.execute(
            "SELECT account_id, role, eth_address, balance, api_key FROM accounts"
        ) as cursor:
            async for row in cursor:
                result[row[0]] = {
                    "account_id": row[0],
                    "role": row[1],
                    "eth_address": row[2],
                    "balance": row[3],
                    "api_key": row[4],
                }
        return result

    async def _record_tx(self, account_id: str, tx_type: str, amount: float, ref: str = ""):
        now = time.time()
        await self._db.execute(
            "INSERT INTO transactions (account_id, type, amount, reference_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (account_id, tx_type, amount, ref, now),
        )
        await self._db.commit()


# ---------------------------------------------------------------------------
# Repository: Workers
# ---------------------------------------------------------------------------

class WorkerRepo:
    """CRUD operations for the workers table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def upsert(
        self,
        worker_id: str,
        eth_address: str = "",
        gpu_count: int = 0,
        total_memory_gb: int = 0,
        gpus: Optional[list] = None,
        version: str = "",
        state: str = "AVAILABLE",
    ) -> dict:
        now = time.time()
        gpus_json = json.dumps(gpus or [])
        await self._db.execute(
            "INSERT INTO workers (worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
            "version, state, last_heartbeat, registered_at, last_online_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(worker_id) DO UPDATE SET "
            "eth_address=excluded.eth_address, gpu_count=excluded.gpu_count, "
            "total_memory_gb=excluded.total_memory_gb, gpus_json=excluded.gpus_json, "
            "version=excluded.version, state=excluded.state, last_heartbeat=excluded.last_heartbeat, "
            "last_online_at=excluded.last_online_at",
            (worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, version, state, now, now, now),
        )
        await self._db.commit()
        return await self.get(worker_id)

    async def get(self, worker_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
            "version, state, hashrate, active_gpus, last_heartbeat, registered_at, "
            "price_per_min, min_duration_sec, max_duration_sec, "
            "total_online_sec, last_online_at, self_blocks_found "
            "FROM workers WHERE worker_id = ?",
            (worker_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "worker_id": row[0],
            "eth_address": row[1],
            "gpu_count": row[2],
            "total_memory_gb": row[3],
            "gpus": json.loads(row[4]),
            "version": row[5],
            "state": row[6],
            "hashrate": row[7],
            "active_gpus": row[8],
            "last_heartbeat": row[9],
            "registered_at": row[10],
            "price_per_min": row[11],
            "min_duration_sec": row[12],
            "max_duration_sec": row[13],
            "total_online_sec": row[14] or 0.0,
            "last_online_at": row[15],
            "self_blocks_found": row[16],
        }

    async def update_heartbeat(self, worker_id: str, hashrate: float, active_gpus: int):
        now = time.time()
        # Accumulate uptime: delta since last_online_at (capped at 60s to avoid
        # counting large gaps from disconnections)
        async with self._db.execute(
            "SELECT last_online_at FROM workers WHERE worker_id = ?", (worker_id,)
        ) as cursor:
            row = await cursor.fetchone()
        delta = 0.0
        if row and row[0] is not None:
            delta = min(now - row[0], 60.0)  # cap at 60s per heartbeat interval
            if delta < 0:
                delta = 0.0
        await self._db.execute(
            "UPDATE workers SET hashrate = ?, active_gpus = ?, last_heartbeat = ?, "
            "total_online_sec = total_online_sec + ?, last_online_at = ? "
            "WHERE worker_id = ?",
            (hashrate, active_gpus, now, delta, now, worker_id),
        )
        await self._db.commit()

    async def update_state(self, worker_id: str, state: str):
        await self._db.execute(
            "UPDATE workers SET state = ? WHERE worker_id = ?",
            (state, worker_id),
        )
        await self._db.commit()

    async def update_pricing(
        self, worker_id: str, price_per_min: float, min_duration_sec: int, max_duration_sec: int
    ):
        await self._db.execute(
            "UPDATE workers SET price_per_min = ?, min_duration_sec = ?, max_duration_sec = ? "
            "WHERE worker_id = ?",
            (price_per_min, min_duration_sec, max_duration_sec, worker_id),
        )
        await self._db.commit()

    async def increment_self_blocks(self, worker_id: str):
        await self._db.execute(
            "UPDATE workers SET self_blocks_found = self_blocks_found + 1 WHERE worker_id = ?",
            (worker_id,),
        )
        await self._db.commit()

    async def list_all(self) -> List[dict]:
        results = []
        async with self._db.execute(
            "SELECT worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
            "version, state, hashrate, active_gpus, last_heartbeat, registered_at, "
            "price_per_min, min_duration_sec, max_duration_sec, "
            "total_online_sec, last_online_at, self_blocks_found "
            "FROM workers"
        ) as cursor:
            async for row in cursor:
                results.append({
                    "worker_id": row[0],
                    "eth_address": row[1],
                    "gpu_count": row[2],
                    "total_memory_gb": row[3],
                    "gpus": json.loads(row[4]),
                    "version": row[5],
                    "state": row[6],
                    "hashrate": row[7],
                    "active_gpus": row[8],
                    "last_heartbeat": row[9],
                    "registered_at": row[10],
                    "price_per_min": row[11],
                    "min_duration_sec": row[12],
                    "max_duration_sec": row[13],
                    "total_online_sec": row[14] or 0.0,
                    "last_online_at": row[15],
                    "self_blocks_found": row[16],
                })
        return results

    async def count(self) -> int:
        async with self._db.execute("SELECT COUNT(*) FROM workers") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def find_available(self, exclude_worker_ids: Optional[List[str]] = None) -> Optional[dict]:
        """Find the first available worker not in the exclusion list."""
        exclude = set(exclude_worker_ids or [])
        async with self._db.execute(
            "SELECT worker_id FROM workers WHERE state = 'AVAILABLE' ORDER BY registered_at"
        ) as cursor:
            async for row in cursor:
                if row[0] not in exclude:
                    return await self.get(row[0])
        return None


# ---------------------------------------------------------------------------
# Repository: Leases
# ---------------------------------------------------------------------------

class LeaseRepo:
    """CRUD operations for the leases table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(
        self,
        lease_id: str,
        worker_id: str,
        consumer_id: str,
        consumer_address: str,
        prefix: str,
        duration_sec: int,
        price_per_sec: float = 0.01,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO leases (lease_id, worker_id, consumer_id, consumer_address, prefix, "
            "duration_sec, price_per_sec, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)",
            (lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, price_per_sec, now),
        )
        await self._db.commit()
        return await self.get(lease_id)

    async def get(self, lease_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, "
            "price_per_sec, state, created_at, ended_at, blocks_found, "
            "total_hashrate_samples, hashrate_count "
            "FROM leases WHERE lease_id = ?",
            (lease_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        created_at = row[8]
        ended_at = row[9]
        hashrate_count = row[12]
        total_hashrate_samples = row[11]
        elapsed = (ended_at or time.time()) - created_at
        avg_hashrate = (total_hashrate_samples / hashrate_count) if hashrate_count > 0 else 0.0
        return {
            "lease_id": row[0],
            "worker_id": row[1],
            "consumer_id": row[2],
            "consumer_address": row[3],
            "prefix": row[4],
            "duration_sec": row[5],
            "price_per_sec": row[6],
            "state": row[7],
            "created_at": created_at,
            "ended_at": ended_at,
            "blocks_found": row[10],
            "total_hashrate_samples": total_hashrate_samples,
            "hashrate_count": hashrate_count,
            "avg_hashrate": avg_hashrate,
            "elapsed_sec": elapsed,
        }

    async def update_state(self, lease_id: str, state: str, ended_at: Optional[float] = None):
        if ended_at is not None:
            await self._db.execute(
                "UPDATE leases SET state = ?, ended_at = ? WHERE lease_id = ?",
                (state, ended_at, lease_id),
            )
        else:
            await self._db.execute(
                "UPDATE leases SET state = ? WHERE lease_id = ?",
                (state, lease_id),
            )
        await self._db.commit()

    async def increment_blocks(self, lease_id: str):
        await self._db.execute(
            "UPDATE leases SET blocks_found = blocks_found + 1 WHERE lease_id = ?",
            (lease_id,),
        )
        await self._db.commit()

    async def update_hashrate_stats(self, lease_id: str, hashrate: float):
        await self._db.execute(
            "UPDATE leases SET total_hashrate_samples = total_hashrate_samples + ?, "
            "hashrate_count = hashrate_count + 1 WHERE lease_id = ?",
            (hashrate, lease_id),
        )
        await self._db.commit()

    async def get_active_lease_for_worker(self, worker_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT lease_id FROM leases WHERE worker_id = ? AND state = 'active'",
            (worker_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return await self.get(row[0])

    async def find_expired(self) -> List[dict]:
        """Find active leases that have exceeded their duration."""
        now = time.time()
        results = []
        async with self._db.execute(
            "SELECT lease_id FROM leases WHERE state = 'active' AND (created_at + duration_sec) < ?",
            (now,),
        ) as cursor:
            async for row in cursor:
                lease = await self.get(row[0])
                if lease:
                    results.append(lease)
        return results

    async def list_all(self, state: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        cols = ("lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, "
                "price_per_sec, state, created_at, ended_at, blocks_found, "
                "total_hashrate_samples, hashrate_count")
        if state:
            query = f"SELECT {cols} FROM leases WHERE state = ? ORDER BY created_at DESC"
            params: tuple = (state,)
        else:
            query = f"SELECT {cols} FROM leases ORDER BY created_at DESC"
            params = ()
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = params + (limit, offset)
        results = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                created_at = row[8]
                ended_at = row[9]
                hashrate_count = row[12]
                total_hashrate_samples = row[11]
                elapsed = (ended_at or time.time()) - created_at
                avg_hashrate = (total_hashrate_samples / hashrate_count) if hashrate_count > 0 else 0.0
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "consumer_id": row[2],
                    "consumer_address": row[3],
                    "prefix": row[4],
                    "duration_sec": row[5],
                    "price_per_sec": row[6],
                    "state": row[7],
                    "created_at": created_at,
                    "ended_at": ended_at,
                    "blocks_found": row[10],
                    "total_hashrate_samples": total_hashrate_samples,
                    "hashrate_count": hashrate_count,
                    "avg_hashrate": avg_hashrate,
                    "elapsed_sec": elapsed,
                })
        return results

    async def count(self, state: Optional[str] = None) -> int:
        if state:
            async with self._db.execute(
                "SELECT COUNT(*) FROM leases WHERE state = ?", (state,)
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self._db.execute("SELECT COUNT(*) FROM leases") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Repository: Blocks
# ---------------------------------------------------------------------------

class BlockRepo:
    """CRUD operations for the blocks table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(
        self,
        lease_id: str,
        worker_id: str,
        block_hash: str,
        key: str,
        account: str = "",
        attempts: int = 0,
        hashrate: str = "0.0",
        prefix_valid: bool = True,
        chain_verified: bool = False,
        chain_block_id: int = None,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO blocks (lease_id, worker_id, block_hash, key, account, attempts, hashrate, prefix_valid, chain_verified, chain_block_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (lease_id, worker_id, block_hash, key, account, attempts, hashrate, int(prefix_valid), int(chain_verified), chain_block_id, now),
        )
        await self._db.commit()
        return {
            "lease_id": lease_id,
            "worker_id": worker_id,
            "hash": block_hash,
            "key": key,
            "account": account,
            "attempts": attempts,
            "hashrate": hashrate,
            "prefix_valid": prefix_valid,
            "chain_verified": chain_verified,
            "chain_block_id": chain_block_id,
            "timestamp": now,
        }

    async def get_for_lease(self, lease_id: str) -> List[dict]:
        results = []
        async with self._db.execute(
            "SELECT lease_id, worker_id, block_hash, key, account, attempts, hashrate, prefix_valid, chain_verified, chain_block_id, created_at "
            "FROM blocks WHERE lease_id = ? ORDER BY created_at",
            (lease_id,),
        ) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "hash": row[2],
                    "key": row[3],
                    "account": row[4],
                    "attempts": row[5],
                    "hashrate": row[6],
                    "prefix_valid": bool(row[7]),
                    "chain_verified": bool(row[8]),
                    "chain_block_id": row[9],
                    "timestamp": row[10],
                })
        return results

    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        results = []
        query = ("SELECT lease_id, worker_id, block_hash, key, account, attempts, hashrate, "
                 "prefix_valid, chain_verified, chain_block_id, created_at "
                 "FROM blocks ORDER BY created_at DESC")
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "hash": row[2],
                    "key": row[3],
                    "account": row[4],
                    "attempts": row[5],
                    "hashrate": row[6],
                    "prefix_valid": bool(row[7]),
                    "chain_verified": bool(row[8]),
                    "chain_block_id": row[9],
                    "timestamp": row[10],
                })
        return results

    async def get_self_mined(self, worker_id: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        query = ("SELECT lease_id, worker_id, block_hash, key, account, attempts, hashrate, "
                 "prefix_valid, chain_verified, chain_block_id, created_at "
                 "FROM blocks WHERE lease_id = ''")
        params: tuple = ()
        if worker_id:
            query += " AND worker_id = ?"
            params = (worker_id,)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = params + (limit, offset)
        results = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "worker_id": row[1],
                    "hash": row[2],
                    "key": row[3],
                    "account": row[4],
                    "attempts": row[5],
                    "hashrate": row[6],
                    "prefix_valid": bool(row[7]),
                    "chain_verified": bool(row[8]),
                    "chain_block_id": row[9],
                    "timestamp": row[10],
                })
        return results

    async def count(self, lease_id: Optional[str] = None) -> int:
        if lease_id:
            async with self._db.execute(
                "SELECT COUNT(*) FROM blocks WHERE lease_id = ?", (lease_id,)
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self._db.execute("SELECT COUNT(*) FROM blocks") as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_self_mined(self, worker_id: Optional[str] = None) -> int:
        if worker_id:
            async with self._db.execute(
                "SELECT COUNT(*) FROM blocks WHERE lease_id = '' AND worker_id = ?", (worker_id,)
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with self._db.execute(
                "SELECT COUNT(*) FROM blocks WHERE lease_id = ''"
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_since(self, since_ts: float) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM blocks WHERE created_at >= ?", (since_ts,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Repository: Settlements
# ---------------------------------------------------------------------------

class SettlementRepo:
    """CRUD operations for the settlements table."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(
        self,
        lease_id: str,
        consumer_id: str,
        worker_id: str,
        duration_sec: float,
        blocks_found: int,
        total_cost: float,
        provider_payout: float,
        platform_fee: float,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO settlements (lease_id, consumer_id, worker_id, duration_sec, blocks_found, "
            "total_cost, provider_payout, platform_fee, settled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (lease_id, consumer_id, worker_id, duration_sec, blocks_found, total_cost, provider_payout, platform_fee, now),
        )
        await self._db.commit()
        return {
            "lease_id": lease_id,
            "consumer_id": consumer_id,
            "worker_id": worker_id,
            "duration_sec": round(duration_sec, 2),
            "blocks_found": blocks_found,
            "total_cost": round(total_cost, 4),
            "provider_payment": round(provider_payout, 4),
            "platform_fee": round(platform_fee, 4),
        }

    async def get(self, lease_id: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT lease_id, consumer_id, worker_id, duration_sec, blocks_found, "
            "total_cost, provider_payout, platform_fee, settled_at "
            "FROM settlements WHERE lease_id = ?",
            (lease_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "lease_id": row[0],
            "consumer_id": row[1],
            "worker_id": row[2],
            "duration_sec": round(row[3], 2),
            "blocks_found": row[4],
            "total_cost": round(row[5], 4),
            "provider_payment": round(row[6], 4),
            "platform_fee": round(row[7], 4),
        }

    async def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        results = []
        query = ("SELECT lease_id, consumer_id, worker_id, duration_sec, blocks_found, "
                 "total_cost, provider_payout, platform_fee, settled_at "
                 "FROM settlements ORDER BY settled_at DESC")
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                results.append({
                    "lease_id": row[0],
                    "consumer_id": row[1],
                    "worker_id": row[2],
                    "duration_sec": round(row[3], 2),
                    "blocks_found": row[4],
                    "total_cost": round(row[5], 4),
                    "provider_payment": round(row[6], 4),
                    "platform_fee": round(row[7], 4),
                })
        return results

    async def count(self) -> int:
        async with self._db.execute("SELECT COUNT(*) FROM settlements") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Repository: Transactions (audit log)
# ---------------------------------------------------------------------------

class TransactionRepo:
    """Read-only queries + insert for the transactions audit log."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def record(self, account_id: str, tx_type: str, amount: float, reference_id: str = ""):
        now = time.time()
        await self._db.execute(
            "INSERT INTO transactions (account_id, type, amount, reference_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (account_id, tx_type, amount, reference_id, now),
        )
        await self._db.commit()

    async def list_for_account(self, account_id: str) -> List[dict]:
        results = []
        async with self._db.execute(
            "SELECT id, account_id, type, amount, reference_id, created_at "
            "FROM transactions WHERE account_id = ? ORDER BY created_at DESC",
            (account_id,),
        ) as cursor:
            async for row in cursor:
                results.append({
                    "id": row[0],
                    "account_id": row[1],
                    "type": row[2],
                    "amount": row[3],
                    "reference_id": row[4],
                    "created_at": row[5],
                })
        return results

    async def list_all(self) -> List[dict]:
        results = []
        async with self._db.execute(
            "SELECT id, account_id, type, amount, reference_id, created_at "
            "FROM transactions ORDER BY created_at DESC"
        ) as cursor:
            async for row in cursor:
                results.append({
                    "id": row[0],
                    "account_id": row[1],
                    "type": row[2],
                    "amount": row[3],
                    "reference_id": row[4],
                    "created_at": row[5],
                })
        return results


# ---------------------------------------------------------------------------
# Repository: Hashrate Snapshots
# ---------------------------------------------------------------------------

class SnapshotRepo:

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def insert_batch(self, rows: List[tuple]):
        now = time.time()
        await self._db.executemany(
            "INSERT INTO hashrate_snapshots (worker_id, hashrate, active_gpus, timestamp) "
            "VALUES (?, ?, ?, ?)",
            [(wid, hr, gpus, now) for wid, hr, gpus in rows],
        )
        await self._db.commit()

    async def query(self, worker_id: Optional[str] = None, hours: float = 1) -> List[dict]:
        cutoff = time.time() - hours * 3600
        if worker_id:
            sql = ("SELECT worker_id, hashrate, active_gpus, timestamp "
                   "FROM hashrate_snapshots WHERE worker_id = ? AND timestamp >= ? "
                   "ORDER BY timestamp")
            params: tuple = (worker_id, cutoff)
        else:
            sql = ("SELECT worker_id, hashrate, active_gpus, timestamp "
                   "FROM hashrate_snapshots WHERE timestamp >= ? "
                   "ORDER BY timestamp")
            params = (cutoff,)
        results = []
        async with self._db.execute(sql, params) as cursor:
            async for row in cursor:
                results.append({
                    "worker_id": row[0],
                    "hashrate": row[1],
                    "active_gpus": row[2],
                    "timestamp": row[3],
                })
        return results

    async def delete_older_than(self, hours: float = 24):
        cutoff = time.time() - hours * 3600
        await self._db.execute(
            "DELETE FROM hashrate_snapshots WHERE timestamp < ?", (cutoff,)
        )
        await self._db.commit()


# ---------------------------------------------------------------------------
# Repository: Wallet Snapshots (per-address aggregated stats)
# ---------------------------------------------------------------------------

class WalletSnapshotRepo:
    """Hourly/daily snapshots per wallet address for historical charts."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def insert(
        self,
        eth_address: str,
        interval_type: str,
        total_hashrate: float,
        online_workers: int,
        total_workers: int,
        blocks_found: int,
        cumulative_blocks: int,
        earnings: float,
    ) -> dict:
        now = time.time()
        await self._db.execute(
            "INSERT INTO wallet_snapshots (eth_address, timestamp, interval_type, total_hashrate, "
            "online_workers, total_workers, blocks_found, cumulative_blocks, earnings) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (eth_address, now, interval_type, total_hashrate, online_workers, total_workers,
             blocks_found, cumulative_blocks, earnings),
        )
        await self._db.commit()
        return {
            "eth_address": eth_address,
            "timestamp": now,
            "interval_type": interval_type,
            "total_hashrate": total_hashrate,
            "online_workers": online_workers,
            "total_workers": total_workers,
            "blocks_found": blocks_found,
            "cumulative_blocks": cumulative_blocks,
            "earnings": earnings,
        }

    async def query(
        self,
        eth_address: str,
        hours: float = 24,
        interval_type: str = "hourly",
    ) -> List[dict]:
        cutoff = time.time() - hours * 3600
        results = []
        async with self._db.execute(
            "SELECT timestamp, total_hashrate, online_workers, total_workers, "
            "blocks_found, cumulative_blocks, earnings "
            "FROM wallet_snapshots "
            "WHERE eth_address = ? COLLATE NOCASE AND timestamp >= ? AND interval_type = ? "
            "ORDER BY timestamp",
            (eth_address, cutoff, interval_type),
        ) as cursor:
            async for row in cursor:
                results.append({
                    "timestamp": row[0],
                    "hashrate": row[1],
                    "online_workers": row[2],
                    "total_workers": row[3],
                    "blocks": row[4],
                    "cumulative_blocks": row[5],
                    "earnings": row[6],
                })
        return results

    async def get_latest(self, eth_address: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT timestamp, total_hashrate, online_workers, total_workers, "
            "blocks_found, cumulative_blocks, earnings "
            "FROM wallet_snapshots "
            "WHERE eth_address = ? COLLATE NOCASE "
            "ORDER BY timestamp DESC LIMIT 1",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "timestamp": row[0],
            "hashrate": row[1],
            "online_workers": row[2],
            "total_workers": row[3],
            "blocks": row[4],
            "cumulative_blocks": row[5],
            "earnings": row[6],
        }

    async def get_achievements(self, eth_address: str) -> dict:
        """Get aggregated achievement stats for a wallet."""
        # Peak hashrate
        async with self._db.execute(
            "SELECT MAX(total_hashrate) FROM wallet_snapshots WHERE eth_address = ? COLLATE NOCASE",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        peak_hashrate = row[0] if row and row[0] else 0

        # Total blocks (latest cumulative)
        async with self._db.execute(
            "SELECT cumulative_blocks FROM wallet_snapshots "
            "WHERE eth_address = ? COLLATE NOCASE ORDER BY timestamp DESC LIMIT 1",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        total_blocks = row[0] if row and row[0] else 0

        # Total earnings
        async with self._db.execute(
            "SELECT SUM(earnings) FROM wallet_snapshots WHERE eth_address = ? COLLATE NOCASE",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        total_earnings = row[0] if row and row[0] else 0

        # First seen (earliest snapshot)
        async with self._db.execute(
            "SELECT MIN(timestamp) FROM wallet_snapshots WHERE eth_address = ? COLLATE NOCASE",
            (eth_address,),
        ) as cursor:
            row = await cursor.fetchone()
        first_seen = row[0] if row and row[0] else None

        return {
            "peak_hashrate": peak_hashrate,
            "total_blocks": total_blocks,
            "total_earnings": round(total_earnings, 4),
            "first_seen": first_seen,
        }

    async def cleanup(self, hourly_retain_hours: float = 168, daily_retain_days: float = 90):
        """Delete old snapshots: hourly older than 7 days, daily older than 90 days."""
        now = time.time()
        await self._db.execute(
            "DELETE FROM wallet_snapshots WHERE interval_type = 'hourly' AND timestamp < ?",
            (now - hourly_retain_hours * 3600,),
        )
        await self._db.execute(
            "DELETE FROM wallet_snapshots WHERE interval_type = 'daily' AND timestamp < ?",
            (now - daily_retain_days * 86400,),
        )
        await self._db.commit()


# ---------------------------------------------------------------------------
# Storage Manager
# ---------------------------------------------------------------------------

class StorageManager:
    """Top-level manager: opens the database, runs migrations, exposes repos."""

    def __init__(self, db_path: str = "platform.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self.accounts: Optional[AccountRepo] = None
        self.workers: Optional[WorkerRepo] = None
        self.leases: Optional[LeaseRepo] = None
        self.blocks: Optional[BlockRepo] = None
        self.settlements: Optional[SettlementRepo] = None
        self.transactions: Optional[TransactionRepo] = None
        self.snapshots: Optional[SnapshotRepo] = None
        self.wallet_snapshots: Optional[WalletSnapshotRepo] = None

    async def initialize(self):
        """Open database, create/migrate schema, instantiate repos."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._migrate()

        self.accounts = AccountRepo(self._db)
        self.workers = WorkerRepo(self._db)
        self.leases = LeaseRepo(self._db)
        self.blocks = BlockRepo(self._db)
        self.settlements = SettlementRepo(self._db)
        self.transactions = TransactionRepo(self._db)
        self.snapshots = SnapshotRepo(self._db)
        self.wallet_snapshots = WalletSnapshotRepo(self._db)

        logger.info("Storage initialized: %s", self.db_path)

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Storage closed")

    async def _migrate(self):
        """Run schema creation / migrations."""
        # Check current version
        current_version = 0
        try:
            async with self._db.execute("SELECT MAX(version) FROM schema_version") as cursor:
                row = await cursor.fetchone()
                if row and row[0] is not None:
                    current_version = row[0]
        except Exception:
            # Table doesn't exist yet
            pass

        if current_version < SCHEMA_VERSION:
            logger.info("Migrating database from v%d to v%d", current_version, SCHEMA_VERSION)
            await self._db.executescript(SCHEMA_SQL)

            # V2 migration: add chain_verified and chain_block_id to blocks
            if current_version < 2:
                # ALTER TABLE is idempotent-safe: columns already exist in fresh schema
                try:
                    await self._db.execute(
                        "ALTER TABLE blocks ADD COLUMN chain_verified INTEGER NOT NULL DEFAULT 0"
                    )
                except Exception:
                    pass  # column already exists from CREATE TABLE
                try:
                    await self._db.execute(
                        "ALTER TABLE blocks ADD COLUMN chain_block_id INTEGER"
                    )
                except Exception:
                    pass

            # V3 migration: add pricing columns to workers
            if current_version < 3:
                for col, typedef in [
                    ("price_per_min", "REAL NOT NULL DEFAULT 0.60"),
                    ("min_duration_sec", "INTEGER NOT NULL DEFAULT 60"),
                    ("max_duration_sec", "INTEGER NOT NULL DEFAULT 86400"),
                ]:
                    try:
                        await self._db.execute(
                            f"ALTER TABLE workers ADD COLUMN {col} {typedef}"
                        )
                    except Exception:
                        pass  # column already exists from CREATE TABLE

            # V4 migration: add uptime tracking columns to workers
            if current_version < 4:
                for col, typedef in [
                    ("total_online_sec", "REAL NOT NULL DEFAULT 0.0"),
                    ("last_online_at", "REAL"),
                ]:
                    try:
                        await self._db.execute(
                            f"ALTER TABLE workers ADD COLUMN {col} {typedef}"
                        )
                    except Exception:
                        pass

            # V5 migration: add api_key column to accounts
            if current_version < 5:
                try:
                    await self._db.execute(
                        "ALTER TABLE accounts ADD COLUMN api_key TEXT NOT NULL DEFAULT ''"
                    )
                except Exception:
                    pass  # column already exists from CREATE TABLE
                try:
                    await self._db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_accounts_api_key ON accounts(api_key)"
                    )
                except Exception:
                    pass

            # V6 migration: add self_blocks_found to workers, remove FK on blocks.lease_id
            if current_version < 6:
                try:
                    await self._db.execute(
                        "ALTER TABLE workers ADD COLUMN self_blocks_found INTEGER NOT NULL DEFAULT 0"
                    )
                except Exception:
                    pass  # column already exists from CREATE TABLE

                # Recreate blocks table without FK constraint
                try:
                    await self._db.execute(
                        "CREATE TABLE IF NOT EXISTS blocks_new ("
                        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        "  lease_id TEXT NOT NULL DEFAULT '',"
                        "  worker_id TEXT NOT NULL,"
                        "  block_hash TEXT NOT NULL,"
                        "  key TEXT NOT NULL,"
                        "  account TEXT NOT NULL DEFAULT '',"
                        "  attempts INTEGER NOT NULL DEFAULT 0,"
                        "  hashrate TEXT NOT NULL DEFAULT '0.0',"
                        "  prefix_valid INTEGER NOT NULL DEFAULT 1,"
                        "  chain_verified INTEGER NOT NULL DEFAULT 0,"
                        "  chain_block_id INTEGER,"
                        "  created_at REAL NOT NULL"
                        ")"
                    )
                    await self._db.execute(
                        "INSERT INTO blocks_new SELECT * FROM blocks"
                    )
                    await self._db.execute("DROP TABLE blocks")
                    await self._db.execute("ALTER TABLE blocks_new RENAME TO blocks")
                    await self._db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blocks_lease ON blocks(lease_id)"
                    )
                    await self._db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blocks_worker ON blocks(worker_id)"
                    )
                except Exception:
                    logger.exception("V6 blocks table migration failed")

            # V7 migration: add performance indexes for dashboard queries
            if current_version < 7:
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_blocks_created ON blocks(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_blocks_self ON blocks(lease_id) WHERE lease_id = ''",
                ]:
                    try:
                        await self._db.execute(idx_sql)
                    except Exception:
                        pass

            # V8 migration: add hashrate_snapshots table
            if current_version < 8:
                try:
                    await self._db.execute(
                        "CREATE TABLE IF NOT EXISTS hashrate_snapshots ("
                        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        "  worker_id TEXT NOT NULL,"
                        "  hashrate REAL NOT NULL,"
                        "  active_gpus INTEGER NOT NULL DEFAULT 0,"
                        "  timestamp REAL NOT NULL"
                        ")"
                    )
                    await self._db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_snapshots_worker_ts "
                        "ON hashrate_snapshots(worker_id, timestamp)"
                    )
                    await self._db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_snapshots_ts "
                        "ON hashrate_snapshots(timestamp)"
                    )
                except Exception:
                    logger.exception("V8 migration failed")

            # V9 migration: add eth_address index for wallet auth
            if current_version < 9:
                try:
                    await self._db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_accounts_eth_address "
                        "ON accounts(eth_address COLLATE NOCASE)"
                    )
                except Exception:
                    logger.exception("V9 migration failed")

            await self._db.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, time.time()),
            )
            await self._db.commit()
            logger.info("Migration complete (v%d)", SCHEMA_VERSION)
        else:
            logger.debug("Database schema up to date (v%d)", current_version)
