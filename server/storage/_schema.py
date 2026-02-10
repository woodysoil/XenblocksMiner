SCHEMA_VERSION = 11

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
CREATE INDEX IF NOT EXISTS idx_workers_state ON workers(state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_leases_worker_active ON leases(worker_id) WHERE state = 'active';
"""
