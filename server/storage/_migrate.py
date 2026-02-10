import logging
import time

from ._schema import SCHEMA_SQL, SCHEMA_VERSION

logger = logging.getLogger("storage")


async def run_migrations(db, logger_override=None):
    log = logger_override or logger
    current_version = 0
    try:
        async with db.execute("SELECT MAX(version) FROM schema_version") as cursor:
            row = await cursor.fetchone()
            if row and row[0] is not None:
                current_version = row[0]
    except Exception:
        pass

    if current_version < SCHEMA_VERSION:
        log.info("Migrating database from v%d to v%d", current_version, SCHEMA_VERSION)
        await db.executescript(SCHEMA_SQL)

        if current_version < 2:
            try:
                await db.execute(
                    "ALTER TABLE blocks ADD COLUMN chain_verified INTEGER NOT NULL DEFAULT 0"
                )
            except Exception:
                pass
            try:
                await db.execute(
                    "ALTER TABLE blocks ADD COLUMN chain_block_id INTEGER"
                )
            except Exception:
                pass

        if current_version < 3:
            for col, typedef in [
                ("price_per_min", "REAL NOT NULL DEFAULT 0.60"),
                ("min_duration_sec", "INTEGER NOT NULL DEFAULT 60"),
                ("max_duration_sec", "INTEGER NOT NULL DEFAULT 86400"),
            ]:
                try:
                    await db.execute(f"ALTER TABLE workers ADD COLUMN {col} {typedef}")
                except Exception:
                    pass

        if current_version < 4:
            for col, typedef in [
                ("total_online_sec", "REAL NOT NULL DEFAULT 0.0"),
                ("last_online_at", "REAL"),
            ]:
                try:
                    await db.execute(f"ALTER TABLE workers ADD COLUMN {col} {typedef}")
                except Exception:
                    pass

        if current_version < 5:
            try:
                await db.execute(
                    "ALTER TABLE accounts ADD COLUMN api_key TEXT NOT NULL DEFAULT ''"
                )
            except Exception:
                pass
            try:
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_accounts_api_key ON accounts(api_key)"
                )
            except Exception:
                pass

        if current_version < 6:
            try:
                await db.execute(
                    "ALTER TABLE workers ADD COLUMN self_blocks_found INTEGER NOT NULL DEFAULT 0"
                )
            except Exception:
                pass
            try:
                await db.execute(
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
                await db.execute("INSERT INTO blocks_new SELECT * FROM blocks")
                await db.execute("DROP TABLE blocks")
                await db.execute("ALTER TABLE blocks_new RENAME TO blocks")
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_blocks_lease ON blocks(lease_id)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_blocks_worker ON blocks(worker_id)"
                )
            except Exception:
                log.exception("V6 blocks table migration failed")

        if current_version < 7:
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_blocks_created ON blocks(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_blocks_self ON blocks(lease_id) WHERE lease_id = ''",
            ]:
                try:
                    await db.execute(idx_sql)
                except Exception:
                    pass

        if current_version < 8:
            try:
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS hashrate_snapshots ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  worker_id TEXT NOT NULL,"
                    "  hashrate REAL NOT NULL,"
                    "  active_gpus INTEGER NOT NULL DEFAULT 0,"
                    "  timestamp REAL NOT NULL"
                    ")"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_snapshots_worker_ts "
                    "ON hashrate_snapshots(worker_id, timestamp)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_snapshots_ts "
                    "ON hashrate_snapshots(timestamp)"
                )
            except Exception:
                log.exception("V8 migration failed")

        if current_version < 9:
            try:
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_accounts_eth_address "
                    "ON accounts(eth_address COLLATE NOCASE)"
                )
            except Exception:
                log.exception("V9 migration failed")

        if current_version < 11:
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_workers_state ON workers(state)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_leases_worker_active "
                "ON leases(worker_id) WHERE state = 'active'",
            ]:
                try:
                    await db.execute(idx_sql)
                except Exception:
                    log.exception("V11 migration index failed: %s", idx_sql)

        await db.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, time.time()),
        )
        await db.commit()
        log.info("Migration complete (v%d)", SCHEMA_VERSION)
    else:
        log.debug("Database schema up to date (v%d)", current_version)
