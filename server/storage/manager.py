import logging
from typing import Optional

try:
    import aiosqlite
except ImportError:
    raise ImportError(
        "aiosqlite is required for the storage layer. "
        "Install with: pip install aiosqlite"
    )

from ._schema import SCHEMA_VERSION
from ._migrate import run_migrations
from .accounts import AccountRepo
from .workers import WorkerRepo
from .leases import LeaseRepo
from .blocks import BlockRepo
from .settlement_repo import SettlementRepo
from .transactions import TransactionRepo
from .snapshots import SnapshotRepo
from .wallet_snapshots import WalletSnapshotRepo

logger = logging.getLogger("storage")


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
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await run_migrations(self._db, logger)

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
