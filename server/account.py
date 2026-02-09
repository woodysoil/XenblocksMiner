"""
account.py - Account service.

Async account management backed by StorageManager's AccountRepo.
Provides the same logical interface for the rest of the server.
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from server.storage import AccountRepo

logger = logging.getLogger("account")


class AccountService:
    """Account service backed by SQLite via AccountRepo."""

    def __init__(self, repo: "AccountRepo"):
        self._repo = repo

    async def setup_defaults(self):
        """Create default test accounts if they don't exist."""
        await self.create_account("consumer-1", "consumer", balance=1000.0,
                                  eth_address="0xaabbccddee1234567890abcdef1234567890abcd")
        await self.create_account("consumer-2", "consumer", balance=500.0,
                                  eth_address="0x1111111111222222222233333333334444444444")

    async def create_account(
        self, account_id: str, role: str,
        balance: float = 0.0, eth_address: str = ""
    ) -> dict:
        acct = await self._repo.create(account_id, role, balance=balance, eth_address=eth_address)
        if acct:
            logger.info("Created/found account %s role=%s balance=%.4f", account_id, role, acct["balance"])
        return acct

    async def get_account(self, account_id: str) -> Optional[dict]:
        return await self._repo.get(account_id)

    async def get_or_create_provider(self, worker_id: str, eth_address: str) -> dict:
        return await self._repo.get_or_create_provider(worker_id, eth_address)

    async def deposit(self, account_id: str, amount: float) -> dict:
        return await self._repo.deposit(account_id, amount)

    async def transfer(self, from_id: str, to_id: str, amount: float):
        await self._repo.transfer(from_id, to_id, amount)

    async def list_accounts(self) -> Dict[str, dict]:
        return await self._repo.list_all()
