import logging
import time
from typing import Dict, Optional

import aiosqlite

logger = logging.getLogger("storage")


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
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")
        now = time.time()
        cursor = await self._db.execute(
            "UPDATE accounts SET balance = balance - ?, updated_at = ? "
            "WHERE account_id = ? AND balance >= ?",
            (amount, now, from_id, amount),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Insufficient balance or account {from_id} not found")
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
