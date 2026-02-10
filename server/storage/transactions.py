import time
from typing import List

import aiosqlite


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
