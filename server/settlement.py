"""
settlement.py - Settlement engine.

When a lease ends, calculates payment: provider gets 95%, platform gets 5%.
Uses a simple per-second pricing model. Backed by SettlementRepo and AccountRepo.
"""

import logging
import time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from server.account import AccountService
    from server.storage import SettlementRepo

logger = logging.getLogger("settlement")

PROVIDER_SHARE = 0.95
PLATFORM_SHARE = 0.05
PLATFORM_ACCOUNT_ID = "platform-treasury"

# Simple pricing: per-second rate
RATE_PER_SECOND = 0.01


class SettlementEngine:
    """Settles completed leases."""

    def __init__(self, accounts: "AccountService", settlement_repo: "SettlementRepo"):
        self.accounts = accounts
        self._settlements = settlement_repo

    async def setup_defaults(self):
        """Ensure platform treasury account exists."""
        await self.accounts.create_account(PLATFORM_ACCOUNT_ID, "consumer", balance=0.0)

    async def settle_lease(self, lease: dict) -> Optional[dict]:
        """Calculate and execute settlement in a single IMMEDIATE transaction.

        All balance changes and transaction records commit atomically —
        no partial state on crash or concurrent access.
        """
        duration = lease["elapsed_sec"]
        rate = lease.get("price_per_sec", RATE_PER_SECOND)
        total_cost = duration * rate
        provider_payment = total_cost * PROVIDER_SHARE
        platform_fee = total_cost * PLATFORM_SHARE
        db = self.accounts._repo._db

        try:
            await db.execute("BEGIN IMMEDIATE")

            # Read consumer balance inside transaction
            async with db.execute(
                "SELECT balance FROM accounts WHERE account_id = ?",
                (lease["consumer_id"],),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                await db.execute("ROLLBACK")
                logger.error("Consumer account %s not found", lease["consumer_id"])
                return None

            if row[0] < total_cost:
                logger.warning(
                    "Consumer %s insufficient balance (%.4f < %.4f) for lease %s",
                    lease["consumer_id"], row[0], total_cost, lease["lease_id"],
                )
                total_cost = max(row[0], 0.0)
                provider_payment = total_cost * PROVIDER_SHARE
                platform_fee = total_cost * PLATFORM_SHARE

            now = time.time()
            await db.execute(
                "UPDATE accounts SET balance = balance - ?, updated_at = ? WHERE account_id = ?",
                (total_cost, now, lease["consumer_id"]),
            )
            await db.execute(
                "UPDATE accounts SET balance = balance + ?, updated_at = ? WHERE account_id = ?",
                (provider_payment, now, lease["worker_id"]),
            )
            await db.execute(
                "UPDATE accounts SET balance = balance + ?, updated_at = ? WHERE account_id = ?",
                (platform_fee, now, PLATFORM_ACCOUNT_ID),
            )
            for acct_id, tx_type, amount in [
                (lease["consumer_id"], "lease_charge", total_cost),
                (lease["worker_id"], "provider_payout", provider_payment),
                (PLATFORM_ACCOUNT_ID, "platform_fee", platform_fee),
            ]:
                await db.execute(
                    "INSERT INTO transactions (account_id, type, amount, reference_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (acct_id, tx_type, amount, lease["lease_id"], now),
                )
            await db.commit()
        except Exception:
            try:
                await db.execute("ROLLBACK")
            except Exception:
                pass
            logger.exception("Settlement failed for lease %s", lease["lease_id"])
            return None

        record = await self._settlements.create(
            lease_id=lease["lease_id"],
            consumer_id=lease["consumer_id"],
            worker_id=lease["worker_id"],
            duration_sec=duration,
            blocks_found=lease["blocks_found"],
            total_cost=total_cost,
            provider_payout=provider_payment,
            platform_fee=platform_fee,
        )
        logger.info(
            "Settled lease %s: cost=%.4f provider=%.4f platform=%.4f blocks=%d",
            lease["lease_id"], total_cost, provider_payment, platform_fee, lease["blocks_found"],
        )
        return record

    async def get_settlement(self, lease_id: str) -> Optional[dict]:
        """Retrieve the settlement record for a lease, if any."""
        return await self._settlements.get(lease_id)

    async def list_settlements(self, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        """Return all settlement records."""
        return await self._settlements.list_all(limit=limit, offset=offset)
