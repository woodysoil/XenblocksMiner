"""
settlement.py - Settlement engine.

When a lease ends, calculates payment: provider gets 95%, platform gets 5%.
Uses a simple per-second pricing model. Backed by SettlementRepo and AccountRepo.
"""

import logging
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
        """Calculate and execute settlement for a completed lease."""
        duration = lease["elapsed_sec"]
        # Use lease-specific pricing if available, fallback to global rate
        rate = lease.get("price_per_sec", RATE_PER_SECOND)
        total_cost = duration * rate

        provider_payment = total_cost * PROVIDER_SHARE
        platform_fee = total_cost * PLATFORM_SHARE

        # Debit consumer, credit provider + platform
        consumer_acct = await self.accounts.get_account(lease["consumer_id"])
        if consumer_acct is None:
            logger.error("Consumer account %s not found for settlement", lease["consumer_id"])
            return None

        if consumer_acct["balance"] < total_cost:
            logger.warning(
                "Consumer %s insufficient balance (%.4f < %.4f) for lease %s",
                lease["consumer_id"], consumer_acct["balance"], total_cost, lease["lease_id"],
            )
            total_cost = consumer_acct["balance"]
            provider_payment = total_cost * PROVIDER_SHARE
            platform_fee = total_cost * PLATFORM_SHARE

        try:
            # Debit consumer
            consumer_acct = await self.accounts.get_account(lease["consumer_id"])
            new_consumer_balance = consumer_acct["balance"] - total_cost
            await self.accounts._repo.update_balance(lease["consumer_id"], new_consumer_balance)

            # Credit provider
            provider_acct = await self.accounts.get_or_create_provider(lease["worker_id"], "")
            new_provider_balance = provider_acct["balance"] + provider_payment
            await self.accounts._repo.update_balance(lease["worker_id"], new_provider_balance)

            # Credit platform
            platform_acct = await self.accounts.get_account(PLATFORM_ACCOUNT_ID)
            if platform_acct:
                new_platform_balance = platform_acct["balance"] + platform_fee
                await self.accounts._repo.update_balance(PLATFORM_ACCOUNT_ID, new_platform_balance)

            # Record transactions
            await self.accounts._repo._record_tx(lease["consumer_id"], "lease_charge", total_cost, lease["lease_id"])
            await self.accounts._repo._record_tx(lease["worker_id"], "provider_payout", provider_payment, lease["lease_id"])
            await self.accounts._repo._record_tx(PLATFORM_ACCOUNT_ID, "platform_fee", platform_fee, lease["lease_id"])
        except Exception:
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

    async def list_settlements(self) -> List[dict]:
        """Return all settlement records."""
        return await self._settlements.list_all()
