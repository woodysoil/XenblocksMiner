"""
watcher.py - Block watcher.

Subscribes to worker block_found messages, verifies prefix matches the lease,
verifies block exists on the chain simulator, and records blocks via BlockRepo
and LeaseRepo.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from server.chain_simulator import ChainSimulator
    from server.storage import BlockRepo, LeaseRepo

logger = logging.getLogger("watcher")

PLATFORM_PREFIX_LENGTH = 16


class BlockWatcher:
    """Watches for block_found messages and verifies them."""

    def __init__(
        self,
        block_repo: "BlockRepo",
        lease_repo: "LeaseRepo",
        chain: Optional["ChainSimulator"] = None,
    ):
        self._blocks = block_repo
        self._leases = lease_repo
        self._chain = chain

    async def handle_block_found(self, msg: dict, worker_id_from_topic: str):
        """Process a block_found message from a worker."""
        worker_id = msg.get("worker_id", worker_id_from_topic)
        lease_id = msg.get("lease_id", "")
        key = msg.get("key", "")
        block_hash = msg.get("hash", "")
        account = msg.get("account", "")
        attempts = msg.get("attempts", 0)
        hashrate = msg.get("hashrate", "0.0")
        timestamp = msg.get("timestamp", 0)

        # Verify the lease exists
        lease = await self._leases.get(lease_id)
        if lease is None:
            logger.warning("Block from %s references unknown lease %s", worker_id, lease_id)
            return

        # Verify key prefix matches the assigned prefix
        prefix_valid = True
        if lease["prefix"]:
            actual_prefix = key[:PLATFORM_PREFIX_LENGTH].lower()
            expected_prefix = lease["prefix"].lower()
            prefix_valid = actual_prefix == expected_prefix
            if not prefix_valid:
                logger.warning(
                    "Prefix mismatch for lease %s: expected=%s got=%s",
                    lease_id, expected_prefix, actual_prefix,
                )

        # Verify block exists on-chain via the chain simulator
        chain_verified = False
        chain_block_id = None
        if self._chain is not None:
            chain_blocks = self._chain.get_blocks_by_key_prefix(key[:PLATFORM_PREFIX_LENGTH])
            for cb in chain_blocks:
                if cb["key"] == key:
                    chain_verified = True
                    chain_block_id = cb["block_id"]
                    break
            if not chain_verified:
                logger.warning(
                    "Block NOT verified on-chain: lease=%s worker=%s key=%s..%s",
                    lease_id, worker_id, key[:8], key[-4:],
                )
        else:
            # No chain simulator available, skip verification
            chain_verified = True

        # Record the block in SQLite
        await self._blocks.create(
            lease_id=lease_id,
            worker_id=worker_id,
            block_hash=block_hash,
            key=key,
            account=account,
            attempts=attempts,
            hashrate=hashrate,
            prefix_valid=prefix_valid,
            chain_verified=chain_verified,
            chain_block_id=chain_block_id,
        )

        # Only increment lease block counter for verified blocks
        if chain_verified and prefix_valid:
            await self._leases.increment_blocks(lease_id)

        # Re-fetch for logging
        updated_lease = await self._leases.get(lease_id)
        blocks_count = updated_lease["blocks_found"] if updated_lease else 0
        logger.info(
            "Block recorded: lease=%s worker=%s hash=%s..%s prefix_valid=%s chain_verified=%s (verified_total=%d)",
            lease_id, worker_id, block_hash[:8], block_hash[-4:],
            prefix_valid, chain_verified, blocks_count,
        )

    async def get_blocks_for_lease(self, lease_id: str) -> List[dict]:
        """Return all recorded blocks for a given lease."""
        return await self._blocks.get_for_lease(lease_id)

    async def get_all_blocks(self) -> List[dict]:
        """Return all recorded blocks."""
        return await self._blocks.get_all()

    async def total_blocks(self) -> int:
        """Return the total number of recorded blocks."""
        return await self._blocks.count()
