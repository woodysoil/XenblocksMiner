"""
test_settlement.py - Settlement and Lease Completion Tests

Tests settlement-related flows:
 - Lease completes (release) → COMPLETED → AVAILABLE
 - Block counts tracked per lease
 - Consumer address associated with blocks
 - Lease duration tracking
 - Settlement calculation readiness (blocks found, duration, worker_id)

Note: Actual settlement calculation is server-side logic (Task #3).
These tests verify the data flow that feeds into settlement.
"""

import time
import uuid
import pytest

from .conftest import (
    WorkerSimulator, PlatformSimulator,
    make_block_found_msg,
    TOPIC_BLOCK, PLATFORM_PREFIX_LENGTH, HASH_LENGTH,
)


class TestLeaseCompletionData:
    """Verify all data needed for settlement is available after a lease completes."""

    def test_completed_lease_has_block_count(self, broker, platform, worker, worker_id):
        """After lease release, platform knows how many blocks were found."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="aabb" * 4)

        # Report 3 blocks
        for i in range(3):
            key = "aabb" * 4 + f"{i:048x}"
            worker.send_block_found(
                hash_val=f"0000{i:060x}", key=key,
                account="0x" + "ee" * 20,
                attempts=100000 * (i + 1), hashrate=1200.0,
            )

        platform.send_release(worker_id, task["lease_id"])

        blocks = platform.get_blocks(worker_id)
        lease_blocks = [b for b in blocks if b["lease_id"] == task["lease_id"]]
        assert len(lease_blocks) == 3

    def test_completed_lease_blocks_have_correct_lease_id(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        lease_id = f"lease-{uuid.uuid4()}"
        platform.send_assign_task(worker_id, lease_id=lease_id, prefix="ccdd" * 4)

        worker.send_block_found(
            hash_val="0000" * 16,
            key="ccdd" * 4 + "0" * 48,
            account="0x" + "ee" * 20, attempts=50000, hashrate=900.0,
        )

        platform.send_release(worker_id, lease_id)

        blocks = platform.get_blocks(worker_id)
        assert all(b["lease_id"] == lease_id for b in blocks)

    def test_zero_blocks_lease(self, broker, platform, worker, worker_id):
        """Lease that finds no blocks should still complete cleanly."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="1111" * 4)

        # No blocks found
        platform.send_release(worker_id, task["lease_id"])

        assert worker.state == "AVAILABLE"
        blocks = platform.get_blocks(worker_id)
        lease_blocks = [b for b in blocks if b["lease_id"] == task["lease_id"]]
        assert len(lease_blocks) == 0


class TestSettlementDataIntegrity:
    """Verify the integrity of settlement-relevant data."""

    def test_block_worker_id_matches_lease_worker(self, broker, platform, worker, worker_id):
        """Every block's worker_id should match the leased worker."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="aabb" * 4)

        worker.send_block_found(
            hash_val="0000" * 16,
            key="aabb" * 4 + "0" * 48,
            account="0x" + "ee" * 20, attempts=100000, hashrate=1000.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 1
        assert blocks[0]["worker_id"] == worker_id

    def test_block_account_matches_consumer(self, broker, platform, worker, worker_id):
        """Block account should match the consumer_address from assign_task."""
        consumer_addr = "0x" + "ff" * 20
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(
            worker_id, consumer_address=consumer_addr, prefix="1234" * 4,
        )

        worker.send_block_found(
            hash_val="0000" * 16,
            key="1234" * 4 + "0" * 48,
            account=consumer_addr, attempts=50000, hashrate=800.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert blocks[0]["account"] == consumer_addr

    def test_block_key_prefix_matches_assignment(self, broker, platform, worker, worker_id):
        """Block key prefix must match the prefix from assign_task."""
        prefix = "deadbeefcafe1234"
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(worker_id, prefix=prefix)

        key = prefix + "a" * (HASH_LENGTH - PLATFORM_PREFIX_LENGTH)
        worker.send_block_found(
            hash_val="0000" * 16, key=key,
            account="0x" + "ee" * 20, attempts=75000, hashrate=950.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert blocks[0]["key"][:PLATFORM_PREFIX_LENGTH] == prefix

    def test_block_timestamps_are_ordered(self, broker, platform, worker, worker_id):
        """Block timestamps should be monotonically non-decreasing."""
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(worker_id, prefix="abcd" * 4)

        for i in range(3):
            key = "abcd" * 4 + f"{i:048x}"
            worker.send_block_found(
                hash_val=f"0000{i:060x}", key=key,
                account="0x" + "ee" * 20,
                attempts=10000 * (i + 1), hashrate=1000.0,
            )

        blocks = platform.get_blocks(worker_id)
        timestamps = [b["timestamp"] for b in blocks]
        assert timestamps == sorted(timestamps)


class TestMultiLeaseSettlement:
    """Test settlement data across multiple leases."""

    def test_blocks_correctly_attributed_across_leases(self, broker, platform, worker, worker_id):
        """Blocks from different leases have correct lease_ids."""
        platform.send_register_ack(worker_id, accepted=True)

        lease_block_counts = {}

        for i in range(3):
            prefix = f"{i:016x}"
            task = platform.send_assign_task(worker_id, prefix=prefix)
            lid = task["lease_id"]

            num_blocks = i + 1
            for j in range(num_blocks):
                key = prefix + f"{j:048x}"
                worker.send_block_found(
                    hash_val=f"0000{j:060x}", key=key,
                    account="0x" + "ee" * 20,
                    attempts=10000, hashrate=1000.0,
                )

            lease_block_counts[lid] = num_blocks
            platform.send_release(worker_id, lid)

        # Verify counts
        all_blocks = platform.get_blocks(worker_id)
        for lid, expected_count in lease_block_counts.items():
            actual = [b for b in all_blocks if b["lease_id"] == lid]
            assert len(actual) == expected_count

    def test_total_blocks_across_leases(self, broker, platform, worker, worker_id):
        """Total block count should be sum across all leases."""
        platform.send_register_ack(worker_id, accepted=True)

        total = 0
        for i in range(3):
            prefix = f"{i:016x}"
            task = platform.send_assign_task(worker_id, prefix=prefix)
            n = i + 1
            total += n
            for j in range(n):
                key = prefix + f"{j:048x}"
                worker.send_block_found(
                    hash_val=f"0000{j:060x}", key=key,
                    account="0x" + "ee" * 20,
                    attempts=10000, hashrate=1000.0,
                )
            platform.send_release(worker_id, task["lease_id"])

        all_blocks = platform.get_blocks(worker_id)
        assert len(all_blocks) == total  # 1+2+3 = 6


class TestStatusFlowForSettlement:
    """Verify status updates that settlement logic might depend on."""

    def test_completed_status_emitted(self, broker, platform, worker, worker_id):
        """After release, COMPLETED status should be visible."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="abcd" * 4)
        platform.send_release(worker_id, task["lease_id"])

        statuses = platform.get_worker_statuses(worker_id)
        states = [s["state"] for s in statuses]
        assert "COMPLETED" in states

    def test_available_status_after_completion(self, broker, platform, worker, worker_id):
        """Worker returns to AVAILABLE after COMPLETED."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="abcd" * 4)
        platform.send_release(worker_id, task["lease_id"])

        statuses = platform.get_worker_statuses(worker_id)
        states = [s["state"] for s in statuses]
        # After COMPLETED, should see AVAILABLE
        completed_idx = len(states) - 1 - states[::-1].index("COMPLETED")
        remaining = states[completed_idx + 1:]
        assert "AVAILABLE" in remaining
