"""
test_block_discovery.py - Block Discovery and Reporting Tests

Tests the block_found reporting flow:
  Worker finds block during MINING → sends block_found → platform receives it

Validates:
 - block_found message structure (proto/worker_to_platform.json#block_found)
 - Block only sent when in MINING state with active lease
 - Key prefix matches the lease prefix
 - Hashrate is formatted as string (2 decimal places)
 - Platform correctly receives and indexes blocks per worker/lease
 - Multiple blocks in a single lease
"""

import re
import uuid
import pytest

from .conftest import (
    WorkerSimulator, PlatformSimulator,
    make_block_found_msg,
    TOPIC_BLOCK, PLATFORM_PREFIX_LENGTH, HASH_LENGTH,
)


class TestBlockFoundMessageStructure:
    """Validate block_found messages match proto schema."""

    def test_block_found_has_all_required_fields(self):
        msg = make_block_found_msg(
            worker_id="w1",
            lease_id="lease-123",
            hash_val="0000abcd" * 8,
            key="a1b2c3d4e5f67890" + "0" * 48,
            account="0x" + "ab" * 20,
            attempts=100000,
            hashrate="1250.75",
            prefix="a1b2c3d4e5f67890",
        )
        required = [
            "worker_id", "lease_id", "hash", "key",
            "account", "attempts", "hashrate", "timestamp",
        ]
        for field in required:
            assert field in msg, f"Missing required field: {field}"

    def test_hashrate_is_string(self):
        """In block_found, hashrate is a string (ostringstream with 2 decimals)."""
        msg = make_block_found_msg("w1", "lease-1")
        assert isinstance(msg["hashrate"], str)

    def test_hashrate_two_decimal_format(self):
        msg = make_block_found_msg("w1", "lease-1", hashrate="1250.75")
        assert re.match(r"^\d+\.\d{2}$", msg["hashrate"])

    def test_attempts_is_integer(self):
        msg = make_block_found_msg("w1", "lease-1", attempts=999999)
        assert isinstance(msg["attempts"], int)
        assert msg["attempts"] == 999999

    def test_timestamp_is_integer(self):
        msg = make_block_found_msg("w1", "lease-1")
        assert isinstance(msg["timestamp"], int)

    def test_key_length_is_hash_length(self):
        prefix = "a1b2c3d4e5f67890"
        msg = make_block_found_msg("w1", "lease-1", prefix=prefix)
        assert len(msg["key"]) == HASH_LENGTH


class TestBlockReportingInMiningState:
    """Test that block_found messages are only sent when the worker is in MINING state."""

    def test_block_reported_during_mining(self, broker, platform, worker, worker_id):
        """Worker in MINING state can send block_found."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="deadbeefcafe1234")
        assert worker.state == "MINING"

        worker.send_block_found(
            hash_val="0000aabb" * 8,
            key="deadbeefcafe1234" + "1234abcd" * 6,
            account="0x" + "ee" * 20,
            attempts=500000,
            hashrate=1250.75,
        )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 1
        assert blocks[0]["lease_id"] == task["lease_id"]

    def test_block_not_reported_when_available(self, broker, platform, worker, worker_id):
        """Worker in AVAILABLE state should NOT send blocks."""
        platform.send_register_ack(worker_id, accepted=True)
        assert worker.state == "AVAILABLE"

        worker.send_block_found(
            hash_val="0000", key="a" * 64,
            account="0x" + "ab" * 20, attempts=100, hashrate=100.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 0

    def test_block_not_reported_when_idle(self, broker, platform, worker_id):
        """Worker in IDLE state should NOT send blocks."""
        w = WorkerSimulator(broker, worker_id)
        # Don't start - stays IDLE
        w.send_block_found(
            hash_val="0000", key="a" * 64,
            account="0x" + "ab" * 20, attempts=100, hashrate=100.0,
        )
        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 0


class TestBlockKeyPrefix:
    """Verify that reported block keys use the correct prefix."""

    def test_block_key_starts_with_lease_prefix(self, broker, platform, worker, worker_id):
        prefix = "a1b2c3d4e5f67890"
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix=prefix)

        key = prefix + "abcdef" * 8  # fill to 64 chars
        key = key[:HASH_LENGTH]
        worker.send_block_found(
            hash_val="0000" * 16,
            key=key,
            account="0x" + "ee" * 20,
            attempts=100000,
            hashrate=1000.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 1
        assert blocks[0]["key"].startswith(prefix)

    def test_verify_prefix_16_chars(self, broker, platform, worker, worker_id):
        """Platform can verify the first 16 chars of the key match the assigned prefix."""
        prefix = "deadbeefcafe1234"
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix=prefix)

        key = prefix + "0" * (HASH_LENGTH - PLATFORM_PREFIX_LENGTH)
        worker.send_block_found(
            hash_val="0000" * 16, key=key,
            account="0x" + "ee" * 20, attempts=50000, hashrate=800.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 1
        reported_key = blocks[0]["key"]
        assert reported_key[:PLATFORM_PREFIX_LENGTH] == prefix
        assert len(reported_key[PLATFORM_PREFIX_LENGTH:]) == HASH_LENGTH - PLATFORM_PREFIX_LENGTH


class TestMultipleBlocks:
    """Test multiple block discoveries within a single lease session."""

    def test_multiple_blocks_same_lease(self, broker, platform, worker, worker_id):
        prefix = "abcdef0123456789"
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix=prefix)

        for i in range(5):
            key = prefix + f"{i:048x}"
            worker.send_block_found(
                hash_val=f"0000{i:060x}",
                key=key,
                account="0x" + "ee" * 20,
                attempts=100000 * (i + 1),
                hashrate=1000.0 + i * 10,
            )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 5
        for block in blocks:
            assert block["lease_id"] == task["lease_id"]
            assert block["key"].startswith(prefix)

    def test_blocks_have_increasing_attempts(self, broker, platform, worker, worker_id):
        """Each block's attempts value should be independently meaningful."""
        prefix = "1111222233334444"
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(worker_id, prefix=prefix)

        attempt_values = [50000, 120000, 200000]
        for attempts in attempt_values:
            key = prefix + uuid.uuid4().hex[:48]
            worker.send_block_found(
                hash_val="0000" * 16, key=key,
                account="0x" + "ee" * 20,
                attempts=attempts, hashrate=1000.0,
            )

        blocks = platform.get_blocks(worker_id)
        reported_attempts = [b["attempts"] for b in blocks]
        assert reported_attempts == attempt_values

    def test_blocks_across_leases_tracked_separately(self, broker, platform, worker, worker_id):
        """Blocks from different leases should have different lease_ids."""
        platform.send_register_ack(worker_id, accepted=True)

        # Lease 1
        task1 = platform.send_assign_task(worker_id, prefix="aaaa" * 4)
        worker.send_block_found(
            hash_val="0000" * 16,
            key="aaaa" * 4 + "0" * 48,
            account="0x" + "ee" * 20, attempts=100, hashrate=100.0,
        )
        platform.send_release(worker_id, task1["lease_id"])

        # Lease 2
        task2 = platform.send_assign_task(worker_id, prefix="bbbb" * 4)
        worker.send_block_found(
            hash_val="0000" * 16,
            key="bbbb" * 4 + "0" * 48,
            account="0x" + "ee" * 20, attempts=200, hashrate=200.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 2
        assert blocks[0]["lease_id"] == task1["lease_id"]
        assert blocks[1]["lease_id"] == task2["lease_id"]


class TestBlockFoundAccount:
    """Verify the account field in block_found matches the consumer's address."""

    def test_account_is_consumer_address(self, broker, platform, worker, worker_id):
        """Account in block_found should be the consumer's address."""
        consumer_addr = "0xaabbccddee1234567890abcdef1234567890abcd"
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(
            worker_id, consumer_address=consumer_addr, prefix="1234567890abcdef",
        )

        worker.send_block_found(
            hash_val="0000" * 16,
            key="1234567890abcdef" + "0" * 48,
            account=consumer_addr,
            attempts=100000, hashrate=1000.0,
        )

        blocks = platform.get_blocks(worker_id)
        assert len(blocks) == 1
        assert blocks[0]["account"] == consumer_addr
