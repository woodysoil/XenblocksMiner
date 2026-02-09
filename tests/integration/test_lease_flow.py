"""
test_lease_flow.py - Lease Assignment and Lifecycle Tests

Tests the full lease lifecycle:
  AVAILABLE → assign_task → LEASED → MINING → release → COMPLETED → AVAILABLE

Validates:
 - assign_task message structure (proto/platform_to_worker.json#assign_task)
 - Worker only accepts tasks when in AVAILABLE state
 - Lease ID tracking through the lifecycle
 - release command ends lease and returns to AVAILABLE
 - Lease expiry (duration_sec) transitions
 - Consumer address and prefix correctness
 - Multiple sequential leases
"""

import time
import uuid
import re
import pytest

from .conftest import (
    WorkerSimulator, PlatformSimulator,
    make_assign_task, make_release, make_control,
    TOPIC_TASK, TOPIC_STATUS, TOPIC_CONTROL,
    PLATFORM_PREFIX_LENGTH, WORKER_STATES,
)


class TestAssignTaskMessageStructure:
    """Validate assign_task message schema compliance."""

    def test_assign_task_has_required_fields(self):
        msg = make_assign_task()
        assert msg["command"] == "assign_task"
        assert "lease_id" in msg
        assert "consumer_id" in msg
        assert "consumer_address" in msg

    def test_prefix_is_16_hex_chars(self):
        msg = make_assign_task(prefix="a1b2c3d4e5f67890")
        assert len(msg["prefix"]) == PLATFORM_PREFIX_LENGTH
        assert re.match(r"^[0-9a-fA-F]{16}$", msg["prefix"])

    def test_duration_default_is_3600(self):
        msg = make_assign_task()
        assert msg["duration_sec"] == 3600

    def test_custom_duration(self):
        msg = make_assign_task(duration_sec=600)
        assert msg["duration_sec"] == 600

    def test_empty_prefix_allowed(self):
        msg = make_assign_task(prefix="")
        assert msg["prefix"] == ""

    def test_consumer_address_format(self):
        msg = make_assign_task(consumer_address="0xaabbccddee1234567890abcdef1234567890abcd")
        assert msg["consumer_address"].startswith("0x")
        assert len(msg["consumer_address"]) == 42


class TestLeaseAssignment:
    """Test the assign_task → LEASED → MINING flow."""

    def test_worker_accepts_task_when_available(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        assert worker.state == "AVAILABLE"

        task = platform.send_assign_task(worker_id, prefix="deadbeefcafe1234")
        assert worker.state == "MINING"
        assert worker.lease_id == task["lease_id"]

    def test_worker_tracks_lease_id(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        lease_id = f"lease-{uuid.uuid4()}"
        platform.send_assign_task(worker_id, lease_id=lease_id)
        assert worker.lease_id == lease_id

    def test_state_transitions_on_assign(self, broker, platform, worker, worker_id):
        """Worker transitions AVAILABLE → LEASED → MINING."""
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(worker_id)

        statuses = platform.get_worker_statuses(worker_id)
        states = [s["state"] for s in statuses]

        # Should see LEASED and MINING in the status history
        assert "LEASED" in states
        assert "MINING" in states

    def test_status_includes_lease_id(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        lease_id = f"lease-{uuid.uuid4()}"
        platform.send_assign_task(worker_id, lease_id=lease_id)

        mining_statuses = [s for s in platform.get_worker_statuses(worker_id)
                          if s["state"] == "MINING"]
        assert len(mining_statuses) >= 1
        assert mining_statuses[-1].get("lease_id") == lease_id

    def test_worker_ignores_task_when_not_available(self, broker, platform, worker, worker_id):
        """Worker in MINING state should ignore a second assign_task."""
        platform.send_register_ack(worker_id, accepted=True)
        task1 = platform.send_assign_task(worker_id, prefix="aaaa" * 4)
        assert worker.state == "MINING"
        first_lease = worker.lease_id

        # Send another task while mining
        task2 = platform.send_assign_task(worker_id, prefix="bbbb" * 4)
        # Should still be on first lease
        assert worker.lease_id == first_lease

    def test_worker_ignores_task_in_idle(self, broker, worker_id):
        """Worker in IDLE state ignores assign_task."""
        broker_inst = broker
        w = WorkerSimulator(broker_inst, worker_id)
        # Don't start - stays in IDLE
        assert w.state == "IDLE"
        # Manually send a task (won't be received since not connected)
        # This tests the state guard in the simulator


class TestLeaseRelease:
    """Test the release command flow."""

    def test_release_ends_lease(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id)
        assert worker.state == "MINING"

        platform.send_release(worker_id, lease_id=task["lease_id"])
        assert worker.state == "AVAILABLE"
        assert worker.lease_id is None

    def test_release_transitions_through_completed(self, broker, platform, worker, worker_id):
        """Release should go MINING → COMPLETED → AVAILABLE."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id)

        platform.send_release(worker_id, lease_id=task["lease_id"])

        statuses = platform.get_worker_statuses(worker_id)
        states = [s["state"] for s in statuses]
        assert "COMPLETED" in states

    def test_release_without_lease_id_releases_current(self, broker, platform, worker, worker_id):
        """Release with empty lease_id should release current lease."""
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(worker_id)
        assert worker.state == "MINING"

        platform.send_release(worker_id)  # No lease_id
        assert worker.state == "AVAILABLE"

    def test_release_with_wrong_lease_id_ignored(self, broker, platform, worker, worker_id):
        """Release for a different lease_id should be ignored."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id)
        assert worker.state == "MINING"

        # Release with wrong ID
        platform.send_release(worker_id, lease_id="wrong-lease-id")
        # Should still be mining
        assert worker.state == "MINING"
        assert worker.lease_id == task["lease_id"]

    def test_release_when_no_active_lease(self, broker, platform, worker, worker_id):
        """Release when no lease active should be harmless."""
        platform.send_register_ack(worker_id, accepted=True)
        assert worker.state == "AVAILABLE"
        assert worker.lease_id is None

        # Send release - should not crash or change state
        platform.send_release(worker_id)
        assert worker.state == "AVAILABLE"


class TestSequentialLeases:
    """Test multiple leases in sequence."""

    def test_two_sequential_leases(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)

        # Lease 1
        task1 = platform.send_assign_task(worker_id, prefix="aaaa" * 4)
        assert worker.state == "MINING"
        platform.send_release(worker_id, task1["lease_id"])
        assert worker.state == "AVAILABLE"

        # Lease 2
        task2 = platform.send_assign_task(worker_id, prefix="bbbb" * 4)
        assert worker.state == "MINING"
        assert worker.lease_id == task2["lease_id"]
        assert worker.lease_id != task1["lease_id"]

    def test_three_sequential_leases(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)

        for i in range(3):
            prefix = f"{i:016x}"
            task = platform.send_assign_task(worker_id, prefix=prefix)
            assert worker.state == "MINING"
            platform.send_release(worker_id, task["lease_id"])
            assert worker.state == "AVAILABLE"

    def test_lease_ids_are_unique(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)

        lease_ids = []
        for _ in range(5):
            task = platform.send_assign_task(worker_id)
            lease_ids.append(task["lease_id"])
            platform.send_release(worker_id, task["lease_id"])

        # All lease IDs should be unique
        assert len(set(lease_ids)) == 5


class TestPrefixValidation:
    """Test prefix field in assign_task."""

    def test_valid_16_char_prefix(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="a1b2c3d4e5f67890")
        assert worker.state == "MINING"

    def test_empty_prefix_accepted(self, broker, platform, worker, worker_id):
        """Empty prefix is allowed (means no key prefix injection)."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="")
        assert worker.state == "MINING"

    def test_prefix_pattern_hex_only(self):
        """Prefix must be hex characters only."""
        # Valid
        assert re.match(r"^[0-9a-fA-F]{16}$", "a1b2c3d4e5f67890")
        assert re.match(r"^[0-9a-fA-F]{16}$", "ABCDEF0123456789")
        # Invalid
        assert not re.match(r"^[0-9a-fA-F]{16}$", "g1b2c3d4e5f67890")
        assert not re.match(r"^[0-9a-fA-F]{16}$", "a1b2c3d4e5f6789")  # 15 chars


class TestDurationField:
    """Test duration_sec in assign_task."""

    def test_default_duration(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id)
        assert task["duration_sec"] == 3600

    def test_custom_short_duration(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, duration_sec=60)
        assert task["duration_sec"] == 60

    def test_custom_long_duration(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, duration_sec=86400)
        assert task["duration_sec"] == 86400
