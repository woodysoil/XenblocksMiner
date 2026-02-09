"""
test_multi_worker.py - Multiple Worker Tests

Tests scenarios with 2+ workers connected simultaneously:
 - Multiple workers register independently
 - Platform assigns lease to specific worker; others stay AVAILABLE
 - Workers have isolated topic namespaces (xenminer/{worker_id}/...)
 - One worker's release doesn't affect another
 - Blocks from different workers tracked separately
 - Heartbeats from multiple workers received independently
"""

import uuid
import pytest

from .conftest import (
    WorkerSimulator, PlatformSimulator,
    TOPIC_REGISTER, TOPIC_HEARTBEAT, TOPIC_STATUS, TOPIC_BLOCK,
    PLATFORM_PREFIX_LENGTH, HASH_LENGTH,
)


class TestMultipleWorkerRegistration:
    """Test independent registration of multiple workers."""

    def test_two_workers_register(self, broker, platform, worker_factory):
        w1 = worker_factory("worker-A")
        w2 = worker_factory("worker-B")

        assert "worker-A" in platform.registered_workers
        assert "worker-B" in platform.registered_workers

    def test_three_workers_register(self, broker, platform, worker_factory):
        workers = [worker_factory(f"worker-{i}") for i in range(3)]

        assert len(platform.registered_workers) == 3
        for i in range(3):
            assert f"worker-{i}" in platform.registered_workers

    def test_workers_have_different_ids(self, broker, platform, worker_factory):
        w1 = worker_factory("worker-X", eth_address="0x" + "11" * 20)
        w2 = worker_factory("worker-Y", eth_address="0x" + "22" * 20)

        reg_x = platform.registered_workers["worker-X"]
        reg_y = platform.registered_workers["worker-Y"]

        assert reg_x["worker_id"] != reg_y["worker_id"]
        assert reg_x["eth_address"] != reg_y["eth_address"]


class TestLeaseIsolation:
    """Test that leases are isolated between workers."""

    def test_assign_to_one_worker_only(self, broker, platform, worker_factory):
        """Assigning task to worker-A doesn't affect worker-B."""
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        task = platform.send_assign_task("worker-A", prefix="aaaa" * 4)

        assert w_a.state == "MINING"
        assert w_a.lease_id == task["lease_id"]
        assert w_b.state == "AVAILABLE"
        assert w_b.lease_id is None

    def test_release_one_worker_doesnt_affect_other(self, broker, platform, worker_factory):
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        task_a = platform.send_assign_task("worker-A", prefix="aaaa" * 4)
        task_b = platform.send_assign_task("worker-B", prefix="bbbb" * 4)

        assert w_a.state == "MINING"
        assert w_b.state == "MINING"

        # Release worker-A only
        platform.send_release("worker-A", task_a["lease_id"])

        assert w_a.state == "AVAILABLE"
        assert w_b.state == "MINING"
        assert w_b.lease_id == task_b["lease_id"]

    def test_both_workers_mining_different_consumers(self, broker, platform, worker_factory):
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        task_a = platform.send_assign_task(
            "worker-A",
            consumer_address="0x" + "aa" * 20,
            prefix="aaaa" * 4,
        )
        task_b = platform.send_assign_task(
            "worker-B",
            consumer_address="0x" + "bb" * 20,
            prefix="bbbb" * 4,
        )

        assert w_a.state == "MINING"
        assert w_b.state == "MINING"
        assert task_a["lease_id"] != task_b["lease_id"]
        assert task_a["consumer_address"] != task_b["consumer_address"]


class TestMultiWorkerBlocks:
    """Test block reporting from multiple workers."""

    def test_blocks_attributed_to_correct_worker(self, broker, platform, worker_factory):
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        platform.send_assign_task("worker-A", prefix="aaaa" * 4)
        platform.send_assign_task("worker-B", prefix="bbbb" * 4)

        # Worker A finds a block
        w_a.send_block_found(
            hash_val="0000" * 16,
            key="aaaa" * 4 + "0" * 48,
            account="0x" + "ee" * 20, attempts=100000, hashrate=1000.0,
        )

        # Worker B finds a block
        w_b.send_block_found(
            hash_val="0000" * 16,
            key="bbbb" * 4 + "0" * 48,
            account="0x" + "ee" * 20, attempts=200000, hashrate=1500.0,
        )

        blocks_a = platform.get_blocks("worker-A")
        blocks_b = platform.get_blocks("worker-B")

        assert len(blocks_a) == 1
        assert len(blocks_b) == 1
        assert blocks_a[0]["worker_id"] == "worker-A"
        assert blocks_b[0]["worker_id"] == "worker-B"

    def test_blocks_have_correct_prefixes(self, broker, platform, worker_factory):
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        platform.send_assign_task("worker-A", prefix="1111222233334444")
        platform.send_assign_task("worker-B", prefix="5555666677778888")

        w_a.send_block_found(
            hash_val="0000" * 16,
            key="1111222233334444" + "0" * 48,
            account="0x" + "ee" * 20, attempts=100000, hashrate=1000.0,
        )
        w_b.send_block_found(
            hash_val="0000" * 16,
            key="5555666677778888" + "0" * 48,
            account="0x" + "ee" * 20, attempts=100000, hashrate=1000.0,
        )

        blocks_a = platform.get_blocks("worker-A")
        blocks_b = platform.get_blocks("worker-B")

        assert blocks_a[0]["key"].startswith("1111222233334444")
        assert blocks_b[0]["key"].startswith("5555666677778888")


class TestMultiWorkerHeartbeats:
    """Test heartbeat isolation between workers."""

    def test_heartbeats_tracked_per_worker(self, broker, platform, worker_factory):
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        w_a.send_heartbeat(hashrate=1000.0, active_gpus=2)
        w_b.send_heartbeat(hashrate=2000.0, active_gpus=4)

        hb_a = platform.get_heartbeats("worker-A")
        hb_b = platform.get_heartbeats("worker-B")

        assert len(hb_a) == 1
        assert len(hb_b) == 1
        assert hb_a[0]["hashrate"] == 1000.0
        assert hb_b[0]["hashrate"] == 2000.0
        assert hb_a[0]["active_gpus"] == 2
        assert hb_b[0]["active_gpus"] == 4

    def test_multiple_heartbeats_per_worker(self, broker, platform, worker_factory):
        w = worker_factory("worker-A")

        for i in range(5):
            w.send_heartbeat(hashrate=1000.0 + i * 100)

        hb = platform.get_heartbeats("worker-A")
        assert len(hb) == 5
        # Hashrates should increase
        rates = [h["hashrate"] for h in hb]
        assert rates == sorted(rates)


class TestMultiWorkerControlIsolation:
    """Test that control commands target only the intended worker."""

    def test_pause_one_worker_only(self, broker, platform, worker_factory):
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        platform.send_control("worker-A", "pause")

        assert w_a.state == "IDLE"
        assert w_b.state == "AVAILABLE"

    def test_shutdown_one_worker_only(self, broker, platform, worker_factory):
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        platform.send_control("worker-A", "shutdown")

        assert not w_a.client.is_connected
        assert w_b.client.is_connected
        assert w_b.state == "AVAILABLE"

    def test_mixed_states_across_workers(self, broker, platform, worker_factory):
        """Workers can be in different states simultaneously."""
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")
        w_c = worker_factory("worker-C")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)
        platform.send_register_ack("worker-C", accepted=True)

        platform.send_assign_task("worker-A", prefix="aaaa" * 4)
        platform.send_control("worker-C", "pause")

        assert w_a.state == "MINING"
        assert w_b.state == "AVAILABLE"
        assert w_c.state == "IDLE"


class TestTopicIsolation:
    """Verify topic namespacing prevents cross-worker interference."""

    def test_worker_only_receives_own_topics(self, broker, platform, worker_factory):
        """Worker-A should not receive messages for worker-B."""
        w_a = worker_factory("worker-A")
        w_b = worker_factory("worker-B")

        platform.send_register_ack("worker-A", accepted=True)
        platform.send_register_ack("worker-B", accepted=True)

        # Send task to worker-B
        platform.send_assign_task("worker-B", prefix="bbbb" * 4)

        # Worker-A should NOT have received the assign_task
        a_tasks = [c for c in w_a.received_commands if c.get("command") == "assign_task"]
        assert len(a_tasks) == 0

        # Worker-B should have received it
        b_tasks = [c for c in w_b.received_commands if c.get("command") == "assign_task"]
        assert len(b_tasks) == 1
