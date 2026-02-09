"""
test_worker_registration.py - Worker Registration Flow Tests

Tests the registration handshake:
  Worker sends 'register' → Platform sends 'register_ack'

Validates:
 - Registration message structure matches proto/worker_to_platform.json#register
 - Platform receives all required fields
 - register_ack accepted flow transitions worker to AVAILABLE
 - register_ack rejected flow transitions worker to ERROR
 - GPU info array correctness
 - Re-registration after error recovery
"""

import time
import pytest

from .conftest import (
    WorkerSimulator, PlatformSimulator,
    make_register_msg, make_register_ack,
    TOPIC_REGISTER, TOPIC_STATUS, TOPIC_TASK,
    PLATFORM_PREFIX_LENGTH, WORKER_STATES,
)


class TestRegistrationMessageStructure:
    """Validate that registration messages match the proto schema."""

    def test_register_has_all_required_fields(self, broker, platform, worker, worker_id):
        """Worker registration must include all fields from proto schema."""
        assert worker_id in platform.registered_workers
        reg = platform.registered_workers[worker_id]

        required_fields = [
            "worker_id", "eth_address", "gpu_count",
            "total_memory_gb", "gpus", "version", "timestamp",
        ]
        for field in required_fields:
            assert field in reg, f"Missing required field: {field}"

    def test_worker_id_matches(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        assert reg["worker_id"] == worker_id

    def test_gpu_count_is_integer(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        assert isinstance(reg["gpu_count"], int)
        assert reg["gpu_count"] > 0

    def test_total_memory_is_integer(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        assert isinstance(reg["total_memory_gb"], int)
        assert reg["total_memory_gb"] > 0

    def test_version_is_string(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        assert isinstance(reg["version"], str)
        assert reg["version"] == "2.0.0"

    def test_timestamp_is_recent(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        assert isinstance(reg["timestamp"], int)
        now = int(time.time())
        assert abs(reg["timestamp"] - now) < 5

    def test_eth_address_format(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        addr = reg["eth_address"]
        assert addr.startswith("0x")
        assert len(addr) == 42  # 0x + 40 hex chars


class TestGpuInfoArray:
    """Validate the gpus array in registration messages."""

    def test_gpus_array_matches_count(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        assert len(reg["gpus"]) == reg["gpu_count"]

    def test_gpu_has_required_fields(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        for gpu in reg["gpus"]:
            assert "index" in gpu
            assert "name" in gpu
            assert "memory_gb" in gpu
            assert "bus_id" in gpu

    def test_gpu_index_is_integer(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        for gpu in reg["gpus"]:
            assert isinstance(gpu["index"], int)

    def test_gpu_memory_is_integer(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        for gpu in reg["gpus"]:
            assert isinstance(gpu["memory_gb"], int)
            assert gpu["memory_gb"] > 0

    def test_gpu_name_is_nonempty_string(self, broker, platform, worker, worker_id):
        reg = platform.registered_workers[worker_id]
        for gpu in reg["gpus"]:
            assert isinstance(gpu["name"], str)
            assert len(gpu["name"]) > 0

    def test_single_gpu_worker(self, broker, platform):
        """Worker with 1 GPU registers correctly."""
        w = WorkerSimulator(broker, "single-gpu-worker", gpu_count=1)
        w.start()
        assert "single-gpu-worker" in platform.registered_workers
        reg = platform.registered_workers["single-gpu-worker"]
        assert reg["gpu_count"] == 1
        assert len(reg["gpus"]) == 1


class TestRegisterAckAccepted:
    """Test the happy-path: platform accepts registration."""

    def test_worker_receives_ack(self, broker, platform, worker, worker_id):
        """Platform sends register_ack(accepted=true), worker transitions to AVAILABLE."""
        platform.send_register_ack(worker_id, accepted=True)

        # Worker should have received the ack
        acks = [c for c in worker.received_commands if c.get("command") == "register_ack"]
        assert len(acks) >= 1
        assert acks[-1]["accepted"] is True

    def test_worker_state_available_after_ack(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        assert worker.state == "AVAILABLE"

    def test_ack_message_structure(self, broker, worker_id):
        """register_ack message has correct structure."""
        msg = make_register_ack(accepted=True)
        assert msg["command"] == "register_ack"
        assert msg["accepted"] is True
        assert "reason" not in msg  # No reason for accepted acks


class TestRegisterAckRejected:
    """Test the rejection path: platform rejects registration."""

    def test_rejected_ack_has_reason(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=False, reason="version_unsupported")

        acks = [c for c in worker.received_commands if c.get("command") == "register_ack"]
        assert len(acks) >= 1
        assert acks[-1]["accepted"] is False
        assert acks[-1]["reason"] == "version_unsupported"

    def test_worker_transitions_to_error_on_rejection(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=False, reason="banned")
        assert worker.state == "ERROR"

    def test_rejected_ack_status_reported(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=False, reason="banned")

        statuses = platform.get_worker_statuses(worker_id)
        error_statuses = [s for s in statuses if s["state"] == "ERROR"]
        assert len(error_statuses) >= 1

    def test_rejected_ack_default_reason(self):
        """When no reason provided, default to 'unknown'."""
        msg = make_register_ack(accepted=False)
        assert msg["reason"] == "unknown"


class TestReRegistration:
    """Test re-registration scenarios."""

    def test_re_register_after_disconnect(self, broker, platform, worker_factory):
        """Worker disconnects and reconnects, sending a new registration."""
        w = worker_factory("rereg-worker")
        assert "rereg-worker" in platform.registered_workers

        # Stop (disconnect)
        w.stop()
        assert not w.client.is_connected

        # Re-start
        w2 = WorkerSimulator(broker, "rereg-worker")
        w2.start()
        assert "rereg-worker" in platform.registered_workers

    def test_re_register_after_resume_control(self, broker, platform, worker, worker_id):
        """After pause + resume, worker re-registers."""
        platform.send_register_ack(worker_id, accepted=True)

        # Pause
        platform.send_control(worker_id, "pause")
        assert worker.state == "IDLE"

        # Resume triggers re-registration
        old_reg_count = len([m for m in platform._all_messages
                            if TOPIC_REGISTER in m.topic])
        platform.send_control(worker_id, "resume")
        assert worker.state == "AVAILABLE"

        new_reg_count = len([m for m in platform._all_messages
                            if TOPIC_REGISTER in m.topic])
        assert new_reg_count > old_reg_count


class TestRegistrationEdgeCases:
    """Edge cases and boundary conditions for registration."""

    def test_worker_with_custom_eth_address(self, broker, platform):
        w = WorkerSimulator(broker, "custom-addr",
                           eth_address="0x1234567890abcdef1234567890abcdef12345678")
        w.start()
        reg = platform.registered_workers["custom-addr"]
        assert reg["eth_address"] == "0x1234567890abcdef1234567890abcdef12345678"

    def test_multiple_registrations_overwrite(self, broker, platform):
        """Second registration from same worker_id overwrites first."""
        w1 = WorkerSimulator(broker, "dup-worker",
                            eth_address="0x" + "11" * 20)
        w1.start()
        first_addr = platform.registered_workers["dup-worker"]["eth_address"]

        # Re-register with different address
        w2 = WorkerSimulator(broker, "dup-worker",
                            eth_address="0x" + "22" * 20)
        w2.start()
        second_addr = platform.registered_workers["dup-worker"]["eth_address"]

        assert first_addr != second_addr
        assert second_addr == "0x" + "22" * 20
