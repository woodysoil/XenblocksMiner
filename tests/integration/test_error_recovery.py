"""
test_error_recovery.py - Error Recovery Flow Tests

Tests error handling and recovery:
 - Registration rejection → ERROR → recovery
 - Control: pause → IDLE → resume → re-register → AVAILABLE
 - Control: shutdown → full stop
 - Invalid messages handled gracefully
 - State machine ERROR → IDLE → AVAILABLE recovery path
 - Lease during error state ignored
"""

import uuid
import pytest

from .conftest import (
    WorkerSimulator, PlatformSimulator,
    make_register_ack, make_assign_task, make_control, make_release,
    TOPIC_TASK, TOPIC_CONTROL, TOPIC_STATUS,
)


class TestRegistrationRejectionRecovery:
    """Test recovery from registration rejection."""

    def test_rejected_then_accepted(self, broker, platform, worker, worker_id):
        """Worker gets rejected, transitions to ERROR, then gets accepted on retry."""
        # First: rejected
        platform.send_register_ack(worker_id, accepted=False, reason="version_mismatch")
        assert worker.state == "ERROR"

        # Simulate recovery: worker re-registers (normally via watchdog)
        # In the simulator, we manually trigger re-registration via resume
        platform.send_control(worker_id, "resume")
        # Resume only works from IDLE, but ERROR needs to go through IDLE first
        # The C++ code does ERROR -> IDLE -> AVAILABLE via watchdog
        # Simulate the watchdog: force to IDLE first
        worker.state = "IDLE"
        platform.send_control(worker_id, "resume")
        assert worker.state == "AVAILABLE"

    def test_error_status_reported_to_platform(self, broker, platform, worker, worker_id):
        """Platform sees ERROR status when registration is rejected."""
        platform.send_register_ack(worker_id, accepted=False, reason="banned")

        statuses = platform.get_worker_statuses(worker_id)
        error_statuses = [s for s in statuses if s["state"] == "ERROR"]
        assert len(error_statuses) >= 1


class TestPauseResumeFlow:
    """Test pause and resume control commands."""

    def test_pause_transitions_to_idle(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        assert worker.state == "AVAILABLE"

        platform.send_control(worker_id, "pause")
        assert worker.state == "IDLE"

    def test_resume_from_idle(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_control(worker_id, "pause")
        assert worker.state == "IDLE"

        platform.send_control(worker_id, "resume")
        assert worker.state == "AVAILABLE"

    def test_resume_triggers_re_registration(self, broker, platform, worker, worker_id):
        """Resume sends a new registration message."""
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_control(worker_id, "pause")

        # Count registrations before resume
        reg_before = len([m for m in platform._all_messages
                         if "register" in m.topic and "status" not in m.topic])

        platform.send_control(worker_id, "resume")

        reg_after = len([m for m in platform._all_messages
                        if "register" in m.topic and "status" not in m.topic])
        assert reg_after > reg_before

    def test_pause_during_mining_releases_lease(self, broker, platform, worker, worker_id):
        """Pause while mining should end the lease and go IDLE."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id, prefix="abcd" * 4)
        assert worker.state == "MINING"

        platform.send_control(worker_id, "pause")
        assert worker.state == "IDLE"
        assert worker.lease_id is None

    def test_pause_resume_cycle(self, broker, platform, worker, worker_id):
        """Multiple pause/resume cycles work correctly."""
        platform.send_register_ack(worker_id, accepted=True)

        for _ in range(3):
            assert worker.state == "AVAILABLE"
            platform.send_control(worker_id, "pause")
            assert worker.state == "IDLE"
            platform.send_control(worker_id, "resume")
            assert worker.state == "AVAILABLE"

    def test_resume_when_not_idle_is_noop(self, broker, platform, worker, worker_id):
        """Resume when already AVAILABLE should not change state."""
        platform.send_register_ack(worker_id, accepted=True)
        assert worker.state == "AVAILABLE"

        # Resume when already AVAILABLE
        platform.send_control(worker_id, "resume")
        assert worker.state == "AVAILABLE"


class TestShutdownFlow:
    """Test shutdown control command."""

    def test_shutdown_disconnects_worker(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_control(worker_id, "shutdown")
        assert worker.state == "IDLE"
        assert not worker.client.is_connected

    def test_shutdown_sends_offline_status(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_control(worker_id, "shutdown")

        statuses = platform.get_worker_statuses(worker_id)
        offline_statuses = [s for s in statuses if s["state"] == "offline"]
        assert len(offline_statuses) >= 1

    def test_shutdown_during_mining(self, broker, platform, worker, worker_id):
        """Shutdown while mining should stop cleanly."""
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_assign_task(worker_id, prefix="abcd" * 4)
        assert worker.state == "MINING"

        platform.send_control(worker_id, "shutdown")
        assert not worker.client.is_connected


class TestAssignTaskInInvalidStates:
    """Test that assign_task is rejected in non-AVAILABLE states."""

    def test_assign_task_ignored_in_idle(self, broker, platform, worker, worker_id):
        """Worker in IDLE should ignore assign_task."""
        platform.send_register_ack(worker_id, accepted=True)
        platform.send_control(worker_id, "pause")  # Go to IDLE
        assert worker.state == "IDLE"

        task = platform.send_assign_task(worker_id)
        # Should still be IDLE
        assert worker.state == "IDLE"
        assert worker.lease_id is None

    def test_assign_task_ignored_in_mining(self, broker, platform, worker, worker_id):
        """Worker already MINING should ignore second assign_task."""
        platform.send_register_ack(worker_id, accepted=True)
        task1 = platform.send_assign_task(worker_id, prefix="aaaa" * 4)
        assert worker.state == "MINING"
        first_lease = worker.lease_id

        task2 = platform.send_assign_task(worker_id, prefix="bbbb" * 4)
        assert worker.lease_id == first_lease


class TestReleaseInInvalidStates:
    """Test release command behavior in various states."""

    def test_release_when_available_is_harmless(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)
        assert worker.state == "AVAILABLE"

        platform.send_release(worker_id)
        assert worker.state == "AVAILABLE"

    def test_release_wrong_id_in_mining(self, broker, platform, worker, worker_id):
        """Release with wrong lease_id should be ignored."""
        platform.send_register_ack(worker_id, accepted=True)
        task = platform.send_assign_task(worker_id)
        assert worker.state == "MINING"

        platform.send_release(worker_id, lease_id="wrong-id")
        assert worker.state == "MINING"
        assert worker.lease_id == task["lease_id"]


class TestControlActions:
    """Verify all three control actions."""

    @pytest.mark.parametrize("action", ["pause", "resume", "shutdown"])
    def test_control_action_valid(self, action):
        msg = make_control(action)
        assert msg["action"] == action

    def test_invalid_control_action_rejected(self):
        with pytest.raises(AssertionError):
            make_control("invalid_action")


class TestGracefulRecoveryAfterLease:
    """Test worker can recover and accept new lease after various disruptions."""

    def test_new_lease_after_release(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)

        # First lease
        task1 = platform.send_assign_task(worker_id)
        platform.send_release(worker_id, task1["lease_id"])
        assert worker.state == "AVAILABLE"

        # Second lease
        task2 = platform.send_assign_task(worker_id)
        assert worker.state == "MINING"
        assert worker.lease_id == task2["lease_id"]

    def test_new_lease_after_pause_resume(self, broker, platform, worker, worker_id):
        platform.send_register_ack(worker_id, accepted=True)

        # Pause and resume
        platform.send_control(worker_id, "pause")
        platform.send_control(worker_id, "resume")
        assert worker.state == "AVAILABLE"

        # Should accept new lease
        task = platform.send_assign_task(worker_id)
        assert worker.state == "MINING"
