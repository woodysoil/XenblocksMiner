"""
test_state_machine.py - PlatformManager State Machine Validation

Tests the 6-state machine for platform leasing:
  IDLE -> AVAILABLE -> LEASED -> MINING -> COMPLETED -> IDLE
                                       \\-> ERROR -> IDLE

Validates:
 - All legal transitions
 - Illegal transitions are rejected
 - Timeout handling
 - State entry/exit callbacks
"""

import time
import enum
import pytest
from typing import Optional, Callable, Dict, Set


# ── Python model of PlatformManager state machine ──────────────────────────

class MinerState(enum.Enum):
    IDLE = "IDLE"
    AVAILABLE = "AVAILABLE"
    LEASED = "LEASED"
    MINING = "MINING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


# Legal transitions: from_state -> set of allowed to_states
LEGAL_TRANSITIONS: Dict[MinerState, Set[MinerState]] = {
    MinerState.IDLE: {MinerState.AVAILABLE},
    MinerState.AVAILABLE: {MinerState.LEASED, MinerState.IDLE, MinerState.ERROR},
    MinerState.LEASED: {MinerState.MINING, MinerState.ERROR, MinerState.IDLE},
    MinerState.MINING: {MinerState.COMPLETED, MinerState.ERROR},
    MinerState.COMPLETED: {MinerState.IDLE, MinerState.AVAILABLE},
    MinerState.ERROR: {MinerState.IDLE},
}


class InvalidTransitionError(Exception):
    pass


class StateTimeoutError(Exception):
    pass


class PlatformStateMachine:
    """Python model of the platform leasing state machine."""

    def __init__(self, timeout_seconds: float = 30.0):
        self._state = MinerState.IDLE
        self._timeout = timeout_seconds
        self._state_entered_at = time.monotonic()
        self._transition_log: list = []
        self._on_enter_callbacks: Dict[MinerState, Callable] = {}
        self._on_exit_callbacks: Dict[MinerState, Callable] = {}

    @property
    def state(self) -> MinerState:
        return self._state

    @property
    def transition_log(self) -> list:
        return list(self._transition_log)

    def register_on_enter(self, state: MinerState, callback: Callable):
        self._on_enter_callbacks[state] = callback

    def register_on_exit(self, state: MinerState, callback: Callable):
        self._on_exit_callbacks[state] = callback

    def transition_to(self, new_state: MinerState):
        if new_state not in LEGAL_TRANSITIONS.get(self._state, set()):
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.value} to {new_state.value}"
            )
        old_state = self._state

        # Exit callback
        if old_state in self._on_exit_callbacks:
            self._on_exit_callbacks[old_state]()

        self._state = new_state
        self._state_entered_at = time.monotonic()
        self._transition_log.append((old_state, new_state))

        # Enter callback
        if new_state in self._on_enter_callbacks:
            self._on_enter_callbacks[new_state]()

    def check_timeout(self) -> bool:
        """Returns True if current state has timed out."""
        elapsed = time.monotonic() - self._state_entered_at
        return elapsed > self._timeout

    def time_in_state(self) -> float:
        return time.monotonic() - self._state_entered_at

    def reset(self):
        """Force reset to IDLE (for error recovery)."""
        old = self._state
        self._state = MinerState.IDLE
        self._state_entered_at = time.monotonic()
        self._transition_log.append((old, MinerState.IDLE))


# ── Tests: legal transitions ───────────────────────────────────────────────

class TestLegalTransitions:
    """Verify all legal state transitions succeed."""

    def test_idle_to_available(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        assert sm.state == MinerState.AVAILABLE

    def test_available_to_leased(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        assert sm.state == MinerState.LEASED

    def test_leased_to_mining(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        assert sm.state == MinerState.MINING

    def test_mining_to_completed(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        assert sm.state == MinerState.COMPLETED

    def test_completed_to_idle(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        sm.transition_to(MinerState.IDLE)
        assert sm.state == MinerState.IDLE

    def test_completed_to_available(self):
        """After completing a task, can go directly to AVAILABLE for next task."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        sm.transition_to(MinerState.AVAILABLE)
        assert sm.state == MinerState.AVAILABLE

    def test_mining_to_error(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.ERROR)
        assert sm.state == MinerState.ERROR

    def test_error_to_idle(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.ERROR)
        sm.transition_to(MinerState.IDLE)
        assert sm.state == MinerState.IDLE

    def test_leased_to_error(self):
        """Lease might fail before mining starts."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.ERROR)
        assert sm.state == MinerState.ERROR

    def test_leased_to_idle(self):
        """Lease cancelled before mining starts."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.IDLE)
        assert sm.state == MinerState.IDLE

    def test_available_to_idle(self):
        """User decides to stop platform mode."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.IDLE)
        assert sm.state == MinerState.IDLE

    def test_available_to_error(self):
        """Registration rejected by platform."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.ERROR)
        assert sm.state == MinerState.ERROR

    def test_full_happy_path(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        sm.transition_to(MinerState.IDLE)

        expected = [
            (MinerState.IDLE, MinerState.AVAILABLE),
            (MinerState.AVAILABLE, MinerState.LEASED),
            (MinerState.LEASED, MinerState.MINING),
            (MinerState.MINING, MinerState.COMPLETED),
            (MinerState.COMPLETED, MinerState.IDLE),
        ]
        assert sm.transition_log == expected

    def test_full_error_recovery_path(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.ERROR)
        sm.transition_to(MinerState.IDLE)
        sm.transition_to(MinerState.AVAILABLE)  # retry
        assert sm.state == MinerState.AVAILABLE


# ── Tests: illegal transitions ─────────────────────────────────────────────

class TestIllegalTransitions:
    """Verify all illegal transitions are rejected."""

    @pytest.mark.parametrize("bad_target", [
        MinerState.LEASED,
        MinerState.MINING,
        MinerState.COMPLETED,
        MinerState.ERROR,
    ])
    def test_idle_cannot_skip(self, bad_target):
        sm = PlatformStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(bad_target)

    def test_available_cannot_go_to_mining(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.MINING)

    def test_available_cannot_go_to_completed(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.COMPLETED)

    def test_mining_cannot_go_to_available(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.AVAILABLE)

    def test_mining_cannot_go_to_idle(self):
        """Must go through COMPLETED or ERROR first."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.IDLE)

    def test_mining_cannot_go_to_leased(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.LEASED)

    def test_error_cannot_go_to_available(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.ERROR)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.AVAILABLE)

    def test_error_cannot_go_to_mining(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.ERROR)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.MINING)

    def test_completed_cannot_go_to_mining(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.MINING)

    def test_completed_cannot_go_to_error(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.ERROR)

    def test_self_transition_not_allowed(self):
        """Cannot transition to the same state."""
        sm = PlatformStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.IDLE)


# ── Tests: timeout handling ────────────────────────────────────────────────

class TestTimeout:
    """Verify timeout detection."""

    def test_no_timeout_immediately(self):
        sm = PlatformStateMachine(timeout_seconds=10.0)
        assert not sm.check_timeout()

    def test_timeout_detected(self):
        sm = PlatformStateMachine(timeout_seconds=0.05)
        time.sleep(0.1)
        assert sm.check_timeout()

    def test_timeout_resets_on_transition(self):
        sm = PlatformStateMachine(timeout_seconds=0.05)
        time.sleep(0.1)
        assert sm.check_timeout()
        sm.transition_to(MinerState.AVAILABLE)
        assert not sm.check_timeout()

    def test_time_in_state(self):
        sm = PlatformStateMachine()
        time.sleep(0.05)
        assert sm.time_in_state() >= 0.04


# ── Tests: callbacks ───────────────────────────────────────────────────────

class TestCallbacks:
    """Verify on_enter and on_exit callbacks fire."""

    def test_on_enter_callback(self):
        sm = PlatformStateMachine()
        entered = []
        sm.register_on_enter(MinerState.AVAILABLE, lambda: entered.append("AVAILABLE"))
        sm.transition_to(MinerState.AVAILABLE)
        assert entered == ["AVAILABLE"]

    def test_on_exit_callback(self):
        sm = PlatformStateMachine()
        exited = []
        sm.register_on_exit(MinerState.IDLE, lambda: exited.append("IDLE"))
        sm.transition_to(MinerState.AVAILABLE)
        assert exited == ["IDLE"]

    def test_both_callbacks_on_transition(self):
        sm = PlatformStateMachine()
        log = []
        sm.register_on_exit(MinerState.IDLE, lambda: log.append("exit_idle"))
        sm.register_on_enter(MinerState.AVAILABLE, lambda: log.append("enter_available"))
        sm.transition_to(MinerState.AVAILABLE)
        assert log == ["exit_idle", "enter_available"]

    def test_callback_not_called_for_other_states(self):
        sm = PlatformStateMachine()
        entered = []
        sm.register_on_enter(MinerState.MINING, lambda: entered.append("MINING"))
        sm.transition_to(MinerState.AVAILABLE)
        assert entered == []


# ── Tests: reset ───────────────────────────────────────────────────────────

class TestReset:
    """Verify force-reset to IDLE."""

    def test_reset_from_error(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.ERROR)
        sm.reset()
        assert sm.state == MinerState.IDLE

    def test_reset_from_mining(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.reset()
        assert sm.state == MinerState.IDLE

    def test_reset_logs_transition(self):
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.reset()
        log = sm.transition_log
        assert log[-1] == (MinerState.AVAILABLE, MinerState.IDLE)


# ── Tests: transition map completeness ─────────────────────────────────────

class TestTransitionMapCompleteness:
    """Every state must be represented in the transition map."""

    def test_all_states_have_entry(self):
        for state in MinerState:
            assert state in LEGAL_TRANSITIONS, f"{state} missing from transition map"

    def test_no_state_can_transition_to_itself(self):
        for state, targets in LEGAL_TRANSITIONS.items():
            assert state not in targets, f"{state} can transition to itself"


# ── Tests: error recovery integration ─────────────────────────────────────

class TestErrorRecoveryIntegration:
    """Verify the full ERROR → IDLE → AVAILABLE recovery path and beyond."""

    def test_full_recovery_through_mining(self):
        """ERROR → IDLE → AVAILABLE → LEASED → MINING (complete recovery)."""
        sm = PlatformStateMachine()
        # Normal startup
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        # Hit error
        sm.transition_to(MinerState.ERROR)
        # Recovery: must go through IDLE first
        sm.transition_to(MinerState.IDLE)
        sm.transition_to(MinerState.AVAILABLE)
        # Re-assigned a new task
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        assert sm.state == MinerState.MINING

        expected = [
            (MinerState.IDLE, MinerState.AVAILABLE),
            (MinerState.AVAILABLE, MinerState.LEASED),
            (MinerState.LEASED, MinerState.MINING),
            (MinerState.MINING, MinerState.ERROR),
            (MinerState.ERROR, MinerState.IDLE),
            (MinerState.IDLE, MinerState.AVAILABLE),
            (MinerState.AVAILABLE, MinerState.LEASED),
            (MinerState.LEASED, MinerState.MINING),
        ]
        assert sm.transition_log == expected

    def test_full_recovery_to_completion(self):
        """Recovery then successfully complete a task."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.ERROR)
        # Recover
        sm.transition_to(MinerState.IDLE)
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        sm.transition_to(MinerState.AVAILABLE)
        assert sm.state == MinerState.AVAILABLE

    def test_consecutive_errors_from_mining(self):
        """Multiple errors in a row, each recovered through IDLE."""
        sm = PlatformStateMachine()
        for i in range(3):
            sm.transition_to(MinerState.AVAILABLE)
            sm.transition_to(MinerState.LEASED)
            sm.transition_to(MinerState.MINING)
            sm.transition_to(MinerState.ERROR)
            sm.transition_to(MinerState.IDLE)

        # After 3 error-recovery cycles, should be back at IDLE
        assert sm.state == MinerState.IDLE
        # Verify we can still proceed normally
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        assert sm.state == MinerState.COMPLETED

    def test_consecutive_errors_from_leased(self):
        """Error during lease setup (before mining starts), repeated."""
        sm = PlatformStateMachine()
        for i in range(3):
            sm.transition_to(MinerState.AVAILABLE)
            sm.transition_to(MinerState.LEASED)
            sm.transition_to(MinerState.ERROR)
            sm.transition_to(MinerState.IDLE)

        assert sm.state == MinerState.IDLE
        sm.transition_to(MinerState.AVAILABLE)
        assert sm.state == MinerState.AVAILABLE

    def test_error_from_available_recovery(self):
        """Registration rejected (AVAILABLE → ERROR), then recover."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.ERROR)
        sm.transition_to(MinerState.IDLE)
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        assert sm.state == MinerState.MINING

    def test_error_from_each_state_then_recover(self):
        """Test ERROR entry from every state that allows it, then full recovery."""
        # States that can transition to ERROR: AVAILABLE, LEASED, MINING
        error_sources = [
            # (setup transitions, state that errors)
            ([MinerState.AVAILABLE], MinerState.AVAILABLE),
            ([MinerState.AVAILABLE, MinerState.LEASED], MinerState.LEASED),
            ([MinerState.AVAILABLE, MinerState.LEASED, MinerState.MINING], MinerState.MINING),
        ]
        for setup, source in error_sources:
            sm = PlatformStateMachine()
            for s in setup:
                sm.transition_to(s)
            assert sm.state == source
            sm.transition_to(MinerState.ERROR)
            assert sm.state == MinerState.ERROR
            sm.transition_to(MinerState.IDLE)
            assert sm.state == MinerState.IDLE
            # Can fully recover
            sm.transition_to(MinerState.AVAILABLE)
            sm.transition_to(MinerState.LEASED)
            sm.transition_to(MinerState.MINING)
            sm.transition_to(MinerState.COMPLETED)
            assert sm.state == MinerState.COMPLETED

    def test_error_recovery_cannot_skip_idle(self):
        """ERROR must go through IDLE; cannot jump directly to AVAILABLE."""
        sm = PlatformStateMachine()
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.ERROR)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(MinerState.AVAILABLE)

    def test_error_recovery_callbacks_fire(self):
        """Verify callbacks fire during ERROR → IDLE → AVAILABLE recovery."""
        sm = PlatformStateMachine()
        log = []
        sm.register_on_exit(MinerState.ERROR, lambda: log.append("exit_error"))
        sm.register_on_enter(MinerState.IDLE, lambda: log.append("enter_idle"))
        sm.register_on_exit(MinerState.IDLE, lambda: log.append("exit_idle"))
        sm.register_on_enter(MinerState.AVAILABLE, lambda: log.append("enter_available"))

        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.ERROR)
        log.clear()

        sm.transition_to(MinerState.IDLE)
        sm.transition_to(MinerState.AVAILABLE)

        assert log == ["exit_error", "enter_idle", "exit_idle", "enter_available"]

    def test_mixed_error_and_completion_cycles(self):
        """Alternate between successful completions and error recoveries."""
        sm = PlatformStateMachine()

        # Cycle 1: success
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        sm.transition_to(MinerState.IDLE)

        # Cycle 2: error
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.ERROR)
        sm.transition_to(MinerState.IDLE)

        # Cycle 3: success again
        sm.transition_to(MinerState.AVAILABLE)
        sm.transition_to(MinerState.LEASED)
        sm.transition_to(MinerState.MINING)
        sm.transition_to(MinerState.COMPLETED)
        sm.transition_to(MinerState.AVAILABLE)

        assert sm.state == MinerState.AVAILABLE
        assert len(sm.transition_log) == 15
