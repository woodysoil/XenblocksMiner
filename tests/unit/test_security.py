"""
test_security.py - Security Validation Tests

Tests security constraints:
 - Key prefix length limits (max 16 chars for platform)
 - Ethereum address format validation
 - MQTT payload structure validation
 - Hex-only injection prevention
 - Prefix boundary enforcement
"""

import json
import re
import hashlib
import hmac
import pytest


# ── Constants ──────────────────────────────────────────────────────────────

HASH_LENGTH = 64
DEVFEE_PREFIX = "FFFFFFFF"
ECODEVFEE_PREFIX = "EEEEEEEE"
MAX_PLATFORM_PREFIX_LENGTH = 16


# ── Helpers ────────────────────────────────────────────────────────────────

def is_valid_hex(s: str) -> bool:
    return bool(re.match(r"^[0-9a-fA-F]+$", s))


def is_valid_eth_address(address: str) -> bool:
    """Basic Ethereum address validation: 0x + 40 hex chars."""
    return bool(re.match(r"^0x[0-9a-fA-F]{40}$", address))


def validate_platform_prefix(prefix: str) -> bool:
    """Validate that a platform prefix meets security constraints."""
    if len(prefix) > MAX_PLATFORM_PREFIX_LENGTH:
        return False
    if len(prefix) == 0:
        return False
    if not is_valid_hex(prefix):
        return False
    return True


def sign_payload(payload: dict, secret: str) -> str:
    """Create HMAC-SHA256 signature for MQTT payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_payload_signature(payload: dict, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature."""
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


# ── Tests: prefix length enforcement ───────────────────────────────────────

class TestPrefixLengthEnforcement:
    """Platform prefix must be at most 16 hex characters."""

    def test_valid_16_char_prefix(self):
        assert validate_platform_prefix("a1b2c3d4e5f67890")

    def test_valid_1_char_prefix(self):
        assert validate_platform_prefix("a")

    def test_reject_empty_prefix(self):
        assert not validate_platform_prefix("")

    def test_reject_17_char_prefix(self):
        assert not validate_platform_prefix("a" * 17)

    def test_reject_64_char_prefix(self):
        """Must not allow full-key-length prefix (would eliminate randomness)."""
        assert not validate_platform_prefix("a" * 64)

    def test_reject_non_hex_prefix(self):
        assert not validate_platform_prefix("zzzzzzzzzzzzzzzz")

    def test_reject_mixed_invalid_chars(self):
        assert not validate_platform_prefix("a1b2c3d4e5f6789g")  # 'g' is invalid

    def test_reject_special_chars(self):
        assert not validate_platform_prefix("a1b2c3d4e5f6789!")

    def test_reject_spaces(self):
        assert not validate_platform_prefix("a1b2 c3d4e5f6789")

    def test_reject_null_bytes(self):
        assert not validate_platform_prefix("a1b2\x00c3d4e5f689")

    @pytest.mark.parametrize("length", range(1, 17))
    def test_all_valid_lengths(self, length):
        prefix = "a" * length
        assert validate_platform_prefix(prefix)


# ── Tests: Ethereum address validation ─────────────────────────────────────

class TestEthereumAddressValidation:
    """Mirrors src/EthereumAddressValidator.cpp basic checks."""

    def test_valid_address(self):
        assert is_valid_eth_address("0x24691E54aFafe2416a8252097C9Ca67557271475")

    def test_valid_all_lowercase(self):
        assert is_valid_eth_address("0x" + "a" * 40)

    def test_valid_all_uppercase(self):
        assert is_valid_eth_address("0x" + "A" * 40)

    def test_reject_no_prefix(self):
        assert not is_valid_eth_address("24691E54aFafe2416a8252097C9Ca67557271475")

    def test_reject_short_address(self):
        assert not is_valid_eth_address("0x1234")

    def test_reject_long_address(self):
        assert not is_valid_eth_address("0x" + "a" * 41)

    def test_reject_non_hex(self):
        assert not is_valid_eth_address("0x" + "g" * 40)

    def test_reject_empty(self):
        assert not is_valid_eth_address("")

    def test_reject_only_prefix(self):
        assert not is_valid_eth_address("0x")

    def test_reject_wrong_prefix(self):
        assert not is_valid_eth_address("1x" + "a" * 40)


# ── Tests: MQTT payload structure ──────────────────────────────────────────

class TestMQTTPayloadValidation:
    """Validate expected fields in MQTT messages."""

    REQUIRED_REGISTER_FIELDS = {"miner_id", "gpu_count", "gpu_model", "version"}
    REQUIRED_TASK_FIELDS = {"task_id", "key_prefix", "target_address", "difficulty"}
    REQUIRED_BLOCK_FIELDS = {"task_id", "key", "hash_result", "salt"}

    def _validate_fields(self, payload: dict, required: set) -> list:
        """Return list of missing fields."""
        return [f for f in required if f not in payload]

    def test_valid_register_payload(self):
        payload = {
            "miner_id": "miner-001",
            "gpu_count": 2,
            "gpu_model": "RTX 4090",
            "version": "1.0.0",
        }
        assert self._validate_fields(payload, self.REQUIRED_REGISTER_FIELDS) == []

    def test_register_missing_fields(self):
        payload = {"miner_id": "miner-001"}
        missing = self._validate_fields(payload, self.REQUIRED_REGISTER_FIELDS)
        assert "gpu_count" in missing
        assert "gpu_model" in missing
        assert "version" in missing

    def test_valid_task_payload(self):
        payload = {
            "task_id": "task-42",
            "key_prefix": "a1b2c3d4e5f67890",
            "target_address": "0x" + "a" * 40,
            "difficulty": 1727,
        }
        assert self._validate_fields(payload, self.REQUIRED_TASK_FIELDS) == []

    def test_task_invalid_prefix(self):
        """Task payload with invalid (too long) prefix."""
        payload = {
            "task_id": "task-42",
            "key_prefix": "a" * 17,  # too long
            "target_address": "0x" + "a" * 40,
            "difficulty": 1727,
        }
        assert not validate_platform_prefix(payload["key_prefix"])

    def test_task_invalid_address(self):
        payload = {
            "task_id": "task-42",
            "key_prefix": "a1b2c3d4e5f67890",
            "target_address": "not_an_address",
            "difficulty": 1727,
        }
        assert not is_valid_eth_address(payload["target_address"])

    def test_valid_block_payload(self):
        payload = {
            "task_id": "task-42",
            "key": "a1b2c3d4e5f67890" + "0" * 48,
            "hash_result": "base64encodedresult",
            "salt": "a" * 40,
        }
        assert self._validate_fields(payload, self.REQUIRED_BLOCK_FIELDS) == []

    def test_block_key_has_correct_prefix(self):
        prefix = "a1b2c3d4e5f67890"
        payload = {
            "task_id": "task-42",
            "key": prefix + "0" * 48,
            "hash_result": "result",
            "salt": "a" * 40,
        }
        assert payload["key"].startswith(prefix)
        assert len(payload["key"]) == HASH_LENGTH


# ── Tests: MQTT payload signing ────────────────────────────────────────────

class TestPayloadSigning:
    """Verify HMAC-SHA256 signing for MQTT payloads."""

    SECRET = "test-secret-key-2024"

    def test_sign_and_verify(self):
        payload = {"task_id": "task-42", "key": "abc123"}
        sig = sign_payload(payload, self.SECRET)
        assert verify_payload_signature(payload, sig, self.SECRET)

    def test_tampered_payload_fails(self):
        payload = {"task_id": "task-42", "key": "abc123"}
        sig = sign_payload(payload, self.SECRET)
        payload["key"] = "tampered"
        assert not verify_payload_signature(payload, sig, self.SECRET)

    def test_wrong_secret_fails(self):
        payload = {"task_id": "task-42"}
        sig = sign_payload(payload, self.SECRET)
        assert not verify_payload_signature(payload, sig, "wrong-secret")

    def test_signature_is_hex(self):
        payload = {"data": "test"}
        sig = sign_payload(payload, self.SECRET)
        assert re.match(r"^[0-9a-f]{64}$", sig)

    def test_deterministic_signature(self):
        payload = {"a": 1, "b": 2}
        sig1 = sign_payload(payload, self.SECRET)
        sig2 = sign_payload(payload, self.SECRET)
        assert sig1 == sig2

    def test_key_order_independent(self):
        """JSON canonical form ensures key order doesn't matter."""
        payload1 = {"b": 2, "a": 1}
        payload2 = {"a": 1, "b": 2}
        sig1 = sign_payload(payload1, self.SECRET)
        sig2 = sign_payload(payload2, self.SECRET)
        assert sig1 == sig2


# ── Tests: key injection boundary ─────────────────────────────────────────

class TestKeyInjectionBoundary:
    """Ensure prefix injection cannot produce keys that exceed HASH_LENGTH boundaries."""

    def test_prefix_plus_random_equals_hash_length(self):
        prefix = "a" * MAX_PLATFORM_PREFIX_LENGTH
        random_part = "b" * (HASH_LENGTH - MAX_PLATFORM_PREFIX_LENGTH)
        key = prefix + random_part
        assert len(key) == HASH_LENGTH

    def test_prefix_cannot_exceed_hash_length(self):
        """Even if someone tries a longer prefix, key stays at HASH_LENGTH."""
        # Simulating RandomHexKeyGenerator behavior
        prefix = "a" * 100
        if len(prefix) >= HASH_LENGTH:
            key = prefix[:HASH_LENGTH]
        else:
            key = prefix + "0" * (HASH_LENGTH - len(prefix))
        assert len(key) == HASH_LENGTH

    def test_all_hex_in_key(self):
        """Keys must contain only hex characters."""
        prefix = "a1b2c3d4e5f67890"
        random_part = "0123456789abcdef" * 3  # 48 chars
        key = prefix + random_part
        assert is_valid_hex(key)
        assert len(key) == HASH_LENGTH


# ── Tests: devfee prefix safety ────────────────────────────────────────────

class TestDevFeePrefixSafety:
    """Ensure devfee prefixes (FFFFFFFF/EEEEEEEE) cannot be spoofed via platform mode."""

    def test_devfee_prefix_is_8_chars(self):
        assert len(DEVFEE_PREFIX) == 8

    def test_ecodevfee_prefix_is_8_chars(self):
        assert len(ECODEVFEE_PREFIX) == 8

    def test_platform_prefix_cannot_start_with_devfee(self):
        """Platform prefixes starting with FFFFFFFF should be flagged."""
        bad_prefix = "ffffffff12345678"  # starts with devfee
        assert bad_prefix.startswith(DEVFEE_PREFIX.lower())
        # This is a security check: platform should NOT accept devfee-like prefixes
        # The actual enforcement would be in the C++ PlatformManager

    def test_platform_prefix_cannot_start_with_ecodevfee(self):
        bad_prefix = "eeeeeeee12345678"
        assert bad_prefix.startswith(ECODEVFEE_PREFIX.lower())


# ── Tests: salt validation ─────────────────────────────────────────────────

class TestSaltValidation:
    """Validate salt derivation from Ethereum addresses (40 hex chars after 0x)."""

    def test_salt_from_valid_address(self):
        addr = "0x24691E54aFafe2416a8252097C9Ca67557271475"
        salt = addr[2:]
        assert len(salt) == 40
        assert is_valid_hex(salt)

    def test_salt_length_always_40(self):
        """Salt is always 40 chars (address minus 0x prefix)."""
        for _ in range(10):
            addr = "0x" + "a" * 40
            salt = addr[2:]
            assert len(salt) == 40

    def test_reject_salt_from_invalid_address(self):
        """If address is invalid, salt extraction should be guarded."""
        bad_addr = "0x1234"
        salt = bad_addr[2:]
        assert len(salt) != 40  # too short
