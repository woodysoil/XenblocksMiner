"""
test_key_prefix.py - Key Prefix Injection Validation

Mirrors the C++ RandomHexKeyGenerator logic in Python and validates:
 - 16 char platform prefix + 48 char random = 64 char total key
 - Prefix correctly injected into generated key
 - DevFee (8-char FFFFFFFF/EEEEEEEE) vs platform mode (16-char) priority
 - Prefix is lowercased
 - Edge cases: empty prefix, prefix == total length, prefix > total length
"""

import random
import string
import re
import pytest


# ── Python port of RandomHexKeyGenerator ────────────────────────────────────

HEX_CHARS_LOWER = "0123456789abcdef"
HASH_LENGTH = 64
DEVFEE_PREFIX = "FFFFFFFF"
ECODEVFEE_PREFIX = "EEEEEEEE"


class RandomHexKeyGenerator:
    """Python equivalent of src/RandomHexKeyGenerator.h."""

    def __init__(self, initial_prefix: str = "", key_length: int = 64):
        self.total_length = key_length
        self.prefix = ""
        self.set_prefix(initial_prefix)

    def set_prefix(self, new_prefix: str):
        self.prefix = new_prefix.lower()

    def next_random_key(self) -> str:
        if len(self.prefix) >= self.total_length:
            # C++ behaviour: truncate prefix to total_length
            return self.prefix[:self.total_length]

        remaining = self.total_length - len(self.prefix)
        suffix = "".join(random.choice(HEX_CHARS_LOWER) for _ in range(remaining))
        return self.prefix + suffix


# ── Tests: basic key generation ─────────────────────────────────────────────

class TestKeyLength:
    """Every generated key must be exactly HASH_LENGTH characters."""

    def test_no_prefix_length(self):
        gen = RandomHexKeyGenerator("", HASH_LENGTH)
        for _ in range(50):
            key = gen.next_random_key()
            assert len(key) == HASH_LENGTH

    def test_8char_prefix_length(self):
        gen = RandomHexKeyGenerator(DEVFEE_PREFIX, HASH_LENGTH)
        for _ in range(50):
            key = gen.next_random_key()
            assert len(key) == HASH_LENGTH

    def test_16char_prefix_length(self):
        prefix = "a1b2c3d4e5f67890"
        gen = RandomHexKeyGenerator(prefix, HASH_LENGTH)
        for _ in range(50):
            key = gen.next_random_key()
            assert len(key) == HASH_LENGTH

    def test_max_prefix_length(self):
        """Prefix == total_length  → key is the prefix itself."""
        prefix = "a" * HASH_LENGTH
        gen = RandomHexKeyGenerator(prefix, HASH_LENGTH)
        key = gen.next_random_key()
        assert key == prefix
        assert len(key) == HASH_LENGTH

    def test_prefix_longer_than_total(self):
        """Prefix > total_length → key is truncated prefix."""
        prefix = "a" * (HASH_LENGTH + 10)
        gen = RandomHexKeyGenerator(prefix, HASH_LENGTH)
        key = gen.next_random_key()
        assert len(key) == HASH_LENGTH
        assert key == "a" * HASH_LENGTH


class TestHexValidity:
    """All generated keys must be valid lowercase hex."""

    HEX_RE = re.compile(r"^[0-9a-f]+$")

    def test_no_prefix_hex(self):
        gen = RandomHexKeyGenerator("", HASH_LENGTH)
        for _ in range(50):
            assert self.HEX_RE.match(gen.next_random_key())

    def test_with_prefix_hex(self):
        gen = RandomHexKeyGenerator("abcdef12", HASH_LENGTH)
        for _ in range(50):
            assert self.HEX_RE.match(gen.next_random_key())


# ── Tests: prefix injection ─────────────────────────────────────────────────

class TestPrefixInjection:
    """Verify prefix appears at the start of each key."""

    def test_devfee_prefix_injected(self):
        prefix = DEVFEE_PREFIX.lower()
        gen = RandomHexKeyGenerator(DEVFEE_PREFIX, HASH_LENGTH)
        for _ in range(50):
            key = gen.next_random_key()
            assert key.startswith(prefix), f"Key {key!r} must start with {prefix!r}"

    def test_ecodevfee_prefix_injected(self):
        prefix = ECODEVFEE_PREFIX.lower()
        gen = RandomHexKeyGenerator(ECODEVFEE_PREFIX, HASH_LENGTH)
        for _ in range(50):
            key = gen.next_random_key()
            assert key.startswith(prefix), f"Key {key!r} must start with {prefix!r}"

    def test_platform_prefix_injected(self):
        prefix = "a1b2c3d4e5f67890"  # 16 chars
        gen = RandomHexKeyGenerator(prefix, HASH_LENGTH)
        for _ in range(50):
            key = gen.next_random_key()
            assert key.startswith(prefix)

    def test_combined_devfee_plus_user_prefix(self):
        """In DevFee mode the C++ code sets prefix = DEVFEE_PREFIX + userAddr.
        The combined prefix is injected at the start of the key."""
        user_addr_hex = "24691e54afafe2416a8252097c9ca67557271475"
        combined = (DEVFEE_PREFIX + user_addr_hex).lower()
        gen = RandomHexKeyGenerator(combined, HASH_LENGTH)
        key = gen.next_random_key()
        assert key.startswith(combined[:HASH_LENGTH])
        assert len(key) == HASH_LENGTH

    def test_set_prefix_changes_output(self):
        gen = RandomHexKeyGenerator("aaaa", HASH_LENGTH)
        key1 = gen.next_random_key()
        assert key1.startswith("aaaa")

        gen.set_prefix("bbbb")
        key2 = gen.next_random_key()
        assert key2.startswith("bbbb")

    def test_clear_prefix(self):
        gen = RandomHexKeyGenerator("aaaa", HASH_LENGTH)
        gen.set_prefix("")
        key = gen.next_random_key()
        # No deterministic prefix; just validate length and hex
        assert len(key) == HASH_LENGTH


class TestPrefixLowercase:
    """C++ lowercases the prefix; verify the Python port does the same."""

    def test_uppercase_prefix_lowered(self):
        gen = RandomHexKeyGenerator("AABBCCDD", HASH_LENGTH)
        key = gen.next_random_key()
        assert key.startswith("aabbccdd")

    def test_mixed_case_prefix_lowered(self):
        gen = RandomHexKeyGenerator("AaBbCcDd", HASH_LENGTH)
        key = gen.next_random_key()
        assert key.startswith("aabbccdd")


# ── Tests: DevFee vs Platform priority simulation ──────────────────────────

class TestDevFeePlatformPriority:
    """
    Simulates the priority logic from MineUnit::runMineLoop().

    The C++ code:
      - Normal mode: prefix = ""
      - DevFee mode: prefix = DEVFEE_PREFIX + userAddr (8 + 40 = 48 chars)
      - EcoDevFee: prefix = ECODEVFEE_PREFIX + userAddr (8 + 40 = 48 chars)

    For platform mode (new feature), prefix = platform_prefix (16 chars).
    When DevFee is active, DevFee prefix takes precedence over platform prefix.
    """

    USER_ADDR = "24691e54afafe2416a8252097c9ca67557271475"  # 40 chars
    PLATFORM_PREFIX = "a1b2c3d4e5f67890"  # 16 chars

    def _resolve_prefix(self, batch_count: int, devfee_permillage: int,
                        platform_prefix: str, has_eco_devfee: bool) -> str:
        """Simulate the prefix resolution logic."""
        if 1000 - batch_count <= devfee_permillage:
            if (1000 - batch_count <= devfee_permillage // 2) and has_eco_devfee:
                return (ECODEVFEE_PREFIX + self.USER_ADDR).lower()
            else:
                return (DEVFEE_PREFIX + self.USER_ADDR).lower()
        elif platform_prefix:
            return platform_prefix.lower()
        else:
            return ""

    def test_normal_mode_no_prefix(self):
        prefix = self._resolve_prefix(batch_count=0, devfee_permillage=1,
                                      platform_prefix="", has_eco_devfee=False)
        assert prefix == ""

    def test_devfee_mode_overrides_platform(self):
        """When devfee is active, devfee prefix takes priority."""
        prefix = self._resolve_prefix(batch_count=999, devfee_permillage=1,
                                      platform_prefix=self.PLATFORM_PREFIX,
                                      has_eco_devfee=False)
        assert prefix.startswith(DEVFEE_PREFIX.lower())
        assert self.PLATFORM_PREFIX not in prefix

    def test_eco_devfee_overrides_platform(self):
        prefix = self._resolve_prefix(batch_count=999, devfee_permillage=2,
                                      platform_prefix=self.PLATFORM_PREFIX,
                                      has_eco_devfee=True)
        assert prefix.startswith(ECODEVFEE_PREFIX.lower())

    def test_platform_prefix_when_no_devfee(self):
        prefix = self._resolve_prefix(batch_count=0, devfee_permillage=1,
                                      platform_prefix=self.PLATFORM_PREFIX,
                                      has_eco_devfee=False)
        assert prefix == self.PLATFORM_PREFIX.lower()

    def test_platform_prefix_16_chars(self):
        """Platform prefix must be exactly 16 hex chars."""
        assert len(self.PLATFORM_PREFIX) == 16
        assert re.match(r"^[0-9a-f]{16}$", self.PLATFORM_PREFIX)

    def test_devfee_prefix_8_chars(self):
        assert len(DEVFEE_PREFIX) == 8

    def test_key_with_platform_prefix_structure(self):
        """16 char prefix + 48 char random = 64 total."""
        gen = RandomHexKeyGenerator(self.PLATFORM_PREFIX, HASH_LENGTH)
        key = gen.next_random_key()
        assert len(key) == 64
        assert key[:16] == self.PLATFORM_PREFIX
        assert len(key[16:]) == 48

    def test_key_with_devfee_prefix_structure(self):
        """8 char devfee + 40 char user addr prefix (48) + 16 random = 64."""
        combined = (DEVFEE_PREFIX + self.USER_ADDR).lower()
        gen = RandomHexKeyGenerator(combined, HASH_LENGTH)
        key = gen.next_random_key()
        assert len(key) == 64
        assert key[:8] == DEVFEE_PREFIX.lower()
        assert key[8:48] == self.USER_ADDR[:40]


# ── Tests: salt construction (mirrors MineUnit::runMineLoop) ───────────────

class TestSaltConstruction:
    """Validate the salt selection logic used for argon2id hashing.

    Mirrors the salt selection in MineUnit::runMineLoop, choosing between
    user address, devfee address, and eco-devfee address based on batch count.
    """

    USER_ADDRESS = "0x24691E54aFafe2416a8252097C9Ca67557271475"
    DEVFEE_ADDRESS = "0xDevFeeAddr000000000000000000000000000000"
    ECO_ADDRESS = "0xEcoFeeAddr000000000000000000000000000000"

    def _get_salt(self, batch_count: int, devfee_permillage: int,
                  has_eco: bool) -> str:
        """Mirrors the salt selection in MineUnit::runMineLoop."""
        if 1000 - batch_count <= devfee_permillage:
            if (1000 - batch_count <= devfee_permillage // 2) and has_eco:
                return self.ECO_ADDRESS[2:]
            else:
                return self.DEVFEE_ADDRESS[2:]
        else:
            return self.USER_ADDRESS[2:]

    def test_normal_salt_is_user_address(self):
        salt = self._get_salt(0, 1, False)
        assert salt == self.USER_ADDRESS[2:]

    def test_devfee_salt_is_devfee_address(self):
        salt = self._get_salt(999, 1, False)
        assert salt == self.DEVFEE_ADDRESS[2:]

    def test_eco_salt_is_eco_address(self):
        salt = self._get_salt(999, 2, True)
        assert salt == self.ECO_ADDRESS[2:]


# ── Tests: randomness sanity ───────────────────────────────────────────────

class TestRandomness:
    """Basic sanity checks that random portion is actually random."""

    def test_different_keys_generated(self):
        gen = RandomHexKeyGenerator("", HASH_LENGTH)
        keys = {gen.next_random_key() for _ in range(100)}
        # With 64 hex chars of randomness, collisions are astronomically unlikely
        assert len(keys) == 100

    def test_random_portion_varies_with_prefix(self):
        prefix = "aabb"
        gen = RandomHexKeyGenerator(prefix, HASH_LENGTH)
        suffixes = set()
        for _ in range(100):
            key = gen.next_random_key()
            suffixes.add(key[len(prefix):])
        assert len(suffixes) == 100
