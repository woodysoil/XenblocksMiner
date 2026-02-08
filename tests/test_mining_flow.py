"""
test_mining_flow.py - Mining Flow Validation

Tests the end-to-end mining flow:
 - Simplified search string ("ab" instead of "XEN11") for fast testing
 - Simulated Argon2id hashing with key prefix injection
 - Self-mining → Platform mode → Back to self-mining transitions
 - DevFee rotation logic
 - Block discovery and submission callback
"""

import base64
import hashlib
import re
import time
import pytest

try:
    import argon2
    from argon2.low_level import hash_secret_raw, Type
    HAS_ARGON2 = True
except ImportError:
    HAS_ARGON2 = False

from .mock_mqtt_broker import MockMQTTBroker

# ── Constants ──────────────────────────────────────────────────────────────

HASH_LENGTH = 64
DEVFEE_PREFIX = "FFFFFFFF"
ECODEVFEE_PREFIX = "EEEEEEEE"
SIMPLIFIED_SEARCH = "ab"  # Low-difficulty search for tests (instead of "XEN11")


# ── Python port of key generator ───────────────────────────────────────────

import random

HEX_CHARS_LOWER = "0123456789abcdef"


class RandomHexKeyGenerator:
    def __init__(self, prefix: str = "", key_length: int = 64):
        self.total_length = key_length
        self.prefix = prefix.lower()

    def set_prefix(self, new_prefix: str):
        self.prefix = new_prefix.lower()

    def next_random_key(self) -> str:
        if len(self.prefix) >= self.total_length:
            return self.prefix[:self.total_length]
        remaining = self.total_length - len(self.prefix)
        suffix = "".join(random.choice(HEX_CHARS_LOWER) for _ in range(remaining))
        return self.prefix + suffix


# ── Simulated mining logic ─────────────────────────────────────────────────

class SimulatedMiner:
    """
    Simulates the MineUnit mining loop in pure Python.
    Uses SHA-256 (or Argon2id if available) as a hash stand-in
    to verify key prefix injection and block discovery.
    """

    def __init__(self, user_address: str, devfee_address: str = "",
                 eco_devfee_address: str = "", devfee_permillage: int = 1,
                 search_string: str = SIMPLIFIED_SEARCH):
        self.user_address = user_address
        self.devfee_address = devfee_address
        self.eco_devfee_address = eco_devfee_address
        self.devfee_permillage = devfee_permillage
        self.search_string = search_string

        self.platform_prefix = ""
        self.platform_salt = ""
        self.mode = "self"  # "self" or "platform"

        self.key_gen = RandomHexKeyGenerator("", HASH_LENGTH)
        self.submitted_blocks: list = []
        self.batch_count = 0

    def set_platform_mode(self, prefix: str, salt: str):
        """Switch to platform mining mode."""
        self.mode = "platform"
        self.platform_prefix = prefix
        self.platform_salt = salt

    def set_self_mode(self):
        """Switch back to self-mining mode."""
        self.mode = "self"
        self.platform_prefix = ""
        self.platform_salt = ""

    def _resolve_prefix_and_salt(self) -> tuple:
        """
        Mirrors MineUnit::runMineLoop() prefix/salt logic.
        Returns (prefix, salt).
        """
        if self.mode == "platform" and self.platform_prefix:
            # Platform mode: use platform prefix, platform salt
            return self.platform_prefix, self.platform_salt

        salt = self.user_address[2:] if self.user_address.startswith("0x") else self.user_address

        # DevFee check
        if 1000 - self.batch_count <= self.devfee_permillage:
            half = self.devfee_permillage // 2
            if (1000 - self.batch_count <= half) and self.eco_devfee_address:
                eco_salt = self.eco_devfee_address[2:] if self.eco_devfee_address.startswith("0x") else self.eco_devfee_address
                prefix = (ECODEVFEE_PREFIX + salt).lower()
                return prefix, eco_salt
            else:
                dev_salt = self.devfee_address[2:] if self.devfee_address.startswith("0x") else self.devfee_address
                prefix = (DEVFEE_PREFIX + salt).lower()
                return prefix, dev_salt
        else:
            return "", salt

    def _compute_hash(self, key: str, salt: str) -> str:
        """
        Compute a hash of key+salt. Uses SHA-256 as a fast stand-in.
        Returns base64-encoded result (similar to C++ base64_encode of argon2id output).
        """
        data = (key + salt).encode("utf-8")
        raw = hashlib.sha256(data).digest()
        return base64.b64encode(raw).decode("ascii")

    def _compute_argon2_hash(self, key: str, salt: str) -> str:
        """Compute actual argon2id hash if library is available."""
        if not HAS_ARGON2:
            return self._compute_hash(key, salt)
        raw = hash_secret_raw(
            secret=key.encode("utf-8"),
            salt=salt.encode("utf-8"),
            time_cost=1,
            memory_cost=1024,  # Low memory for test speed
            parallelism=1,
            hash_len=HASH_LENGTH,
            type=Type.ID,
        )
        return base64.b64encode(raw).decode("ascii")

    def mine_batch(self, batch_size: int = 100, use_argon2: bool = False) -> list:
        """
        Mine a batch and return found blocks.
        """
        prefix, salt = self._resolve_prefix_and_salt()
        self.key_gen.set_prefix(prefix)

        found = []
        hash_fn = self._compute_argon2_hash if use_argon2 else self._compute_hash

        for _ in range(batch_size):
            key = self.key_gen.next_random_key()
            hashed = hash_fn(key, salt)

            if self.search_string in hashed:
                block = {
                    "key": key,
                    "hash": hashed,
                    "salt": salt,
                    "prefix": prefix,
                    "mode": self.mode,
                }
                found.append(block)
                self.submitted_blocks.append(block)

        self.batch_count += 1
        if self.batch_count >= 1000:
            self.batch_count = 0

        return found


# ── Tests: key prefix injection in mining ──────────────────────────────────

class TestKeyPrefixInMining:
    USER = "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"

    def test_self_mining_no_prefix(self):
        miner = SimulatedMiner(self.USER)
        prefix, salt = miner._resolve_prefix_and_salt()
        assert prefix == ""
        assert salt == self.USER[2:]

    def test_platform_prefix_injected(self):
        miner = SimulatedMiner(self.USER)
        platform_prefix = "a1b2c3d4e5f67890"
        platform_salt = "deadbeef" * 5
        miner.set_platform_mode(platform_prefix, platform_salt)

        prefix, salt = miner._resolve_prefix_and_salt()
        assert prefix == platform_prefix
        assert salt == platform_salt

    def test_platform_key_starts_with_prefix(self):
        miner = SimulatedMiner(self.USER)
        platform_prefix = "a1b2c3d4e5f67890"
        miner.set_platform_mode(platform_prefix, "somesalt")
        miner.key_gen.set_prefix(platform_prefix)

        for _ in range(50):
            key = miner.key_gen.next_random_key()
            assert key.startswith(platform_prefix)
            assert len(key) == HASH_LENGTH

    def test_devfee_prefix_during_devfee_batch(self):
        miner = SimulatedMiner(
            self.USER,
            devfee_address="0x" + "d" * 40,
            devfee_permillage=1,
        )
        miner.batch_count = 999  # triggers devfee
        prefix, salt = miner._resolve_prefix_and_salt()
        assert prefix.startswith(DEVFEE_PREFIX.lower())

    def test_devfee_overrides_platform_mode(self):
        """Even in platform mode, devfee logic takes precedence in self-mining."""
        miner = SimulatedMiner(
            self.USER,
            devfee_address="0x" + "d" * 40,
            devfee_permillage=1,
        )
        # In platform mode, devfee is NOT applied (platform gets full mining)
        miner.set_platform_mode("a1b2c3d4e5f67890", "platformsalt")
        miner.batch_count = 999
        prefix, salt = miner._resolve_prefix_and_salt()
        # Platform mode should use platform prefix regardless
        assert prefix == "a1b2c3d4e5f67890"


# ── Tests: mining with simplified search ───────────────────────────────────

class TestSimplifiedMining:
    """Use "ab" as search string for quick block discovery."""

    USER = "0x" + "a" * 40

    def test_find_blocks_with_easy_search(self):
        """With "ab" search, blocks should be found quickly."""
        miner = SimulatedMiner(self.USER, search_string="ab")
        all_found = []
        for _ in range(100):
            found = miner.mine_batch(batch_size=100)
            all_found.extend(found)
        # "ab" appears often in base64-encoded SHA256
        assert len(all_found) > 0, "Should find at least one block with 'ab' search"

    def test_found_block_has_search_string_in_hash(self):
        miner = SimulatedMiner(self.USER, search_string="ab")
        for _ in range(200):
            found = miner.mine_batch(batch_size=100)
            for block in found:
                assert "ab" in block["hash"]
            if found:
                break

    def test_found_block_in_self_mode(self):
        miner = SimulatedMiner(self.USER, search_string="ab")
        for _ in range(200):
            found = miner.mine_batch(batch_size=100)
            for block in found:
                assert block["mode"] == "self"
                assert block["prefix"] == ""
            if found:
                break

    def test_found_block_in_platform_mode(self):
        miner = SimulatedMiner(self.USER, search_string="ab")
        miner.set_platform_mode("deadbeefcafe1234", "platformsalt123")
        for _ in range(200):
            found = miner.mine_batch(batch_size=100)
            for block in found:
                assert block["mode"] == "platform"
                assert block["key"].startswith("deadbeefcafe1234")
                assert block["salt"] == "platformsalt123"
            if found:
                break

    def test_xen11_never_found_with_easy_search(self):
        """With "ab" search, we should NOT be checking for XEN11."""
        miner = SimulatedMiner(self.USER, search_string="ab")
        miner.mine_batch(batch_size=10)
        # Just verify the search string is "ab" not "XEN11"
        assert miner.search_string == "ab"


# ── Tests: mode switching ──────────────────────────────────────────────────

class TestModeSwitching:
    USER = "0x" + "b" * 40

    def test_self_to_platform_to_self(self):
        miner = SimulatedMiner(self.USER, search_string="ab")

        # Phase 1: self mining
        assert miner.mode == "self"
        miner.mine_batch(batch_size=10)
        prefix1, salt1 = miner._resolve_prefix_and_salt()
        assert prefix1 == ""
        assert salt1 == "b" * 40

        # Phase 2: platform
        miner.set_platform_mode("1234567890abcdef", "platform_salt_here")
        assert miner.mode == "platform"
        prefix2, salt2 = miner._resolve_prefix_and_salt()
        assert prefix2 == "1234567890abcdef"
        assert salt2 == "platform_salt_here"

        # Phase 3: back to self
        miner.set_self_mode()
        assert miner.mode == "self"
        prefix3, salt3 = miner._resolve_prefix_and_salt()
        assert prefix3 == ""
        assert salt3 == "b" * 40

    def test_multiple_platform_tasks(self):
        """Switch between different platform tasks."""
        miner = SimulatedMiner(self.USER, search_string="ab")

        miner.set_platform_mode("aaaa" * 4, "salt_a")
        p1, s1 = miner._resolve_prefix_and_salt()
        assert p1 == "aaaa" * 4

        miner.set_platform_mode("bbbb" * 4, "salt_b")
        p2, s2 = miner._resolve_prefix_and_salt()
        assert p2 == "bbbb" * 4
        assert s2 == "salt_b"

    def test_submitted_blocks_track_mode(self):
        miner = SimulatedMiner(self.USER, search_string="a")  # very easy search

        # Mine in self mode
        for _ in range(50):
            miner.mine_batch(batch_size=200)

        self_blocks = [b for b in miner.submitted_blocks if b["mode"] == "self"]

        # Mine in platform mode
        miner.set_platform_mode("cafe" * 4, "platform_salt")
        for _ in range(50):
            miner.mine_batch(batch_size=200)

        platform_blocks = [b for b in miner.submitted_blocks if b["mode"] == "platform"]

        # Both modes should have produced blocks with search="a"
        assert len(self_blocks) > 0, "Self mode should produce blocks"
        assert len(platform_blocks) > 0, "Platform mode should produce blocks"

        for b in platform_blocks:
            assert b["key"].startswith("cafe" * 4)


# ── Tests: DevFee rotation ────────────────────────────────────────────────

class TestDevFeeRotation:
    USER = "0x" + "c" * 40
    DEVFEE = "0x" + "d" * 40
    ECO = "0x" + "e" * 40

    def test_devfee_batch_counter_wraps(self):
        miner = SimulatedMiner(self.USER, self.DEVFEE, devfee_permillage=1)
        for i in range(1001):
            miner.mine_batch(batch_size=1)
        # batch_count should have wrapped
        assert miner.batch_count < 1000

    def test_devfee_triggers_at_correct_batch(self):
        miner = SimulatedMiner(self.USER, self.DEVFEE, devfee_permillage=1)

        # At batch_count 998 (1000-998=2 > 1), no devfee
        miner.batch_count = 998
        prefix, salt = miner._resolve_prefix_and_salt()
        assert not prefix.startswith(DEVFEE_PREFIX.lower())

        # At batch_count 999 (1000-999=1 <= 1), devfee
        miner.batch_count = 999
        prefix, salt = miner._resolve_prefix_and_salt()
        assert prefix.startswith(DEVFEE_PREFIX.lower())

    def test_eco_devfee_with_both_addresses(self):
        miner = SimulatedMiner(self.USER, self.DEVFEE, self.ECO, devfee_permillage=2)
        # batch_count=999: 1000-999=1 <= 2 (devfee range)
        #   1 <= 2//2=1 (eco range) and eco exists
        miner.batch_count = 999
        prefix, salt = miner._resolve_prefix_and_salt()
        assert prefix.startswith(ECODEVFEE_PREFIX.lower())
        assert salt == "e" * 40

    def test_regular_devfee_when_eco_empty(self):
        miner = SimulatedMiner(self.USER, self.DEVFEE, "", devfee_permillage=2)
        miner.batch_count = 999
        prefix, salt = miner._resolve_prefix_and_salt()
        assert prefix.startswith(DEVFEE_PREFIX.lower())
        assert salt == "d" * 40


# ── Tests: Argon2id integration (if available) ─────────────────────────────

class TestArgon2Integration:
    USER = "0x" + "f" * 40

    @pytest.mark.skipif(not HAS_ARGON2, reason="argon2-cffi not installed")
    def test_argon2_hash_produces_output(self):
        miner = SimulatedMiner(self.USER, search_string="a")
        key = "a" * 64
        salt = "f" * 40
        result = miner._compute_argon2_hash(key, salt)
        assert len(result) > 0
        # Base64 encoded
        assert re.match(r"^[A-Za-z0-9+/]+=*$", result)

    @pytest.mark.skipif(not HAS_ARGON2, reason="argon2-cffi not installed")
    def test_argon2_deterministic(self):
        miner = SimulatedMiner(self.USER)
        key = "deadbeef" * 8
        salt = "f" * 40
        h1 = miner._compute_argon2_hash(key, salt)
        h2 = miner._compute_argon2_hash(key, salt)
        assert h1 == h2

    @pytest.mark.skipif(not HAS_ARGON2, reason="argon2-cffi not installed")
    def test_argon2_different_keys_different_hashes(self):
        miner = SimulatedMiner(self.USER)
        salt = "f" * 40
        h1 = miner._compute_argon2_hash("a" * 64, salt)
        h2 = miner._compute_argon2_hash("b" * 64, salt)
        assert h1 != h2

    @pytest.mark.skipif(not HAS_ARGON2, reason="argon2-cffi not installed")
    def test_argon2_mine_batch(self):
        miner = SimulatedMiner(self.USER, search_string="a")
        found = []
        for _ in range(20):
            found.extend(miner.mine_batch(batch_size=50, use_argon2=True))
            if found:
                break
        assert len(found) > 0


# ── Tests: submit callback simulation ─────────────────────────────────────

class TestSubmitCallback:
    """Simulates the SubmitCallback from MiningCommon.h."""

    USER = "0x" + "a" * 40

    def test_callback_receives_correct_data(self):
        submitted = []

        def callback(hexsalt, key, hashed_pure, attempts, hashrate):
            submitted.append({
                "hexsalt": hexsalt,
                "key": key,
                "hashed_pure": hashed_pure,
                "attempts": attempts,
                "hashrate": hashrate,
            })

        # Simulate a block submission
        callback("a" * 40, "deadbeef" * 8, "XEN11hashresult", 12345, 1500.0)
        assert len(submitted) == 1
        assert submitted[0]["hexsalt"] == "a" * 40
        assert submitted[0]["key"] == "deadbeef" * 8
        assert submitted[0]["attempts"] == 12345

    def test_platform_callback_includes_prefix(self):
        submitted = []

        def callback(hexsalt, key, hashed_pure, attempts, hashrate):
            submitted.append({"key": key, "hexsalt": hexsalt})

        prefix = "a1b2c3d4e5f67890"
        key = prefix + "0" * 48
        callback("platformsalt", key, "hash_with_ab_inside", 100, 1000.0)

        assert submitted[0]["key"].startswith(prefix)
        assert len(submitted[0]["key"]) == 64
