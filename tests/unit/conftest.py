"""Shared fixtures for XenblocksMiner test suite."""

import pytest
import re
import string


# ── Constants mirroring C++ MiningCommon.h ──────────────────────────────────

HASH_LENGTH = 64
DEVFEE_PREFIX = "FFFFFFFF"
ECODEVFEE_PREFIX = "EEEEEEEE"

# Platform mode uses 16-char hex prefix
PLATFORM_PREFIX_LENGTH = 16
DEVFEE_PREFIX_LENGTH = 8

# Valid hex character set
HEX_CHARS = set(string.hexdigits.lower())


# ── Helpers ─────────────────────────────────────────────────────────────────

def is_valid_hex(s: str) -> bool:
    """Check if string contains only lowercase hex characters."""
    return all(c in HEX_CHARS for c in s)


def is_valid_eth_address(address: str) -> bool:
    """Basic Ethereum address format validation (0x + 40 hex chars)."""
    return bool(re.match(r"^0x[0-9a-fA-F]{40}$", address))


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_eth_address():
    """A valid-format Ethereum address for testing."""
    return "0x24691E54aFafe2416a8252097C9Ca67557271475"


@pytest.fixture
def sample_user_address():
    """A valid-format user Ethereum address for testing."""
    return "0xAbCdEf0123456789AbCdEf0123456789AbCdEf01"


@pytest.fixture
def platform_prefix():
    """A 16-character hex platform prefix."""
    return "a1b2c3d4e5f67890"


@pytest.fixture
def short_search_string():
    """A simplified search string for fast testing (instead of XEN11)."""
    return "ab"
