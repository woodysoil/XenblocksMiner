"""
chain_simulator.py - XenBlocks blockchain/RPC simulator.

Simulates the actual XenBlocks RPC server for fully offline testing:
 - GET  /difficulty           -> current network difficulty
 - POST /verify               -> validate and accept block submission
 - GET  /getblocks/lastblock  -> recent block records
 - GET  /balance/{address}    -> XNM balance for an address
 - POST /send_pow             -> PoW submission (merkle root, always accepted in mock)

Block validation mirrors real XenBlocks:
 - Key must be 64 hex characters
 - Hash must contain "XEN11" (normal block) or match XUNI\\d (xuni block)
 - Superblock: XEN11 in hash + 50+ uppercase characters
 - XUNI blocks only accepted within 5 minutes of the hour

Difficulty auto-adjusts every N blocks (configurable).

Usage (standalone):
    python -m server.chain_simulator --port 8545
    python server/chain_simulator.py --port 8545

Usage (integrated into mock platform server):
    from server.chain_simulator import ChainSimulator
    chain = ChainSimulator()
    chain.register_routes(fastapi_app)
"""

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger("chain")

# ---------------------------------------------------------------------------
# Constants (matching C++ MiningCommon.h)
# ---------------------------------------------------------------------------

KEY_HEX_LENGTH = 64          # Full key: 64 hex chars (prefix + random)
HASH_LENGTH = 64             # Argon2id output length
DEFAULT_DIFFICULTY = 1727    # Default network difficulty
DIFFICULTY_ADJUST_INTERVAL = 50    # Adjust difficulty every N blocks
DIFFICULTY_ADJUST_FACTOR = 1.05    # Increase by 5% per interval
MIN_DIFFICULTY = 100
MAX_DIFFICULTY = 100000

XUNI_PATTERN = re.compile(r"XUNI\d")
SUPERBLOCK_UPPERCASE_THRESHOLD = 50

# Block type search strings
NORMAL_BLOCK_MARKER = "XEN11"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ChainBlock:
    block_id: int
    account: str
    key: str
    hash_to_verify: str
    block_type: str  # "normal", "super", "xuni"
    attempts: int = 0
    hashes_per_second: str = "0.0"
    worker: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "account": self.account,
            "key": self.key,
            "hash_to_verify": self.hash_to_verify,
            "block_type": self.block_type,
            "attempts": self.attempts,
            "hashes_per_second": self.hashes_per_second,
            "worker": self.worker,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Chain Simulator
# ---------------------------------------------------------------------------


class ChainSimulator:
    """Mock XenBlocks blockchain with RPC endpoints."""

    def __init__(
        self,
        base_difficulty: int = DEFAULT_DIFFICULTY,
        difficulty_adjust_interval: int = DIFFICULTY_ADJUST_INTERVAL,
        xuni_always_valid: bool = True,
        block_marker: str = "",
    ):
        self._difficulty = base_difficulty
        self._base_difficulty = base_difficulty
        self._adjust_interval = difficulty_adjust_interval
        self._xuni_always_valid = xuni_always_valid  # Skip time-window check for testing
        self._block_marker = block_marker or NORMAL_BLOCK_MARKER

        self._blocks: List[ChainBlock] = []
        self._next_block_id = 1
        self._balances: Dict[str, float] = {}
        self._known_keys: set = set()  # Deduplication

        logger.info(
            "Chain simulator initialized (difficulty=%d, adjust_interval=%d)",
            self._difficulty, self._adjust_interval,
        )

    # -------------------------------------------------------------------
    # Core chain operations
    # -------------------------------------------------------------------

    @property
    def difficulty(self) -> int:
        return self._difficulty

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    def classify_block(self, hash_value: str) -> Optional[str]:
        """Determine block type from hash content. Returns None if invalid."""
        if self._block_marker in hash_value:
            uppercase_count = sum(1 for c in hash_value if c.isupper())
            if uppercase_count >= SUPERBLOCK_UPPERCASE_THRESHOLD:
                return "super"
            return "normal"
        if XUNI_PATTERN.search(hash_value):
            if not self._xuni_always_valid and not self._is_xuni_time_window():
                return None  # Outside time window
            return "xuni"
        return None

    def validate_key(self, key: str) -> tuple:
        """Validate key format. Returns (valid, error_message)."""
        if not key:
            return False, "Key is empty"
        # Keys can be longer than 64 chars in practice (prefix + random),
        # but must be at least 64 hex chars
        if len(key) < KEY_HEX_LENGTH:
            return False, f"Key too short: {len(key)} chars (need >= {KEY_HEX_LENGTH})"
        try:
            int(key, 16)
        except ValueError:
            return False, "Key contains non-hex characters"
        return True, ""

    def submit_block(
        self,
        hash_to_verify: str,
        key: str,
        account: str,
        attempts: int = 0,
        hashes_per_second: str = "0.0",
        worker: str = "",
    ) -> tuple:
        """
        Validate and submit a block.
        Returns (success: bool, message: str, block_type: str or None).
        """
        # Validate key
        key_valid, key_error = self.validate_key(key)
        if not key_valid:
            return False, key_error, None

        # Check duplicate
        if key in self._known_keys:
            return False, "Key already exists", None

        # Classify block by hash
        block_type = self.classify_block(hash_to_verify)
        if block_type is None:
            return False, "Hash does not contain valid block marker (XEN11 or XUNI)", None

        # Accept the block
        block = ChainBlock(
            block_id=self._next_block_id,
            account=account,
            key=key,
            hash_to_verify=hash_to_verify,
            block_type=block_type,
            attempts=attempts,
            hashes_per_second=hashes_per_second,
            worker=worker,
        )
        self._blocks.append(block)
        self._known_keys.add(key)
        self._next_block_id += 1

        # Credit miner balance
        reward = self._get_reward(block_type)
        addr_lower = account.lower()
        self._balances[addr_lower] = self._balances.get(addr_lower, 0.0) + reward

        # Maybe adjust difficulty
        if len(self._blocks) % self._adjust_interval == 0:
            self._adjust_difficulty()

        logger.info(
            "Block #%d accepted: type=%s account=%s key=%s..%s reward=%.4f",
            block.block_id, block_type, account[:12], key[:8], key[-4:], reward,
        )
        return True, f"Block accepted: {block_type} #{block.block_id}", block_type

    def get_balance(self, address: str) -> float:
        return self._balances.get(address.lower(), 0.0)

    def get_last_blocks(self, count: int = 100) -> List[dict]:
        """Return the last N blocks (like /getblocks/lastblock)."""
        return [b.to_dict() for b in self._blocks[-count:]]

    def get_blocks_by_account(self, account: str, limit: int = 100) -> List[dict]:
        addr_lower = account.lower()
        return [
            b.to_dict()
            for b in reversed(self._blocks)
            if b.account.lower() == addr_lower
        ][:limit]

    def get_blocks_by_key_prefix(self, prefix: str, limit: int = 100) -> List[dict]:
        prefix_lower = prefix.lower()
        return [
            b.to_dict()
            for b in reversed(self._blocks)
            if b.key.lower().startswith(prefix_lower)
        ][:limit]

    def get_stats(self) -> dict:
        normal = sum(1 for b in self._blocks if b.block_type == "normal")
        super_blocks = sum(1 for b in self._blocks if b.block_type == "super")
        xuni = sum(1 for b in self._blocks if b.block_type == "xuni")
        return {
            "total_blocks": len(self._blocks),
            "normal_blocks": normal,
            "super_blocks": super_blocks,
            "xuni_blocks": xuni,
            "difficulty": self._difficulty,
            "unique_miners": len(self._balances),
            "next_block_id": self._next_block_id,
        }

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    @staticmethod
    def _is_xuni_time_window() -> bool:
        """XUNI blocks only valid within 5 minutes of the hour."""
        now = datetime.now()
        minutes = now.minute
        return minutes < 5 or minutes >= 55

    @staticmethod
    def _get_reward(block_type: str) -> float:
        if block_type == "super":
            return 50.0
        if block_type == "normal":
            return 10.0
        if block_type == "xuni":
            return 1.0
        return 0.0

    def _adjust_difficulty(self):
        old = self._difficulty
        # Simple linear increase for mock
        self._difficulty = int(self._difficulty * DIFFICULTY_ADJUST_FACTOR)
        self._difficulty = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, self._difficulty))
        if self._difficulty != old:
            logger.info("Difficulty adjusted: %d -> %d (at block %d)", old, self._difficulty, len(self._blocks))

    # -------------------------------------------------------------------
    # FastAPI route registration
    # -------------------------------------------------------------------

    def register_routes(self, app):
        """Register XenBlocks RPC endpoints on an existing FastAPI app."""

        @app.get("/difficulty")
        async def get_difficulty():
            # Match real XenBlocks: returns difficulty as string
            return {"difficulty": str(self._difficulty)}

        @app.post("/verify")
        async def verify_block(payload: dict):
            hash_to_verify = payload.get("hash_to_verify", "")
            key = payload.get("key", "")
            account = payload.get("account", "")
            attempts = payload.get("attempts", "0")
            hashes_per_second = payload.get("hashes_per_second", "0.0")
            worker = payload.get("worker", "")

            # Parse attempts (C++ sends as string)
            try:
                attempts_int = int(attempts)
            except (ValueError, TypeError):
                attempts_int = 0

            success, message, block_type = self.submit_block(
                hash_to_verify=hash_to_verify,
                key=key,
                account=account,
                attempts=attempts_int,
                hashes_per_second=str(hashes_per_second),
                worker=worker,
            )

            if success:
                return {
                    "status": "success",
                    "message": message,
                    "block_type": block_type,
                    "block_id": self._next_block_id - 1,
                }
            else:
                # Real server returns various error strings the C++ client checks for
                from fastapi.responses import JSONResponse
                status_code = 500
                if "already exists" in message:
                    status_code = 409
                elif "outside of time window" in message.lower() or "time window" in message.lower():
                    status_code = 400
                return JSONResponse(
                    status_code=status_code,
                    content={"status": "error", "message": message},
                )

        @app.get("/getblocks/lastblock")
        async def get_last_blocks():
            return self.get_last_blocks(100)

        @app.post("/send_pow")
        async def send_pow(payload: dict):
            # PoW submission - always accept in mock
            return {
                "status": "success",
                "message": "PoW accepted (mock)",
                "block_id": payload.get("block_id", 0),
            }

        @app.get("/balance/{address}")
        async def get_balance(address: str):
            return {
                "address": address,
                "balance": self.get_balance(address),
            }

        # --- Extra endpoints for the mock platform ---

        @app.get("/chain/stats")
        async def chain_stats():
            return self.get_stats()

        @app.get("/chain/blocks")
        async def chain_blocks(
            account: Optional[str] = None,
            prefix: Optional[str] = None,
            limit: int = 100,
        ):
            if account:
                return self.get_blocks_by_account(account, limit)
            if prefix:
                return self.get_blocks_by_key_prefix(prefix, limit)
            return self.get_last_blocks(limit)

        logger.info("Chain simulator routes registered on FastAPI app")


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)-10s] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="XenBlocks Chain Simulator (standalone)")
    parser.add_argument("--port", type=int, default=8545, help="HTTP port (default: 8545)")
    parser.add_argument("--difficulty", type=int, default=DEFAULT_DIFFICULTY, help=f"Base difficulty (default: {DEFAULT_DIFFICULTY})")
    parser.add_argument("--adjust-interval", type=int, default=DIFFICULTY_ADJUST_INTERVAL, help=f"Blocks between difficulty adjustments (default: {DIFFICULTY_ADJUST_INTERVAL})")
    args = parser.parse_args()

    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        print("ERROR: FastAPI and uvicorn are required.")
        sys.exit(1)

    app = FastAPI(title="XenBlocks Chain Simulator", version="0.1.0")
    chain = ChainSimulator(
        base_difficulty=args.difficulty,
        difficulty_adjust_interval=args.adjust_interval,
    )
    chain.register_routes(app)

    @app.get("/")
    async def root():
        stats = chain.get_stats()
        stats["service"] = "XenBlocks Chain Simulator"
        stats["port"] = args.port
        return stats

    logger.info("=" * 50)
    logger.info("  XenBlocks Chain Simulator")
    logger.info("  Port: %d", args.port)
    logger.info("  Difficulty: %d", args.difficulty)
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
