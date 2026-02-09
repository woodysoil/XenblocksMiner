"""
server.py - Mock platform server entry point.

Single-process server combining:
 - Embedded MQTT broker (pure-Python async, port 1883)
 - SQLite persistent storage via StorageManager
 - Platform services (account, matcher, watcher, settlement)
 - REST control API (FastAPI on uvicorn, port 8080)

Usage:
    python -m server.server [--mqtt-port 1883] [--api-port 8080] [--db-path data/marketplace.db]
    python server/server.py [--mqtt-port 1883] [--api-port 8080] [--db-path data/marketplace.db]
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so imports work both ways
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: FastAPI and uvicorn are required. Install with:")
    print("  pip install fastapi uvicorn pydantic")
    sys.exit(1)

from server.account import AccountService
from server.auth import AuthService, ensure_api_keys_for_defaults
from server.broker import MQTTBroker
from server.chain_simulator import ChainSimulator
from server.dashboard import register_dashboard
from server.matcher import MatchingEngine
from server.pricing import PricingEngine
from server.reputation import ReputationEngine
from server.settlement import SettlementEngine
from server.storage import StorageManager
from server.watcher import BlockWatcher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-10s] %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Pydantic models for REST API
# ---------------------------------------------------------------------------

class RentRequest(BaseModel):
    """Request body for renting hashpower from a worker."""

    consumer_id: str
    consumer_address: str
    duration_sec: int = 3600
    worker_id: Optional[str] = None

class StopRequest(BaseModel):
    """Request body for stopping an active lease."""

    lease_id: str

class DepositRequest(BaseModel):
    """Request body for depositing funds into an account."""

    amount: float

class PricingRequest(BaseModel):
    """Request body for setting worker pricing parameters."""

    price_per_min: float
    min_duration_sec: int = 60
    max_duration_sec: int = 86400

class RegisterRequest(BaseModel):
    """Request body for registering a new account."""

    account_id: str
    role: str  # "provider" or "consumer"
    eth_address: str = ""
    balance: float = 0.0

class LoginRequest(BaseModel):
    """Request body for logging in to an existing account."""

    account_id: str

# ---------------------------------------------------------------------------
# MQTT topic parser
# ---------------------------------------------------------------------------

_TOPIC_RE = re.compile(r"^xenminer/([^/]+)/(\w+)$")


def parse_topic(topic: str):
    """Parse an MQTT topic into (worker_id, message_type), or (None, None) if invalid."""
    m = _TOPIC_RE.match(topic)
    if m:
        return m.group(1), m.group(2)
    return None, None

# ---------------------------------------------------------------------------
# Platform server
# ---------------------------------------------------------------------------

class PlatformServer:
    """Single-process mock platform server combining MQTT broker, REST API, and services."""

    def __init__(self, mqtt_port: int = 1883, api_port: int = 8080, db_path: str = "data/marketplace.db",
                 enable_chain: bool = True, block_marker: str = ""):
        self.mqtt_port = mqtt_port
        self.api_port = api_port
        self.db_path = db_path
        self._block_marker = block_marker

        # Broker (created immediately, no async needed)
        self.broker = MQTTBroker(port=mqtt_port)

        # Storage + services are initialized async in start()
        self.storage: Optional[StorageManager] = None
        self.accounts: Optional[AccountService] = None
        self.matcher: Optional[MatchingEngine] = None
        self.watcher: Optional[BlockWatcher] = None
        self.settlement: Optional[SettlementEngine] = None

        # Pricing engine
        self.pricing: Optional[PricingEngine] = None

        # Reputation engine
        self.reputation: Optional[ReputationEngine] = None

        # Auth service
        self.auth: Optional[AuthService] = None

        # Chain simulator (embedded, provides /difficulty, /verify, etc.)
        self.chain: Optional[ChainSimulator] = None
        self._enable_chain = enable_chain

        # FastAPI app
        self.app = FastAPI(title="XenMiner Mock Platform", version="0.3.0")
        self._register_routes()
        register_dashboard(self.app)

        # Embed chain simulator routes on the same app
        if self._enable_chain:
            self.chain = ChainSimulator(block_marker=self._block_marker)
            self.chain.register_routes(self.app)
            logger.info("Chain simulator embedded on platform server")

        # Register MQTT message handler
        self.broker.on_message(self._on_mqtt_message)

    async def _init_services(self):
        """Initialize storage and wire up services (must be called in async context)."""
        # Ensure data directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.storage = StorageManager(self.db_path)
        await self.storage.initialize()

        self.accounts = AccountService(self.storage.accounts)
        await self.accounts.setup_defaults()

        self.matcher = MatchingEngine(
            self.broker, self.accounts,
            self.storage.workers, self.storage.leases,
        )
        self.watcher = BlockWatcher(self.storage.blocks, self.storage.leases, chain=self.chain)
        self.settlement = SettlementEngine(self.accounts, self.storage.settlements)
        await self.settlement.setup_defaults()
        self.pricing = PricingEngine(self.storage.workers)
        self.reputation = ReputationEngine(
            self.storage.workers, self.storage.leases, self.storage.blocks,
        )
        self.auth = AuthService(self.storage.accounts)
        await ensure_api_keys_for_defaults(self.auth)

        logger.info("Services initialized (db=%s)", self.db_path)

    # -------------------------------------------------------------------
    # MQTT message dispatcher
    # -------------------------------------------------------------------

    async def _on_mqtt_message(self, topic: str, payload: dict, client_id: str):
        worker_id, msg_type = parse_topic(topic)
        if worker_id is None:
            return

        if msg_type == "register":
            await self.matcher.register_worker(payload)
        elif msg_type == "heartbeat":
            await self.matcher.update_heartbeat(payload)
        elif msg_type == "status":
            await self.matcher.update_worker_state(payload)
        elif msg_type == "block":
            await self.watcher.handle_block_found(payload, worker_id)
        else:
            logger.debug("Unhandled topic: %s", topic)

    # -------------------------------------------------------------------
    # Lease watchdog
    # -------------------------------------------------------------------

    async def _lease_watchdog(self):
        """Periodically check for expired leases and settle them."""
        while True:
            try:
                expired = await self.matcher.check_expired_leases()
                for lease in expired:
                    await self.settlement.settle_lease(lease)
            except Exception:
                logger.exception("Error in lease watchdog")
            await asyncio.sleep(5)

    # -------------------------------------------------------------------
    # FastAPI routes
    # -------------------------------------------------------------------

    def _register_routes(self):
        app = self.app

        # --- Auth helper: optional auth (returns None if no key) ---
        async def optional_account(x_api_key: str = Header(default="")) -> Optional[dict]:
            if self.auth is None or not x_api_key:
                return None
            return await self.auth.resolve_account(x_api_key)

        # --- Public endpoints (no auth required) ---

        @app.get("/")
        async def root():
            return {
                "service": "XenMiner Mock Platform",
                "mqtt_port": self.mqtt_port,
                "api_port": self.api_port,
                "connected_workers": len(self.broker.connected_client_ids),
                "uptime": "running",
            }

        @app.get("/api/workers")
        async def list_workers():
            return await self.matcher.get_available_workers()

        @app.get("/api/marketplace")
        async def browse_marketplace(
            sort_by: str = "price",
            gpu_type: Optional[str] = None,
            min_hashrate: Optional[float] = None,
            max_price: Optional[float] = None,
            min_gpus: Optional[int] = None,
            available_only: bool = True,
        ):
            return await self.pricing.browse_marketplace(
                sort_by=sort_by,
                gpu_type=gpu_type,
                min_hashrate=min_hashrate,
                max_price=max_price,
                min_gpus=min_gpus,
                available_only=available_only,
            )

        @app.get("/api/marketplace/estimate")
        async def estimate_cost(
            duration_sec: int,
            worker_id: Optional[str] = None,
            min_hashrate: Optional[float] = None,
        ):
            result = await self.pricing.estimate_cost(
                duration_sec=duration_sec,
                worker_id=worker_id,
                min_hashrate=min_hashrate,
            )
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return result

        @app.get("/api/workers/{worker_id}/pricing")
        async def get_worker_pricing(worker_id: str):
            result = await self.pricing.get_pricing(worker_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Worker not found")
            return result

        @app.get("/api/workers/{worker_id}/pricing/suggest")
        async def suggest_worker_pricing(worker_id: str):
            result = await self.pricing.suggest_pricing(worker_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Worker not found")
            return result

        @app.get("/api/workers/{worker_id}/reputation")
        async def get_worker_reputation(worker_id: str):
            result = await self.reputation.get_score(worker_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Worker not found")
            return result

        @app.get("/api/leases")
        async def list_leases(state: Optional[str] = None):
            return await self.matcher.list_leases(state=state)

        @app.get("/api/leases/{lease_id}")
        async def get_lease(lease_id: str):
            lease = await self.matcher.get_lease(lease_id)
            if lease is None:
                raise HTTPException(status_code=404, detail="Lease not found")
            result = {
                "lease_id": lease["lease_id"],
                "worker_id": lease["worker_id"],
                "consumer_id": lease["consumer_id"],
                "consumer_address": lease["consumer_address"],
                "prefix": lease["prefix"],
                "duration_sec": lease["duration_sec"],
                "state": lease["state"],
                "created_at": lease["created_at"],
                "ended_at": lease["ended_at"],
                "blocks_found": lease["blocks_found"],
                "avg_hashrate": lease["avg_hashrate"],
                "elapsed_sec": lease["elapsed_sec"],
            }
            blocks = await self.watcher.get_blocks_for_lease(lease_id)
            if blocks:
                result["blocks"] = blocks
            settlement = await self.settlement.get_settlement(lease_id)
            if settlement:
                result["settlement"] = settlement
            return result

        @app.get("/api/blocks")
        async def list_blocks(lease_id: Optional[str] = None):
            if lease_id:
                return await self.watcher.get_blocks_for_lease(lease_id)
            return await self.watcher.get_all_blocks()

        @app.get("/api/status")
        async def server_status():
            return {
                "mqtt_clients": self.broker.connected_client_ids,
                "workers": len(await self.matcher.get_available_workers()),
                "active_leases": len(await self.matcher.list_leases(state="active")),
                "total_blocks": await self.watcher.total_blocks(),
                "total_settlements": len(await self.settlement.list_settlements()),
            }

        # --- Auth endpoints (no auth required) ---

        @app.post("/api/auth/register")
        async def auth_register(req: RegisterRequest):
            try:
                acct = await self.auth.register(
                    account_id=req.account_id,
                    role=req.role,
                    eth_address=req.eth_address,
                    balance=req.balance,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))
            return {
                "account_id": acct["account_id"],
                "role": acct["role"],
                "eth_address": acct["eth_address"],
                "balance": acct["balance"],
                "api_key": acct["api_key"],
            }

        @app.post("/api/auth/login")
        async def auth_login(req: LoginRequest):
            try:
                acct = await self.auth.login(req.account_id)
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return {
                "account_id": acct["account_id"],
                "role": acct["role"],
                "api_key": acct["api_key"],
            }

        @app.get("/api/auth/me")
        async def auth_me(x_api_key: str = Header(default="")):
            acct = await self.auth.resolve_account(x_api_key)
            if acct is None:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
            return {
                "account_id": acct["account_id"],
                "role": acct["role"],
                "eth_address": acct.get("eth_address", ""),
                "balance": acct.get("balance", 0.0),
            }

        # --- Protected consumer endpoints ---

        @app.post("/api/rent")
        async def rent_hashpower(req: RentRequest, x_api_key: str = Header(default="")):
            # Auth is optional for backward compatibility; if key is provided, validate
            caller = await optional_account(x_api_key)
            if caller and caller["role"] not in ("consumer", "admin"):
                raise HTTPException(status_code=403, detail="Consumer account required to rent")
            lease = await self.matcher.rent_hashpower(
                consumer_id=req.consumer_id,
                consumer_address=req.consumer_address,
                duration_sec=req.duration_sec,
                worker_id=req.worker_id,
            )
            if lease is None:
                raise HTTPException(status_code=404, detail="No available workers")
            return {
                "lease_id": lease["lease_id"],
                "worker_id": lease["worker_id"],
                "prefix": lease["prefix"],
                "duration_sec": lease["duration_sec"],
                "consumer_id": lease["consumer_id"],
                "consumer_address": lease["consumer_address"],
                "created_at": lease["created_at"],
            }

        @app.post("/api/stop")
        async def stop_lease(req: StopRequest, x_api_key: str = Header(default="")):
            caller = await optional_account(x_api_key)
            lease = await self.matcher.stop_lease(req.lease_id)
            if lease is None:
                raise HTTPException(status_code=404, detail="Lease not found or not active")
            # If authenticated, verify ownership (consumer or admin)
            if caller and caller["role"] != "admin":
                if caller["role"] == "consumer" and caller["account_id"] != lease["consumer_id"]:
                    raise HTTPException(status_code=403, detail="You can only stop your own leases")
            record = await self.settlement.settle_lease(lease)
            result = {
                "lease_id": lease["lease_id"],
                "state": lease["state"],
                "blocks_found": lease["blocks_found"],
            }
            if record:
                result["settlement"] = record
            return result

        # --- Protected provider endpoints ---

        @app.put("/api/workers/{worker_id}/pricing")
        async def set_worker_pricing(worker_id: str, req: PricingRequest, x_api_key: str = Header(default="")):
            # Auth optional for backward compat; if present, check provider/admin
            caller = await optional_account(x_api_key)
            if caller and caller["role"] not in ("provider", "admin"):
                raise HTTPException(status_code=403, detail="Provider account required")
            if caller and caller["role"] == "provider" and caller["account_id"] != worker_id:
                raise HTTPException(status_code=403, detail="You can only set pricing for your own worker")
            try:
                result = await self.pricing.set_pricing(
                    worker_id=worker_id,
                    price_per_min=req.price_per_min,
                    min_duration_sec=req.min_duration_sec,
                    max_duration_sec=req.max_duration_sec,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if result is None:
                raise HTTPException(status_code=404, detail="Worker not found")
            return {
                "worker_id": result["worker_id"],
                "price_per_min": result["price_per_min"],
                "min_duration_sec": result["min_duration_sec"],
                "max_duration_sec": result["max_duration_sec"],
            }

        # --- Protected account endpoints ---

        @app.get("/api/accounts/{account_id}/balance")
        async def get_balance(account_id: str, x_api_key: str = Header(default="")):
            # Optional auth: if key provided, verify it's the account owner or admin
            caller = await optional_account(x_api_key)
            if caller and caller["role"] != "admin" and caller["account_id"] != account_id:
                raise HTTPException(status_code=403, detail="You can only view your own balance")
            acct = await self.accounts.get_account(account_id)
            if acct is None:
                raise HTTPException(status_code=404, detail="Account not found")
            return {
                "account_id": acct["account_id"],
                "role": acct["role"],
                "balance": acct["balance"],
                "eth_address": acct["eth_address"],
            }

        @app.post("/api/accounts/{account_id}/deposit")
        async def deposit(account_id: str, req: DepositRequest, x_api_key: str = Header(default="")):
            caller = await optional_account(x_api_key)
            if caller and caller["role"] != "admin" and caller["account_id"] != account_id:
                raise HTTPException(status_code=403, detail="You can only deposit to your own account")
            try:
                acct = await self.accounts.deposit(account_id, req.amount)
            except KeyError:
                raise HTTPException(status_code=404, detail="Account not found")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {
                "account_id": acct["account_id"],
                "balance": acct["balance"],
            }

        # --- Admin endpoints ---

        @app.get("/api/accounts")
        async def list_accounts(x_api_key: str = Header(default="")):
            # Optional auth: if key provided, must be admin
            caller = await optional_account(x_api_key)
            if caller and caller["role"] != "admin":
                raise HTTPException(status_code=403, detail="Admin access required")
            accounts = await self.accounts.list_accounts()
            # Strip api_key from response for security
            sanitized = {}
            for k, v in accounts.items():
                sanitized[k] = {
                    "account_id": v["account_id"],
                    "role": v["role"],
                    "eth_address": v["eth_address"],
                    "balance": v["balance"],
                }
            return sanitized

        @app.get("/api/settlements")
        async def list_settlements(x_api_key: str = Header(default="")):
            caller = await optional_account(x_api_key)
            if caller and caller["role"] != "admin":
                raise HTTPException(status_code=403, detail="Admin access required")
            return await self.settlement.list_settlements()

    # -------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------

    async def start(self):
        """Start storage, broker, watchdog, and API server."""
        # Initialize storage and services
        await self._init_services()

        # Start MQTT broker
        await self.broker.start()
        logger.info("MQTT broker started on port %d", self.mqtt_port)

        # Start watchdog
        self._watchdog_task = asyncio.create_task(self._lease_watchdog())

        # Start uvicorn
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.api_port,
            log_level="info",
        )
        self._uvicorn_server = uvicorn.Server(config)
        logger.info("REST API starting on port %d", self.api_port)
        await self._uvicorn_server.serve()

    async def stop(self):
        """Stop all services, broker, and the API server."""
        self._watchdog_task.cancel()
        await self.broker.stop()
        if self.storage:
            await self.storage.close()
        self._uvicorn_server.should_exit = True


def main():
    """CLI entry point for the mock platform server."""
    parser = argparse.ArgumentParser(description="XenMiner Mock Platform Server")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    parser.add_argument("--api-port", type=int, default=8080, help="REST API port (default: 8080)")
    parser.add_argument("--db-path", default="data/marketplace.db", help="SQLite database path (default: data/marketplace.db)")
    parser.add_argument("--no-chain", action="store_true", help="Disable embedded chain simulator")
    parser.add_argument("--block-marker", default="", help="Override block detection marker for testing (default: XEN11)")
    args = parser.parse_args()

    server = PlatformServer(
        mqtt_port=args.mqtt_port, api_port=args.api_port,
        db_path=args.db_path, enable_chain=not args.no_chain,
        block_marker=args.block_marker,
    )

    logger.info("=" * 60)
    logger.info("  XenMiner Mock Platform Server")
    logger.info("  MQTT broker: localhost:%d", args.mqtt_port)
    logger.info("  REST API:    http://localhost:%d", args.api_port)
    logger.info("  Dashboard:   http://localhost:%d/dashboard", args.api_port)
    logger.info("  Database:    %s", args.db_path)
    logger.info("  Chain sim:   %s", "enabled" if not args.no_chain else "disabled")
    if args.block_marker:
        logger.info("  Block marker: %s (test override)", args.block_marker)
    logger.info("=" * 60)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
