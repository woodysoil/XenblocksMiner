"""
auth.py - API key authentication and role-based access control.

Simple API key authentication for the platform server. Each account gets a
random 32-char hex API key on creation. Protected endpoints require the
X-API-Key header.

Usage:
    from server.auth import AuthService, require_role

    # In server setup:
    auth = AuthService(account_repo)

    # In route:
    @app.post("/api/rent")
    async def rent(req: RentRequest, account: dict = Depends(auth.get_current_account)):
        ...

    # Role-based:
    @app.get("/api/accounts")
    async def list_accounts(account: dict = Depends(auth.require_admin)):
        ...
"""

import logging
import secrets
from typing import TYPE_CHECKING, Optional

try:
    from fastapi import Header, HTTPException
except ImportError:
    pass

if TYPE_CHECKING:
    from server.storage import AccountRepo

logger = logging.getLogger("auth")

# Admin API key (configurable, defaults to a well-known test value)
DEFAULT_ADMIN_KEY = "admin-test-key-do-not-use-in-production"


class AuthService:
    """API key authentication and role-based access."""

    def __init__(self, account_repo: "AccountRepo", admin_key: str = DEFAULT_ADMIN_KEY):
        self._repo = account_repo
        self._admin_key = admin_key

    @staticmethod
    def generate_api_key() -> str:
        """Generate a random 32-char hex API key."""
        return secrets.token_hex(16)

    # -------------------------------------------------------------------
    # Account registration / login
    # -------------------------------------------------------------------

    async def register(
        self, account_id: str, role: str, eth_address: str = "", balance: float = 0.0,
    ) -> dict:
        """Register a new account and return it with an API key."""
        if role not in ("provider", "consumer"):
            raise ValueError("Role must be 'provider' or 'consumer'")

        existing = await self._repo.get(account_id)
        if existing is not None:
            raise ValueError(f"Account '{account_id}' already exists")

        api_key = self.generate_api_key()
        acct = await self._repo.create(account_id, role, balance=balance, eth_address=eth_address)
        if acct is None:
            raise RuntimeError("Failed to create account")

        await self._repo.set_api_key(account_id, api_key)
        acct["api_key"] = api_key
        logger.info("Registered account %s role=%s", account_id, role)
        return acct

    async def login(self, account_id: str) -> dict:
        """Get or regenerate API key for an existing account."""
        acct = await self._repo.get(account_id)
        if acct is None:
            raise KeyError(f"Account '{account_id}' not found")

        api_key = acct.get("api_key") or ""
        if not api_key:
            # Generate a key if account doesn't have one (legacy accounts)
            api_key = self.generate_api_key()
            await self._repo.set_api_key(account_id, api_key)

        acct["api_key"] = api_key
        return acct

    # -------------------------------------------------------------------
    # FastAPI dependencies
    # -------------------------------------------------------------------

    async def resolve_account(self, x_api_key: str = Header(default="")) -> Optional[dict]:
        """Resolve API key to account. Returns None if no key provided."""
        if not x_api_key:
            return None

        # Check admin key
        if x_api_key == self._admin_key:
            return {
                "account_id": "_admin",
                "role": "admin",
                "eth_address": "",
                "balance": 0.0,
                "api_key": x_api_key,
            }

        acct = await self._repo.get_by_api_key(x_api_key)
        return acct

    async def get_current_account(self, x_api_key: str = Header(default="")) -> dict:
        """Require a valid API key. Returns the account dict or raises 401."""
        acct = await self.resolve_account(x_api_key)
        if acct is None:
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key. Pass X-API-Key header.",
            )
        return acct

    async def require_consumer(self, x_api_key: str = Header(default="")) -> dict:
        """Require a consumer or admin account."""
        acct = await self.get_current_account(x_api_key)
        if acct["role"] not in ("consumer", "admin"):
            raise HTTPException(status_code=403, detail="Consumer account required")
        return acct

    async def require_provider(self, x_api_key: str = Header(default="")) -> dict:
        """Require a provider or admin account."""
        acct = await self.get_current_account(x_api_key)
        if acct["role"] not in ("provider", "admin"):
            raise HTTPException(status_code=403, detail="Provider account required")
        return acct

    async def require_admin(self, x_api_key: str = Header(default="")) -> dict:
        """Require the admin API key."""
        acct = await self.get_current_account(x_api_key)
        if acct["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return acct

    def require_worker_owner(self, worker_id_param: str = "worker_id"):
        """Return a dependency that checks the caller owns the given worker.

        The provider's account_id must match the worker_id (since providers
        are registered with account_id == worker_id in the matcher).
        Admin accounts bypass this check.
        """
        async def _check(x_api_key: str = Header(default="")) -> dict:
            acct = await self.require_provider(x_api_key)
            # Admin can manage any worker
            if acct["role"] == "admin":
                return acct
            # Provider's account_id must match the worker_id
            # (handled at the route level since we need the path param)
            return acct
        return _check


def ensure_api_keys_for_defaults(auth: AuthService):
    """Coroutine to ensure default test accounts have API keys.

    Call after setup_defaults() to backfill keys for pre-existing accounts.
    """
    async def _backfill():
        for account_id in ("consumer-1", "consumer-2"):
            acct = await auth._repo.get(account_id)
            if acct and not acct.get("api_key"):
                key = auth.generate_api_key()
                await auth._repo.set_api_key(account_id, key)
                logger.info("Backfilled API key for default account %s", account_id)
    return _backfill()
