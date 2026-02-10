"""
auth.py - Wallet-based (EIP-191) + API key authentication.

Supports two auth flows:
  1. Wallet: GET /api/auth/nonce → sign → POST /api/auth/verify → JWT
  2. Legacy: X-API-Key header (backward compatible)

resolve_account() checks Authorization: Bearer <jwt> first, then X-API-Key.
"""

import logging
import secrets
import time
from typing import TYPE_CHECKING, Optional

try:
    from fastapi import Header, HTTPException
except ImportError:
    pass

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

try:
    from eth_account import Account as EthAccount
    from eth_account.messages import encode_defunct
except ImportError:
    EthAccount = None
    encode_defunct = None

if TYPE_CHECKING:
    from server.storage import AccountRepo

logger = logging.getLogger("auth")

DEFAULT_ADMIN_KEY = "admin-test-key-do-not-use-in-production"
NONCE_TTL = 300  # 5 minutes
JWT_TTL = 86400  # 24 hours
SIGN_MESSAGE_TEMPLATE = (
    "Sign this message to authenticate with XenBlocks.\n\nNonce: {nonce}"
)


class AuthService:
    """Wallet + API key authentication and role-based access."""

    def __init__(
        self,
        account_repo: "AccountRepo",
        admin_key: str = DEFAULT_ADMIN_KEY,
        jwt_secret: str = "",
    ):
        self._repo = account_repo
        self._admin_key = admin_key
        self._jwt_secret = jwt_secret or secrets.token_hex(32)
        if not jwt_secret:
            logger.warning(
                "No --jwt-secret provided; generated ephemeral secret "
                "(JWTs will invalidate on restart)"
            )
        # In-memory nonce store: address -> (nonce, expiry_timestamp)
        self._nonces: dict[str, tuple[str, float]] = {}

    @staticmethod
    def generate_api_key() -> str:
        return secrets.token_hex(16)

    # -------------------------------------------------------------------
    # Nonce / Wallet auth
    # -------------------------------------------------------------------

    def generate_nonce(self, address: str) -> str:
        nonce = secrets.token_hex(16)
        self._nonces[address.lower()] = (nonce, time.time() + NONCE_TTL)
        return nonce

    def verify_signature(self, address: str, signature: str, nonce: str) -> bool:
        addr_lower = address.lower()
        stored = self._nonces.get(addr_lower)
        if stored is None or stored[0] != nonce:
            return False
        if time.time() > stored[1]:
            self._nonces.pop(addr_lower, None)
            return False

        if EthAccount is None or encode_defunct is None:
            raise RuntimeError("eth-account not installed")

        message_text = SIGN_MESSAGE_TEMPLATE.format(nonce=nonce)
        msg = encode_defunct(text=message_text)
        try:
            recovered = EthAccount.recover_message(msg, signature=signature)
        except Exception:
            return False

        if recovered.lower() != addr_lower:
            return False

        # Consume nonce
        self._nonces.pop(addr_lower, None)
        return True

    def issue_jwt(self, address: str, role: str, account_id: str) -> str:
        if pyjwt is None:
            raise RuntimeError("PyJWT not installed")
        payload = {
            "sub": address.lower(),
            "role": role,
            "account_id": account_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + JWT_TTL,
        }
        return pyjwt.encode(payload, self._jwt_secret, algorithm="HS256")

    def decode_jwt(self, token: str) -> Optional[dict]:
        if pyjwt is None:
            return None
        try:
            return pyjwt.decode(token, self._jwt_secret, algorithms=["HS256"])
        except pyjwt.ExpiredSignatureError:
            return None
        except pyjwt.InvalidTokenError:
            return None

    # -------------------------------------------------------------------
    # Account registration / login (legacy API key flow)
    # -------------------------------------------------------------------

    async def register(
        self, account_id: str, role: str, eth_address: str = "", balance: float = 0.0,
    ) -> dict:
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
        acct = await self._repo.get(account_id)
        if acct is None:
            raise KeyError(f"Account '{account_id}' not found")
        api_key = acct.get("api_key") or ""
        if not api_key:
            api_key = self.generate_api_key()
            await self._repo.set_api_key(account_id, api_key)
        acct["api_key"] = api_key
        return acct

    # -------------------------------------------------------------------
    # FastAPI dependencies
    # -------------------------------------------------------------------

    async def resolve_account(
        self,
        x_api_key: str = Header(default=""),
        authorization: str = Header(default=""),
    ) -> Optional[dict]:
        """Resolve JWT or API key to account. Returns None if no credentials."""
        # Try JWT first
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            claims = self.decode_jwt(token)
            if claims:
                acct = await self._repo.get(claims.get("account_id", ""))
                if acct:
                    return acct

        if not x_api_key:
            return None

        # Admin key
        if x_api_key == self._admin_key:
            return {
                "account_id": "_admin",
                "role": "admin",
                "eth_address": "",
                "balance": 0.0,
                "api_key": x_api_key,
            }

        return await self._repo.get_by_api_key(x_api_key)

    async def get_current_account(
        self,
        x_api_key: str = Header(default=""),
        authorization: str = Header(default=""),
    ) -> dict:
        acct = await self.resolve_account(x_api_key, authorization)
        if acct is None:
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid credentials. Pass Authorization: Bearer <jwt> or X-API-Key header.",
            )
        return acct

    async def require_consumer(
        self,
        x_api_key: str = Header(default=""),
        authorization: str = Header(default=""),
    ) -> dict:
        acct = await self.get_current_account(x_api_key, authorization)
        if acct["role"] not in ("consumer", "admin"):
            raise HTTPException(status_code=403, detail="Consumer account required")
        return acct

    async def require_provider(
        self,
        x_api_key: str = Header(default=""),
        authorization: str = Header(default=""),
    ) -> dict:
        acct = await self.get_current_account(x_api_key, authorization)
        if acct["role"] not in ("provider", "admin"):
            raise HTTPException(status_code=403, detail="Provider account required")
        return acct

    async def require_admin(
        self,
        x_api_key: str = Header(default=""),
        authorization: str = Header(default=""),
    ) -> dict:
        acct = await self.get_current_account(x_api_key, authorization)
        if acct["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return acct

    def require_worker_owner(self, worker_id_param: str = "worker_id"):
        async def _check(
            x_api_key: str = Header(default=""),
            authorization: str = Header(default=""),
        ) -> dict:
            acct = await self.require_provider(x_api_key, authorization)
            if acct["role"] == "admin":
                return acct
            return acct
        return _check


def ensure_api_keys_for_defaults(auth: AuthService):
    async def _backfill():
        for account_id in ("consumer-1", "consumer-2"):
            acct = await auth._repo.get(account_id)
            if acct and not acct.get("api_key"):
                key = auth.generate_api_key()
                await auth._repo.set_api_key(account_id, key)
                logger.info("Backfilled API key for default account %s", account_id)
    return _backfill()
