"""Account router — /api/auth/*, /api/accounts/{id}/* endpoints."""

from fastapi import APIRouter, Header, HTTPException
from starlette.requests import Request

from server.deps import get_server
from server.models import RegisterRequest, LoginRequest, DepositRequest

router = APIRouter()


@router.post("/api/auth/register")
async def auth_register(request: Request, req: RegisterRequest):
    srv = get_server(request)
    try:
        acct = await srv.auth.register(
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


@router.post("/api/auth/login")
async def auth_login(request: Request, req: LoginRequest):
    srv = get_server(request)
    try:
        acct = await srv.auth.login(req.account_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "account_id": acct["account_id"],
        "role": acct["role"],
        "api_key": acct["api_key"],
    }


@router.get("/api/auth/me")
async def auth_me(request: Request, x_api_key: str = Header(default="")):
    srv = get_server(request)
    acct = await srv.auth.resolve_account(x_api_key)
    if acct is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return {
        "account_id": acct["account_id"],
        "role": acct["role"],
        "eth_address": acct.get("eth_address", ""),
        "balance": acct.get("balance", 0.0),
    }


@router.get("/api/accounts/{account_id}/balance")
async def get_balance(request: Request, account_id: str, x_api_key: str = Header(default="")):
    srv = get_server(request)
    caller = None
    if srv.auth and x_api_key:
        caller = await srv.auth.resolve_account(x_api_key)
    if caller and caller["role"] != "admin" and caller["account_id"] != account_id:
        raise HTTPException(status_code=403, detail="You can only view your own balance")
    acct = await srv.accounts.get_account(account_id)
    if acct is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return {
        "account_id": acct["account_id"],
        "role": acct["role"],
        "balance": acct["balance"],
        "eth_address": acct["eth_address"],
    }


@router.post("/api/accounts/{account_id}/deposit")
async def deposit(request: Request, account_id: str, req: DepositRequest, x_api_key: str = Header(default="")):
    srv = get_server(request)
    caller = None
    if srv.auth and x_api_key:
        caller = await srv.auth.resolve_account(x_api_key)
    if caller and caller["role"] != "admin" and caller["account_id"] != account_id:
        raise HTTPException(status_code=403, detail="You can only deposit to your own account")
    try:
        acct = await srv.accounts.deposit(account_id, req.amount)
    except KeyError:
        raise HTTPException(status_code=404, detail="Account not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "account_id": acct["account_id"],
        "balance": acct["balance"],
    }
