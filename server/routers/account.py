"""Account router — /api/auth/*, /api/accounts/{id}/* endpoints."""

from fastapi import APIRouter, Header, HTTPException, Query
from starlette.requests import Request

from server.auth import SIGN_MESSAGE_TEMPLATE
from server.deps import get_server
from server.models import RegisterRequest, LoginRequest, DepositRequest, WithdrawRequest, WalletVerifyRequest

router = APIRouter()


@router.get("/api/auth/nonce")
async def auth_nonce(request: Request, address: str = Query(...)):
    srv = get_server(request)
    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address")
    nonce = srv.auth.generate_nonce(address)
    return {
        "nonce": nonce,
        "message": SIGN_MESSAGE_TEMPLATE.format(nonce=nonce),
    }


@router.post("/api/auth/verify")
async def auth_verify(request: Request, req: WalletVerifyRequest):
    srv = get_server(request)
    if not srv.auth.verify_signature(req.address, req.signature, req.nonce):
        raise HTTPException(status_code=401, detail="Invalid signature or expired nonce")
    acct = await srv.accounts.get_or_create_by_eth_address(req.address)
    if acct is None:
        raise HTTPException(status_code=500, detail="Failed to create account")
    token = srv.auth.issue_jwt(req.address, acct["role"], acct["account_id"])
    return {
        "token": token,
        "address": req.address,
        "account_id": acct["account_id"],
        "role": acct["role"],
    }


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
async def auth_me(
    request: Request,
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    acct = await srv.auth.resolve_account(x_api_key, authorization)
    if acct is None:
        raise HTTPException(status_code=401, detail="Invalid or missing credentials")
    return {
        "account_id": acct["account_id"],
        "role": acct["role"],
        "eth_address": acct.get("eth_address", ""),
        "balance": acct.get("balance", 0.0),
    }


@router.get("/api/accounts/{account_id}/balance")
async def get_balance(
    request: Request,
    account_id: str,
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    caller = None
    if srv.auth and (x_api_key or authorization):
        caller = await srv.auth.resolve_account(x_api_key, authorization)
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
async def deposit(
    request: Request,
    account_id: str,
    req: DepositRequest,
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    caller = None
    if srv.auth and (x_api_key or authorization):
        caller = await srv.auth.resolve_account(x_api_key, authorization)
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


@router.post("/api/accounts/{account_id}/withdraw")
async def withdraw(
    request: Request,
    account_id: str,
    req: WithdrawRequest,
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    srv = get_server(request)
    caller = await srv.auth.get_current_account(x_api_key=x_api_key, authorization=authorization)
    if caller["role"] != "admin" and caller["account_id"] != account_id:
        raise HTTPException(status_code=403, detail="You can only withdraw from your own account")
    try:
        acct = await srv.accounts.withdraw(account_id, req.amount)
    except KeyError:
        raise HTTPException(status_code=404, detail="Account not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "account_id": acct["account_id"],
        "balance": acct["balance"],
        "withdrawn": req.amount,
    }
