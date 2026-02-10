"""Pydantic request models for the REST API."""

from typing import Optional
from pydantic import BaseModel


class RentRequest(BaseModel):
    consumer_id: str
    consumer_address: str
    duration_sec: int = 3600
    worker_id: Optional[str] = None


class StopRequest(BaseModel):
    lease_id: str


class DepositRequest(BaseModel):
    amount: float


class PricingRequest(BaseModel):
    price_per_min: float
    min_duration_sec: int = 60
    max_duration_sec: int = 86400


class ControlRequest(BaseModel):
    action: str = "set_config"
    config: dict = {}


class RegisterRequest(BaseModel):
    account_id: str
    role: str
    eth_address: str = ""
    balance: float = 0.0


class LoginRequest(BaseModel):
    account_id: str
