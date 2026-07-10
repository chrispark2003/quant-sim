"""Order types for the simulated execution engine.

These are pure data objects -- creating one does not submit anything
anywhere. Only execution/simulator.py's ExecutionSimulator turns an Order
into a simulated Fill against the virtual ledger.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    asset_class: str = "equity"
    rationale: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    strategy: str = "unassigned"

    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("stop orders require stop_price")
        if self.quantity == 0:
            raise ValueError("order quantity must be nonzero")


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    fees: float
    slippage: float
    timestamp: datetime
    rationale: str
    strategy: str = "unassigned"
