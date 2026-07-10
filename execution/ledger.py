"""Virtual ledger: cash balance, open positions, realized/unrealized P&L,
and the full trade blotter. This is the ONLY place "money" exists in this
project -- it is a purely simulated bookkeeping object with no connection
to any real account.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from execution.orders import Fill, OrderSide

DISCLAIMER = "SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE"


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_entry_price) * self.quantity

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price


@dataclass
class BlotterEntry:
    timestamp: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fees: float
    slippage: float
    realized_pnl: float
    rationale: str
    strategy: str
    cash_after: float


class Ledger:
    """Tracks cash, positions, and the trade blotter. Persisted to disk as
    JSON so the live loop can resume state across restarts without
    re-entering or double-submitting positions."""

    def __init__(self, starting_cash: float = 100000.0, state_path: str | Path | None = None):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.blotter: list[BlotterEntry] = []
        self.equity_history: list[dict] = []
        self.state_path = Path(state_path) if state_path else None

    def apply_fill(self, fill: Fill) -> BlotterEntry:
        pos = self.positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        trade_cost = signed_qty * fill.fill_price
        self.cash -= trade_cost
        self.cash -= fill.fees

        realized_this_fill = 0.0
        if pos.quantity == 0 or (pos.quantity > 0) == (signed_qty > 0):
            # Adding to (or opening) a position in the same direction: blend entry price.
            new_qty = pos.quantity + signed_qty
            if new_qty != 0:
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.quantity + fill.fill_price * signed_qty) / new_qty
                )
            pos.quantity = new_qty
        else:
            # Reducing or flipping a position: realize P&L on the closed portion.
            closing_qty = min(abs(signed_qty), abs(pos.quantity))
            direction = 1 if pos.quantity > 0 else -1
            realized_this_fill = (fill.fill_price - pos.avg_entry_price) * closing_qty * direction
            pos.realized_pnl += realized_this_fill

            remaining = pos.quantity + signed_qty
            if abs(signed_qty) > abs(pos.quantity):
                # Flipped through zero into the opposite direction.
                pos.avg_entry_price = fill.fill_price
            pos.quantity = remaining

        entry = BlotterEntry(
            timestamp=fill.timestamp.isoformat() if isinstance(fill.timestamp, datetime) else str(fill.timestamp),
            symbol=fill.symbol,
            side=fill.side.value if hasattr(fill.side, "value") else str(fill.side),
            quantity=fill.quantity,
            fill_price=fill.fill_price,
            fees=fill.fees,
            slippage=fill.slippage,
            realized_pnl=realized_this_fill,
            rationale=fill.rationale,
            strategy=fill.strategy,
            cash_after=self.cash,
        )
        self.blotter.append(entry)
        return entry

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def total_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        return sum(
            p.unrealized_pnl(current_prices.get(sym, p.avg_entry_price))
            for sym, p in self.positions.items() if p.quantity != 0
        )

    def equity(self, current_prices: dict[str, float]) -> float:
        market_value = sum(
            p.market_value(current_prices.get(sym, p.avg_entry_price))
            for sym, p in self.positions.items()
        )
        return self.cash + market_value

    def snapshot_equity(self, timestamp, current_prices: dict[str, float]) -> None:
        eq = self.equity(current_prices)
        self.equity_history.append({
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            "equity": eq,
            "cash": self.cash,
            "realized_pnl": self.total_realized_pnl(),
            "unrealized_pnl": self.total_unrealized_pnl(current_prices),
        })

    def open_positions(self, current_prices: dict[str, float]) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_entry_price": p.avg_entry_price,
                "current_price": current_prices.get(p.symbol, p.avg_entry_price),
                "unrealized_pnl": p.unrealized_pnl(current_prices.get(p.symbol, p.avg_entry_price)),
                "market_value": p.market_value(current_prices.get(p.symbol, p.avg_entry_price)),
            }
            for p in self.positions.values() if p.quantity != 0
        ]

    def position_notional_map(self, current_prices: dict[str, float]) -> dict[str, float]:
        return {
            sym: p.market_value(current_prices.get(sym, p.avg_entry_price))
            for sym, p in self.positions.items() if p.quantity != 0
        }

    def blotter_df(self) -> pd.DataFrame:
        if not self.blotter:
            return pd.DataFrame(columns=[f.name for f in BlotterEntry.__dataclass_fields__.values()])
        return pd.DataFrame([asdict(b) for b in self.blotter])

    def equity_curve(self) -> pd.Series:
        if not self.equity_history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")["equity"]

    # --- persistence ---

    def to_dict(self) -> dict:
        return {
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "blotter": [asdict(b) for b in self.blotter],
            "equity_history": self.equity_history,
        }

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.state_path
        if target is None:
            raise ValueError("no state_path configured for this ledger")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        ledger = cls(starting_cash=data["starting_cash"], state_path=path)
        ledger.cash = data["cash"]
        ledger.positions = {s: Position(**p) for s, p in data["positions"].items()}
        ledger.blotter = [BlotterEntry(**b) for b in data["blotter"]]
        ledger.equity_history = data["equity_history"]
        return ledger

    @classmethod
    def load_or_create(cls, path: str | Path, starting_cash: float = 100000.0) -> "Ledger":
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls(starting_cash=starting_cash, state_path=path)
