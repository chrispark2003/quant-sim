"""Simulated execution engine: turns Orders into Fills against synthetic bar
data, applying slippage, transaction costs, volume-participation caps, and
an optional signal-to-fill latency model.

No real brokerage or exchange is ever contacted here -- this module only
computes hypothetical fill prices from historical/simulated OHLCV bars and
hands them to the ledger.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from execution.orders import Fill, Order, OrderSide, OrderStatus, OrderType


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class ExecutionSimulator:
    def __init__(self, config: dict):
        self.equities_cfg = config.get("equities", {})
        self.crypto_cfg = config.get("crypto", {})
        self.latency_cfg = config.get("latency", {"enabled": False})

    def _cost_config(self, asset_class: str) -> dict:
        return self.crypto_cfg if asset_class == "crypto" else self.equities_cfg

    def _slippage_bps(self, asset_class: str) -> float:
        return self._cost_config(asset_class).get("slippage_bps", 5)

    def _fees(self, asset_class: str, quantity: float, fill_price: float) -> float:
        cfg = self._cost_config(asset_class)
        notional = abs(quantity) * fill_price
        if asset_class == "crypto":
            return notional * cfg.get("taker_fee_bps", 10.0) / 10000.0
        per_share = cfg.get("commission_per_share", 0.005) * abs(quantity)
        bps_fee = notional * cfg.get("taker_fee_bps", 1.0) / 10000.0
        commission = max(per_share, cfg.get("commission_min", 1.0))
        return commission + bps_fee

    def simulated_latency(self) -> timedelta:
        if not self.latency_cfg.get("enabled", False):
            return timedelta(0)
        mean_ms = self.latency_cfg.get("mean_ms", 250)
        std_ms = self.latency_cfg.get("std_ms", 75)
        ms = max(0.0, random.gauss(mean_ms, std_ms))
        return timedelta(milliseconds=ms)

    def _apply_slippage(self, price: float, side: OrderSide, asset_class: str) -> tuple[float, float]:
        bps = self._slippage_bps(asset_class)
        direction = 1 if side == OrderSide.BUY else -1
        slip_amount = price * (bps / 10000.0) * direction
        return price + slip_amount, abs(slip_amount)

    def _volume_cap_quantity(self, requested_qty: float, bar: Bar, asset_class: str) -> float:
        cfg = self._cost_config(asset_class)
        cap_pct = cfg.get("volume_participation_cap", None)
        if not cap_pct or bar.volume <= 0:
            return requested_qty
        max_qty = bar.volume * cap_pct
        if abs(requested_qty) > max_qty:
            return max_qty * np.sign(requested_qty)
        return requested_qty

    def fill_market_order(self, order: Order, next_bar: Bar) -> Fill | None:
        """Market orders fill at next-bar open plus slippage, with the fill
        size capped by the volume-participation limit."""
        filled_qty = self._volume_cap_quantity(order.quantity, next_bar, order.asset_class)
        if filled_qty == 0:
            return None

        fill_price, slippage = self._apply_slippage(next_bar.open, order.side, order.asset_class)
        fees = self._fees(order.asset_class, filled_qty, fill_price)

        if abs(filled_qty) < abs(order.quantity):
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.FILLED

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=filled_qty,
            fill_price=fill_price,
            fees=fees,
            slippage=slippage,
            timestamp=next_bar.timestamp,
            rationale=order.rationale,
            strategy=order.strategy,
        )

    def fill_limit_order(self, order: Order, bar: Bar) -> Fill | None:
        """Limit orders fill only if the bar's range trades through the
        limit price (buy limit fills if low <= limit; sell limit fills if
        high >= limit), at the limit price (no favorable slippage assumed,
        conservative fill-at-limit)."""
        if order.side == OrderSide.BUY and bar.low > order.limit_price:
            return None
        if order.side == OrderSide.SELL and bar.high < order.limit_price:
            return None

        filled_qty = self._volume_cap_quantity(order.quantity, bar, order.asset_class)
        if filled_qty == 0:
            return None

        fill_price = order.limit_price
        fees = self._fees(order.asset_class, filled_qty, fill_price)
        order.status = OrderStatus.FILLED if abs(filled_qty) >= abs(order.quantity) else OrderStatus.PARTIALLY_FILLED

        return Fill(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            quantity=filled_qty, fill_price=fill_price, fees=fees, slippage=0.0,
            timestamp=bar.timestamp, rationale=order.rationale, strategy=order.strategy,
        )

    def fill_stop_order(self, order: Order, bar: Bar) -> Fill | None:
        """Stop orders trigger when the bar's range touches the stop price,
        then fill (with slippage) at the stop price -- simulating a stop
        converting to a market order at trigger."""
        triggered = (
            (order.side == OrderSide.SELL and bar.low <= order.stop_price) or
            (order.side == OrderSide.BUY and bar.high >= order.stop_price)
        )
        if not triggered:
            return None

        filled_qty = self._volume_cap_quantity(order.quantity, bar, order.asset_class)
        if filled_qty == 0:
            return None

        fill_price, slippage = self._apply_slippage(order.stop_price, order.side, order.asset_class)
        fees = self._fees(order.asset_class, filled_qty, fill_price)
        order.status = OrderStatus.FILLED if abs(filled_qty) >= abs(order.quantity) else OrderStatus.PARTIALLY_FILLED

        return Fill(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            quantity=filled_qty, fill_price=fill_price, fees=fees, slippage=slippage,
            timestamp=bar.timestamp, rationale=order.rationale, strategy=order.strategy,
        )

    def check_stop_loss_take_profit(self, order: Order, bar: Bar, entry_side: OrderSide) -> Fill | None:
        """Evaluate an open position's attached stop-loss/take-profit levels
        against a bar's OHLC range during event-driven backtests. `entry_side`
        is the side of the original entry (exit is the opposite side)."""
        exit_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY
        trigger_price = None

        if order.stop_loss_price is not None:
            hit_stop = (
                (entry_side == OrderSide.BUY and bar.low <= order.stop_loss_price) or
                (entry_side == OrderSide.SELL and bar.high >= order.stop_loss_price)
            )
            if hit_stop:
                trigger_price = order.stop_loss_price

        if trigger_price is None and order.take_profit_price is not None:
            hit_tp = (
                (entry_side == OrderSide.BUY and bar.high >= order.take_profit_price) or
                (entry_side == OrderSide.SELL and bar.low <= order.take_profit_price)
            )
            if hit_tp:
                trigger_price = order.take_profit_price

        if trigger_price is None:
            return None

        fill_price, slippage = self._apply_slippage(trigger_price, exit_side, order.asset_class)
        fees = self._fees(order.asset_class, order.quantity, fill_price)
        return Fill(
            order_id=order.order_id, symbol=order.symbol, side=exit_side,
            quantity=order.quantity, fill_price=fill_price, fees=fees, slippage=slippage,
            timestamp=bar.timestamp, rationale="stop-loss/take-profit exit", strategy=order.strategy,
        )

    def execute(self, order: Order, bar: Bar) -> Fill | None:
        """Dispatch to the correct fill routine based on order type. `bar`
        represents the next available bar the order can trade against."""
        if order.order_type == OrderType.MARKET:
            return self.fill_market_order(order, bar)
        if order.order_type == OrderType.LIMIT:
            return self.fill_limit_order(order, bar)
        if order.order_type == OrderType.STOP:
            return self.fill_stop_order(order, bar)
        raise ValueError(f"unsupported order type: {order.order_type}")
