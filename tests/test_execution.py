"""Unit tests for the execution simulator: fill logic, slippage, fee
calculation, and stop/limit trigger conditions.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from execution.orders import Order, OrderSide, OrderType
from execution.simulator import Bar, ExecutionSimulator

CONFIG = {
    "equities": {"slippage_bps": 5, "commission_per_share": 0.005, "commission_min": 1.0,
                 "taker_fee_bps": 1.0, "volume_participation_cap": 0.10},
    "crypto": {"slippage_bps": 10, "taker_fee_bps": 10.0, "volume_participation_cap": 0.10},
    "latency": {"enabled": False},
}


def make_bar(open_=100.0, high=101.0, low=99.0, close=100.5, volume=10000.0) -> Bar:
    return Bar(timestamp=datetime(2024, 1, 1), open=open_, high=high, low=low, close=close, volume=volume)


class TestMarketOrderFills:
    def test_buy_fills_at_next_bar_open_with_positive_slippage(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET, asset_class="equity")
        bar = make_bar(open_=100.0)
        fill = sim.fill_market_order(order, bar)
        assert fill is not None
        assert fill.fill_price > bar.open  # buy slippage pushes price up

    def test_sell_fills_at_next_bar_open_with_negative_slippage(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=10, order_type=OrderType.MARKET, asset_class="equity")
        bar = make_bar(open_=100.0)
        fill = sim.fill_market_order(order, bar)
        assert fill is not None
        assert fill.fill_price < bar.open  # sell slippage pushes price down

    def test_volume_participation_cap_limits_fill_size(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100000, order_type=OrderType.MARKET, asset_class="equity")
        bar = make_bar(volume=10000.0)
        fill = sim.fill_market_order(order, bar)
        assert fill.quantity == pytest.approx(1000.0)  # 10% of 10000 volume

    def test_equity_commission_fee(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET, asset_class="equity")
        bar = make_bar(open_=100.0, volume=1_000_000)
        fill = sim.fill_market_order(order, bar)
        # commission_per_share*qty = 0.05, but commission_min=1.0 floors it, plus bps fee
        assert fill.fees >= 1.0

    def test_crypto_taker_fee(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="BTCUSDT", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, asset_class="crypto")
        bar = make_bar(open_=50000.0, volume=1_000_000)
        fill = sim.fill_market_order(order, bar)
        expected_fee = fill.fill_price * 1 * (10.0 / 10000.0)
        assert fill.fees == pytest.approx(expected_fee, rel=1e-6)


class TestLimitOrders:
    def test_buy_limit_fills_when_price_trades_through(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.LIMIT,
                      limit_price=99.5, asset_class="equity")
        bar = make_bar(low=99.0, high=101.0)
        fill = sim.fill_limit_order(order, bar)
        assert fill is not None
        assert fill.fill_price == 99.5

    def test_buy_limit_does_not_fill_when_price_never_reaches(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.LIMIT,
                      limit_price=90.0, asset_class="equity")
        bar = make_bar(low=99.0, high=101.0)
        fill = sim.fill_limit_order(order, bar)
        assert fill is None


class TestStopOrders:
    def test_sell_stop_triggers_when_price_falls_through(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=10, order_type=OrderType.STOP,
                      stop_price=98.0, asset_class="equity")
        bar = make_bar(low=97.0, high=101.0)
        fill = sim.fill_stop_order(order, bar)
        assert fill is not None

    def test_sell_stop_does_not_trigger_above_stop(self):
        sim = ExecutionSimulator(CONFIG)
        order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=10, order_type=OrderType.STOP,
                      stop_price=90.0, asset_class="equity")
        bar = make_bar(low=95.0, high=101.0)
        fill = sim.fill_stop_order(order, bar)
        assert fill is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
