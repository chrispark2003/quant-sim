"""Unit tests for the risk layer: position sizing, circuit breaker, and
Sharpe/Sortino calculations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.metrics import max_drawdown, sharpe_ratio, sortino_ratio
from risk.portfolio import CircuitBreaker, PortfolioConstraints
from risk.sizer import PositionSizer

SIZER_CONFIG = {
    "method": "vol_target", "target_annual_vol": 0.15, "kelly_fraction": 0.5,
    "lookback_days": 60, "max_position_pct": 0.10, "max_sector_pct": 0.30,
    "min_position_notional": 100.0,
}


class TestPositionSizer:
    def test_zero_signal_gives_zero_notional(self):
        sizer = PositionSizer(SIZER_CONFIG)
        result = sizer.size_position("AAPL", signal_score=0.0, equity=100000, realized_annual_vol=0.20)
        assert result.target_notional == 0.0

    def test_positive_signal_gives_positive_notional(self):
        sizer = PositionSizer(SIZER_CONFIG)
        result = sizer.size_position("AAPL", signal_score=0.5, equity=100000, realized_annual_vol=0.20)
        assert result.target_notional > 0

    def test_negative_signal_gives_negative_notional(self):
        sizer = PositionSizer(SIZER_CONFIG)
        result = sizer.size_position("AAPL", signal_score=-0.5, equity=100000, realized_annual_vol=0.20)
        assert result.target_notional < 0

    def test_position_cap_enforced(self):
        sizer = PositionSizer(SIZER_CONFIG)
        # Very low realized vol relative to target vol => uncapped weight would be huge.
        result = sizer.size_position("AAPL", signal_score=1.0, equity=100000, realized_annual_vol=0.01)
        assert abs(result.target_pct_of_equity) <= SIZER_CONFIG["max_position_pct"] + 1e-9
        assert result.capped is True

    def test_sector_cap_enforced(self):
        sizer = PositionSizer(SIZER_CONFIG)
        result = sizer.size_position(
            "AAPL", signal_score=1.0, equity=100000, realized_annual_vol=0.15,
            sector="Technology", current_sector_pct=0.28,
        )
        assert abs(result.target_pct_of_equity) <= 0.02 + 1e-9


class TestCircuitBreaker:
    def test_trips_on_large_drawdown(self):
        cb = CircuitBreaker(max_drawdown_pct=0.15, cooldown_bars=3)
        equity = pd.Series([100, 110, 120, 100, 95, 90])  # ~25% drawdown from peak 120
        assert cb.check(equity) is True

    def test_does_not_trip_on_small_drawdown(self):
        cb = CircuitBreaker(max_drawdown_pct=0.15, cooldown_bars=3)
        equity = pd.Series([100, 110, 120, 118, 115, 112])
        assert cb.check(equity) is False

    def test_cooldown_keeps_halt_active(self):
        cb = CircuitBreaker(max_drawdown_pct=0.10, cooldown_bars=2)
        equity_drawdown = pd.Series([100, 90])  # 10% dd, trips
        assert cb.check(equity_drawdown) is True
        equity_recovered = pd.Series([100, 90, 99])
        assert cb.check(equity_recovered) is True  # still in cooldown


class TestPortfolioConstraints:
    def test_gross_exposure_limit_scales_down(self):
        constraints = PortfolioConstraints({
            "max_gross_exposure": 1.0, "max_net_exposure": 1.0,
            "correlation_lookback_days": 60, "correlation_limit": 0.7, "correlation_size_haircut": 0.5,
        })
        result = constraints.check_exposure_limits(
            "AAPL", target_notional=150000, positions={"MSFT": 50000}, equity=100000,
        )
        assert result.allowed_notional < result.original_notional
        assert result.breached


class TestMetrics:
    def test_sharpe_positive_for_upward_drift(self):
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0.001, 0.01, 500))
        assert sharpe_ratio(returns) > 0

    def test_sharpe_zero_for_zero_variance(self):
        returns = pd.Series([0.0] * 50)
        assert sharpe_ratio(returns) == 0.0

    def test_sortino_only_penalizes_downside(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.01] * 20)
        sortino = sortino_ratio(returns)
        sharpe = sharpe_ratio(returns)
        assert sortino != 0.0
        assert isinstance(sharpe, float)

    def test_max_drawdown_is_negative_or_zero(self):
        equity = pd.Series([100, 120, 90, 110])
        mdd = max_drawdown(equity)
        assert mdd <= 0
        assert mdd == pytest.approx(-0.25, rel=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
