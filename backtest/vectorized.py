"""Fast vectorized backtester.

Applies a signal's scores directly to historical bars, sizes positions with
simple next-bar execution assumptions, and computes P&L with a flat cost
model. This is intended for rapid strategy iteration, NOT the realistic
fill/slippage simulation done in the event-driven backtester.

*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk.sizer import PositionSizer

BACKTEST_DISCLAIMER = "BACKTEST RESULTS -- NOT LIVE PERFORMANCE"


class VectorizedBacktester:
    def __init__(self, sizer_config: dict, cost_bps: float = 5.0, starting_cash: float = 100000.0):
        self.sizer = PositionSizer(sizer_config)
        self.cost_bps = cost_bps
        self.starting_cash = starting_cash

    def run(self, features: pd.DataFrame, signal_scores: pd.Series) -> dict:
        """features: OHLCV+feature DataFrame indexed by timestamp for a single
        symbol (must include 'close' and 'volatility'). signal_scores: same
        index, values in [-1, 1].

        Position sizing uses volatility targeting; weights are computed at
        bar t and applied to bar t+1's return (avoiding lookahead), with a
        flat transaction cost charged on turnover.
        """
        df = features.copy()
        df["signal"] = signal_scores.reindex(df.index).fillna(0.0)

        weights = []
        for ts, row in df.iterrows():
            vol = row.get("volatility", np.nan)
            result = self.sizer.size_position(
                symbol="BACKTEST", signal_score=row["signal"], equity=1.0,
                realized_annual_vol=vol if pd.notna(vol) else self.sizer.target_annual_vol,
            )
            weights.append(result.target_pct_of_equity)
        df["target_weight"] = weights

        # Applied weight lags by one bar: decide at close of t, earn t->t+1 return.
        df["applied_weight"] = df["target_weight"].shift(1).fillna(0.0)
        df["forward_return"] = df["close"].pct_change().fillna(0.0)

        turnover = df["applied_weight"].diff().abs().fillna(df["applied_weight"].abs())
        transaction_cost = turnover * (self.cost_bps / 10000.0)

        df["strategy_return"] = df["applied_weight"] * df["forward_return"] - transaction_cost
        df["equity_curve"] = self.starting_cash * (1 + df["strategy_return"]).cumprod()

        return {
            "equity_curve": df["equity_curve"],
            "returns": df["strategy_return"],
            "weights": df["applied_weight"],
            "turnover": turnover,
            "detail": df,
        }

    def run_universe(self, features_by_symbol: dict[str, pd.DataFrame], scores_by_symbol: dict[str, pd.Series]) -> dict:
        """Run the vectorized backtest independently per symbol, then combine
        into an equal-weighted portfolio equity curve."""
        results = {
            sym: self.run(features_by_symbol[sym], scores_by_symbol[sym])
            for sym in features_by_symbol if sym in scores_by_symbol
        }
        returns = pd.DataFrame({sym: r["returns"] for sym, r in results.items()}).fillna(0.0)
        portfolio_returns = returns.mean(axis=1)
        portfolio_equity = self.starting_cash * (1 + portfolio_returns).cumprod()
        return {
            "per_symbol": results,
            "portfolio_returns": portfolio_returns,
            "portfolio_equity": portfolio_equity,
        }
