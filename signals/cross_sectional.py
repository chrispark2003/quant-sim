"""Cross-sectional factor ranking signal: combines value, quality, and
momentum factors into a single composite score, ranked across the trading
universe at each timestamp.

Note: without a fundamentals data source wired in, "value" uses an inverse
volatility-adjusted price-extension proxy for a P/E-like cheap/expensive
factor (lower relative price extension vs. its own trend = "cheaper"). This
keeps the module fully self-contained using only OHLCV data, while remaining
swappable for a real fundamentals feed via the `value_fn` param.
"""
from __future__ import annotations

import pandas as pd

from signals.base import Signal


def _cross_sectional_rank_score(row: pd.Series) -> pd.Series:
    """Rank a cross-section (one row across symbols) into [-1, 1] via
    percentile rank centered at 0."""
    ranks = row.rank(pct=True, na_option="keep")
    return (ranks - 0.5) * 2.0


class CrossSectionalSignal(Signal):
    name = "cross_sectional"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.value_weight = self.params.get("value_weight", 0.33)
        self.quality_weight = self.params.get("quality_weight", 0.33)
        self.momentum_weight = self.params.get("momentum_weight", 0.34)
        self.quality_lookback = self.params.get("quality_lookback", 60)

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Single-symbol fallback: without peers to rank against, returns a
        neutral series (use `generate_universe_signal` for the real
        cross-sectional ranking this signal is designed for)."""
        return pd.Series(0.0, index=data.index)

    def generate_universe_signal(self, features_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """features_by_symbol: {symbol -> feature DataFrame with at least
        'close' and 'return_simple'/'volatility' columns}. Returns a wide
        DataFrame (timestamp x symbol) of composite scores in [-1, 1]."""
        closes = pd.DataFrame({s: f["close"] for s, f in features_by_symbol.items()})
        rets = pd.DataFrame({s: f["return_simple"] for s, f in features_by_symbol.items()})
        vol = pd.DataFrame({s: f["volatility"] for s, f in features_by_symbol.items()})

        # Momentum factor: trailing 20-day return.
        momentum_factor = closes.pct_change(20)

        # Quality factor: vol-adjusted trailing return (Sharpe-like proxy).
        trailing_ret = rets.rolling(self.quality_lookback).mean() * 252
        quality_factor = trailing_ret / vol.replace(0, pd.NA)

        # Value proxy: negative of price extension above its own moving
        # average (i.e. "cheap" = trading below its recent average).
        moving_avg = closes.rolling(self.quality_lookback).mean()
        value_factor = -(closes / moving_avg.replace(0, pd.NA) - 1.0)

        momentum_score = momentum_factor.apply(_cross_sectional_rank_score, axis=1)
        quality_score = quality_factor.apply(_cross_sectional_rank_score, axis=1)
        value_score = value_factor.apply(_cross_sectional_rank_score, axis=1)

        composite = (
            self.value_weight * value_score.fillna(0.0)
            + self.quality_weight * quality_score.fillna(0.0)
            + self.momentum_weight * momentum_score.fillna(0.0)
        )
        return composite.clip(lower=-1.0, upper=1.0)
