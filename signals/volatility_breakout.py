"""Volatility breakout signal: ATR-based breakout detection after a period
of volatility compression (a classic "squeeze then expand" setup).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Signal


class VolatilityBreakoutSignal(Signal):
    name = "volatility_breakout"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.compression_lookback = self.params.get("compression_lookback", 20)
        self.compression_percentile = self.params.get("compression_percentile", 0.25)
        self.breakout_multiple = self.params.get("breakout_multiple", 1.5)

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """`data` must contain 'atr' and 'close' columns. Detects: (1) ATR
        currently in the bottom `compression_percentile` of its own rolling
        history (a "compression" state), followed by (2) a bar-over-bar price
        move exceeding `breakout_multiple` * ATR (an "expansion" breakout).
        Direction follows the breakout's sign."""
        atr = data["atr"]
        close = data["close"]

        atr_rank = atr.rolling(self.compression_lookback).apply(
            lambda x: (x[-1] <= x).mean() if len(x) else np.nan, raw=True
        )
        was_compressed = atr_rank.shift(1) <= self.compression_percentile

        price_move = close.diff()
        breakout_threshold = self.breakout_multiple * atr.shift(1)
        breakout_up = was_compressed & (price_move > breakout_threshold)
        breakout_down = was_compressed & (-price_move > breakout_threshold)

        magnitude = (price_move.abs() / breakout_threshold.replace(0, np.nan)).clip(upper=3.0) / 3.0
        magnitude = magnitude.fillna(0.0)

        score = pd.Series(0.0, index=data.index)
        score = score.where(~breakout_up, magnitude)
        score = score.where(~breakout_down, -magnitude)
        return self.clip(score)
