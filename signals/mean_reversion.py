"""Mean reversion signal: Bollinger Band z-score combined with RSI divergence.

Intuition: price stretched far above the upper band with an overbought RSI
is a short signal (expect reversion down); the symmetric case is a long
signal.
"""
from __future__ import annotations

import pandas as pd

from signals.base import Signal


class MeanReversionSignal(Signal):
    name = "mean_reversion"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """`data` must already contain 'bb_pct_b' and 'rsi' columns (computed
        by the feature store: Bollinger %b and RSI)."""
        pct_b = data["bb_pct_b"].fillna(0.5)
        rsi = data["rsi"].fillna(50.0)

        # %b of 0.5 is the midline (neutral); 1.0/0.0 are the bands.
        # Map so above the upper band (%b > 1) => bearish (-1 direction), and
        # below the lower band (%b < 0) => bullish (+1 direction).
        bb_component = -(pct_b - 0.5) * 2.0

        # RSI overbought => bearish, oversold => bullish, scaled to [-1, 1].
        rsi_component = pd.Series(0.0, index=rsi.index)
        rsi_component = rsi_component.where(
            rsi <= self.rsi_overbought,
            -((rsi - self.rsi_overbought) / (100 - self.rsi_overbought)).clip(upper=1.0),
        )
        rsi_component = rsi_component.where(
            rsi >= self.rsi_oversold,
            ((self.rsi_oversold - rsi) / self.rsi_oversold).clip(upper=1.0),
        )

        combined = 0.6 * bb_component + 0.4 * rsi_component
        return self.clip(combined)
