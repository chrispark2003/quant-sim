"""Momentum signal: rate-of-change over multiple windows, cross-sectionally
z-scored so relative momentum drives sign/magnitude of the score.
"""
from __future__ import annotations

import pandas as pd

from signals.base import Signal


class MomentumSignal(Signal):
    name = "momentum"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.windows = self.params.get("windows", [1, 5, 20])
        self.zscore_lookback = self.params.get("zscore_lookback", 60)

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """`data` must contain a 'close' column indexed by timestamp for a
        single symbol. Combines multi-window rate-of-change and squashes the
        average into [-1, 1].

        Note: raw ROC is used directly here rather than z-scored against its
        *own* rolling history -- self z-scoring a monotonic trend inverts
        sign (a steady linear uptrend has decelerating pct-change, which
        z-scores negative at the end), which is the opposite of what a
        trend-following momentum signal should say. Z-scoring is instead
        applied cross-sectionally in `generate_cross_sectional`, matching
        the intended "z-scored cross-sectionally" semantics."""
        close = data["close"]
        roc_components = [close.pct_change(w) for w in self.windows]
        combined = pd.concat(roc_components, axis=1).mean(axis=1).fillna(0.0)
        return self.clip(self.squash(combined, scale=0.05))

    def generate_cross_sectional(self, closes_wide: pd.DataFrame) -> pd.DataFrame:
        """Given a wide DataFrame of close prices (columns = symbols), compute
        momentum per symbol then z-score cross-sectionally at each timestamp
        (across symbols) -- used by the ensemble/cross-sectional strategies."""
        roc_by_window = [closes_wide.pct_change(w) for w in self.windows]
        roc_avg = sum(roc_by_window) / len(roc_by_window)

        row_mean = roc_avg.mean(axis=1)
        row_std = roc_avg.std(axis=1).replace(0, pd.NA)
        cross_z = roc_avg.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)

        squashed = cross_z.apply(lambda col: self.squash(col, scale=2.0))
        return squashed.clip(lower=-1.0, upper=1.0)
