"""Abstract base class for alpha signals.

Every signal consumes a feature-enriched OHLCV DataFrame (as produced by
data.feature_store.FeatureStore) and emits a score Series indexed by
timestamp, bounded to [-1, +1]: -1 is maximally short, +1 is maximally long,
0 is neutral/no opinion.
"""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class Signal(abc.ABC):
    name: str = "signal"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abc.abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Return a score Series in [-1, 1] indexed the same as `data`."""
        raise NotImplementedError

    @staticmethod
    def clip(scores: pd.Series, lo: float = -1.0, hi: float = 1.0) -> pd.Series:
        return scores.clip(lower=lo, upper=hi)

    @staticmethod
    def zscore(series: pd.Series, window: int | None = None) -> pd.Series:
        """Z-score a series, either over a rolling window or the full sample."""
        if window:
            mean = series.rolling(window).mean()
            std = series.rolling(window).std()
        else:
            mean = series.mean()
            std = series.std()
        z = (series - mean) / std.replace(0, np.nan) if hasattr(std, "replace") else (series - mean) / (std or np.nan)
        return z.fillna(0.0)

    @staticmethod
    def squash(series: pd.Series, scale: float = 1.0) -> pd.Series:
        """Map an unbounded z-score-like series into (-1, 1) via tanh."""
        return np.tanh(series / scale)
