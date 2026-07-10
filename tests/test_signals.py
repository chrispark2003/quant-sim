"""Unit tests for the signal layer: verify each signal's output stays within
[-1, 1] on synthetic data and moves in the expected direction for an obvious
trend/reversion/breakout setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from signals.volatility_breakout import VolatilityBreakoutSignal


def _make_ohlcv(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes
    volume = np.full(len(closes), 1_000_000.0)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume}, index=idx)


def _with_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    from data.feature_store import FeatureStore
    return FeatureStore(use_disk_cache=False).compute("TEST", ohlcv)


class TestMomentumSignal:
    def test_output_range(self):
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = MomentumSignal({"windows": [1, 5, 20], "zscore_lookback": 30})
        scores = signal.generate_signal(ohlcv)
        assert scores.between(-1, 1).all()

    def test_uptrend_is_positive(self):
        closes = np.linspace(100, 200, 100)
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = MomentumSignal({"windows": [1, 5, 20], "zscore_lookback": 30})
        scores = signal.generate_signal(ohlcv)
        assert scores.iloc[-10:].mean() > 0

    def test_downtrend_is_negative(self):
        closes = np.linspace(200, 100, 100)
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = MomentumSignal({"windows": [1, 5, 20], "zscore_lookback": 30})
        scores = signal.generate_signal(ohlcv)
        assert scores.iloc[-10:].mean() < 0


class TestMeanReversionSignal:
    def test_output_range(self):
        closes = 100 + 5 * np.sin(np.linspace(0, 10, 100))
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = MeanReversionSignal({"rsi_overbought": 70, "rsi_oversold": 30})
        scores = signal.generate_signal(ohlcv)
        assert scores.between(-1, 1).all()

    def test_extreme_overbought_is_negative(self):
        # Sharp spike above the Bollinger band -> expect a bearish (negative) score.
        closes = np.concatenate([np.full(30, 100.0), np.linspace(100, 160, 10)])
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = MeanReversionSignal({"rsi_overbought": 70, "rsi_oversold": 30})
        scores = signal.generate_signal(ohlcv)
        assert scores.iloc[-1] < 0

    def test_extreme_oversold_is_positive(self):
        closes = np.concatenate([np.full(30, 100.0), np.linspace(100, 40, 10)])
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = MeanReversionSignal({"rsi_overbought": 70, "rsi_oversold": 30})
        scores = signal.generate_signal(ohlcv)
        assert scores.iloc[-1] > 0


class TestVolatilityBreakoutSignal:
    def test_output_range(self):
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = VolatilityBreakoutSignal({
            "compression_lookback": 20, "compression_percentile": 0.5, "breakout_multiple": 0.5,
        })
        scores = signal.generate_signal(ohlcv)
        assert scores.between(-1, 1).all()

    def test_breakout_after_compression_is_directional(self):
        # Flat/compressed period followed by a sharp upward jump.
        flat = np.full(40, 100.0) + np.random.normal(0, 0.05, 40)
        jump = np.concatenate([[100.0], np.full(9, 130.0)])
        closes = np.concatenate([flat, jump])
        ohlcv = _with_features(_make_ohlcv(closes))
        signal = VolatilityBreakoutSignal({
            "compression_lookback": 20, "compression_percentile": 0.8, "breakout_multiple": 1.0,
        })
        scores = signal.generate_signal(ohlcv)
        assert scores.iloc[40:45].max() > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
