"""Feature store: computes technical features once and caches them so every
signal/strategy reuses the same computation instead of recomputing indicators
independently.

Features computed per symbol from OHLCV bars:
  - returns (simple, log)
  - rolling realized volatility (annualized)
  - RSI
  - MACD (line, signal, histogram)
  - Bollinger Bands (mid, upper, lower, %b, bandwidth)
  - rolling cross-sectional correlation matrix

Results are cached in-memory (keyed by symbol + data hash) and persisted to
parquet under data/cache/ so repeated backtests don't recompute features.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from settings import cache_dir


def _hash_frame(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def compute_returns(close: pd.Series) -> pd.DataFrame:
    simple = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    return pd.DataFrame({"return_simple": simple, "return_log": log_ret})


def compute_volatility(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_pct_b": pct_b, "bb_bandwidth": bandwidth,
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def compute_rolling_correlation(returns_wide: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling pairwise correlation, returned as the correlation matrix as of
    the last available timestamp (most callers only need the latest matrix
    for correlation-aware position sizing)."""
    return returns_wide.tail(window).corr()


class FeatureStore:
    """Compute-once, reuse-everywhere feature cache for a single symbol's
    OHLCV history. Keyed by a hash of the input bars so unchanged data never
    triggers recomputation."""

    def __init__(self, use_disk_cache: bool = True):
        self._mem_cache: dict[str, pd.DataFrame] = {}
        self.use_disk_cache = use_disk_cache

    def _cache_path(self, symbol: str, key: str) -> Path:
        return cache_dir() / f"features_{symbol}_{key}.parquet"

    def compute(self, symbol: str, ohlcv: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        """ohlcv must be indexed by timestamp with columns open/high/low/close/volume."""
        params = params or {}
        key = f"{_hash_frame(ohlcv)}_{hash(frozenset(params.items())) & 0xffffffff}"
        cache_key = f"{symbol}:{key}"

        if cache_key in self._mem_cache:
            return self._mem_cache[cache_key]

        disk_path = self._cache_path(symbol, key)
        if self.use_disk_cache and disk_path.exists():
            df = pd.read_parquet(disk_path)
            self._mem_cache[cache_key] = df
            return df

        close, high, low = ohlcv["close"], ohlcv["high"], ohlcv["low"]
        rsi_window = params.get("rsi_window", 14)
        bb_window = params.get("bollinger_window", 20)
        bb_std = params.get("bollinger_num_std", 2.0)
        atr_window = params.get("atr_window", 14)
        vol_window = params.get("vol_window", 20)

        feats = pd.concat([
            ohlcv,
            compute_returns(close),
            compute_rsi(close, rsi_window).rename("rsi"),
            compute_macd(close),
            compute_bollinger(close, bb_window, bb_std),
            compute_atr(high, low, close, atr_window).rename("atr"),
        ], axis=1)
        feats["volatility"] = compute_volatility(feats["return_simple"], vol_window)

        if self.use_disk_cache:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            feats.to_parquet(disk_path)
        self._mem_cache[cache_key] = feats
        return feats

    def compute_universe(self, ohlcv_by_symbol: dict[str, pd.DataFrame], params: dict | None = None) -> dict[str, pd.DataFrame]:
        return {sym: self.compute(sym, df, params) for sym, df in ohlcv_by_symbol.items()}

    def clear_cache(self) -> None:
        self._mem_cache.clear()


_default_feature_store: FeatureStore | None = None


def get_feature_store() -> FeatureStore:
    global _default_feature_store
    if _default_feature_store is None:
        _default_feature_store = FeatureStore()
    return _default_feature_store
