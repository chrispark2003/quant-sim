"""Normalized schema used by every data adapter and the time-series store.

Every observation flowing through the system -- price bars, computed
features, or news sentiment scores -- is represented as a row of:

    timestamp    : UTC datetime
    symbol       : e.g. "AAPL", "BTCUSDT"
    asset_class  : "equity" | "crypto" | "news"
    field        : e.g. "open", "close", "volume", "rsi_14", "sentiment"
    value        : float

This long/tidy format lets equities, crypto, and alternative data (news
sentiment) live in the same table without needing asset-class-specific
columns, and makes it trivial to pivot into wide feature matrices on demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

SCHEMA_COLUMNS = ["timestamp", "symbol", "asset_class", "field", "value"]


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    NEWS = "news"


class Field(str, Enum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    SENTIMENT = "sentiment"


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    symbol: str
    asset_class: AssetClass
    field: str
    value: float

    def as_tuple(self) -> tuple:
        return (self.timestamp, self.symbol, str(self.asset_class), self.field, self.value)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DataFrame conforms to the normalized schema, coercing types."""
    missing = set(SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required schema columns: {missing}")
    out = df[SCHEMA_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["symbol"] = out["symbol"].astype(str)
    out["asset_class"] = out["asset_class"].astype(str)
    out["field"] = out["field"].astype(str)
    out["value"] = out["value"].astype(float)
    return out


def ohlcv_to_long(df: pd.DataFrame, symbol: str, asset_class: str) -> pd.DataFrame:
    """Convert a wide OHLCV DataFrame (indexed or columned by timestamp) into
    the normalized long schema."""
    df = df.copy()
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    value_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    long_df = df.melt(id_vars=["timestamp"], value_vars=value_cols,
                       var_name="field", value_name="value")
    long_df["symbol"] = symbol
    long_df["asset_class"] = asset_class
    return validate_schema(long_df)


def long_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the normalized long schema back into a wide OHLCV DataFrame."""
    wide = df.pivot_table(index="timestamp", columns="field", values="value", aggfunc="last")
    wide = wide.sort_index()
    return wide
