"""Abstract base class for all market/alt-data adapters.

Every adapter normalizes its output to the schema defined in data/schema.py.
Adapters are read-only data sources -- none of them are capable of placing
orders. Order execution only ever happens in execution/simulator.py against
the virtual ledger.
"""
from __future__ import annotations

import abc
from datetime import datetime
from typing import Callable

import pandas as pd


class BaseAdapter(abc.ABC):
    """Abstract data adapter. Subclasses implement historical fetch + a live
    streaming hook. All adapters are data-only (read side); no adapter is
    permitted to submit orders anywhere."""

    asset_class: str = "unknown"

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars, normalized to the long schema
        (timestamp, symbol, asset_class, field, value)."""
        raise NotImplementedError

    @abc.abstractmethod
    def stream_live(self, symbol: str, callback: Callable[[pd.DataFrame], None]) -> None:
        """Stream/poll live data for `symbol`, invoking `callback` with a
        normalized long-schema DataFrame for each new observation/bar.
        Implementations may poll a REST endpoint or open a websocket; either
        way this must never touch an order-placement endpoint."""
        raise NotImplementedError

    def _empty_frame(self) -> pd.DataFrame:
        from data.schema import SCHEMA_COLUMNS
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
