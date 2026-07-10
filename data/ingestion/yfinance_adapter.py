"""Yahoo Finance adapter for historical equity OHLCV data.

No API key required. Used as the primary equities-historical source for
backtesting; `stream_live` polls on an interval since yfinance has no
native websocket.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

import pandas as pd

from data.ingestion.base import BaseAdapter
from data.schema import ohlcv_to_long

_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "1d": "1d",
}


class YFinanceAdapter(BaseAdapter):
    asset_class = "equity"

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        import yfinance as yf

        interval = _TIMEFRAME_MAP.get(timeframe, "1d")
        ticker = yf.Ticker(symbol)
        raw = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        if raw.empty:
            return self._empty_frame()
        raw = raw.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        raw.index.name = "timestamp"
        return ohlcv_to_long(raw.reset_index(), symbol=symbol, asset_class=self.asset_class)

    def stream_live(self, symbol: str, callback: Callable[[pd.DataFrame], None], poll_seconds: int = 60) -> None:
        """Polling-based 'live stream' -- fetches the latest bar every
        `poll_seconds` and invokes callback. Intended to run in its own
        thread/process from the live loop, not to block the main process."""
        last_ts = None
        while True:
            end = datetime.utcnow()
            start = end - pd.Timedelta(minutes=10)
            df = self.fetch_ohlcv(symbol, "1m", start, end)
            if not df.empty:
                latest_ts = df["timestamp"].max()
                if latest_ts != last_ts:
                    last_ts = latest_ts
                    callback(df[df["timestamp"] == latest_ts])
            time.sleep(poll_seconds)
