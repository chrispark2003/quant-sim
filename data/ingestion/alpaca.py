"""Alpaca market-data adapter (data API only).

This adapter exclusively uses Alpaca's *market data* endpoints. It never
imports or calls alpaca-py's trading/order client, and it is not wired to
any execution path in this project -- all simulated fills happen in
execution/simulator.py against the virtual ledger.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd
import requests

from data.ingestion.base import BaseAdapter
from data.schema import ohlcv_to_long
from settings import env

_DATA_BASE_URL = "https://data.alpaca.markets"

_TIMEFRAME_MAP = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day",
}


class AlpacaAdapter(BaseAdapter):
    asset_class = "equity"

    def _headers(self) -> dict:
        key = env("ALPACA_API_KEY")
        secret = env("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in environment (.env)")
        return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        tf = _TIMEFRAME_MAP.get(timeframe, "1Day")
        params = {
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "timeframe": tf,
            "limit": 10000,
            "adjustment": "raw",
        }
        url = f"{_DATA_BASE_URL}/v2/stocks/{symbol}/bars"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=15)
        resp.raise_for_status()
        bars = resp.json().get("bars", [])
        if not bars:
            return self._empty_frame()

        rows = [{
            "timestamp": pd.to_datetime(b["t"], utc=True),
            "open": b["o"], "high": b["h"], "low": b["l"],
            "close": b["c"], "volume": b["v"],
        } for b in bars]
        wide = pd.DataFrame(rows)
        return ohlcv_to_long(wide, symbol=symbol, asset_class=self.asset_class)

    def stream_live(self, symbol: str, callback: Callable[[pd.DataFrame], None]) -> None:
        """Polling fallback for live bars (avoids requiring the alpaca-py
        websocket client as a hard dependency)."""
        import time
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
            time.sleep(60)
