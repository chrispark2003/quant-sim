"""Polygon.io adapter for historical/live equity OHLCV data (aggregates API).

Read-only market-data adapter -- no order placement capability.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

import pandas as pd
import requests

from data.ingestion.base import BaseAdapter
from data.schema import ohlcv_to_long
from settings import env

_BASE_URL = "https://api.polygon.io"

_MULTIPLIER_TIMESPAN = {
    "1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"),
    "1h": (1, "hour"), "1d": (1, "day"),
}


class PolygonAdapter(BaseAdapter):
    asset_class = "equity"

    def _api_key(self) -> str:
        key = env("POLYGON_API_KEY")
        if not key:
            raise RuntimeError("POLYGON_API_KEY not set in environment (.env)")
        return key

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        mult, span = _MULTIPLIER_TIMESPAN.get(timeframe, (1, "day"))
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        url = f"{_BASE_URL}/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start_str}/{end_str}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self._api_key()}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return self._empty_frame()

        rows = [{
            "timestamp": pd.to_datetime(r["t"], unit="ms", utc=True),
            "open": r["o"], "high": r["h"], "low": r["l"],
            "close": r["c"], "volume": r["v"],
        } for r in results]
        wide = pd.DataFrame(rows)
        return ohlcv_to_long(wide, symbol=symbol, asset_class=self.asset_class)

    def stream_live(self, symbol: str, callback: Callable[[pd.DataFrame], None], poll_seconds: int = 60) -> None:
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
