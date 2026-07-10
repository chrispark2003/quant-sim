"""Binance public REST/WebSocket adapter for crypto OHLCV data.

Uses only public market-data endpoints (klines + websocket kline stream).
No API key is required for market data, and this adapter never touches
any Binance order-placement endpoint -- it is read-only by construction.

binance.com returns HTTP 451 from US IPs (geoblock). Set in .env:
    BINANCE_BASE_URL=https://api.binance.us
    BINANCE_WS_URL=wss://stream.binance.us:9443/ws
to use the US-accessible endpoint; the kline API shape is identical.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Callable

import pandas as pd
import requests

from data.ingestion.base import BaseAdapter
from data.schema import ohlcv_to_long
from settings import env

_BASE_URL = env("BINANCE_BASE_URL", "https://api.binance.com")
_WS_BASE = env("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws")

_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d",
}


class BinanceAdapter(BaseAdapter):
    asset_class = "crypto"

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        interval = _INTERVAL_MAP.get(timeframe, "1h")
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 1000,
        }
        resp = requests.get(f"{_BASE_URL}/api/v3/klines", params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            return self._empty_frame()

        rows = []
        for k in raw:
            rows.append({
                "timestamp": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        wide = pd.DataFrame(rows)
        return ohlcv_to_long(wide, symbol=symbol.upper(), asset_class=self.asset_class)

    def stream_live(self, symbol: str, callback: Callable[[pd.DataFrame], None], interval: str = "1m") -> None:
        """Open a websocket kline stream and invoke callback on each closed
        candle. Requires the `websocket-client` package."""
        import websocket

        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"{_WS_BASE}/{stream_name}"

        def on_message(_ws, message):
            data = json.loads(message)
            k = data.get("k", {})
            if not k.get("x"):  # only emit when the candle is closed
                return
            row = {
                "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            wide = pd.DataFrame([row])
            long_df = ohlcv_to_long(wide, symbol=symbol.upper(), asset_class=self.asset_class)
            callback(long_df)

        ws = websocket.WebSocketApp(url, on_message=on_message)
        ws.run_forever()
