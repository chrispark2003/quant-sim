"""NewsAPI.org adapter for headline ingestion, used by the news-sentiment signal.

Read-only alt-data source. Requires NEWSAPI_API_KEY in the environment.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

import pandas as pd
import requests

from data.ingestion.base import BaseAdapter
from settings import env

_BASE_URL = "https://newsapi.org/v2/everything"


class NewsAPIAdapter(BaseAdapter):
    asset_class = "news"

    def _api_key(self) -> str:
        key = env("NEWSAPI_API_KEY")
        if not key:
            raise RuntimeError("NEWSAPI_API_KEY is not set in the environment (.env)")
        return key

    def fetch_headlines(self, query: str, start: datetime, end: datetime, page_size: int = 50) -> pd.DataFrame:
        params = {
            "q": query,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self._api_key(),
        }
        resp = requests.get(_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        articles = payload.get("articles", [])
        if not articles:
            return pd.DataFrame(columns=["timestamp", "query", "title", "description", "source"])

        rows = [{
            "timestamp": pd.to_datetime(a["publishedAt"], utc=True),
            "query": query,
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "source": (a.get("source") or {}).get("name", ""),
        } for a in articles]
        return pd.DataFrame(rows)

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError("NewsAPIAdapter provides headlines, not OHLCV bars")

    def stream_live(self, symbol: str, callback: Callable[[pd.DataFrame], None], poll_seconds: int = 300) -> None:
        last_ts = None
        while True:
            end = datetime.utcnow()
            start = end - pd.Timedelta(hours=1)
            df = self.fetch_headlines(symbol, start, end)
            if not df.empty:
                newest = df["timestamp"].max()
                if newest != last_ts:
                    last_ts = newest
                    callback(df)
            time.sleep(poll_seconds)
