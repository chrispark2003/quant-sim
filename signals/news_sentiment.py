"""News sentiment signal: NLP scoring of headlines via a lightweight
finetuned sentiment model (FinBERT via HuggingFace Transformers by default),
mapped to a -1/+1 signal and decayed over time so stale headlines lose
influence.

Results are cached to disk (parquet) so the same headline is never re-scored
by the model twice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signals.base import Signal
from settings import cache_dir

_LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class NewsSentimentSignal(Signal):
    name = "news_sentiment"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.model_name = self.params.get("model_name", "ProsusAI/finbert")
        self.decay_half_life_hours = self.params.get("decay_half_life_hours", 12)
        self.cache_path = Path(self.params.get("cache_path", cache_dir() / "news_sentiment_cache.parquet"))
        self._pipeline = None

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_path.exists():
            return pd.read_parquet(self.cache_path)
        return pd.DataFrame(columns=["title", "score", "label"])

    def _save_cache(self, df: pd.DataFrame) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path)

    def _get_pipeline(self):
        """Lazily load the HuggingFace sentiment pipeline. Import is deferred
        so the rest of the project works without `transformers`/`torch`
        installed unless this signal is actually used."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
        return self._pipeline

    def score_headlines(self, headlines: pd.DataFrame) -> pd.DataFrame:
        """headlines: DataFrame with columns [timestamp, title]. Returns the
        same frame with added 'score' in [-1, 1] and 'label' columns, using
        the disk cache to avoid re-scoring repeated headlines."""
        if headlines.empty:
            return headlines.assign(score=pd.Series(dtype=float), label=pd.Series(dtype=str))

        cache = self._load_cache()
        cached_titles = set(cache["title"]) if not cache.empty else set()
        to_score = headlines[~headlines["title"].isin(cached_titles)]

        if not to_score.empty:
            clf = self._get_pipeline()
            results = clf(to_score["title"].tolist(), truncation=True)
            new_rows = pd.DataFrame({
                "title": to_score["title"].values,
                "label": [r["label"].lower() for r in results],
                "score": [
                    _LABEL_SIGN.get(r["label"].lower(), 0.0) * r["score"] for r in results
                ],
            })
            cache = pd.concat([cache, new_rows], ignore_index=True).drop_duplicates("title")
            self._save_cache(cache)

        merged = headlines.merge(cache[["title", "score", "label"]], on="title", how="left")
        return merged

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """`data` is expected to be a headlines-shaped DataFrame with columns
        [timestamp, title] (and optionally a target index of bar timestamps
        to reindex onto). Applies exponential time-decay so recent headlines
        dominate, then squashes the decayed weighted-average sentiment into
        [-1, 1]."""
        if data.empty or "title" not in data.columns:
            return pd.Series(dtype=float)

        scored = self.score_headlines(data)
        scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True)
        now = scored["timestamp"].max()
        age_hours = (now - scored["timestamp"]).dt.total_seconds() / 3600.0
        decay = np.power(0.5, age_hours / self.decay_half_life_hours)

        weighted_score = (scored["score"] * decay).sum() / decay.sum() if decay.sum() > 0 else 0.0
        idx = pd.DatetimeIndex(scored["timestamp"].sort_values().unique())
        return self.clip(pd.Series(weighted_score, index=idx))
