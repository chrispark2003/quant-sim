"""Ensemble combinator: weighted average of individual signal scores, with
weights configurable via config/strategies.yaml (signals.ensemble.weights).
"""
from __future__ import annotations

import pandas as pd

from signals.base import Signal
from signals.cross_sectional import CrossSectionalSignal
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from signals.news_sentiment import NewsSentimentSignal
from signals.volatility_breakout import VolatilityBreakoutSignal

_SIGNAL_REGISTRY = {
    "momentum": MomentumSignal,
    "mean_reversion": MeanReversionSignal,
    "cross_sectional": CrossSectionalSignal,
    "volatility_breakout": VolatilityBreakoutSignal,
    "news_sentiment": NewsSentimentSignal,
}


class EnsembleSignal(Signal):
    name = "ensemble"

    def __init__(self, weights: dict[str, float], signal_params: dict[str, dict] | None = None, clip_range=(-1.0, 1.0)):
        super().__init__()
        self.weights = weights
        self.clip_range = clip_range
        signal_params = signal_params or {}
        self.signals: dict[str, Signal] = {
            key: cls(signal_params.get(key, {}))
            for key, cls in _SIGNAL_REGISTRY.items()
            if key in weights
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Runs each enabled component signal against `data`, then computes
        the weighted average, renormalizing weights over whichever component
        signals produced non-null output for a given timestamp."""
        component_scores = {}
        for key, sig in self.signals.items():
            try:
                component_scores[key] = sig.generate_signal(data)
            except Exception:
                continue

        if not component_scores:
            return pd.Series(0.0, index=data.index)

        scores_df = pd.DataFrame(component_scores).reindex(data.index)
        weight_series = pd.Series({k: self.weights.get(k, 0.0) for k in scores_df.columns})

        available_mask = scores_df.notna()
        effective_weights = available_mask.mul(weight_series, axis=1)
        weight_sums = effective_weights.sum(axis=1).replace(0, pd.NA)

        weighted = (scores_df.fillna(0.0) * effective_weights).sum(axis=1) / weight_sums
        weighted = weighted.fillna(0.0)
        return weighted.clip(lower=self.clip_range[0], upper=self.clip_range[1])

    @classmethod
    def from_config(cls, config: dict) -> "EnsembleSignal":
        ensemble_cfg = config.get("ensemble", {})
        signal_params = {k: v for k, v in config.items() if k != "ensemble"}
        return cls(
            weights=ensemble_cfg.get("weights", {}),
            signal_params=signal_params,
            clip_range=tuple(ensemble_cfg.get("clip_range", [-1.0, 1.0])),
        )
