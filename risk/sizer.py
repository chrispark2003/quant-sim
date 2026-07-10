"""Position sizing: volatility targeting with optional Kelly-fraction sizing,
subject to hard per-position and per-sector caps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SizingResult:
    symbol: str
    target_notional: float
    target_pct_of_equity: float
    method: str
    capped: bool
    rationale: str


class PositionSizer:
    def __init__(self, config: dict):
        self.method = config.get("method", "vol_target")
        self.target_annual_vol = config.get("target_annual_vol", 0.15)
        self.kelly_fraction = config.get("kelly_fraction", 0.5)
        self.lookback_days = config.get("lookback_days", 60)
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_sector_pct = config.get("max_sector_pct", 0.30)
        self.min_position_notional = config.get("min_position_notional", 100.0)

    def _vol_target_weight(self, signal_score: float, realized_annual_vol: float) -> float:
        if realized_annual_vol <= 0 or np.isnan(realized_annual_vol):
            return 0.0
        raw_weight = (self.target_annual_vol / realized_annual_vol) * signal_score
        return raw_weight

    def _kelly_weight(self, signal_score: float, expected_return: float, variance: float) -> float:
        if variance <= 0 or np.isnan(variance):
            return 0.0
        full_kelly = expected_return / variance
        return full_kelly * self.kelly_fraction * np.sign(signal_score) * min(abs(signal_score), 1.0)

    def size_position(
        self,
        symbol: str,
        signal_score: float,
        equity: float,
        realized_annual_vol: float,
        expected_return: float | None = None,
        return_variance: float | None = None,
        sector: str | None = None,
        current_sector_pct: float = 0.0,
    ) -> SizingResult:
        """Returns the target notional (signed: + long, - short) for a single
        position, after applying vol-target/Kelly sizing and hard caps."""
        signal_score = float(np.clip(signal_score, -1.0, 1.0))

        if self.method == "kelly" and expected_return is not None and return_variance is not None:
            weight = self._kelly_weight(signal_score, expected_return, return_variance)
            method_used = "kelly"
        elif self.method == "hybrid" and expected_return is not None and return_variance is not None:
            vol_w = self._vol_target_weight(signal_score, realized_annual_vol)
            kelly_w = self._kelly_weight(signal_score, expected_return, return_variance)
            weight = 0.5 * vol_w + 0.5 * kelly_w
            method_used = "hybrid"
        else:
            weight = self._vol_target_weight(signal_score, realized_annual_vol)
            method_used = "vol_target"

        capped = False
        if abs(weight) > self.max_position_pct:
            weight = np.sign(weight) * self.max_position_pct
            capped = True

        if sector and (current_sector_pct + abs(weight)) > self.max_sector_pct:
            allowed = max(self.max_sector_pct - current_sector_pct, 0.0)
            if allowed < abs(weight):
                weight = np.sign(weight) * allowed
                capped = True

        notional = weight * equity
        if abs(notional) < self.min_position_notional:
            notional = 0.0

        rationale = (
            f"{method_used} sizing: signal={signal_score:.3f}, ann_vol={realized_annual_vol:.3f}, "
            f"target_vol={self.target_annual_vol:.3f}, weight={weight:.4f}"
            + (" [capped]" if capped else "")
        )

        return SizingResult(
            symbol=symbol,
            target_notional=float(notional),
            target_pct_of_equity=float(weight),
            method=method_used,
            capped=capped,
            rationale=rationale,
        )
