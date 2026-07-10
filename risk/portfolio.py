"""Portfolio-level risk constraints, enforced before any order is submitted
to the execution simulator:

  - max gross exposure (sum of |position notional| / equity)
  - max net exposure (sum of signed position notional / equity)
  - correlation-aware sizing (haircut new positions correlated with existing
    book beyond a threshold)
  - circuit breaker: halts all new orders once drawdown exceeds a threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConstraintResult:
    allowed_notional: float
    original_notional: float
    breached: list[str] = field(default_factory=list)
    halted: bool = False
    rationale: str = ""


class CircuitBreaker:
    def __init__(self, max_drawdown_pct: float = 0.15, cooldown_bars: int = 5):
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_bars = cooldown_bars
        self._tripped = False
        self._cooldown_remaining = 0

    def check(self, equity_curve: pd.Series) -> bool:
        """Returns True if trading should be halted. Once tripped, stays
        halted for `cooldown_bars` bars even if drawdown recovers, to avoid
        immediately re-entering into continued volatility."""
        if equity_curve.empty:
            return self._tripped

        running_max = equity_curve.cummax()
        current_dd = equity_curve.iloc[-1] / running_max.iloc[-1] - 1.0

        # Small epsilon guards against floating-point rounding at the exact
        # threshold (e.g. 90/100 - 1.0 evaluates to -0.09999999999999998).
        if current_dd <= -self.max_drawdown_pct + 1e-9:
            self._tripped = True
            self._cooldown_remaining = self.cooldown_bars
        elif self._tripped:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self._tripped = False

        return self._tripped

    def reset(self) -> None:
        self._tripped = False
        self._cooldown_remaining = 0

    @property
    def is_tripped(self) -> bool:
        return self._tripped


class PortfolioConstraints:
    def __init__(self, config: dict):
        self.max_gross_exposure = config.get("max_gross_exposure", 2.0)
        self.max_net_exposure = config.get("max_net_exposure", 1.0)
        self.correlation_lookback_days = config.get("correlation_lookback_days", 60)
        self.correlation_limit = config.get("correlation_limit", 0.7)
        self.correlation_size_haircut = config.get("correlation_size_haircut", 0.5)

    def _current_gross_net(self, positions: dict[str, float], equity: float) -> tuple[float, float]:
        if equity <= 0:
            return 0.0, 0.0
        gross = sum(abs(v) for v in positions.values()) / equity
        net = sum(positions.values()) / equity
        return gross, net

    def apply_correlation_haircut(
        self,
        symbol: str,
        target_notional: float,
        positions: dict[str, float],
        returns_corr_matrix: pd.DataFrame | None,
    ) -> tuple[float, list[str]]:
        breached = []
        if returns_corr_matrix is None or symbol not in returns_corr_matrix.columns:
            return target_notional, breached

        held_symbols = [s for s, notional in positions.items() if abs(notional) > 0 and s != symbol]
        if not held_symbols:
            return target_notional, breached

        correlated_exposure = False
        for held in held_symbols:
            if held not in returns_corr_matrix.columns:
                continue
            corr = returns_corr_matrix.loc[symbol, held]
            if pd.notna(corr) and abs(corr) > self.correlation_limit:
                correlated_exposure = True
                breached.append(f"correlation({symbol},{held})={corr:.2f} > {self.correlation_limit}")

        if correlated_exposure:
            target_notional *= (1 - self.correlation_size_haircut)
        return target_notional, breached

    def check_exposure_limits(
        self,
        symbol: str,
        target_notional: float,
        positions: dict[str, float],
        equity: float,
    ) -> ConstraintResult:
        breached = []
        other_positions = {s: v for s, v in positions.items() if s != symbol}
        hypothetical = dict(other_positions)
        hypothetical[symbol] = target_notional

        gross, net = self._current_gross_net(hypothetical, equity)
        allowed_notional = target_notional

        if gross > self.max_gross_exposure and equity > 0:
            scale = self.max_gross_exposure / gross
            allowed_notional *= scale
            breached.append(f"gross_exposure {gross:.2f} > {self.max_gross_exposure}")

        recomputed = dict(other_positions)
        recomputed[symbol] = allowed_notional
        _, net_after = self._current_gross_net(recomputed, equity)
        if abs(net_after) > self.max_net_exposure and equity > 0:
            excess = abs(net_after) - self.max_net_exposure
            reduction = excess * equity * np.sign(allowed_notional if allowed_notional != 0 else 1)
            allowed_notional -= reduction
            breached.append(f"net_exposure {net_after:.2f} > {self.max_net_exposure}")

        return ConstraintResult(
            allowed_notional=allowed_notional,
            original_notional=target_notional,
            breached=breached,
            rationale="; ".join(breached) if breached else "within limits",
        )

    def evaluate(
        self,
        symbol: str,
        target_notional: float,
        positions: dict[str, float],
        equity: float,
        returns_corr_matrix: pd.DataFrame | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        equity_curve: pd.Series | None = None,
    ) -> ConstraintResult:
        """Full pre-trade risk gate: circuit breaker -> correlation haircut ->
        exposure limits. Returns the allowed notional (possibly reduced or
        zeroed) plus a human-readable rationale for the blotter."""
        if circuit_breaker is not None and equity_curve is not None:
            if circuit_breaker.check(equity_curve):
                return ConstraintResult(
                    allowed_notional=0.0,
                    original_notional=target_notional,
                    breached=["circuit_breaker_tripped"],
                    halted=True,
                    rationale=f"circuit breaker tripped (drawdown > {circuit_breaker.max_drawdown_pct:.0%}); all new orders halted",
                )

        haircut_notional, corr_breaches = self.apply_correlation_haircut(
            symbol, target_notional, positions, returns_corr_matrix
        )
        exposure_result = self.check_exposure_limits(symbol, haircut_notional, positions, equity)
        exposure_result.breached = corr_breaches + exposure_result.breached
        exposure_result.original_notional = target_notional
        if corr_breaches:
            exposure_result.rationale = "; ".join(exposure_result.breached)
        return exposure_result
