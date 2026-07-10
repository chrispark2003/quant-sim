"""Performance/risk metrics computed strictly from actual trade/equity
history -- never from assumed or theoretical distributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def returns_from_equity(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = _TRADING_DAYS, risk_free: float = 0.0) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, periods_per_year: int = _TRADING_DAYS, risk_free: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    if not downside_std or np.isnan(downside_std) or downside_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = _TRADING_DAYS) -> float:
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return 0.0
    cagr = (1 + total_return) ** (1 / years) - 1
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(cagr / mdd)


def cagr(equity_curve: pd.Series, periods_per_year: int = _TRADING_DAYS) -> float:
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def historical_var(returns: pd.Series, confidence: float = 0.95, window: int = 30) -> float:
    """Historical (empirical) rolling VaR, expressed as a positive loss
    fraction. E.g. 0.03 means a 3% loss at the given confidence level."""
    recent = returns.tail(window)
    if recent.empty:
        return 0.0
    return float(-np.percentile(recent, (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95, window: int = 30) -> float:
    """Conditional VaR (expected shortfall): average loss beyond the VaR
    threshold."""
    recent = returns.tail(window)
    if recent.empty:
        return 0.0
    var_threshold = -historical_var(recent, confidence, window)
    tail = recent[recent <= var_threshold]
    if tail.empty:
        return float(-var_threshold)
    return float(-tail.mean())


def beta_to_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    cov = aligned.cov().iloc[0, 1]
    bench_var = aligned.iloc[:, 1].var()
    if bench_var == 0:
        return 0.0
    return float(cov / bench_var)


def hit_rate(trade_pnls: pd.Series) -> float:
    if trade_pnls.empty:
        return 0.0
    return float((trade_pnls > 0).mean())


def avg_win_loss(trade_pnls: pd.Series) -> dict[str, float]:
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    return {
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "win_loss_ratio": float(wins.mean() / abs(losses.mean())) if not losses.empty and losses.mean() != 0 and not wins.empty else 0.0,
    }


def full_metrics_report(equity_curve: pd.Series, trade_pnls: pd.Series | None = None,
                         benchmark_equity: pd.Series | None = None,
                         periods_per_year: int = _TRADING_DAYS) -> dict:
    """Compute the full metrics suite from an equity curve (and optionally
    per-trade P&L / a benchmark equity curve for beta)."""
    returns = returns_from_equity(equity_curve)
    report = {
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "calmar": calmar_ratio(equity_curve, periods_per_year),
        "cagr": cagr(equity_curve, periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "var_95": historical_var(returns, 0.95, window=30),
        "var_99": historical_var(returns, 0.99, window=30),
        "cvar_95": historical_cvar(returns, 0.95, window=30),
        "cvar_99": historical_cvar(returns, 0.99, window=30),
    }
    if benchmark_equity is not None:
        bench_returns = returns_from_equity(benchmark_equity)
        report["beta"] = beta_to_benchmark(returns, bench_returns)
    if trade_pnls is not None and not trade_pnls.empty:
        report["hit_rate"] = hit_rate(trade_pnls)
        report.update(avg_win_loss(trade_pnls))
    return report
