"""Backtest report generator: performance summary, benchmark comparison, and
parameter sensitivity tables.

*** ALL OUTPUT FROM THIS MODULE IS BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from risk.metrics import full_metrics_report

BACKTEST_DISCLAIMER = "*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***"


@dataclass
class BacktestReport:
    strategy_metrics: dict
    benchmark_metrics: dict | None = None
    sensitivity_table: pd.DataFrame | None = None
    disclaimer: str = BACKTEST_DISCLAIMER

    def to_markdown(self) -> str:
        lines = [f"# {self.disclaimer}", ""]
        lines.append("## Strategy Performance")
        lines.append(_metrics_to_md_table(self.strategy_metrics))

        if self.benchmark_metrics:
            lines.append("")
            lines.append("## Benchmark (Buy & Hold) Comparison")
            comparison = pd.DataFrame({
                "strategy": self.strategy_metrics,
                "benchmark": self.benchmark_metrics,
            })
            lines.append(comparison.to_markdown())

        if self.sensitivity_table is not None and not self.sensitivity_table.empty:
            lines.append("")
            lines.append("## Parameter Sensitivity")
            lines.append(self.sensitivity_table.to_markdown(index=False))

        lines.append("")
        lines.append(f"_{self.disclaimer}_")
        return "\n".join(lines)


def _metrics_to_md_table(metrics: dict) -> str:
    df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    return df.to_markdown(index=False)


def build_report(
    equity_curve: pd.Series,
    trade_pnls: pd.Series | None = None,
    benchmark_equity: pd.Series | None = None,
    sensitivity_results: list[dict] | None = None,
) -> BacktestReport:
    strategy_metrics = full_metrics_report(equity_curve, trade_pnls, benchmark_equity)
    benchmark_metrics = full_metrics_report(benchmark_equity) if benchmark_equity is not None else None
    sensitivity_df = pd.DataFrame(sensitivity_results) if sensitivity_results else None
    return BacktestReport(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        sensitivity_table=sensitivity_df,
    )


def buy_and_hold_equity(close_prices: pd.Series, starting_cash: float = 100000.0) -> pd.Series:
    """Construct a simple buy-and-hold benchmark equity curve from a close
    price series (e.g. SPY or BTCUSDT), for comparison against the strategy."""
    normalized = close_prices / close_prices.iloc[0]
    return normalized * starting_cash


def parameter_sensitivity_table(
    backtest_fn, features: pd.DataFrame, param_grid: dict[str, list],
) -> pd.DataFrame:
    """Runs `backtest_fn(features, **params) -> {'equity_curve': pd.Series}`
    over a Cartesian grid of parameters and reports Sharpe/CAGR/max-drawdown
    per combination."""
    import itertools

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        result = backtest_fn(features, **params)
        metrics = full_metrics_report(result["equity_curve"])
        rows.append({**params, **metrics})
    return pd.DataFrame(rows)
