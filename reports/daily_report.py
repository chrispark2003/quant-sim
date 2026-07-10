"""Daily/weekly performance report generator (Markdown, with optional PDF
export), built strictly from the actual ledger/trade history.

*** SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE ***
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from live.state import LiveState
from risk.metrics import full_metrics_report
from settings import strategies_config

DISCLAIMER = "SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE"


def generate_report(state: LiveState, period_days: int = 1) -> str:
    ledger = state.ledger
    equity_curve = ledger.equity_curve()
    blotter = ledger.blotter_df()

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
    if not blotter.empty:
        blotter["timestamp"] = pd.to_datetime(blotter["timestamp"], utc=True, errors="coerce")
        period_trades = blotter[blotter["timestamp"] >= cutoff]
    else:
        period_trades = blotter

    trade_pnls = period_trades["realized_pnl"] if not period_trades.empty else pd.Series(dtype=float)
    metrics = full_metrics_report(equity_curve, trade_pnls) if not equity_curve.empty else {}

    lines = [
        f"# Daily Performance Report -- {datetime.now(timezone.utc).date().isoformat()}",
        "",
        f"**{DISCLAIMER}**",
        "",
        "## Account Summary",
        f"- Starting cash: ${ledger.starting_cash:,.2f}",
        f"- Current cash: ${ledger.cash:,.2f}",
        f"- Realized P&L (all-time): ${ledger.total_realized_pnl():,.2f}",
        f"- Trades in period: {len(period_trades)}",
        "",
        "## Risk / Performance Metrics",
    ]
    if metrics:
        for k, v in metrics.items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    else:
        lines.append("- Not enough history yet to compute metrics.")

    lines.append("")
    lines.append("## Trade Blotter (period)")
    if not period_trades.empty:
        lines.append(period_trades.to_markdown(index=False))
    else:
        lines.append("_No trades in this period._")

    lines.append("")
    lines.append(f"**{DISCLAIMER}**")
    return "\n".join(lines)


def write_report(output_path: str, period_days: int = 1) -> str:
    config = strategies_config()
    state = LiveState(starting_cash=config["account"]["starting_cash"])
    report_md = generate_report(state, period_days=period_days)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_md)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the daily/weekly paper-trading performance report.")
    parser.add_argument("--period-days", type=int, default=1)
    parser.add_argument("--output", type=str, default="reports/daily_report.md")
    args = parser.parse_args()
    path = write_report(args.output, period_days=args.period_days)
    print(f"Report written to {path}. {DISCLAIMER}")
