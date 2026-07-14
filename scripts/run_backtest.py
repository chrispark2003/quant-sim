"""Sample backtest runner for `make backtest`: runs the event-driven
backtester against the last 1yr of stored data for the equities universe
using the ensemble signal, then writes a markdown report.

*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***
"""
from __future__ import annotations

from data.feature_store import FeatureStore
from data.schema import long_to_ohlcv
from data.store import get_store
from backtest.event_driven import EventDrivenBacktester
from backtest.report import build_report, buy_and_hold_equity
from settings import strategies_config, symbols_config
from signals.ensemble import EnsembleSignal

DISCLAIMER = "*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***"


def main(output_path: str = "reports/backtest_report.md") -> None:
    config = strategies_config()
    symbols = symbols_config()
    store = get_store()

    equity_symbols = [e["symbol"] for e in symbols.get("equities", [])]
    sector_by_symbol = {e["symbol"]: e["sector"] for e in symbols.get("equities", [])}

    feature_store = FeatureStore()
    ensemble = EnsembleSignal.from_config(config["signals"])

    bars_by_symbol, scores_by_symbol = {}, {}
    for symbol in equity_symbols:
        raw = store.query(symbols=[symbol], asset_class="equity")
        if raw.empty:
            print(f"No stored data for {symbol}, skipping (run `make setup` first).")
            continue
        ohlcv = long_to_ohlcv(raw)
        features = feature_store.compute(symbol, ohlcv, config["signals"]["mean_reversion"])
        bars_by_symbol[symbol] = features
        scores_by_symbol[symbol] = ensemble.generate_signal(features)

    if not bars_by_symbol:
        print("No data available -- run `make setup` first to fetch historical data.")
        return

    backtester = EventDrivenBacktester(
        execution_config=config["execution"],
        sizer_config=config["risk"]["sizing"],
        portfolio_config=config["risk"]["portfolio"],
        circuit_breaker_config=config["risk"]["circuit_breaker"],
        starting_cash=config["account"]["starting_cash"],
    )
    result = backtester.run(bars_by_symbol, scores_by_symbol, asset_class="equity", sector_by_symbol=sector_by_symbol)

    benchmark_symbol = symbols.get("benchmarks", {}).get("equity")
    benchmark_equity = None
    if benchmark_symbol:
        bench_raw = store.query(symbols=[benchmark_symbol], asset_class="equity")
        if not bench_raw.empty:
            bench_ohlcv = long_to_ohlcv(bench_raw)
            benchmark_equity = buy_and_hold_equity(bench_ohlcv["close"], config["account"]["starting_cash"])

    trade_pnls = result["blotter"]["realized_pnl"] if not result["blotter"].empty else None
    report = build_report(result["equity_curve"], trade_pnls=trade_pnls, benchmark_equity=benchmark_equity)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report.to_markdown())

    _write_dashboard_summary(result, report, equity_symbols)

    print(DISCLAIMER)
    print(f"Backtest report written to {output_path}")


def _write_dashboard_summary(result: dict, report, universe: list[str]) -> None:
    """Persist a small JSON summary for the dashboard's /backtest endpoint --
    metrics plus a downsampled equity curve for the archive sparkline."""
    import json

    from settings import state_dir

    curve = result["equity_curve"]
    n_points = 90
    if len(curve) > n_points:
        step = max(1, len(curve) // n_points)
        curve = curve.iloc[::step]
    trades = int(len(result["blotter"])) if not result["blotter"].empty else 0
    start = str(result["equity_curve"].index[0].date()) if len(result["equity_curve"]) else ""
    end = str(result["equity_curve"].index[-1].date()) if len(result["equity_curve"]) else ""

    summary = {
        "label": f"Event-driven backtest -- {start} -> {end} · {len(universe)}-symbol universe",
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "metrics": {**{k: (None if v != v else v) for k, v in report.strategy_metrics.items()},
                    "trades": trades},
        "curve": [float(v) for v in curve.values],
    }
    out = state_dir() / "backtest_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Dashboard backtest summary written to {out}")


if __name__ == "__main__":
    main()
