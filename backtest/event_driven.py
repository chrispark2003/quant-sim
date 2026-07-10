"""Event-driven backtester: bar-by-bar simulation that routes every order
through the same ExecutionSimulator and Ledger used by live paper trading,
for a much more realistic picture of fills, slippage, and costs than the
vectorized backtester.

*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***
"""
from __future__ import annotations

import pandas as pd

from execution.ledger import Ledger
from execution.orders import Order, OrderSide, OrderType
from execution.simulator import Bar, ExecutionSimulator
from risk.portfolio import CircuitBreaker, PortfolioConstraints
from risk.sizer import PositionSizer

BACKTEST_DISCLAIMER = "BACKTEST RESULTS -- NOT LIVE PERFORMANCE"


class EventDrivenBacktester:
    def __init__(self, execution_config: dict, sizer_config: dict, portfolio_config: dict,
                 circuit_breaker_config: dict, starting_cash: float = 100000.0):
        self.simulator = ExecutionSimulator(execution_config)
        self.sizer = PositionSizer(sizer_config)
        self.constraints = PortfolioConstraints(portfolio_config)
        self.circuit_breaker = CircuitBreaker(
            max_drawdown_pct=circuit_breaker_config.get("max_drawdown_pct", 0.15),
            cooldown_bars=circuit_breaker_config.get("cooldown_bars", 5),
        )
        self.ledger = Ledger(starting_cash=starting_cash)

    def run(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        scores_by_symbol: dict[str, pd.Series],
        asset_class: str = "equity",
        sector_by_symbol: dict[str, str] | None = None,
        corr_matrix: pd.DataFrame | None = None,
    ) -> dict:
        """bars_by_symbol: {symbol -> OHLCV DataFrame indexed by timestamp}.
        scores_by_symbol: {symbol -> signal score Series, same index}.
        Iterates bar-by-bar in time order across the whole universe, sizing
        and routing orders through the execution simulator each step."""
        sector_by_symbol = sector_by_symbol or {}
        all_timestamps = sorted(set().union(*[df.index for df in bars_by_symbol.values()]))

        returns_wide = pd.DataFrame({
            sym: df["close"].pct_change() for sym, df in bars_by_symbol.items()
        })

        for i, ts in enumerate(all_timestamps):
            current_prices = {}
            for sym, df in bars_by_symbol.items():
                if ts in df.index:
                    current_prices[sym] = df.loc[ts, "close"]
                elif sym in self.ledger.positions:
                    current_prices[sym] = self.ledger.positions[sym].avg_entry_price

            self.ledger.snapshot_equity(ts, current_prices)
            equity_curve = self.ledger.equity_curve()
            equity = equity_curve.iloc[-1] if not equity_curve.empty else self.ledger.starting_cash

            if i + 1 >= len(all_timestamps):
                continue
            next_ts = all_timestamps[i + 1]

            for sym, df in bars_by_symbol.items():
                if ts not in df.index or next_ts not in df.index:
                    continue
                score = scores_by_symbol.get(sym, pd.Series(dtype=float)).get(ts, 0.0)
                if pd.isna(score):
                    score = 0.0

                realized_vol = df.loc[ts].get("volatility", self.sizer.target_annual_vol)
                if pd.isna(realized_vol) or realized_vol <= 0:
                    realized_vol = self.sizer.target_annual_vol

                positions_notional = self.ledger.position_notional_map(current_prices)
                sector = sector_by_symbol.get(sym)
                sector_pct = 0.0
                if sector:
                    sector_notional = sum(
                        abs(v) for s, v in positions_notional.items()
                        if sector_by_symbol.get(s) == sector and s != sym
                    )
                    sector_pct = sector_notional / equity if equity > 0 else 0.0

                sizing = self.sizer.size_position(
                    symbol=sym, signal_score=score, equity=equity,
                    realized_annual_vol=realized_vol, sector=sector, current_sector_pct=sector_pct,
                )

                constraint_result = self.constraints.evaluate(
                    symbol=sym, target_notional=sizing.target_notional,
                    positions=positions_notional, equity=equity,
                    returns_corr_matrix=corr_matrix, circuit_breaker=self.circuit_breaker,
                    equity_curve=equity_curve,
                )

                current_notional = positions_notional.get(sym, 0.0)
                delta_notional = constraint_result.allowed_notional - current_notional
                price_now = current_prices.get(sym)
                if not price_now or abs(delta_notional) < 1e-6:
                    continue

                qty = delta_notional / price_now
                side = OrderSide.BUY if qty > 0 else OrderSide.SELL
                order = Order(
                    symbol=sym, side=side, quantity=abs(qty), order_type=OrderType.MARKET,
                    asset_class=asset_class, rationale=f"{sizing.rationale} | {constraint_result.rationale}",
                    strategy="event_driven_backtest",
                )
                next_bar = Bar(
                    timestamp=next_ts, open=df.loc[next_ts, "open"], high=df.loc[next_ts, "high"],
                    low=df.loc[next_ts, "low"], close=df.loc[next_ts, "close"], volume=df.loc[next_ts, "volume"],
                )
                fill = self.simulator.execute(order, next_bar)
                if fill:
                    self.ledger.apply_fill(fill)

        return {
            "ledger": self.ledger,
            "equity_curve": self.ledger.equity_curve(),
            "blotter": self.ledger.blotter_df(),
            "disclaimer": BACKTEST_DISCLAIMER,
        }
