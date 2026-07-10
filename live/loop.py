"""Autonomous paper-trading event loop.

Fully autonomous: fetch live data -> compute features -> run signals -> run
risk checks -> size positions -> submit to the execution simulator -> log
fill -> repeat. No human approval is requested per trade -- this is by
design for a paper-trading bot, and it NEVER connects to a real brokerage;
every "fill" is computed by execution/simulator.py against the virtual
ledger in live/state.py.

*** SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE ***
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from data.feature_store import FeatureStore
from data.ingestion.binance_adapter import BinanceAdapter
from data.ingestion.yfinance_adapter import YFinanceAdapter
from data.store import get_store
from execution.orders import Order, OrderSide, OrderType
from execution.simulator import Bar, ExecutionSimulator
from live.kill_switch import get_kill_switch
from live.scheduler import Scheduler
from live.state import LiveState
from risk.portfolio import CircuitBreaker, PortfolioConstraints
from risk.sizer import PositionSizer
from settings import strategies_config, symbols_config
from signals.ensemble import EnsembleSignal

DISCLAIMER = "SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE"

logger = logging.getLogger("quant_sim.live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AutonomousPaperTradingLoop:
    def __init__(self):
        self.config = strategies_config()
        self.symbols_cfg = symbols_config()
        self.store = get_store()
        self.feature_store = FeatureStore()
        self.state = LiveState(starting_cash=self.config["account"]["starting_cash"])
        self.kill_switch = get_kill_switch()

        self.scheduler = Scheduler(self.config["live"]["cadence"], self.config["live"]["equity_market_hours"])
        self.ensemble = EnsembleSignal.from_config(self.config["signals"])
        self.sizer = PositionSizer(self.config["risk"]["sizing"])
        self.constraints = PortfolioConstraints(self.config["risk"]["portfolio"])
        cb_cfg = self.config["risk"]["circuit_breaker"]
        self.circuit_breaker = CircuitBreaker(
            max_drawdown_pct=cb_cfg["max_drawdown_pct"],
            cooldown_bars=cb_cfg["cooldown_bars"],
        )
        self.simulator = ExecutionSimulator(self.config["execution"])

        self.equity_adapter = YFinanceAdapter()
        self.crypto_adapter = BinanceAdapter()

        self.sector_by_symbol = {
            e["symbol"]: e["sector"] for e in self.symbols_cfg.get("equities", [])
        }
        self.sector_by_symbol.update({
            c["symbol"]: c["sector"] for c in self.symbols_cfg.get("crypto", [])
        })

    def _adapter_for(self, asset_class: str):
        return self.crypto_adapter if asset_class == "crypto" else self.equity_adapter

    _TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}

    def _fetch_recent_bars(self, symbol: str, asset_class: str, n_bars: int = 500) -> pd.DataFrame:
        """Fetch the most recent ~n_bars bars, anchored to now.

        The window must be sized to the bars needed (not a fixed day count):
        yfinance refuses 5m requests older than 60 days, and Binance's kline
        endpoint returns the FIRST 1000 bars after startTime -- so an oversized
        window silently yields stale data instead of the latest bars. Equities
        get a 4x span buffer to cover weekends/market-closed hours."""
        adapter = self._adapter_for(asset_class)
        end = datetime.now(timezone.utc)
        timeframe = "1m" if asset_class == "crypto" else "5m"
        span_seconds = n_bars * self._TIMEFRAME_SECONDS[timeframe]
        if asset_class == "equity":
            span_seconds *= 4
        start = end - pd.Timedelta(seconds=span_seconds)
        long_df = adapter.fetch_ohlcv(symbol, timeframe, start, end)
        if long_df.empty:
            return long_df
        self.store.write(long_df)
        from data.schema import long_to_ohlcv
        return long_to_ohlcv(long_df)

    def process_symbol(self, symbol: str, asset_class: str) -> None:
        if self.kill_switch.is_halted():
            logger.warning("[%s] kill switch engaged -- skipping. %s", symbol, DISCLAIMER)
            return

        ohlcv = self._fetch_recent_bars(symbol, asset_class)
        if ohlcv.empty or len(ohlcv) < 5:
            logger.info("[%s] insufficient data this cycle", symbol)
            return

        latest_ts = ohlcv.index[-1]
        if self.state.already_processed(symbol, latest_ts):
            logger.debug("[%s] bar %s already processed, skipping", symbol, latest_ts)
            return

        features = self.feature_store.compute(symbol, ohlcv, self.config["signals"]["mean_reversion"])
        score_series = self.ensemble.generate_signal(features)
        score = float(score_series.iloc[-1]) if not score_series.empty else 0.0

        current_price = float(ohlcv["close"].iloc[-1])
        realized_vol = features["volatility"].iloc[-1]
        if pd.isna(realized_vol) or realized_vol <= 0:
            realized_vol = self.sizer.target_annual_vol

        current_prices = {symbol: current_price}
        equity_curve = self.state.ledger.equity_curve()
        equity = equity_curve.iloc[-1] if not equity_curve.empty else self.state.ledger.starting_cash
        positions_notional = self.state.ledger.position_notional_map(current_prices)

        sizing = self.sizer.size_position(
            symbol=symbol, signal_score=score, equity=equity,
            realized_annual_vol=realized_vol, sector=self.sector_by_symbol.get(symbol),
        )

        constraint_result = self.constraints.evaluate(
            symbol=symbol, target_notional=sizing.target_notional,
            positions=positions_notional, equity=equity,
            circuit_breaker=self.circuit_breaker, equity_curve=equity_curve,
        )

        current_notional = positions_notional.get(symbol, 0.0)
        delta_notional = constraint_result.allowed_notional - current_notional

        rationale = f"score={score:.3f} | {sizing.rationale} | {constraint_result.rationale}"

        if constraint_result.halted or abs(delta_notional) < self.sizer.min_position_notional:
            logger.info("[%s] no trade this cycle: %s", symbol, rationale)
            self.state.mark_processed(symbol, latest_ts)
            self.state.ledger.snapshot_equity(latest_ts, current_prices)
            self.state.save()
            return

        qty = delta_notional / current_price
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        order = Order(
            symbol=symbol, side=side, quantity=abs(qty), order_type=OrderType.MARKET,
            asset_class=asset_class, rationale=rationale, strategy="ensemble_live",
        )
        bar = Bar(
            timestamp=latest_ts, open=float(ohlcv["open"].iloc[-1]), high=float(ohlcv["high"].iloc[-1]),
            low=float(ohlcv["low"].iloc[-1]), close=current_price, volume=float(ohlcv["volume"].iloc[-1]),
        )

        fill = self.simulator.execute(order, bar)
        if fill:
            entry = self.state.ledger.apply_fill(fill)
            logger.info("[%s] FILLED %s %.4f @ %.4f | fees=%.4f | %s | %s",
                        symbol, side.value, fill.quantity, fill.fill_price, fill.fees, rationale, DISCLAIMER)
        else:
            logger.info("[%s] order not filled this cycle (volume cap or no book): %s", symbol, rationale)

        self.state.mark_processed(symbol, latest_ts)
        self.state.ledger.snapshot_equity(latest_ts, current_prices)
        self.state.save()

    def _process_symbol_safe(self, symbol: str, asset_class: str) -> None:
        """One symbol's failure (network error, geoblock, bad data) must not
        abort the rest of the cycle for other symbols."""
        try:
            self.process_symbol(symbol, asset_class)
        except Exception as e:
            logger.error("[%s] skipping this cycle after error: %s", symbol, e)

    def run_once(self) -> None:
        now = datetime.now(timezone.utc)
        for equity in self.symbols_cfg.get("equities", []):
            if self.scheduler.should_run("equity", now):
                self._process_symbol_safe(equity["symbol"], "equity")
        if self.scheduler.should_run("equity", now):
            self.scheduler.mark_ran("equity", now)

        for crypto in self.symbols_cfg.get("crypto", []):
            if self.scheduler.should_run("crypto", now):
                self._process_symbol_safe(crypto["symbol"], "crypto")
        if self.scheduler.should_run("crypto", now):
            self.scheduler.mark_ran("crypto", now)

    def run_forever(self, poll_interval_seconds: int | None = None) -> None:
        interval = poll_interval_seconds or self.config["live"]["poll_interval_seconds"]
        logger.info("Starting autonomous paper-trading loop. %s", DISCLAIMER)
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("error during live loop iteration -- continuing")
            time.sleep(interval)


if __name__ == "__main__":
    AutonomousPaperTradingLoop().run_forever()
