"""FastAPI backend for the paper-trading dashboard.

Every endpoint here reads from the virtual ledger / live state only.
No endpoint places, modifies, or cancels a real-world order -- /kill and
/resume only toggle the local kill-switch flag file.

*** SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE ***
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from live.kill_switch import get_kill_switch
from live.scheduler import Scheduler
from live.state import LiveState
from risk.metrics import full_metrics_report
from settings import state_dir, strategies_config

DISCLAIMER = "SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE"

app = FastAPI(title="quant-sim paper trading dashboard", description=DISCLAIMER)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_config = strategies_config()
_state = LiveState(starting_cash=_config["account"]["starting_cash"])
_kill_switch = get_kill_switch()
_scheduler = Scheduler(_config["live"]["cadence"], _config["live"]["equity_market_hours"])

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_ledger_mtime: float = 0.0


def _refresh_state() -> None:
    """The live loop runs in a separate process and persists the ledger to
    disk after every fill; reload it here whenever the file changes so the
    API never serves a stale snapshot from its own startup time."""
    global _state, _ledger_mtime
    try:
        mtime = _state.ledger_path.stat().st_mtime
    except FileNotFoundError:
        return
    if mtime != _ledger_mtime:
        _state = LiveState(starting_cash=_config["account"]["starting_cash"])
        _ledger_mtime = mtime


_price_cache: dict = {"ts": 0.0, "prices": {}, "sparks": {}}
_PRICE_CACHE_SECONDS = 10
_SPARK_POINTS = 44


def _current_prices() -> dict[str, float]:
    """Mark open positions at the latest close the live loop wrote to the
    DuckDB store (falling back to entry price when a symbol has no bars yet).
    Cached briefly so dashboard polling doesn't hammer the store."""
    now = time.time()
    open_syms = [s for s, p in _state.ledger.positions.items() if p.quantity != 0]
    if now - _price_cache["ts"] > _PRICE_CACHE_SECONDS or set(open_syms) - set(_price_cache["sparks"]):
        prices: dict[str, float] = {}
        sparks: dict[str, list[float]] = {}
        if open_syms:
            try:
                from data.store import get_store
                df = get_store().query(symbols=open_syms, fields=["close"])
                if not df.empty:
                    df = df.sort_values("timestamp")
                    for sym, grp in df.groupby("symbol"):
                        closes = grp["value"].tail(_SPARK_POINTS).tolist()
                        if closes:
                            prices[sym] = float(closes[-1])
                            sparks[sym] = [float(c) for c in closes]
            except Exception:
                prices, sparks = {}, {}
        _price_cache.update(ts=now, prices=prices, sparks=sparks)
    return {
        sym: _price_cache["prices"].get(sym, pos.avg_entry_price)
        for sym, pos in _state.ledger.positions.items()
    }


def _sparklines() -> dict[str, list[float]]:
    _current_prices()  # ensure cache is fresh
    return _price_cache["sparks"]


class StatusResponse(BaseModel):
    disclaimer: str
    kill_switch_halted: bool
    kill_switch_reason: str
    last_data_fetch: Optional[str]
    next_scheduled_run_equity: Optional[str]
    next_scheduled_run_crypto: Optional[str]


@app.get("/performance")
def get_performance():
    _refresh_state()
    prices = _current_prices()
    equity_curve = _state.ledger.equity_curve()
    return {
        "disclaimer": DISCLAIMER,
        "starting_cash": _state.ledger.starting_cash,
        "current_cash": _state.ledger.cash,
        "realized_pnl": _state.ledger.total_realized_pnl(),
        "unrealized_pnl": _state.ledger.total_unrealized_pnl(prices),
        "equity_curve": [
            {"timestamp": str(ts), "equity": float(v)} for ts, v in equity_curve.items()
        ],
        "max_drawdown": float((equity_curve / equity_curve.cummax() - 1.0).min()) if not equity_curve.empty else 0.0,
    }


@app.get("/positions")
def get_positions():
    _refresh_state()
    prices = _current_prices()
    sparks = _sparklines()
    positions = _state.ledger.open_positions(prices)
    for pos in positions:
        pos["spark"] = sparks.get(pos["symbol"], [])
    return {
        "disclaimer": DISCLAIMER,
        "positions": positions,
    }


@app.get("/blotter")
def get_blotter():
    _refresh_state()
    df = _state.ledger.blotter_df()
    return {
        "disclaimer": DISCLAIMER,
        "trades": df.to_dict(orient="records") if not df.empty else [],
    }


@app.get("/risk")
def get_risk():
    _refresh_state()
    equity_curve = _state.ledger.equity_curve()
    blotter = _state.ledger.blotter_df()
    trade_pnls = blotter["realized_pnl"] if not blotter.empty else pd.Series(dtype=float)
    metrics = full_metrics_report(equity_curve, trade_pnls) if not equity_curve.empty else {}
    return {"disclaimer": DISCLAIMER, "metrics": metrics}


@app.get("/attribution")
def get_attribution():
    _refresh_state()
    df = _state.ledger.blotter_df()
    if df.empty:
        return {"disclaimer": DISCLAIMER, "attribution": []}
    grouped = df.groupby("strategy")["realized_pnl"].sum().reset_index()
    grouped["fees_paid"] = df.groupby("strategy")["fees"].sum().values
    return {"disclaimer": DISCLAIMER, "attribution": grouped.to_dict(orient="records")}


@app.get("/status", response_model=StatusResponse)
def get_status():
    ks_status = _kill_switch.status()
    return StatusResponse(
        disclaimer=DISCLAIMER,
        kill_switch_halted=ks_status.get("halted", False),
        kill_switch_reason=ks_status.get("reason", ""),
        last_data_fetch=ks_status.get("updated_at"),
        next_scheduled_run_equity=str(_scheduler.next_run_estimate("equity")) if _scheduler.next_run_estimate("equity") else None,
        next_scheduled_run_crypto=str(_scheduler.next_run_estimate("crypto")) if _scheduler.next_run_estimate("crypto") else None,
    )


@app.post("/kill")
def post_kill():
    _kill_switch.halt(reason="halted via dashboard /kill endpoint")
    return {"disclaimer": DISCLAIMER, "status": "halted"}


@app.post("/resume")
def post_resume():
    _kill_switch.resume()
    return {"disclaimer": DISCLAIMER, "status": "resumed"}


@app.get("/backtest")
def get_backtest():
    """Latest saved backtest summary (written by scripts/run_backtest.py).
    Always labeled as historical simulation, never blended with live data."""
    summary_path = state_dir() / "backtest_summary.json"
    if not summary_path.exists():
        return {"disclaimer": DISCLAIMER, "available": False}
    with open(summary_path) as f:
        summary = json.load(f)
    return {"disclaimer": DISCLAIMER, "available": True, **summary}


@app.get("/")
def root():
    """Serve the Ledger dashboard UI."""
    ledger_html = _FRONTEND_DIR / "ledger.html"
    if ledger_html.exists():
        return FileResponse(ledger_html, media_type="text/html")
    return {"service": "quant-sim dashboard API", "disclaimer": DISCLAIMER,
            "note": "frontend/ledger.html not found; API-only mode", "endpoints_index": "/api"}


@app.get("/api")
def api_index():
    return {
        "service": "quant-sim dashboard API",
        "disclaimer": DISCLAIMER,
        "endpoints": [
            "/performance", "/positions", "/blotter", "/risk",
            "/attribution", "/status", "/backtest", "/kill (POST)", "/resume (POST)",
        ],
    }
