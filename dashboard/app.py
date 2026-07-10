"""FastAPI backend for the paper-trading dashboard.

Every endpoint here reads from the virtual ledger / live state only.
No endpoint places, modifies, or cancels a real-world order -- /kill and
/resume only toggle the local kill-switch flag file.

*** SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE ***
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from live.kill_switch import get_kill_switch
from live.scheduler import Scheduler
from live.state import LiveState
from risk.metrics import full_metrics_report
from settings import strategies_config

DISCLAIMER = "SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE"

app = FastAPI(title="quant-sim paper trading dashboard", description=DISCLAIMER)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_config = strategies_config()
_state = LiveState(starting_cash=_config["account"]["starting_cash"])
_kill_switch = get_kill_switch()
_scheduler = Scheduler(_config["live"]["cadence"], _config["live"]["equity_market_hours"])


def _current_prices() -> dict[str, float]:
    """Best-effort current price map from each position's last known price;
    the dashboard reads whatever the live loop last recorded rather than
    fetching fresh quotes itself."""
    return {
        sym: pos.avg_entry_price
        for sym, pos in _state.ledger.positions.items()
    }


class StatusResponse(BaseModel):
    disclaimer: str
    kill_switch_halted: bool
    kill_switch_reason: str
    last_data_fetch: Optional[str]
    next_scheduled_run_equity: Optional[str]
    next_scheduled_run_crypto: Optional[str]


@app.get("/performance")
def get_performance():
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
    prices = _current_prices()
    return {
        "disclaimer": DISCLAIMER,
        "positions": _state.ledger.open_positions(prices),
    }


@app.get("/blotter")
def get_blotter():
    df = _state.ledger.blotter_df()
    return {
        "disclaimer": DISCLAIMER,
        "trades": df.to_dict(orient="records") if not df.empty else [],
    }


@app.get("/risk")
def get_risk():
    equity_curve = _state.ledger.equity_curve()
    blotter = _state.ledger.blotter_df()
    trade_pnls = blotter["realized_pnl"] if not blotter.empty else pd.Series(dtype=float)
    metrics = full_metrics_report(equity_curve, trade_pnls) if not equity_curve.empty else {}
    return {"disclaimer": DISCLAIMER, "metrics": metrics}


@app.get("/attribution")
def get_attribution():
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


@app.get("/")
def root():
    return {
        "service": "quant-sim dashboard API",
        "disclaimer": DISCLAIMER,
        "endpoints": [
            "/performance", "/positions", "/blotter", "/risk",
            "/attribution", "/status", "/kill (POST)", "/resume (POST)",
        ],
    }
