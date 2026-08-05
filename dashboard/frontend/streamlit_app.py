"""Streamlit dashboard frontend for the paper-trading simulator.

Reads exclusively from the FastAPI backend (dashboard/app.py), which in turn
reads only from the virtual ledger / live state. Nothing on this page can
place, modify, or cancel a real order.

*** SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE ***
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

DISCLAIMER = "SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE"
API_BASE = os.environ.get("QUANT_SIM_API_BASE", "http://localhost:8000")
CONTROL_TOKEN = os.environ.get("DASHBOARD_CONTROL_TOKEN")

st.set_page_config(page_title="quant-sim paper trading", layout="wide")

st_autorefresh(interval=30_000, key="dashboard_refresh")

st.title("quant-sim -- Autonomous Paper Trading Bot")
st.error(f"⚠️ {DISCLAIMER}", icon="⚠️")


def api_get(path: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Could not reach API at {API_BASE}{path}: {e}")
        return {}


def api_post(path: str) -> dict:
    try:
        headers = {"Authorization": f"Bearer {CONTROL_TOKEN}"} if CONTROL_TOKEN else {}
        resp = requests.post(f"{API_BASE}{path}", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Could not reach API at {API_BASE}{path}: {e}")
        return {}


# --- Status / kill switch bar ---
status = api_get("/status")
col1, col2, col3, col4 = st.columns(4)
halted = status.get("kill_switch_halted", False)
col1.metric("Kill Switch", "HALTED" if halted else "ACTIVE")
col2.metric("Next Equity Run", status.get("next_scheduled_run_equity") or "n/a")
col3.metric("Next Crypto Run", status.get("next_scheduled_run_crypto") or "n/a")
with col4:
    if halted:
        if st.button("Resume Trading"):
            api_post("/resume")
            st.rerun()
    else:
        if st.button("Engage Kill Switch"):
            api_post("/kill")
            st.rerun()

st.divider()

# --- Live performance (SIMULATED) ---
st.header("📈 Live Performance -- SIMULATED / PAPER TRADING")
perf = api_get("/performance")
if perf:
    curve = pd.DataFrame(perf.get("equity_curve", []))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Starting Cash", f"${perf.get('starting_cash', 0):,.2f}")
    m2.metric("Realized P&L", f"${perf.get('realized_pnl', 0):,.2f}")
    m3.metric("Unrealized P&L", f"${perf.get('unrealized_pnl', 0):,.2f}")
    m4.metric("Max Drawdown", f"{perf.get('max_drawdown', 0):.2%}")

    if not curve.empty:
        curve["timestamp"] = pd.to_datetime(curve["timestamp"])
        st.line_chart(curve.set_index("timestamp")["equity"])
    else:
        st.info("No equity history yet -- start the paper-trading loop with `make paper`.")

st.subheader("Open Positions")
positions = api_get("/positions").get("positions", [])
st.dataframe(pd.DataFrame(positions) if positions else pd.DataFrame(
    columns=["symbol", "quantity", "avg_entry_price", "current_price", "unrealized_pnl", "market_value"]
), use_container_width=True)

st.subheader("Trade Blotter (every autonomous fill)")
blotter = api_get("/blotter").get("trades", [])
st.dataframe(pd.DataFrame(blotter) if blotter else pd.DataFrame(
    columns=["timestamp", "symbol", "side", "quantity", "fill_price", "fees", "rationale"]
), use_container_width=True)

st.subheader("Risk Metrics")
risk = api_get("/risk").get("metrics", {})
if risk:
    st.dataframe(pd.DataFrame([risk]), use_container_width=True)
else:
    st.info("Not enough trade history yet to compute risk metrics.")

st.subheader("Strategy Attribution")
attribution = api_get("/attribution").get("attribution", [])
st.dataframe(pd.DataFrame(attribution) if attribution else pd.DataFrame(
    columns=["strategy", "realized_pnl", "fees_paid"]
), use_container_width=True)

st.divider()

# --- Backtest section, visually separated ---
st.header("🧪 Backtest Results")
st.warning("*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***")
st.markdown(
    "Run `make backtest` to generate a report, then load its markdown/JSON "
    "output here. Backtest results are historical simulations and are kept "
    "visually separate from the live paper-trading performance above."
)

backtest_report_path = st.text_input("Path to backtest report (markdown)", "reports/backtest_report.md")
if st.button("Load Backtest Report"):
    try:
        with open(backtest_report_path, "r") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.error(f"No report found at {backtest_report_path}. Run `make backtest` first.")

st.caption(DISCLAIMER)
