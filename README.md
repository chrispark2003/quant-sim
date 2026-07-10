# quant-sim

**SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE.**

A quant-grade paper trading bot simulator. Every "trade" in this project is
executed against a local, virtual ledger. **No component in this codebase
connects to a real brokerage or exchange order-placement endpoint** -- data
adapters are read-only, and the only thing that can create a "fill" is
`execution/simulator.py`, which computes a hypothetical fill price from bar
data and hands it to `execution/ledger.py`.

## Architecture overview

```
data/        ingestion adapters + normalized schema + DuckDB store + feature store
signals/     alpha signals (momentum, mean-reversion, cross-sectional, vol breakout,
             news sentiment) + a weighted ensemble combinator
risk/        position sizing (vol-target / Kelly), portfolio constraints,
             circuit breaker, performance/risk metrics
execution/   simulated order book: order types, fill logic, virtual ledger
backtest/    vectorized + event-driven backtesters, walk-forward / purged k-fold
             validation, report generator
live/        autonomous paper-trading event loop, scheduler, state persistence,
             kill switch
dashboard/   FastAPI backend + Streamlit frontend
reports/     daily/weekly performance report generator
tests/       pytest suite covering signals, execution, risk, and the API
```

Data flow, end to end:

1. **Ingestion** (`data/ingestion/*`) pulls OHLCV bars (yfinance, Alpaca,
   Polygon, Binance) and headlines (NewsAPI), normalizes everything to a
   long schema (`timestamp, symbol, asset_class, field, value`), and writes
   it to a local DuckDB file (with optional Parquet export).
2. **Feature store** (`data/feature_store.py`) computes returns, realized
   volatility, RSI, MACD, Bollinger Bands, ATR, and rolling correlation once
   per symbol and caches the result so every signal reuses it.
3. **Signals** (`signals/*`) turn features into a score in `[-1, 1]` per
   symbol; the `EnsembleSignal` combines them with configurable weights.
4. **Risk** (`risk/*`) turns a signal score into a target position size
   (vol-targeting and/or Kelly fraction), then enforces portfolio-level
   gross/net exposure limits, correlation-aware haircuts, and a drawdown
   circuit breaker before anything is allowed to trade.
5. **Execution** (`execution/*`) simulates the fill (slippage, fees, volume
   participation, optional latency) and books it against the virtual ledger.
6. **Backtest** (`backtest/*`) replays this whole pipeline historically,
   either vectorized (fast iteration) or event-driven (realistic, routes
   through the same execution simulator as live).
7. **Live** (`live/*`) runs the same pipeline autonomously on a schedule,
   persisting state so it can resume after a restart without double-trading.
8. **Dashboard** (`dashboard/*`) exposes the ledger/blotter/risk metrics over
   a small FastAPI backend, rendered by a Streamlit frontend that clearly
   separates live paper-trading performance from historical backtest results.

## Data sources

| Source   | Used for                        | Auth |
|----------|----------------------------------|------|
| yfinance | Equity historical OHLCV          | none |
| Alpaca   | Equity market data (data API only) | `ALPACA_API_KEY` / `ALPACA_API_SECRET` |
| Polygon  | Equity market data (aggregates)  | `POLYGON_API_KEY` |
| Binance  | Crypto OHLCV (REST + websocket)  | none (public market data) |
| NewsAPI  | Headlines for the sentiment signal | `NEWSAPI_API_KEY` |

All keys are read from environment variables only (see `.env.example`) --
none are hardcoded anywhere in source.

## Running

```bash
cp .env.example .env        # fill in whichever API keys you have
make setup                  # install deps, init DuckDB, fetch ~1yr of history
make backtest                # run the event-driven backtest + write a report
make paper                   # start the autonomous paper-trading loop
make dashboard                # start the FastAPI backend + Streamlit frontend
make test                    # run the pytest suite
```

`requirements.txt` covers everything except the news-sentiment signal;
`transformers`/`torch` are pulled in separately via `requirements-sentiment.txt`
since the ensemble runs fine without them (weights renormalize automatically):

```bash
pip install -r requirements.txt -r requirements-sentiment.txt
```

## Deploying to EC2

See `deploy/README.md` for the one-time instance setup and
`.github/workflows/deploy.yml` for the CI/CD workflow: on every push to
`main`, tests run first, and the instance's systemd units are only restarted
if they pass.

### Backtest vs. paper-trading mode

- **Backtest** (`make backtest` / `backtest/*`) replays history. All output
  is explicitly labeled `*** BACKTEST RESULTS -- NOT LIVE PERFORMANCE ***`.
- **Paper trading** (`make paper` / `live/loop.py`) runs the same
  signal -> risk -> execution pipeline autonomously against live market data,
  but every fill is still simulated against the virtual ledger. All output is
  labeled `SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE`.

The live loop requires **no human approval per trade** -- that's the point
of a paper-trading bot -- but it never touches real money or a real broker.

## Guardrails

- No adapter, signal, risk module, or execution module ever imports or calls
  a real order-placement endpoint. Read-only market data in; simulated fills
  out.
- `execution/simulator.py` + `execution/ledger.py` are the only places
  "trades" happen, and they only ever mutate an in-memory/on-disk virtual
  ledger.
- `live/kill_switch.py` can halt all new order submission instantly
  (`POST /kill` on the dashboard API), independent of any per-trade approval.
- Every dashboard panel, report, and structured log line is labeled
  `SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE`.
- All API keys come from environment variables (`.env`, never committed) --
  see `settings.py`.

## Known limitations

- The cross-sectional "value" factor is a price-extension proxy, not a real
  fundamentals-based P/E ratio -- no fundamentals data source is wired in.
- `news_sentiment.py` lazily imports `transformers`/`torch`; if those aren't
  installed the ensemble will simply skip that component (renormalizing the
  remaining weights) rather than failing.
- The event-driven backtester assumes each symbol's bars share a common
  timestamp grid; sparse/irregular data across symbols may need resampling.
- Live streaming for yfinance/Alpaca/Polygon is polling-based (no native
  websocket in the free tiers); Binance uses a real public websocket for
  crypto klines.
- The Streamlit dashboard reads position "current price" from each
  position's last recorded ledger price, not a fresh live quote -- refresh
  cadence is tied to how often the live loop last ran.
- This is a simulator for research/education. It is not connected to, and
  must never be connected to, a live brokerage or exchange trading endpoint.
