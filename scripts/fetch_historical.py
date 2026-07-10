"""One-shot historical data fetch for `make setup`: pulls ~1yr of daily bars
for the configured equities universe (via yfinance) and crypto universe (via
Binance public REST), writing everything into the DuckDB store.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data.ingestion.binance_adapter import BinanceAdapter
from data.ingestion.yfinance_adapter import YFinanceAdapter
from data.store import get_store
from settings import symbols_config


def main(lookback_days: int = 365) -> None:
    store = get_store()
    symbols = symbols_config()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    equity_adapter = YFinanceAdapter()
    for entry in symbols.get("equities", []):
        symbol = entry["symbol"]
        print(f"Fetching equity {symbol}...")
        df = equity_adapter.fetch_ohlcv(symbol, "1d", start, end)
        n = store.write(df)
        print(f"  wrote {n} rows for {symbol}")

    benchmark = symbols.get("benchmarks", {}).get("equity")
    if benchmark:
        print(f"Fetching benchmark {benchmark}...")
        df = equity_adapter.fetch_ohlcv(benchmark, "1d", start, end)
        store.write(df)

    crypto_adapter = BinanceAdapter()
    for entry in symbols.get("crypto", []):
        symbol = entry["symbol"]
        print(f"Fetching crypto {symbol}...")
        df = crypto_adapter.fetch_ohlcv(symbol, "1d", start, end)
        n = store.write(df)
        print(f"  wrote {n} rows for {symbol}")

    print("Historical data fetch complete.")


if __name__ == "__main__":
    main()
