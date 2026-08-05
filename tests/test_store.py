"""Tests for DuckDB-backed time-series storage."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.store import TimeSeriesStore


def _sample_observations(symbol: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "symbol": symbol,
                "asset_class": "equity",
                "field": "close",
                "value": 100.0,
            }
        ]
    )


class TestParquetExport:
    def test_rejects_symbols_with_sql_or_path_metacharacters(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PARQUET_DIR", str(tmp_path / "parquet"))
        store = TimeSeriesStore(db_path=str(tmp_path / "test.duckdb"))

        with pytest.raises(ValueError):
            store.export_parquet("AAPL'; DROP TABLE observations; --")

        with pytest.raises(ValueError):
            store.export_parquet("../AAPL")

    def test_exports_supported_symbol(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PARQUET_DIR", str(tmp_path / "parquet"))
        store = TimeSeriesStore(db_path=str(tmp_path / "test.duckdb"))
        store.write(_sample_observations())

        out_path = store.export_parquet("AAPL")

        assert out_path.endswith("AAPL.parquet")
        assert (tmp_path / "parquet" / "AAPL.parquet").exists()
