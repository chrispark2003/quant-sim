"""Unified time-series store backed by DuckDB with optional Parquet export.

Zero-infra: DuckDB is an embedded, file-based OLAP database, so the whole
project runs with no external services. All rows conform to the normalized
long schema (timestamp, symbol, asset_class, field, value).
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import duckdb
import pandas as pd

from data.schema import SCHEMA_COLUMNS, validate_schema
from settings import duckdb_path, parquet_dir

_TABLE = "observations"
_lock = threading.Lock()
_EXPORT_SYMBOL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class TimeSeriesStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or duckdb_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def _init_schema(self) -> None:
        with _lock, self._connect() as con:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    timestamp TIMESTAMP WITH TIME ZONE,
                    symbol VARCHAR,
                    asset_class VARCHAR,
                    field VARCHAR,
                    value DOUBLE
                )
            """)
            con.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_obs_unique
                ON {_TABLE} (timestamp, symbol, field)
            """)

    def write(self, df: pd.DataFrame) -> int:
        """Upsert rows into the store (dedup on timestamp, symbol, field)."""
        if df.empty:
            return 0
        clean = validate_schema(df)
        with _lock, self._connect() as con:
            con.register("incoming", clean)
            con.execute(f"""
                DELETE FROM {_TABLE}
                WHERE (timestamp, symbol, field) IN (
                    SELECT timestamp, symbol, field FROM incoming
                )
            """)
            con.execute(f"INSERT INTO {_TABLE} SELECT * FROM incoming")
        return len(clean)

    def query(
        self,
        symbols: list[str] | None = None,
        fields: list[str] | None = None,
        asset_class: str | None = None,
        start=None,
        end=None,
    ) -> pd.DataFrame:
        clauses = []
        params: list = []
        if symbols:
            clauses.append(f"symbol IN ({','.join(['?'] * len(symbols))})")
            params.extend(symbols)
        if fields:
            clauses.append(f"field IN ({','.join(['?'] * len(fields))})")
            params.extend(fields)
        if asset_class:
            clauses.append("asset_class = ?")
            params.append(asset_class)
        if start:
            clauses.append("timestamp >= ?")
            params.append(pd.Timestamp(start, tz="UTC"))
        if end:
            clauses.append("timestamp <= ?")
            params.append(pd.Timestamp(end, tz="UTC"))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM {_TABLE} {where} ORDER BY timestamp"
        with _lock, self._connect() as con:
            return con.execute(sql, params).fetchdf()

    def export_parquet(self, symbol: str | None = None) -> str:
        out_dir = Path(parquet_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        if symbol is not None and not _EXPORT_SYMBOL_RE.fullmatch(symbol):
            raise ValueError("symbol contains unsupported characters for parquet export")
        fname = f"{symbol}.parquet" if symbol else "all_observations.parquet"
        out_path = out_dir / fname
        safe_out_path = str(out_path).replace("'", "''")
        where = f"WHERE symbol = '{symbol}'" if symbol else ""
        with _lock, self._connect() as con:
            con.execute(f"COPY (SELECT * FROM {_TABLE} {where}) TO '{safe_out_path}' (FORMAT PARQUET)")
        return str(out_path)

    def symbols(self) -> list[str]:
        with _lock, self._connect() as con:
            return [r[0] for r in con.execute(f"SELECT DISTINCT symbol FROM {_TABLE}").fetchall()]


_default_store: TimeSeriesStore | None = None


def get_store() -> TimeSeriesStore:
    global _default_store
    if _default_store is None:
        _default_store = TimeSeriesStore()
    return _default_store
