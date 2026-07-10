"""Persist + resume live paper-trading state across restarts.

Wraps the ledger's own persistence plus loop-level bookkeeping (last
processed bar per symbol) so the autonomous loop never double-submits an
order for a bar it already processed.
"""
from __future__ import annotations

import json
from pathlib import Path

from execution.ledger import Ledger
from settings import state_dir


class LiveState:
    def __init__(self, ledger_path: str | Path | None = None, cursor_path: str | Path | None = None,
                 starting_cash: float = 100000.0):
        self.ledger_path = Path(ledger_path) if ledger_path else state_dir() / "ledger.json"
        self.cursor_path = Path(cursor_path) if cursor_path else state_dir() / "cursor.json"
        self.starting_cash = starting_cash
        self.ledger = Ledger.load_or_create(self.ledger_path, starting_cash=starting_cash)
        self.last_processed_bar: dict[str, str] = self._load_cursor()

    def _load_cursor(self) -> dict[str, str]:
        if self.cursor_path.exists():
            with open(self.cursor_path, "r") as f:
                return json.load(f)
        return {}

    def already_processed(self, symbol: str, bar_timestamp) -> bool:
        ts_str = str(bar_timestamp)
        return self.last_processed_bar.get(symbol) == ts_str

    def mark_processed(self, symbol: str, bar_timestamp) -> None:
        self.last_processed_bar[symbol] = str(bar_timestamp)

    def save(self) -> None:
        self.ledger.save(self.ledger_path)
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cursor_path, "w") as f:
            json.dump(self.last_processed_bar, f, indent=2)
