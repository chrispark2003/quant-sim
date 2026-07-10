"""Configurable per-asset-class trading cadence and market-hours calendar.

Equities respect regular trading hours (default 9:30-16:00 ET, Mon-Fri);
crypto runs 24/7. Cadence strings follow the pattern "<N><unit>" where unit
is m (minutes) or h (hours) or d (days), e.g. "5m", "1h", "1d".
"""
from __future__ import annotations

import re
from datetime import datetime, time

import pytz


def parse_cadence_seconds(cadence: str) -> int:
    match = re.match(r"^(\d+)([mhd])$", cadence.strip())
    if not match:
        raise ValueError(f"invalid cadence string: {cadence}")
    value, unit = int(match.group(1)), match.group(2)
    multiplier = {"m": 60, "h": 3600, "d": 86400}[unit]
    return value * multiplier


class MarketCalendar:
    def __init__(self, config: dict):
        self.timezone = pytz.timezone(config.get("timezone", "America/New_York"))
        self.open_time = self._parse_time(config.get("open", "09:30"))
        self.close_time = self._parse_time(config.get("close", "16:00"))
        self.trading_days = set(config.get("days", ["Mon", "Tue", "Wed", "Thu", "Fri"]))

    @staticmethod
    def _parse_time(hhmm: str) -> time:
        h, m = hhmm.split(":")
        return time(int(h), int(m))

    def is_open(self, now_utc: datetime | None = None) -> bool:
        now_utc = now_utc or datetime.now(pytz.UTC)
        local = now_utc.astimezone(self.timezone)
        day_abbr = local.strftime("%a")
        if day_abbr not in self.trading_days:
            return False
        return self.open_time <= local.time() <= self.close_time


class Scheduler:
    """Tracks the last-run time per asset class and decides whether it's
    time to run again, respecting cadence + market hours."""

    def __init__(self, cadence_config: dict, market_hours_config: dict):
        self.cadence_seconds = {k: parse_cadence_seconds(v) for k, v in cadence_config.items()}
        self.calendar = MarketCalendar(market_hours_config)
        self._last_run: dict[str, datetime] = {}

    def should_run(self, asset_class: str, now: datetime | None = None) -> bool:
        now = now or datetime.now(pytz.UTC)

        if asset_class == "equity" and not self.calendar.is_open(now):
            return False

        interval = self.cadence_seconds.get(asset_class, 300)
        last = self._last_run.get(asset_class)
        if last is None:
            return True
        return (now - last).total_seconds() >= interval

    def mark_ran(self, asset_class: str, now: datetime | None = None) -> None:
        self._last_run[asset_class] = now or datetime.now(pytz.UTC)

    def next_run_estimate(self, asset_class: str) -> datetime | None:
        last = self._last_run.get(asset_class)
        if last is None:
            return None
        interval = self.cadence_seconds.get(asset_class, 300)
        return last + __import__("datetime").timedelta(seconds=interval)
