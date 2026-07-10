"""Shared configuration loader and constants for the quant-sim project.

SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE. No module in this project
ever talks to a real brokerage or exchange order endpoint.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"

load_dotenv(ROOT_DIR / ".env")

DISCLAIMER = "SIMULATED / PAPER TRADING -- NOT FINANCIAL ADVICE"


@functools.lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config file from the config/ directory, cached."""
    path = CONFIG_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


def strategies_config() -> dict[str, Any]:
    return load_yaml("strategies.yaml")


def symbols_config() -> dict[str, Any]:
    return load_yaml("symbols.yaml")


def env(key: str, default: str | None = None) -> str | None:
    """Read a value from the environment only -- never hardcode secrets."""
    return os.environ.get(key, default)


def duckdb_path() -> str:
    return env("DUCKDB_PATH", str(DATA_DIR / "quant_sim.duckdb"))


def parquet_dir() -> str:
    p = env("PARQUET_DIR", str(DATA_DIR / "parquet"))
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def state_dir() -> Path:
    p = DATA_DIR / "state"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = DATA_DIR / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p
