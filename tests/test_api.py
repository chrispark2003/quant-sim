"""Tests for the FastAPI dashboard backend: each endpoint should return the
expected schema/shape, always including the SIMULATED/PAPER TRADING
disclaimer.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.app import DISCLAIMER, _downsample_equity_curve, _EQUITY_CURVE_MAX_POINTS, app

client = TestClient(app)


def _equity_series(n: int) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=max(n, 1), freq="min", tz="UTC")
    vals = [100_000 + i * 0.5 for i in range(max(n, 1))]
    return pd.Series(vals[:n], index=idx[:n])


class TestEndpoints:
    def test_root_serves_ledger_ui(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "SIMULATED / PAPER TRADING" in resp.text

    def test_api_index(self):
        resp = client.get("/api")
        assert resp.status_code == 200
        body = resp.json()
        assert body["disclaimer"] == DISCLAIMER

    def test_backtest_schema(self):
        resp = client.get("/backtest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["disclaimer"] == DISCLAIMER
        assert "available" in body
        if body["available"]:
            assert "metrics" in body and "curve" in body

    def test_positions_include_spark(self):
        resp = client.get("/positions")
        assert resp.status_code == 200
        for pos in resp.json()["positions"]:
            assert "spark" in pos
            assert isinstance(pos["spark"], list)

    def test_performance_schema(self):
        resp = client.get("/performance")
        assert resp.status_code == 200
        body = resp.json()
        for key in ["disclaimer", "starting_cash", "current_cash", "realized_pnl", "unrealized_pnl", "equity_curve", "max_drawdown"]:
            assert key in body
        assert body["disclaimer"] == DISCLAIMER

    def test_positions_schema(self):
        resp = client.get("/positions")
        assert resp.status_code == 200
        body = resp.json()
        assert "positions" in body
        assert isinstance(body["positions"], list)
        assert body["disclaimer"] == DISCLAIMER

    def test_blotter_schema(self):
        resp = client.get("/blotter")
        assert resp.status_code == 200
        body = resp.json()
        assert "trades" in body
        assert isinstance(body["trades"], list)

    def test_risk_schema(self):
        resp = client.get("/risk")
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body
        assert body["disclaimer"] == DISCLAIMER

    def test_attribution_schema(self):
        resp = client.get("/attribution")
        assert resp.status_code == 200
        body = resp.json()
        assert "attribution" in body

    def test_status_schema(self):
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        for key in ["disclaimer", "kill_switch_halted", "kill_switch_reason"]:
            assert key in body

    def test_kill_and_resume_roundtrip(self):
        resp = client.post("/kill")
        assert resp.status_code == 200
        assert resp.json()["status"] == "halted"

        status_after_kill = client.get("/status").json()
        assert status_after_kill["kill_switch_halted"] is True

        resp = client.post("/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "resumed"

        status_after_resume = client.get("/status").json()
        assert status_after_resume["kill_switch_halted"] is False


class TestEquityCurveDownsampling:
    """Regression coverage for the /performance payload-size fix: the equity
    curve grows one point per live-loop cycle with no retention limit (it
    reached ~14k points / ~1MB after 3 days of live trading), so it must
    always be capped and must never drop the most recent point -- the
    frontend reads that point as "current equity"."""

    @pytest.mark.parametrize("n", [0, 1, 499, 500, 501, 502, 600, 14_477])
    def test_never_exceeds_cap(self, n):
        out = _downsample_equity_curve(_equity_series(n))
        assert len(out) <= _EQUITY_CURVE_MAX_POINTS

    @pytest.mark.parametrize("n", [1, 499, 500, 501, 600, 14_477])
    def test_preserves_most_recent_point(self, n):
        s = _equity_series(n)
        out = _downsample_equity_curve(s)
        assert out.iloc[-1] == s.iloc[-1]
        assert out.index[-1] == s.index[-1]

    def test_noop_when_already_under_cap(self):
        s = _equity_series(50)
        out = _downsample_equity_curve(s)
        assert len(out) == 50
        pd.testing.assert_series_equal(out, s)

    def test_stays_near_cap_just_above_threshold(self):
        # A fixed stride (n // max_points) undersamples badly just above the
        # cap -- e.g. n=501 with a stride of 2 would keep only ~251 rows.
        # Evenly spaced positions should stay close to the intended cap.
        out = _downsample_equity_curve(_equity_series(501))
        assert len(out) >= _EQUITY_CURVE_MAX_POINTS - 1

    def test_max_points_one_returns_most_recent_point(self):
        s = _equity_series(600)
        out = _downsample_equity_curve(s, max_points=1)
        assert len(out) == 1
        assert out.iloc[-1] == s.iloc[-1]
        assert out.index[-1] == s.index[-1]

    @pytest.mark.parametrize("max_points", [0, -1, -10])
    def test_non_positive_max_points_returns_empty(self, max_points):
        out = _downsample_equity_curve(_equity_series(600), max_points=max_points)
        assert out.empty

    def test_performance_endpoint_curve_is_capped(self):
        resp = client.get("/performance")
        assert resp.status_code == 200
        assert len(resp.json()["equity_curve"]) <= _EQUITY_CURVE_MAX_POINTS


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
