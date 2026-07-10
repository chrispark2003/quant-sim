"""Tests for the FastAPI dashboard backend: each endpoint should return the
expected schema/shape, always including the SIMULATED/PAPER TRADING
disclaimer.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dashboard.app import DISCLAIMER, app

client = TestClient(app)


class TestEndpoints:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["disclaimer"] == DISCLAIMER

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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
