#!/usr/bin/env bash
# Bootstrap script for a fresh Ubuntu 22.04/24.04 EC2 instance.
# Run as ubuntu: bash deploy/setup_ec2.sh
#
# Installs system deps, creates the venv, installs Python requirements,
# fetches initial historical data, and installs the systemd units.
#
# SIMULATED / PAPER TRADING ONLY -- this deploys a paper-trading bot with
# no real brokerage connectivity.
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/quant-sim}"
cd "$APP_DIR"
mkdir -p logs

echo "==> Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip make

echo "==> Creating venv and installing requirements..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
    echo "==> No .env found; copying .env.example (fill in your keys!)"
    cp .env.example .env
fi

echo "==> Initializing DuckDB store and fetching ~1yr of history..."
python -c "from data.store import get_store; get_store()"
python scripts/fetch_historical.py || echo "WARNING: historical fetch failed (check network/geoblock); continuing"

echo "==> Installing systemd units..."
sudo cp deploy/quant-sim-live.service /etc/systemd/system/
sudo cp deploy/quant-sim-api.service /etc/systemd/system/
sudo cp deploy/quant-sim-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quant-sim-live quant-sim-api quant-sim-dashboard

echo "==> Done. Start everything with:"
echo "    sudo systemctl start quant-sim-live quant-sim-api quant-sim-dashboard"
