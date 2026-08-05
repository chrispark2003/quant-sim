.PHONY: setup backtest paper dashboard test

PYTHON := python3

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -c "from data.store import get_store; get_store()"
	$(PYTHON) scripts/fetch_historical.py

backtest:
	$(PYTHON) scripts/run_backtest.py

paper:
	$(PYTHON) -m live.loop

dashboard:
	uvicorn dashboard.app:app --host 127.0.0.1 --port 8000 --reload &
	streamlit run dashboard/frontend/streamlit_app.py

test:
	pytest tests/ -v
