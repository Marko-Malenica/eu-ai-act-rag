.PHONY: setup start stop ingest

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

start:
	docker compose up -d
	. venv/bin/activate && uvicorn backend.main:app --reload &
	. venv/bin/activate && streamlit run frontend/app.py

ingest:
	. venv/bin/activate && python scripts/ingest.py

stop:
	docker compose down