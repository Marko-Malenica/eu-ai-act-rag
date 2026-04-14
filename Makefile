.PHONY: setup start stop ingest logs

setup:
	cp .env.example .env
	@echo "Fill in .env before continuing"

start:
	ollama serve &
	docker compose up -d

ingest:
	docker compose --profile setup run --rm ingest

stop:
	docker compose down

logs:
	docker compose logs -f