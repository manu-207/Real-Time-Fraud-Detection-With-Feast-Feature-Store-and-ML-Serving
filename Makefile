.PHONY: setup infra pipeline train serve test clean

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# ── Infrastructure ────────────────────────────────────────────────────────────
infra:
	docker-compose up -d redis mlflow

infra-down:
	docker-compose down

# ── Full Pipeline ─────────────────────────────────────────────────────────────
pipeline: data features feast train evaluate monitor
	@echo "✅ Full pipeline complete!"

data:
	python src/data_prep.py

features:
	python src/feature_engineering.py

feast:
	feast -c feature_repo apply
	feast -c feature_repo materialize-incremental $$(date -u +"%Y-%m-%dT%H:%M:%S")

train:
	python src/train.py

evaluate:
	python src/evaluate.py

monitor:
	python src/monitor.py

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

serve-docker:
	docker-compose up --build api

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf data/raw data/processed models/ reports/ mlruns/
	rm -rf feature_repo/data/*.parquet
	rm -rf __pycache__ .pytest_cache
