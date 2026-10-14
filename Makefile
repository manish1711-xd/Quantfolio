# QuantFolio developer tasks.
#
# Works on Linux, macOS and Windows (Git Bash / WSL). On Windows the venv
# binaries live in .venv/Scripts, elsewhere in .venv/bin - VENV_BIN resolves
# whichever exists so the same targets work on every machine.

VENV_BIN := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts,.venv/bin)
PY       := $(VENV_BIN)/python
PIP      := $(PY) -m pip

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev run universe universe-dry mail-list mail-dry mail-send mail-alerts test test-cov lint fmt fmt-check type check clean docker-build docker-run

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	python -m venv .venv

install: ## Install runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install ## Install runtime + development dependencies
	$(PIP) install -r requirements-dev.txt

run: ## Launch the dashboard
	$(VENV_BIN)/streamlit run app.py

universe: ## Rebuild the stock master from the NSE/BSE listings + bhavcopy
	$(PY) -m src.universe_build

universe-dry: ## Report what a universe refresh would change, writing nothing
	$(PY) -m src.universe_build --dry-run

mail-list: ## Show configured email schedules and which are due
	$(PY) -m src.reporting.cli --list

mail-dry: ## Build any due tearsheets and validate SMTP, sending nothing
	$(PY) -m src.reporting.cli --dry-run

mail-send: ## Send tearsheets and price alerts that are due (what cron should run)
	$(PY) -m src.reporting.cli

mail-alerts: ## Check watchlist price alerts only
	$(PY) -m src.reporting.cli --alerts

test: ## Run the offline test suite
	$(PY) -m pytest

test-cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-report=html

lint: ## Lint with ruff
	$(PY) -m ruff check src tests app.py

fmt: ## Format with ruff
	$(PY) -m ruff format src tests app.py
	$(PY) -m ruff check --fix src tests app.py

fmt-check: ## Verify formatting without writing
	$(PY) -m ruff format --check src tests app.py

type: ## Type-check with mypy
	$(PY) -m mypy src

check: lint fmt-check type test ## Everything CI runs

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

docker-build: ## Build the container image
	docker build -t quantfolio:latest .

docker-run: ## Run the container on http://localhost:8501
	docker run --rm -p 8501:8501 -v "$(CURDIR)/data:/app/data" quantfolio:latest
