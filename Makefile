.PHONY: install lint format typecheck test test-unit test-integration coverage clean all help train train-smoke train-full mlflow-ui

# Default target
.DEFAULT_GOAL := help

PYTHON ?= python
TRAIN_ENGINE := $(PYTHON) -m Project.SubProject.engine.train_engine
TRAIN_DATA_DIR ?= data/redsm5
TRAIN_DSM5_DIR ?= data/DSM5
TRAIN_OUTPUT_DIR ?= outputs
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
PROJECT_ROOT := $(shell pwd)
export PYTHONPATH := $(PROJECT_ROOT)/src$(if $(PYTHONPATH),:$(PYTHONPATH),)

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip
	pip install -e '.[dev]'
	@echo "Installation complete!"

lint:  ## Run linters (ruff, black check)
	@echo "Running linters..."
	@echo "→ Running ruff..."
	ruff check src tests
	@echo "→ Running black check..."
	black --check src tests
	@echo "✓ Linting complete!"

format:  ## Auto-format code
	@echo "Auto-formatting code..."
	@echo "→ Running black..."
	black src tests
	@echo "→ Running isort..."
	isort src tests
	@echo "✓ Formatting complete!"

typecheck:  ## Run static type checks (mypy)
	@echo "Running mypy..."
	mypy src
	@echo "✓ Type checking complete!"

test:  ## Run all tests
	@echo "Running all tests..."
	pytest tests/ -v --tb=short
	@echo "✓ Tests complete!"

test-unit:  ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration:  ## Run integration tests only
	@echo "Running integration tests..."
	pytest tests/integration/ -v --tb=short

coverage:  ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated: htmlcov/index.html"

clean:  ## Clean temporary files
	@echo "Cleaning temporary files..."
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete!"

all: format lint typecheck test  ## Format, lint, type-check, and test

train: train-full  ## Run the default full training pipeline

train-smoke:  ## Run a quick 2-fold training smoke test
	@echo "Running smoke-test training (2 folds, 1 epoch)..."
	$(TRAIN_ENGINE) \
		--data-dir $(TRAIN_DATA_DIR) \
		--dsm5-dir $(TRAIN_DSM5_DIR) \
		--output-dir $(TRAIN_OUTPUT_DIR)/smoke_test \
		--n-folds 2 \
		--num-epochs 1 \
		--batch-size 1 \
		--learning-rate 1e-4 \
		--experiment-name mentallama-smoke \
		--tracking-uri $(MLFLOW_TRACKING_URI)
	@echo "✓ Smoke-test training complete!"

train-full:  ## Run full 5-fold cross-validation training
	@echo "Running full 5-fold training..."
	CUBLAS_WORKSPACE_CONFIG=:4096:8 $(TRAIN_ENGINE) \
		--data-dir $(TRAIN_DATA_DIR) \
		--dsm5-dir $(TRAIN_DSM5_DIR) \
		--output-dir $(TRAIN_OUTPUT_DIR)/runs \
		--n-folds 5 \
		--num-epochs 10 \
		--batch-size 4 \
		--learning-rate 1e-4 \
		--experiment-name mentallama-cv-prod \
		--tracking-uri $(MLFLOW_TRACKING_URI)
	@echo "✓ Full training complete!"

mlflow-ui:  ## Start MLflow UI server
	@echo "Starting MLflow UI..."
	@echo "→ Navigate to: http://localhost:5000"
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000
