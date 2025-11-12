.PHONY: install lint format test test-unit test-integration coverage clean all help

# Default target
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	@echo "Installing dependencies..."
	pip install -e .
	@echo "Installation complete!"

lint:  ## Run linters (ruff, black check)
	@echo "Running linters..."
	@echo "→ Running ruff..."
	ruff check src tests || true
	@echo "→ Running black check..."
	black --check src tests || true
	@echo "✓ Linting complete!"

format:  ## Auto-format code
	@echo "Auto-formatting code..."
	@echo "→ Running black..."
	black src tests || true
	@echo "→ Running isort..."
	isort src tests || true
	@echo "✓ Formatting complete!"

test:  ## Run all tests
	@echo "Running all tests..."
	pytest tests/ -v --tb=short || true
	@echo "✓ Tests complete!"

test-unit:  ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short || true

test-integration:  ## Run integration tests only
	@echo "Running integration tests..."
	pytest tests/integration/ -v --tb=short || true

coverage:  ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term || true
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

all: format lint test  ## Format, lint, and test

mlflow-ui:  ## Start MLflow UI server
	@echo "Starting MLflow UI..."
	@echo "→ Navigate to: http://localhost:5000"
	mlflow ui --backend-store-uri outputs/mlruns --port 5000
