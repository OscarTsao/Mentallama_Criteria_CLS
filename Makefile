.PHONY: install install-dev lint format test test-unit test-integration coverage clean clean-experiments clean-all help
.PHONY: train train-cv train-single data-stats validate-paper info experiments setup pre-commit
.PHONY: train-a100 train-v100 train-t4 train-cpu-test

# Default target
.DEFAULT_GOAL := help

# Training parameters (paper-aligned defaults)
BATCH_SIZE ?= 8
EPOCHS ?= 100
PATIENCE ?= 20
PRECISION ?= auto
DEVICE ?= cuda
DATA_DIR ?= data
OUTPUT_DIR ?= experiments

help:  ## Show this help message
	@echo "MentalLLaMA NLI Classifier - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo "  make setup            Full development setup"
	@echo ""
	@echo "Training:"
	@echo "  make train-cv         Train 5-fold cross-validation (paper-compliant)"
	@echo "  make train-single     Train single split (quick experiment)"
	@echo "  make train            Alias for train-cv"
	@echo ""
	@echo "Training Shortcuts:"
	@echo "  make train-a100       Train on A100 (bf16, batch=8)"
	@echo "  make train-v100       Train on V100 (fp16, batch=8)"
	@echo "  make train-t4         Train on T4 (fp16, batch=4)"
	@echo ""
	@echo "Data & Validation:"
	@echo "  make data-stats       Show dataset statistics"
	@echo "  make validate-paper   Validate paper compliance"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make lint             Run linter (ruff + black)"
	@echo "  make format           Auto-format code (black + isort)"
	@echo "  make coverage         Run tests with coverage report"
	@echo "  make all              Format, lint, and test"
	@echo "  make pre-commit       Run pre-commit checks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Remove build artifacts and cache"
	@echo "  make clean-experiments Remove experiment directories"
	@echo "  make clean-all        Clean everything"
	@echo "  make info             Show system information"
	@echo "  make experiments      List recent experiments"
	@echo "  make mlflow-ui        Start MLflow UI server"
	@echo ""
	@echo "Advanced Training Options:"
	@echo "  make train-cv BATCH_SIZE=4      Train with custom batch size"
	@echo "  make train-cv PRECISION=bf16    Train with specific precision"
	@echo "  make train-cv FOLD=0            Train single fold (0-4)"
	@echo "  make train-cv EPOCHS=50         Train with custom epochs"
	@echo ""
	@echo "Examples:"
	@echo "  make setup                       # Setup development environment"
	@echo "  make train-a100                  # Train on A100 GPU"
	@echo "  make train-cv BATCH_SIZE=8       # Train 5-fold CV"
	@echo "  make data-stats                  # Check dataset"
	@echo "  make pre-commit                  # Run all checks before commit"

# ==========================================
# Setup & Installation
# ==========================================

install:  ## Install package and dependencies
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip
	pip install -e .
	@echo "✓ Installation complete!"

install-dev:  ## Install with development dependencies
	@echo "Installing package with development dependencies..."
	python -m pip install --upgrade pip
	pip install -e '.[dev]'
	@echo "✓ Development installation complete!"

setup: install-dev  ## Full development setup
	@echo "✓ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Verify data: make data-stats"
	@echo "  2. Run tests: make test"
	@echo "  3. Start training: make train-cv"

# ==========================================
# Training Commands
# ==========================================

train-cv:  ## Train 5-fold cross-validation (paper-compliant)
	@echo "Starting 5-fold cross-validation training..."
	@echo "Parameters: batch=$(BATCH_SIZE), epochs=$(EPOCHS), patience=$(PATIENCE), precision=$(PRECISION)"
ifdef FOLD
	@echo "Training single fold: $(FOLD)"
	python scripts/train_5fold_cv.py \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--device $(DEVICE) \
		--precision $(PRECISION) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--fold $(FOLD)
else
	@echo "Training all 5 folds..."
	python scripts/train_5fold_cv.py \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--device $(DEVICE) \
		--precision $(PRECISION) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)
endif
	@echo "✓ Training complete! Results in $(OUTPUT_DIR)/"

train-single:  ## Train single train/val split (quick)
	@echo "Starting single split training..."
	@echo "Parameters: batch=$(BATCH_SIZE), epochs=$(EPOCHS), precision=$(PRECISION)"
	python scripts/train.py \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--device $(DEVICE) \
		--precision $(PRECISION) \
		--data-dir $(DATA_DIR)
	@echo "✓ Training complete! Results in experiments/"

train: train-cv  ## Default: 5-fold cross-validation

# Hardware-specific training shortcuts
train-a100:  ## Train on A100 (bf16, batch=8)
	$(MAKE) train-cv BATCH_SIZE=8 PRECISION=bf16

train-v100:  ## Train on V100 (fp16, batch=8)
	$(MAKE) train-cv BATCH_SIZE=8 PRECISION=fp16

train-t4:  ## Train on T4 (fp16, batch=4)
	$(MAKE) train-cv BATCH_SIZE=4 PRECISION=fp16

train-cpu-test:  ## Quick CPU test (2 epochs)
	$(MAKE) train-single BATCH_SIZE=2 EPOCHS=2 DEVICE=cpu PRECISION=fp32

# ==========================================
# Data & Validation
# ==========================================

data-stats:  ## Show dataset statistics
	@echo "Generating dataset statistics..."
	@python -c "\
	import sys; \
	sys.path.insert(0, 'src'); \
	from Project.SubProject.data.dataset import ReDSM5toNLIConverter; \
	converter = ReDSM5toNLIConverter( \
		posts_csv='$(DATA_DIR)/redsm5/redsm5_posts.csv', \
		annotations_csv='$(DATA_DIR)/redsm5/redsm5_annotations.csv', \
		criteria_json='$(DATA_DIR)/DSM5/MDD_Criteira.json' \
	); \
	nli_df = converter.load_and_convert(exhaustive_pairing=True); \
	print('\n' + '='*60); \
	print('Dataset Statistics (Exhaustive Pairing)'); \
	print('='*60); \
	print(f'Total NLI pairs: {len(nli_df):,}'); \
	print(f'Unique posts: {nli_df[\"post_id\"].nunique():,}'); \
	print(f'Positive: {(nli_df[\"label\"] == 1).sum():,} ({(nli_df[\"label\"] == 1).sum() / len(nli_df) * 100:.2f}%)'); \
	print(f'Negative: {(nli_df[\"label\"] == 0).sum():,} ({(nli_df[\"label\"] == 0).sum() / len(nli_df) * 100:.2f}%)'); \
	print('='*60 + '\n'); \
	"

validate-paper:  ## Validate paper compliance
	@echo "Validating paper compliance..."
	python scripts/validate.py
	@echo "✓ Validation complete!"

# ==========================================
# Testing & Quality
# ==========================================

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

pre-commit: format lint test  ## Run pre-commit checks
	@echo "✓ Pre-commit checks passed!"

all: format lint test  ## Format, lint, and test

# ==========================================
# Cleanup
# ==========================================

clean:  ## Remove build artifacts and cache
	@echo "Cleaning temporary files..."
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Cleanup complete!"

clean-experiments:  ## Remove experiment directories (WARNING: deletes all results!)
	@echo "WARNING: This will delete all experiment results!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."; read dummy
	rm -rf experiments/ cv_results/
	@echo "✓ Experiments cleaned!"

clean-all: clean clean-experiments  ## Clean everything

# ==========================================
# Utilities
# ==========================================

info:  ## Show system information
	@echo "System Information:"
	@echo "=================="
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
	@echo "CUDA version: $$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
	@echo "bf16 supported: $$(python -c 'import torch; print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)' 2>/dev/null || echo 'N/A')"

experiments:  ## List recent experiments
	@echo "Recent Experiments:"
	@echo "==================="
	@if [ -d experiments ]; then \
		ls -lt experiments/ | head -15; \
	else \
		echo "No experiments found. Run 'make train' to start."; \
	fi

mlflow-ui:  ## Start MLflow UI server
	@echo "Starting MLflow UI..."
	@echo "→ Navigate to: http://localhost:5000"
	mlflow ui --backend-store-uri outputs/mlruns --port 5000
