# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI/ML experiment template for building mental health classification models, specifically focused on using MentaLLaMA for Major Depressive Disorder (MDD) criteria classification. The project uses PyTorch, Transformers, MLflow for experiment tracking, and Optuna for hyperparameter optimization.

**Primary Use Case**: Binary classification of (post, criterion) pairs to determine if a social media post matches DSM-5 diagnostic criteria for mental health conditions.

## Development Setup

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Key Commands

### Testing and Quality
```bash
# Run all tests
pytest

# Lint and format code
ruff check src tests
black src tests

# Type checking
mypy src
```

### MLflow Tracking
The project uses MLflow with SQLite backend (`sqlite:///mlflow.db`) for experiment tracking:
- Runs are stored in `mlruns/` directory
- Artifacts are logged locally
- UI can be accessed via `mlflow ui` (if needed)

## Architecture

### Module Structure

The codebase follows a nested package structure: `Project.SubProject.*`

**Core Modules**:
- `src/Project/SubProject/models/model.py`: Model definitions
  - `Model`: Wrapper around HuggingFace transformers with custom classification head
  - `classification_head`: Standalone classification layer

- `src/Project/SubProject/utils/`: Shared utilities
  - `mlflow_utils.py`: MLflow configuration and context managers
    - `configure_mlflow()`: Set tracking URI and experiment
    - `enable_autologging()`: Enable framework-specific autologging
    - `mlflow_run()`: Context manager for MLflow runs with automatic tag/param logging
  - `log.py`: Centralized logging via `get_logger()`
  - `seed.py`: Reproducibility via `set_seed()` with PyTorch determinism support

- `src/Project/SubProject/engine/`: Training and evaluation engines (placeholder structure)
- `src/Project/SubProject/data/dataset.py`: Dataset implementations (placeholder structure)

### Data Structure

**Mental Health Classification Data**:
- `data/DSM5/MDD_Criteria.json`: DSM-5 Major Depressive Disorder diagnostic criteria (A.1-A.9)
- `data/redsm5/`: Social media posts with ground truth labels linking posts to criteria

**Expected Data Format**:
- Posts: `{post_id, post}`
- Ground truth: `{post_id, criterion_id, label}` where label is "yes"/"no" → 1/0
- Criteria: `{criterion_id, criterion}`

### Speckit Workflow Integration

This project uses a feature-driven development workflow via `.specify/` and `.claude/commands/`:
- Feature branches follow pattern: `NNN-feature-name` (e.g., `001-model-use-mentallam`)
- Each feature has a spec directory under `specs/NNN-feature-name/` containing:
  - `spec.md`: Feature specification with requirements and acceptance criteria
  - `plan.md`: Implementation plan
  - `tasks.md`: Task breakdown
  - `data-model.md`: Data schemas
  - `research.md`: Research notes

**Speckit Commands** (use via `/speckit.*`):
- `/speckit.specify`: Create/update feature specification
- `/speckit.clarify`: Resolve ambiguities in spec
- `/speckit.plan`: Generate implementation plan
- `/speckit.tasks`: Generate task list
- `/speckit.implement`: Execute implementation
- `/speckit.analyze`: Cross-artifact consistency check

## MLflow Integration Pattern

Standard pattern for experiment tracking:

```python
from Project.SubProject.utils import configure_mlflow, enable_autologging, mlflow_run

# One-time setup
configure_mlflow(tracking_uri="sqlite:///mlflow.db", experiment="your-experiment")
enable_autologging()

# Per-run
with mlflow_run("run-name", tags={"stage": "dev"}, params={"lr": 1e-4}):
    # Training loop here
    pass
```

## Model Architecture Notes

**MentaLLaMA Integration**:
- Primary model: `klyang/MentaLLaMA-chat-7B`
- Use `LlamaForSequenceClassification.from_pretrained()` with `num_labels=2`
- Input format: `"post: {post}, criterion: {criterion} Does the post match the criterion description? Output yes or no"`
- Tokenization: max_length=512, truncation=longest_first, pad to max length

**Training Configuration** (from spec):
- Cross-validation: 5-fold StratifiedGroupKFold grouped by post_id
- PEFT: DoRA (target_modules: q/k/v/o_proj, gate/up/down_proj, r=8, alpha=16, dropout=0.05)
- Training: up to 100 epochs, early stopping patience=20
- Optimization: Select best by validation F1 with threshold tuning (0.00-1.00, step 0.01)
- Compute: gradient checkpointing, grad_accum=4, bf16 AMP when available

## Reproducibility

Use `set_seed()` from `utils.seed` for consistent results:
- Default seed: 42
- Supports environment variable override
- Enables PyTorch deterministic algorithms when `deterministic=True`
- Affects Python random, NumPy, and PyTorch RNG states

## Configuration Management

- `pyproject.toml`: Python package configuration, dependencies, tool settings
- `configs/`: Intended for Hydra configuration files (currently placeholder)
- `.devcontainer/`: VS Code devcontainer setup with Python extensions (Pylance, Ruff, Jupyter)

## Important Conventions

- Line length: 100 characters (enforced by ruff/black)
- Target Python version: 3.10+
- Package namespace: All imports use `Project.SubProject.*` structure
- Logging: Use `get_logger()` for consistent formatting across modules
- MLflow runs: Always use context manager `mlflow_run()` to ensure proper cleanup
