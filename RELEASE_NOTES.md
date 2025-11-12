# Release Notes

## Version 1.0.0 (2025-11-13)

**First production release of the MentaLLaMA Binary Classifier for DSM-5 Criteria Matching.**

### Overview

This release provides a complete end-to-end solution for training, evaluating, and deploying a binary classifier that determines whether a Reddit-style post matches a DSM-5 Major Depressive Disorder (MDD) criterion. The system uses the `klyang/MentaLLaMA-chat-7B` model fine-tuned with DoRA (Weight-Decomposed Low-Rank Adaptation) PEFT adapters.

### Key Features

#### 1. Training Pipeline
- **5-Fold Cross-Validation**: StratifiedGroupKFold with `post_id` grouping to prevent data leakage
- **DoRA PEFT Adapters**: Memory-efficient fine-tuning with configurable rank (default r=8)
- **Hydra Configuration**: Hierarchical YAML-based configuration system for all hyperparameters
- **MLflow Tracking**: Comprehensive experiment tracking with parent/child run hierarchy
- **Early Stopping**: Patience-based early stopping (default: 20 epochs)
- **Threshold Tuning**: Automatic threshold optimization per fold based on F1 score
- **Gradient Checkpointing**: Memory optimization for training on 80GB GPUs
- **Mixed Precision**: BFloat16 support for faster training

#### 2. Data Processing
- **Dataset Validation**: Automatic cardinality checking (expected: 1484 posts, 9 criteria, 13356 pairs)
- **Schema Validation**: Strict validation of CSV schema and data types
- **Label Mapping**: Automatic conversion from yes/no to 0/1 labels
- **Text Cleaning**: Unicode normalization and whitespace handling
- **Fold Persistence**: JSON artifacts with complete fold metadata and dataset hashes

#### 3. Evaluation & Metrics
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, ROC AUC
- **Confusion Matrices**: Per-fold confusion matrix visualization
- **PR Curves**: Precision-recall curves for threshold analysis
- **Aggregation**: Statistical aggregation across folds (mean, std, best fold)
- **Threshold Summary**: Per-fold and global threshold recommendations

#### 4. Inference
- **Single Sample Prediction**: CLI for one-off predictions
- **Batch Inference**: JSONL input/output for bulk predictions
- **Latency Benchmarking**: CPU and GPU latency measurement with p50/p95/p99 percentiles
- **Tuned Thresholds**: Automatic loading of fold-specific or global thresholds
- **MLflow Logging**: Optional logging of inference runs

#### 5. Developer Experience
- **Comprehensive Documentation**: User guide (850+ lines), quickstart, API docs
- **Automated Testing**: Unit and integration tests with pytest
- **Code Quality**: Linting (ruff), formatting (black, isort), type checking (mypy)
- **CI/CD Pipeline**: GitHub Actions workflow with 6 jobs (lint, test, integration, validate, security, summary)
- **Makefile**: Common tasks (install, lint, format, test, coverage, mlflow-ui)
- **Validation Script**: Automated quickstart validation with 13 checks

#### 6. Model Registry
- **MLflow Model Registry**: Model versioning and staging (Staging, Production, Archived)
- **Model Signatures**: Input/output schema definitions
- **Example Inputs**: Documented example inputs for testing
- **Model Metadata**: Metrics, thresholds, fold information, tags
- **Registration Script**: Automated model registration from run IDs

### Performance Targets

- **F1 Score**: Mean F1 ≥ 0.80 across 5 folds
- **CPU Inference Latency**: p95 ≤ 1000 ms (for inputs ≤256 tokens)
- **GPU Memory**: <50GB during training with DoRA r=8, batch_size=8, grad_accum=4
- **Training Time**: ~2-3 hours per fold on A100 80GB GPU

### Architecture

#### Model
- **Base**: `klyang/MentaLLaMA-chat-7B` (7B parameters)
- **Adapter**: DoRA with r=8, alpha=16, dropout=0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision**: BFloat16
- **Max Length**: 512 tokens

#### Prompt Template
```
post: {post}
criterion: {criterion}
Does the post match the criterion description? Output yes or no
```

#### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5 with cosine schedule
- **Warmup**: 10% of steps
- **Weight Decay**: 0.01
- **Batch Size**: 8 × 4 grad_accum = effective 32
- **Max Epochs**: 100 (with early stopping patience=20)

### Project Structure

```
Mentallama_Criteria_CLS/
├── src/Project/SubProject/
│   ├── data/
│   │   ├── dataset.py          # Dataset loader with validation
│   │   └── splits.py           # StratifiedGroupKFold helper
│   ├── models/
│   │   ├── model.py            # MentallamClassifier wrapper
│   │   └── prompt_builder.py   # Prompt formatting
│   ├── engine/
│   │   ├── train_engine.py     # Training CLI
│   │   ├── eval_engine.py      # Evaluation & inference CLI
│   │   └── metrics.py          # Metrics computation
│   └── utils/
│       ├── mlflow_utils.py     # MLflow helpers
│       ├── seed.py             # Reproducibility
│       └── log.py              # Logging
├── configs/
│   ├── config.yaml             # Root config
│   ├── data/redsm5.yaml
│   ├── model/mentallam.yaml
│   ├── training/cv.yaml
│   ├── logging/mlflow.yaml
│   └── inference/base.yaml
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── scripts/
│   ├── validate_quickstart.sh  # Quickstart validation
│   └── register_model.py       # Model registration
├── docs/
│   └── user_guide.md           # Comprehensive documentation
├── .github/workflows/
│   └── ci.yml                  # CI/CD pipeline
├── Makefile                    # Build automation
├── pyproject.toml              # Python project config
└── README.md                   # Project overview
```

### Installation

#### Prerequisites
- Python 3.10 or higher
- CUDA 11.8+ (for GPU training)
- 80GB+ disk space
- Linux or macOS (Windows via WSL2)

#### Basic Installation
```bash
# Clone repository
git clone <repo-url>
cd Mentallama_Criteria_CLS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e '.[dev]'
```

#### Optional Dependencies
```bash
# Flash Attention (faster training)
pip install flash-attn --no-build-isolation

# BitsAndBytes (memory optimization)
pip install bitsandbytes
```

### Quick Start

#### 1. Verify Installation
```bash
make install
python --version  # Should be >=3.10
```

#### 2. Run Smoke Test
```bash
python -m Project.SubProject.engine.train_engine \
  training.folds=2 \
  training.max_steps=10 \
  data.sample_fraction=0.01
```

#### 3. Run Full Training
```bash
python -m Project.SubProject.engine.train_engine \
  experiment.name=mentallam_cv_prod \
  training.batch_size=8 \
  training.max_epochs=100
```

#### 4. Aggregate Metrics
```bash
python -m Project.SubProject.engine.eval_engine aggregate \
  parent_run_id=<PARENT_RUN_ID>
```

#### 5. Run Inference
```bash
python -m Project.SubProject.engine.eval_engine infer \
  checkpoint=outputs/checkpoints/fold_0/best.pt \
  post="I feel sad all the time" \
  criterion="Depressed mood most of the day"
```

#### 6. View MLflow UI
```bash
make mlflow-ui
# Navigate to http://localhost:5000
```

### Configuration

All parameters can be overridden via Hydra:

```bash
# Override batch size and learning rate
python -m Project.SubProject.engine.train_engine \
  training.batch_size=16 \
  training.lr=1e-5

# Override model config
python -m Project.SubProject.engine.train_engine \
  model.peft.r=16 \
  model.peft.alpha=32

# Use debug overrides
python -m Project.SubProject.engine.train_engine \
  training.folds=2 \
  training.max_epochs=3 \
  data.sample_fraction=0.01
```

### Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run with coverage
make coverage
```

### Code Quality

```bash
# Run linters
make lint

# Auto-format code
make format

# All checks (format + lint + test)
make all
```

### Validation

```bash
# Validate quickstart workflow
bash scripts/validate_quickstart.sh

# CI mode (includes lint/test)
VALIDATE_MODE=ci bash scripts/validate_quickstart.sh
```

### Model Registry

```bash
# Register best model
python scripts/register_model.py \
  --run-id <BEST_FOLD_RUN_ID> \
  --model-name mentallama-criteria-cls \
  --stage Production

# Load registered model
import mlflow
model = mlflow.pyfunc.load_model("models:/mentallama-criteria-cls/Production")
```

### Troubleshooting

#### CUDA Out of Memory
```bash
# Reduce batch size
python -m Project.SubProject.engine.train_engine training.batch_size=4

# Increase gradient accumulation
python -m Project.SubProject.engine.train_engine \
  training.batch_size=4 \
  training.grad_accum_steps=8

# Reduce DoRA rank
python -m Project.SubProject.engine.train_engine model.peft.r=4
```

#### MLflow SQLite Locks
```bash
# Set pool size
export MLFLOW_SQLALCHEMY_POOL_SIZE=1

# Or use server mode
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

#### Slow Training
```bash
# Install Flash Attention
pip install flash-attn --no-build-isolation

# Increase batch size
python -m Project.SubProject.engine.train_engine training.batch_size=16

# Use more workers
python -m Project.SubProject.engine.train_engine training.dataloader_num_workers=4
```

### Known Limitations

1. **GPU Memory**: Requires at least 24GB GPU memory for training with default config
2. **SQLite Concurrency**: MLflow SQLite backend has limited concurrency; consider server mode for parallel experiments
3. **CPU Inference**: CPU inference is slow (~1 second per sample); GPU recommended for production
4. **Model Size**: Base model is 7B parameters (~14GB), requires sufficient disk space

### Breaking Changes from Previous Versions

This is the first release (v1.0.0), no breaking changes.

### Deprecated Features

None in this release.

### Migration Guide

Not applicable (first release).

### Contributors

- MLOps Team
- Modeling Team
- Data Science Team

### License

MIT License (see LICENSE file)

### Support

- **Documentation**: See `docs/user_guide.md`
- **Issues**: <repo-url>/issues
- **Discussions**: <repo-url>/discussions

### Acknowledgments

- **MentaLLaMA**: klyang/MentaLLaMA-chat-7B base model
- **Hugging Face**: Transformers, PEFT, Accelerate libraries
- **MLflow**: Experiment tracking and model registry
- **Hydra**: Configuration management

### Roadmap for v1.1.0

Planned features for next release:

- Multi-GPU distributed training with DeepSpeed
- Hyperparameter tuning with Optuna integration
- Model interpretability (SHAP, attention visualization)
- FastAPI inference server with Docker deployment
- Support for additional mental health criteria beyond MDD
- TensorRT optimization for production inference
- A/B testing framework for model evaluation
- Streaming inference for real-time applications

### Changelog

#### [1.0.0] - 2025-11-13

**Added**
- Initial release of training pipeline with 5-fold CV
- DoRA PEFT adapter fine-tuning
- Hydra configuration system
- MLflow experiment tracking
- Comprehensive metrics and evaluation
- Inference CLI with latency benchmarking
- Model registry integration
- Documentation (user guide, quickstart, API docs)
- Unit and integration tests
- CI/CD pipeline with GitHub Actions
- Makefile for common tasks
- Validation script for quickstart workflow
- Code quality tools (ruff, black, isort, mypy)

**Fixed**
- N/A (first release)

**Changed**
- N/A (first release)

**Deprecated**
- N/A (first release)

**Removed**
- N/A (first release)

**Security**
- N/A (first release)

---

**Thank you for using MentaLLaMA Binary Classifier!**

For questions, feedback, or contributions, please visit our GitHub repository.
