# MentaLLaMA Binary Classifier for DSM-5 Criteria

[![CI](https://github.com/<user>/<repo>/workflows/CI%20Pipeline/badge.svg)](https://github.com/<user>/<repo>/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Binary classifier that determines whether a Reddit-style post matches a DSM-5 Major Depressive Disorder (MDD) criterion. Fine-tunes `klyang/MentaLLaMA-chat-7B` using DoRA PEFT adapters with 5-fold cross-validation.

## Features

- **5-Fold Cross-Validation**: StratifiedGroupKFold with post_id grouping to prevent data leakage
- **DoRA PEFT**: Memory-efficient fine-tuning with configurable rank
- **Hydra Configuration**: Hierarchical YAML-based configuration system
- **MLflow Tracking**: Comprehensive experiment tracking with parent/child run hierarchy
- **Automated Threshold Tuning**: Per-fold F1-optimized thresholds
- **Inference CLI**: Single sample and batch JSONL inference with latency benchmarking
- **Model Registry**: MLflow Model Registry integration for versioning and staging

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd Mentallama_Criteria_CLS
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install
pip install -e '.[dev]'
```

### Training (Smoke Test)

```bash
# Run 2-fold training on 1% of data (validates setup)
python -m Project.SubProject.engine.train_engine \
  training.folds=2 \
  training.max_steps=10 \
  data.sample_fraction=0.01
```

### Training (Full 5-Fold CV)

```bash
python -m Project.SubProject.engine.train_engine \
  experiment.name=mentallam_cv_prod \
  training.batch_size=8 \
  training.max_epochs=100
```

### Aggregate Results

```bash
python -m Project.SubProject.engine.eval_engine aggregate \
  parent_run_id=<MLFLOW_PARENT_RUN_ID>
```

### Inference

```bash
# Single prediction
python -m Project.SubProject.engine.eval_engine infer \
  checkpoint=outputs/checkpoints/fold_0/best.pt \
  post="I feel sad all the time" \
  criterion="Depressed mood most of the day"

# Batch prediction
python -m Project.SubProject.engine.eval_engine infer \
  checkpoint=outputs/checkpoints/fold_0/best.pt \
  input_jsonl=data/samples.jsonl \
  output_jsonl=data/predictions.jsonl
```

### MLflow UI

```bash
make mlflow-ui
# Navigate to http://localhost:5000
```

## Project Structure

```
├── src/Project/SubProject/
│   ├── data/              # Dataset loaders and splits
│   ├── models/            # Model wrappers and prompt builders
│   ├── engine/            # Training, evaluation, inference
│   └── utils/             # MLflow, logging, seed helpers
├── configs/               # Hydra configuration files
├── tests/                 # Unit and integration tests
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── Makefile              # Common tasks
└── pyproject.toml        # Python project config
```

## Configuration

All parameters are configurable via Hydra YAML files or command-line overrides:

```bash
# Override batch size and learning rate
python -m Project.SubProject.engine.train_engine \
  training.batch_size=16 \
  training.lr=1e-5

# Override model config
python -m Project.SubProject.engine.train_engine \
  model.peft.r=16 \
  model.peft.alpha=32
```

See `configs/` for all available parameters.

## Model Registry

### Registering a Model

After training, register the best model in MLflow Model Registry:

```bash
python scripts/register_model.py \
  --run-id <BEST_FOLD_RUN_ID> \
  --model-name mentallama-criteria-cls \
  --stage Production
```

### Loading a Registered Model

```python
import mlflow

# Load production model
model_uri = "models:/mentallama-criteria-cls/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Run inference
result = model.predict({
    "post": "I can't sleep at night",
    "criterion": "Insomnia or hypersomnia nearly every day"
})
print(result)  # {'prediction': 'yes', 'probability': 0.89, ...}
```

### Model Versioning

```bash
# List all versions
mlflow models list --name mentallama-criteria-cls

# Transition to stage
mlflow models transition-version --name mentallama-criteria-cls \
  --version 2 --stage Production --archive-existing-versions

# Get model metadata
mlflow models get --name mentallama-criteria-cls --version 2
```

### Model Metadata

Each registered model includes:
- **Metrics**: F1, accuracy, precision, recall, ROC AUC
- **Threshold**: Tuned decision threshold
- **Fold Information**: Which fold produced this model
- **Model Signature**: Input/output schema
- **Example Inputs**: Test cases for validation
- **Tags**: task, domain, base_model, peft_method, etc.

## Development

### Running Tests

```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make coverage          # Tests with coverage report
```

### Code Quality

```bash
make lint    # Run ruff and black checks
make format  # Auto-format with black and isort
make all     # Format, lint, and test
```

### Validation

```bash
# Validate quickstart workflow
bash scripts/validate_quickstart.sh

# CI mode (includes lint/test)
VALIDATE_MODE=ci bash scripts/validate_quickstart.sh
```

## Documentation

- **User Guide**: `docs/user_guide.md` - Comprehensive documentation (850+ lines)
- **Quickstart**: `specs/001-model-use-mentallam/quickstart.md` - Step-by-step guide
- **Plan**: `specs/001-model-use-mentallam/plan.md` - Implementation plan
- **Tasks**: `specs/001-model-use-mentallam/tasks.md` - Task breakdown

## Performance

- **F1 Score**: Mean F1 ≥ 0.80 across 5 folds
- **CPU Inference**: p95 ≤ 1000 ms (for inputs ≤256 tokens)
- **GPU Memory**: <50GB during training (DoRA r=8, batch_size=8)
- **Training Time**: ~2-3 hours per fold on A100 80GB

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 80GB+ disk space
- 24GB+ GPU memory (training) or 8GB+ (inference)

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python -m Project.SubProject.engine.train_engine training.batch_size=4

# Or reduce DoRA rank
python -m Project.SubProject.engine.train_engine model.peft.r=4
```

### MLflow SQLite Locks

```bash
# Set pool size
export MLFLOW_SQLALCHEMY_POOL_SIZE=1

# Or use server mode
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

See `docs/user_guide.md` for more troubleshooting tips.

## Citation

If you use this work, please cite:

```bibtex
@software{mentallama_criteria_cls,
  title = {MentaLLaMA Binary Classifier for DSM-5 Criteria},
  author = {MLOps Team},
  year = {2025},
  version = {1.0.0}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **MentaLLaMA**: [klyang/MentaLLaMA-chat-7B](https://huggingface.co/klyang/MentaLLaMA-chat-7B)
- **Hugging Face**: Transformers, PEFT, Accelerate
- **MLflow**: Experiment tracking and model registry
- **Hydra**: Configuration management

