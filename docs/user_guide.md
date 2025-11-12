# Mentallama Binary Classifier – User Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-13
**Maintainer**: MLOps Team

## Table of Contents
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Training Workflow](#training-workflow)
4. [Evaluation & Metrics](#evaluation--metrics)
5. [Inference](#inference)
6. [MLflow Tracking](#mlflow-tracking)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)
9. [Advanced Topics](#advanced-topics)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA 11.8+ (for GPU training)
- 80GB+ disk space for models and artifacts
- Linux or macOS (Windows via WSL2)

### Basic Installation
```bash
# Clone repository
git clone <repo-url>
cd Mentallama_Criteria_CLS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install package with dev dependencies
pip install -e '.[dev]'
```

### Optional Dependencies

**Flash Attention** (faster training):
```bash
pip install flash-attn --no-build-isolation
```

**BitsAndBytes** (memory optimization):
```bash
pip install bitsandbytes
```

**GPU Verification**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Configuration

### Hydra Configuration System

This project uses [Hydra](https://hydra.cc/) for hierarchical configuration management. All parameters are defined in YAML files under `configs/`.

#### Configuration Structure
```
configs/
├── config.yaml              # Root config with defaults list
├── data/
│   ├── redsm5.yaml         # RedSM5 dataset config
│   └── dsm5.yaml           # DSM5 criteria config
├── model/
│   ├── mentallam.yaml      # MentaLLaMA model config
│   └── dora.yaml           # DoRA adapter config
├── training/
│   ├── cv.yaml             # Cross-validation training
│   └── single.yaml         # Single-fold training
├── logging/
│   └── mlflow.yaml         # MLflow tracking config
├── inference/
│   └── base.yaml           # Inference defaults
└── overrides/
    └── debug.yaml          # Debug/smoke test overrides
```

#### Root Config (`configs/config.yaml`)
```yaml
defaults:
  - data: redsm5
  - model: mentallam
  - training: cv
  - logging: mlflow
  - inference: base
  - _self_

# Global settings
seed: 42
device: cuda
output_dir: outputs
```

#### Model Config (`configs/model/mentallam.yaml`)
```yaml
model:
  name: klyang/MentaLLaMA-chat-7B
  num_labels: 2
  max_length: 512

  # Prompt template
  prompt_template: |
    post: {post}
    criterion: {criterion}
    Does the post match the criterion description? Output yes or no

  # DoRA configuration
  peft:
    type: dora
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    r: 8                    # LoRA rank
    alpha: 16               # Scaling factor
    dropout: 0.05
    bias: none
    task_type: SEQ_CLS

  # Memory optimizations
  grad_checkpointing: true
  use_cache: false
  torch_dtype: bfloat16
```

**Key Parameters Explained**:
- `r`: LoRA rank (higher = more parameters, better fit but slower)
- `alpha`: Scaling factor (typically 2×r)
- `target_modules`: Which layers to apply adapters to
- `grad_checkpointing`: Trade compute for memory

#### Training Config (`configs/training/cv.yaml`)
```yaml
training:
  # Cross-validation
  folds: 5
  seed: 42
  stratify_by: label
  group_by: post_id

  # Training parameters
  batch_size: 8
  grad_accum_steps: 4       # Effective batch = 8 × 4 = 32
  max_epochs: 100
  patience: 20              # Early stopping patience

  # Optimizer
  optimizer: adamw
  lr: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0

  # Learning rate schedule
  lr_scheduler: cosine
  num_cycles: 0.5

  # Evaluation
  eval_strategy: epoch
  eval_steps: null
  save_strategy: epoch
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: f1
  greater_is_better: true

  # Threshold tuning
  threshold_sweep:
    enabled: true
    start: 0.0
    end: 1.0
    step: 0.01
    metric: f1              # Optimize for F1 score
```

**Key Parameters Explained**:
- `batch_size × grad_accum_steps`: Effective batch size
- `patience`: Stop if no improvement for N epochs
- `warmup_ratio`: Fraction of steps for LR warmup
- `threshold_sweep`: Tune decision threshold per fold

#### Logging Config (`configs/logging/mlflow.yaml`)
```yaml
logging:
  tracking_uri: sqlite:///mlflow.db
  artifact_root: mlruns
  experiment: mentallam_cv

  # Run naming
  run_name_template: "fold_{fold_idx}"
  parent_run_name: "cv_experiment_{timestamp}"

  # Logging frequency
  log_every_n_steps: 10
  log_gradients: false      # Set true for debugging
  log_model_checkpoints: true

  # Artifact organization
  artifacts:
    checkpoints_dir: checkpoints
    metrics_dir: metrics
    curves_dir: curves
    configs_dir: configs
    folds_dir: folds
```

### Overriding Configuration

**Command-line Overrides**:
```bash
# Single parameter
python -m Project.SubProject.engine.train_engine training.batch_size=16

# Multiple parameters
python -m Project.SubProject.engine.train_engine \
  training.batch_size=16 \
  training.lr=1e-5 \
  training.max_epochs=50

# Nested parameters
python -m Project.SubProject.engine.train_engine \
  model.peft.r=16 \
  model.peft.alpha=32

# Config group override
python -m Project.SubProject.engine.train_engine \
  training=single \
  data=dsm5
```

**Debug/Smoke Testing**:
```bash
python -m Project.SubProject.engine.train_engine \
  training.folds=2 \
  training.max_epochs=3 \
  data.sample_fraction=0.01 \
  hydra.mode=debug
```

---

## Training Workflow

### Step 1: Prepare Data

**Verify Data Files**:
```bash
ls -lh data/DSM5/
ls -lh data/redsm5/
```

Expected structure:
```
data/
├── DSM5/
│   ├── criteria.csv
│   └── metadata.json
└── redsm5/
    ├── posts.csv
    ├── labels.csv
    └── metadata.json
```

**Data Validation**:
```bash
python -c "
from Project.SubProject.data.dataset import MentalHealthDataset
dataset = MentalHealthDataset(
    redsm5_path='data/redsm5',
    dsm5_path='data/DSM5'
)
print(f'Total samples: {len(dataset)}')
print(f'Label distribution: {dataset.get_label_distribution()}')
"
```

### Step 2: Generate Fold Splits

```bash
python -m Project.SubProject.data.splits \
  data.redsm5_path=data/redsm5 \
  data.dsm5_path=data/DSM5 \
  training.folds=5 \
  training.seed=42 \
  output.folds_dir=artifacts/folds
```

This creates:
```
artifacts/folds/
├── fold_0.json
├── fold_1.json
├── fold_2.json
├── fold_3.json
├── fold_4.json
└── splits_metadata.json
```

Each `fold_*.json` contains:
```json
{
  "fold_index": 0,
  "train_indices": [...],
  "val_indices": [...],
  "train_groups": [...],
  "val_groups": [...],
  "label_distribution": {
    "train": {"0": 5000, "1": 5000},
    "val": {"0": 1178, "1": 1178}
  },
  "seed": 42,
  "dataset_hash": "a3f5e8d..."
}
```

### Step 3: Run Training

**Full 5-Fold Training**:
```bash
python -m Project.SubProject.engine.train_engine \
  experiment.name=mentallam_cv_$(date +%Y%m%d) \
  training.batch_size=8 \
  training.grad_accum_steps=4 \
  training.max_epochs=100 \
  training.patience=20
```

**Output Structure**:
```
outputs/
├── runs/
│   └── 2025-11-13/
│       └── 001/
│           ├── .hydra/
│           │   ├── config.yaml
│           │   └── overrides.yaml
│           └── checkpoints/
│               ├── fold_0/
│               │   ├── best.pt
│               │   └── last.pt
│               ├── fold_1/
│               ⋮
└── mlruns/
    └── 0/
        ├── <parent_run_id>/
        └── <child_run_ids>/
```

**Training Logs**:
```
[2025-11-13 10:15:32] INFO - Starting 5-fold cross-validation
[2025-11-13 10:15:33] INFO - Fold 0/5: train=10678, val=2678
[2025-11-13 10:15:45] INFO - Epoch 1/100: train_loss=0.523, val_loss=0.412, val_f1=0.765
[2025-11-13 10:16:12] INFO - Epoch 2/100: train_loss=0.398, val_loss=0.365, val_f1=0.812
...
[2025-11-13 10:45:23] INFO - Early stopping triggered at epoch 42
[2025-11-13 10:45:24] INFO - Best checkpoint: epoch=22, val_f1=0.847
[2025-11-13 10:45:25] INFO - Fold 0 complete: best_f1=0.847, threshold=0.52
```

### Step 4: Monitor Progress

**MLflow UI**:
```bash
make mlflow-ui
# Navigate to http://localhost:5000
```

**Key Metrics to Monitor**:
- `train_loss`: Should decrease steadily
- `val_loss`: Should decrease then plateau
- `val_f1`: Should increase then plateau
- `learning_rate`: Should follow schedule
- `gpu_memory_allocated`: Should be stable

**View Artifacts**:
- Confusion matrices: `artifacts/metrics/fold_{i}/confusion_matrix.png`
- PR curves: `artifacts/curves/fold_{i}/pr_curve.png`
- Checkpoints: `outputs/checkpoints/fold_{i}/best.pt`

---

## Evaluation & Metrics

### Aggregating Fold Results

After training completes, aggregate metrics across all folds:

```bash
python -m Project.SubProject.engine.eval_engine aggregate \
  parent_run_id=<MLFLOW_PARENT_RUN_ID>
```

**Output**: `artifacts/cv_summary.json`
```json
{
  "mean_metrics": {
    "accuracy": 0.863,
    "precision": 0.858,
    "recall": 0.871,
    "f1": 0.864,
    "roc_auc": 0.921
  },
  "std_metrics": {
    "accuracy": 0.012,
    "precision": 0.015,
    "recall": 0.018,
    "f1": 0.014,
    "roc_auc": 0.008
  },
  "best_fold": {
    "fold_index": 2,
    "f1": 0.882,
    "threshold": 0.51
  },
  "fold_results": [
    {
      "fold_index": 0,
      "accuracy": 0.855,
      "precision": 0.847,
      "recall": 0.869,
      "f1": 0.858,
      "threshold": 0.52
    },
    ...
  ],
  "thresholds": [0.52, 0.49, 0.51, 0.50, 0.53]
}
```

### Understanding Metrics

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**: Of predicted positives, how many are correct?
```
Precision = TP / (TP + FP)
```

**Recall**: Of actual positives, how many did we catch?
```
Recall = TP / (TP + FN)
```

**F1 Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**ROC AUC**: Area under receiver operating characteristic curve (threshold-invariant)

### Confusion Matrix Interpretation

```
                Predicted
                No    Yes
Actual  No    [ TN  | FP ]
        Yes   [ FN  | TP ]
```

Example:
```
                Predicted
                No    Yes
Actual  No    [1150 | 28 ]
        Yes   [ 35  | 1143]
```

- **TN (1150)**: Correctly identified non-matches
- **FP (28)**: False alarms (predicted match, actually non-match)
- **FN (35)**: Missed cases (predicted non-match, actually match)
- **TP (1143)**: Correctly identified matches

### Threshold Tuning

The classifier outputs probabilities `[0, 1]`. The threshold determines the decision boundary.

**Default**: 0.5
**Tuned**: Optimizes F1 score on validation set

```python
# Threshold sweep example
thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = []
for t in thresholds:
    preds = (probas >= t).astype(int)
    f1 = f1_score(labels, preds)
    f1_scores.append(f1)

best_threshold = thresholds[np.argmax(f1_scores)]
```

**When to Adjust**:
- High precision needed (medical diagnosis): Increase threshold
- High recall needed (screening): Decrease threshold
- Balanced: Use F1-optimal threshold

---

## Inference

### Single Sample Prediction

```bash
python -m Project.SubProject.engine.eval_engine infer \
  checkpoint=outputs/checkpoints/fold_0/best.pt \
  post="I can't sleep at night and feel tired all day" \
  criterion="Insomnia or hypersomnia nearly every day" \
  threshold=0.52
```

**Output**:
```json
{
  "prediction": "yes",
  "probability": 0.847,
  "threshold": 0.52,
  "confidence": "high"
}
```

### Batch Inference

**Input JSONL** (`data/batch_inputs.jsonl`):
```jsonl
{"post_id": "p001", "criterion_id": "c001", "post": "I feel sad all the time", "criterion": "Depressed mood most of the day"}
{"post_id": "p002", "criterion_id": "c002", "post": "I love my job", "criterion": "Loss of interest or pleasure"}
```

**Command**:
```bash
python -m Project.SubProject.engine.eval_engine infer \
  checkpoint=outputs/checkpoints/fold_0/best.pt \
  input_jsonl=data/batch_inputs.jsonl \
  output_jsonl=data/batch_outputs.jsonl \
  thresholds_file=artifacts/thresholds.json
```

**Output JSONL** (`data/batch_outputs.jsonl`):
```jsonl
{"post_id": "p001", "criterion_id": "c001", "prediction": "yes", "probability": 0.923, "threshold": 0.52}
{"post_id": "p002", "criterion_id": "c002", "prediction": "no", "probability": 0.134, "threshold": 0.52}
```

### Latency Benchmarking

```bash
python -m Project.SubProject.engine.eval_engine infer \
  checkpoint=outputs/checkpoints/fold_0/best.pt \
  benchmark=true \
  benchmark_samples=100 \
  device=cpu
```

**Output**:
```
Latency Benchmark (CPU):
  Samples: 100
  p50: 234 ms
  p95: 456 ms
  p99: 612 ms
  Mean: 267 ms
```

**Performance Targets**:
- CPU p95: ≤ 1000 ms (for inputs ≤256 tokens)
- GPU p95: ≤ 50 ms

---

## MLflow Tracking

### Understanding Run Hierarchy

```
Parent Run (cv_experiment_20251113)
├── Child Run (fold_0)
│   ├── Metrics: train_loss, val_loss, val_f1, ...
│   ├── Params: lr, batch_size, ...
│   └── Artifacts: checkpoint, confusion_matrix, pr_curve
├── Child Run (fold_1)
├── Child Run (fold_2)
├── Child Run (fold_3)
└── Child Run (fold_4)
```

### Accessing Runs Programmatically

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Get experiment
experiment = mlflow.get_experiment_by_name("mentallam_cv")

# Get all runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

# Filter by parent run
parent_run_id = "<PARENT_RUN_ID>"
child_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
)

# Get best fold by F1
best_fold = child_runs.loc[child_runs['metrics.val_f1'].idxmax()]
print(f"Best fold: {best_fold['tags.fold_index']}")
print(f"Best F1: {best_fold['metrics.val_f1']:.3f}")

# Download artifact
artifact_path = mlflow.artifacts.download_artifacts(
    run_id=best_fold['run_id'],
    artifact_path="checkpoints/best.pt"
)
```

### Comparing Experiments

```python
# Compare two experiments
exp1_runs = mlflow.search_runs(experiment_ids=["1"])
exp2_runs = mlflow.search_runs(experiment_ids=["2"])

# Aggregate metrics
import pandas as pd
comparison = pd.DataFrame({
    "Experiment 1": exp1_runs['metrics.val_f1'].describe(),
    "Experiment 2": exp2_runs['metrics.val_f1'].describe()
})
print(comparison)
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:

1. **Reduce batch size**:
   ```bash
   python -m Project.SubProject.engine.train_engine training.batch_size=4
   ```

2. **Increase gradient accumulation**:
   ```bash
   python -m Project.SubProject.engine.train_engine \
     training.batch_size=4 \
     training.grad_accum_steps=8
   ```

3. **Enable gradient checkpointing** (already default):
   ```yaml
   model:
     grad_checkpointing: true
   ```

4. **Reduce model precision**:
   ```yaml
   model:
     torch_dtype: float16  # or bfloat16
   ```

5. **Reduce DoRA rank**:
   ```yaml
   model:
     peft:
       r: 4
       alpha: 8
   ```

6. **Use 8-bit optimizer**:
   ```bash
   pip install bitsandbytes
   ```
   ```yaml
   training:
     optimizer: adamw_8bit
   ```

### Issue 2: MLflow SQLite Lock Errors

**Symptoms**:
```
OperationalError: database is locked
```

**Solutions**:

1. **Set pool size**:
   ```bash
   export MLFLOW_SQLALCHEMY_POOL_SIZE=1
   ```

2. **Use file locking**:
   ```bash
   export MLFLOW_SQLITE_TIMEOUT=60
   ```

3. **Upgrade to server mode**:
   ```bash
   # Terminal 1: Start server
   mlflow server \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root mlruns \
     --host 127.0.0.1 \
     --port 5000

   # Terminal 2: Update config
   tracking_uri: http://127.0.0.1:5000
   ```

4. **Close runs promptly**:
   ```python
   with mlflow.start_run():
       # Your code
       pass  # Run auto-closes here
   ```

### Issue 3: Slow Training

**Symptoms**:
- Training takes >1 hour per epoch
- GPU utilization <50%

**Solutions**:

1. **Increase batch size**:
   ```bash
   python -m Project.SubProject.engine.train_engine training.batch_size=16
   ```

2. **Enable mixed precision** (already default):
   ```yaml
   model:
     torch_dtype: bfloat16
   ```

3. **Install Flash Attention**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

4. **Profile dataloading**:
   ```python
   from torch.profiler import profile, ProfilerActivity
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       train_one_epoch()
   print(prof.key_averages().table())
   ```

5. **Enable DataLoader num_workers**:
   ```yaml
   training:
     dataloader_num_workers: 4
   ```

### Issue 4: Data Validation Failures

**Symptoms**:
```
AssertionError: Expected 13356 samples, got 12000
```

**Solutions**:

1. **Check data integrity**:
   ```bash
   python -c "
   import pandas as pd
   posts = pd.read_csv('data/redsm5/posts.csv')
   labels = pd.read_csv('data/redsm5/labels.csv')
   criteria = pd.read_csv('data/DSM5/criteria.csv')
   print(f'Posts: {len(posts)}')
   print(f'Labels: {len(labels)}')
   print(f'Criteria: {len(criteria)}')
   "
   ```

2. **Override validation** (use with caution):
   ```bash
   python -m Project.SubProject.engine.train_engine \
     data.validate_cardinality=false \
     data.override_reason="Using subset for debugging"
   ```

3. **Check for duplicates**:
   ```python
   labels = pd.read_csv('data/redsm5/labels.csv')
   print(labels.duplicated().sum())
   ```

### Issue 5: Threshold Tuning Instability

**Symptoms**:
- Thresholds vary widely across folds (e.g., 0.3, 0.7, 0.4)
- F1 scores inconsistent

**Solutions**:

1. **Use median threshold**:
   ```python
   thresholds = [0.52, 0.49, 0.51, 0.50, 0.53]
   final_threshold = np.median(thresholds)  # 0.51
   ```

2. **Manual override**:
   ```bash
   python -m Project.SubProject.engine.eval_engine infer \
     checkpoint=... \
     threshold=0.50
   ```

3. **Examine PR curves**:
   - Open `artifacts/curves/fold_{i}/pr_curve.png`
   - Look for steep drop-offs indicating instability

---

## FAQ

### Q1: How long does training take?

**A**: On a single A100 GPU (80GB):
- Per fold: ~2-3 hours
- Full 5-fold CV: ~10-15 hours

Factors:
- Dataset size: 13,356 samples
- Epochs: ~30-50 (with early stopping)
- Batch size: 8 × 4 grad_accum = effective 32

### Q2: Can I resume training from a checkpoint?

**A**: Yes, add `resume_from_checkpoint` parameter:
```bash
python -m Project.SubProject.engine.train_engine \
  resume_from_checkpoint=outputs/checkpoints/fold_0/last.pt
```

### Q3: How do I use a different model?

**A**: Update `configs/model/mentallam.yaml`:
```yaml
model:
  name: meta-llama/Llama-2-7b-hf  # or any HF model
  # ... rest of config
```

### Q4: Can I run on CPU?

**A**: Yes, but it will be very slow (~50x slower than GPU):
```bash
python -m Project.SubProject.engine.train_engine device=cpu
```

### Q5: How do I export the model for deployment?

**A**: Use the model registry script:
```bash
python scripts/register_model.py \
  --run-id <BEST_FOLD_RUN_ID> \
  --model-name mentallama-criteria-cls \
  --stage Production
```

### Q6: What if I want to use my own dataset?

**A**: Create a custom dataset loader:
```python
# src/Project/SubProject/data/custom_dataset.py
from .dataset import MentalHealthDataset

class CustomDataset(MentalHealthDataset):
    def load_data(self):
        # Load your CSV/JSON files
        # Return list of samples
        pass
```

Update config:
```yaml
data:
  loader: custom
  path: /path/to/your/data
```

### Q7: How do I tune hyperparameters?

**A**: Use Hydra's multirun feature:
```bash
python -m Project.SubProject.engine.train_engine \
  --multirun \
  training.lr=1e-5,2e-5,5e-5 \
  model.peft.r=4,8,16
```

Or use Optuna integration (see Advanced Topics).

### Q8: Can I use LoRA instead of DoRA?

**A**: Yes, update PEFT config:
```yaml
model:
  peft:
    type: lora  # Change from dora to lora
    # ... rest of config
```

### Q9: How do I interpret the confusion matrix?

**A**: See [Confusion Matrix Interpretation](#confusion-matrix-interpretation).

### Q10: What's the difference between `best.pt` and `last.pt`?

**A**:
- `best.pt`: Checkpoint with highest val_f1 during training
- `last.pt`: Checkpoint from the final epoch (for resuming)

Always use `best.pt` for inference.

### Q11: Can I run multiple experiments in parallel?

**A**: Yes, but be cautious with MLflow SQLite:
```bash
# Terminal 1
python -m Project.SubProject.engine.train_engine experiment.name=exp1

# Terminal 2
python -m Project.SubProject.engine.train_engine experiment.name=exp2
```

Consider using MLflow server mode for better concurrency.

### Q12: How do I visualize training curves?

**A**: Use MLflow UI or TensorBoard:
```bash
# MLflow
make mlflow-ui

# Or export to TensorBoard
mlflow deployments run-local -t tensorboard \
  --model-uri mlruns/0/<run_id>
```

### Q13: What's the minimum GPU memory required?

**A**:
- Training: 24GB (with default config)
- Inference: 8GB

To reduce memory:
- Lower batch size
- Reduce DoRA rank
- Use quantization (8-bit, 4-bit)

### Q14: How do I handle class imbalance?

**A**: The dataset is already balanced (50/50 matched/unmatched).

If you have custom imbalanced data:
```yaml
training:
  class_weights: [0.3, 0.7]  # Weight for [class 0, class 1]
```

### Q15: Can I use custom metrics?

**A**: Yes, extend `engine/metrics.py`:
```python
def custom_metric(y_true, y_pred):
    # Your metric logic
    return score

# Register in MetricsComputer
metrics_computer.register_metric("custom", custom_metric)
```

### Q16: How do I debug training issues?

**A**: Enable verbose logging:
```bash
python -m Project.SubProject.engine.train_engine \
  logging.level=DEBUG \
  logging.log_gradients=true
```

### Q17: What's the model size?

**A**:
- Base model: ~7B parameters (~14GB)
- DoRA adapters: ~8M parameters (~16MB)
- Total: ~7.008B parameters

### Q18: Can I use this for multi-class classification?

**A**: Yes, update:
```yaml
model:
  num_labels: 5  # Number of classes
```

And adjust dataset loader to output multi-class labels.

### Q19: How do I cite this work?

**A**: (Add citation format here)

### Q20: Where can I get help?

**A**:
- GitHub Issues: <repo-url>/issues
- Slack: #mentallama-support
- Docs: <repo-url>/docs

---

## Advanced Topics

### Distributed Training

(Coming soon: multi-GPU, multi-node training with DeepSpeed/FSDP)

### Hyperparameter Tuning with Optuna

(Coming soon: automated HP search)

### Model Interpretability

(Coming soon: SHAP, attention visualization)

### Production Deployment

(Coming soon: FastAPI serving, Docker containers)

---

**End of User Guide**
