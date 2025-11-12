# Quickstart – Mentallam Binary Classifier

## 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
pip install peft accelerate bitsandbytes hydra-core==1.3 mlflow scikit-learn
```
Optional: add `pip install flash-attn` if available for faster training.

## 2. Hydra Configs
Create `configs/config.yaml`:
```yaml
defaults:
  - data: redsm5
  - model: mentallam
  - training: cv
  - logging: mlflow
  - inference: base
  - _self_
```
Example `configs/model/mentallam.yaml`:
```yaml
model:
  name: klyang/MentaLLaMA-chat-7B
  max_length: 512
  prompt_template: "post: {post}, criterion: {criterion} Does the post match the criterion description? Output yes or no"
  peft:
    type: dora
    target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
    r: 8
    alpha: 16
    dropout: 0.05
  grad_checkpointing: true
```
`configs/training/cv.yaml` snippet:
```yaml
training:
  folds: 5
  seed: 42
  batch_size: 8
  grad_accum: 4
  max_epochs: 100
  patience: 20
  lr: 2e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
```
`configs/logging/mlflow.yaml`:
```yaml
logging:
  tracking_uri: sqlite:///mlflow.db
  artifact_root: mlruns
  experiment: mentallam_cv
```

## 3. Generate Folds (optional but recommended)
```bash
python -m Project.SubProject.data.splits \
  data.redsm5_path=data/redsm5/posts.csv \
  data.dsm5_path=data/dsm5/criteria.csv \
  outputs.folds_dir=artifacts/folds
```
Inspect `artifacts/folds/fold_0.json` to verify grouping.

## 4. Run 5-fold Training
```bash
python -m Project.SubProject.engine.train_engine \
  experiment.name=mentallam_cv \
  training.batch_size=8 training.grad_accum=4 \
  training.max_epochs=100 training.patience=20 \
  logging.tracking_uri=sqlite:///mlflow.db \
  logging.artifact_root=mlruns \
  hydra.run.dir=outputs/runs/${now:%Y-%m-%d}/${hydra.job.num}
```
- Produces nested MLflow runs (parent summary + child per fold).
- Saves checkpoints under `outputs/checkpoints/fold_{i}`.
- Persists Hydra config to each run.

### Smoke Test
Before full CV, run:
```bash
python -m Project.SubProject.engine.train_engine \
  cv.folds=2 training.max_steps=50 data.sample_fraction=0.01 hydra.mode=debug
```
This validates wiring without long runtimes.

## 5. Aggregate Metrics
```bash
python -m Project.SubProject.engine.eval_engine aggregate \
  parent_run_id=<PARENT_RUN_ID>
```
Writes summary JSON/Markdown to `mlruns/.../artifacts/summary/` and `artifacts/thresholds.json`.

## 6. Inference CLI
```bash
python -m Project.SubProject.engine.eval_engine infer \
  checkpoint=outputs/checkpoints/fold_0/best.pt \
  criterion="Patient reports insomnia" \
  post="I have trouble sleeping every night" \
  thresholds_file=artifacts/thresholds.json \
  log_to_mlflow=true
```
Returns `label=yes|no` and probability. For batch mode provide `--input_jsonl data/samples.jsonl`.

## 7. QA & Tooling
```bash
ruff check src tests
black --check src tests
pytest -q
```
CI/smoke suggestions:
- `make smoke` → runs 2-fold tiny training.
- `make lint` → runs ruff + black + mypy.

## 8. Troubleshooting
- **CUDA OOM**: Lower `training.batch_size` or reduce DoRA rank. Ensure `bitsandbytes` installed.
- **MLflow lock**: Delete `mlflow.db-journal`, rerun with `MLFLOW_SQLALCHEMY_POOL_SIZE=1`.
- **Slow tokenization**: Enable caching via `data.cache_tokenized=true` (to be implemented in loader).
