# Checklist – Mentallam Binary Classifier (001-model-use-mentallam)

## 0. Preflight
- [ ] On feature branch `001-model-use-mentallam`.
- [ ] Python 3.10 virtualenv active; deps installed (`pip install -e '.[dev]' peft accelerate bitsandbytes hydra-core mlflow scikit-learn`).
- [ ] Hydra config tree present under `configs/` with validated defaults list.
- [ ] CLI contract `contracts/cli.md` updated for current commands.

## 1. Data Integrity & Logging
- [ ] Loader merges `data/redsm5` + `data/dsm5`; schema validated, text cleaned.
- [ ] Counts match spec (posts=1484, criteria=9, pairs=13,356) OR `dataset_override_reason` tag recorded.
- [ ] MLflow parent run logs metrics: `dataset_posts`, `dataset_criteria`, `dataset_pairs`.
- [ ] Dataset hash + manifest (`dataset/manifest.csv`, `dataset/stats.json`) attached as artifacts.
- [ ] StratifiedGroupKFold JSON (`artifacts/folds/fold_{i}.json`) includes seed, group IDs, dataset hash.

## 2. Model & Config
- [ ] Model loads `klyang/MentaLLaMA-chat-7B` via `LlamaForSequenceClassification` with DoRA (r=8, alpha=16, dropout=0.05) + gradient checkpointing.
- [ ] Prompt builder uses template `post: {post}, criterion: {criterion} Does the post match the criterion description? Output yes or no` with max_len=512, truncation=`longest_first`.
- [ ] Hydra configs expose tunables for model/training/cv/logging/inference; resolved config saved per run.
- [ ] Retry/backoff utility wired for Hugging Face + MLflow (base 250 ms, cap 8 s, max attempts 3, jitter, retry on 408/409/429/5xx/timeouts only).

## 3. Training Loop (US1)
- [ ] Training CLI accepts Hydra overrides (`cv.folds`, `training.*`, etc.) and seeds via `utils.set_seed`.
- [ ] Each fold logs metrics (precision, recall, F1, accuracy, ROC-AUC), tuned threshold, confusion matrix, PR curve, checkpoint path.
- [ ] Parent run aggregates child run IDs (nested MLflow) and records Hydra config snapshot, package versions, GPU stats, retry config, dataset stats.
- [ ] Smoke test (`cv.folds=2`, small subset) executed and documented.

## 4. Aggregation (US2)
- [ ] `aggregate` subcommand reads child runs, computes mean/std metrics + best fold, produces Markdown/JSON summary.
- [ ] `artifacts/thresholds.json` generated with per-fold and global thresholds; attached to MLflow parent run.
- [ ] Unit tests cover aggregation math (mock MLflow).

## 5. Inference & Latency (US3)
- [ ] `infer` CLI supports single pair + batch JSONL, loads tuned threshold, logs inference metadata optionally.
- [ ] CPU latency bench harness (≤256 tokens, single thread) reports `latency_cpu_ms_p50/p95` to MLflow; build fails if p95>1000 ms (unless override tag present).
- [ ] Inference CLI documented in quickstart/README, including latency expectations and override instructions.
- [ ] Integration tests cover inference CLI + latency harness (mock timers for determinism).

## 6. Testing & Quality
- [ ] Unit tests: dataset loader, split helper, prompt builder, metrics/thresholds, retry utility.
- [ ] Integration tests: train smoke, inference, aggregation.
- [ ] Lint/format/type-check (ruff, black, pytest, mypy) passing locally/CI.
- [ ] Logs contain fold index, loss curves, LR schedule, GPU mem (nvidia-smi), deterministic flags noted.

## 7. Documentation & Release
- [ ] `research.md` updated with dataset stats, prompt experiments, DoRA VRAM notes, retry policy reference.
- [ ] `quickstart.md` verified end-to-end; troubleshooting section includes OOM, MLflow locks, latency guidance.
- [ ] README references CLI contract, dataset stats, latency KPI.
- [ ] Best checkpoint registered in MLflow Model Registry (`mentallama-criteria-cls`) with signature + example IO.
