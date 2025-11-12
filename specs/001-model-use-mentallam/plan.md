# Implementation Plan: Mentallam Binary Classifier with 5-Fold CV

**Branch**: `001-model-use-mentallam` | **Date**: 2025-11-12 | **Spec**: `specs/001-model-use-mentallam/spec.md`
**Input**: Clarified spec describing mentallam fine-tune, Hydra configs, MLflow tracking

## Summary
Train and evaluate a Transformer-based binary classifier that determines whether a Reddit-style post
matches a DSM-5 criterion. The solution fine-tunes `klyang/MentaLLaMA-chat-7B` with DoRA PEFT,
deterministically merges RedSM5/DSM5 datasets, performs StratifiedGroupKFold CV grouped by `post_id`,
validates dataset cardinalities, logs stats/hashes to MLflow, tunes thresholds per fold, and logs all
runs via Hydra+MLflow. Deliverables include reproducible configs, a retry-aware training stack,
training/aggregation/inference CLIs, documentation, a CPU latency bench harness, and tests.

## Technical Context

| Field | Value |
| --- | --- |
| Language / Version | Python 3.10 |
| Primary Dependencies | PyTorch 2.2+, Transformers>=4.40, peft (DoRA), accelerate, Hydra 1.3, MLflow 2.8, scikit-learn |
| Storage | Source CSVs under `data/redsm5` & `data/dsm5`; MLflow metadata `sqlite:///mlflow.db`; artifacts in `mlruns/` |
| Testing | pytest (unit+integration), ruff, black, mypy (critical modules) |
| Target Platform | Linux GPU node (≥1×A100 80GB or equivalent) with bf16 support |
| Project Type | Single ML training project under `src/Project/SubProject/` |
| Performance Goals | Mean F1 ≥ 0.80; CPU inference p95 ≤ 1.0 s for ≤256-token inputs; A100 latency reported but non-blocking |
| Constraints | Hydra required for all params; MLflow logging mandatory; deterministic seeds; DoRA memory budget <50GB; retry/backoff wrapper for HF/MLflow |
| Scale / Scope | ~10k labelled pairs; 5-fold CV (~5× training cost); experiments tracked via MLflow |

## Constitution Check (Gate)
- **Hydra Config Enforcement** – Config tree under `configs/` with defaults for data/model/training/logging/inference; resolved config logged per run ✔️
- **MLflow Tracking** – All runs use `sqlite:///mlflow.db` + `mlruns/`; nested parent/child runs and artifact logging ✔️
- **Binary Matched/Unmatched Task** – Prompt encodes `(post, criterion)`; outputs logits for {matched, unmatched}; tuned thresholds persisted ✔️
- **Reproducibility & Logging** – `utils.set_seed` invoked, Hydra + MLflow log configs/versions, `utils.get_logger` used ✔️
_Gate status: PASS_

## Project Structure
```
specs/001-model-use-mentallam/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/

src/Project/SubProject/
├── data/
│   ├── dataset.py          # loaders + schema validation
│   └── splits.py           # StratifiedGroupKFold helper
├── models/
│   ├── model.py            # MentallamClassifier + DoRA
│   └── prompt_builder.py   # prompt formatting utilities
├── engine/
│   ├── train_engine.py     # Hydra CLI for training
│   ├── eval_engine.py      # aggregation + inference commands
│   └── metrics.py          # metrics + threshold sweep helpers
└── utils/
    ├── mlflow_utils.py
    ├── seed.py
    └── log.py

configs/
├── config.yaml
├── data/redsm5.yaml
├── model/mentallam.yaml
├── training/cv.yaml
├── logging/mlflow.yaml
├── inference/base.yaml
└── overrides/debug.yaml

tests/
├── unit/test_data_pipeline.py
├── unit/test_metrics.py
├── unit/test_prompt_builder.py
└── integration/test_train_smoke.py
```

## Phase 0 – Research & Data Prep
1. Audit RedSM5/DSM5 CSVs (schema, encoding, class balance, duplicates). Capture tables/plots + counts in `research.md` and define expected hash/checksum procedure.
2. Prototype prompt variants (current vs criterion-first) on a tiny labeled slice; document accuracy delta.
3. Measure DoRA memory footprint (r=8, alpha=16, grad_accum=4, bf16) on available GPU; adjust if needed.
4. Prototype threshold sweep via scikit-learn `precision_recall_curve` vs manual torch sweep; choose final helper.
5. Decide whether to cache tokenized tensors per fold; document tradeoffs.
Deliverable: Updated `research.md` with evidence, dataset stats tables, checksum approach, and retry/backoff policy reference.

## Phase 1 – Design & Contracts
1. Finalize entity definitions (`Sample`, `FoldAssignment`, `RunArtifact`, `HydraConfigSnapshot`) plus serialization helpers (documented in `data-model.md`).
2. Author CLI contracts (`contracts/cli.md`) describing Hydra commands for `train`, `aggregate`, `infer`, required overrides, sample outputs, and ownership (MLOps primary, Modeling reviewer). Reference from README and quickstart.
3. Create Hydra config tree with validation (OmegaConf + dataclasses), defaults list, debug override (`overrides/debug.yaml`).
4. Design MLflow artifact layout (configs/, checkpoints/, metrics/, curves/, folds/, dataset stats, latency harness) and failure-handling rules (retry vs fail fast using shared utility).
5. Update `quickstart.md` with environment commands, Hydra overrides, MLflow setup, smoke-test instructions.
Deliverables: data-model, contracts doc, quickstart, config scaffold PR.

## Phase 2 – Foundation (Blocking)
1. Implement dataset loader (`data/dataset.py`) merging posts/criteria/labels; enforce schema validation, text cleaning, label mapping, and compute cardinalities (posts, criteria, pairs). Fail fast when counts ≠ 1484/9/13356 unless override provided; emit stats artifact + MLflow metrics/tags.
2. Implement StratifiedGroupKFold helper (`data/splits.py`) that persists `artifacts/folds/fold_{i}.json` with metadata (seed, tuned threshold placeholder, dataset hash, override reason when applicable).
3. Build prompt builder/formatter module with unit tests covering edge cases (empty text, long sequences, truncation behavior).
4. Implement `MentallamClassifier` wrapper (DoRA adapter loading, gradient checkpointing, bf16 autocast, inference helper).
5. Create shared retry/backoff utility (`src/infra/retries.py`) with Hydra-configurable parameters (base_ms, cap_s, max_attempts) applying exponential backoff w/ full jitter and HTTP status selection. Use in HF download + MLflow calls.
6. Add metrics/threshold module with utilities for confusion matrix, PR curve, F1 sweep (0.00–1.00 step 0.01).
7. Extend utility tests + lint setups (pytest/ruff/black pre-commit or CI script).
Checkpoint: Data/model infrastructure ready, tests passing.

## Phase 3 – User Story 1 (Train 5-fold)
1. Build Hydra-driven training CLI (`train_engine.py`). Flow: resolve config → init MLflow parent run → loop folds.
2. Fold loop: create datasets/dataloaders, configure optimizer/lr scheduler, run accelerate Trainer or custom loop with early stopping (patience=20).
3. Post-epoch evaluation: run validation, compute metrics, sweep thresholds, log confusion matrix + PR curve artifacts, save checkpoint to `outputs/checkpoints/fold_{i}`.
4. Log Hydra config snapshot, fold metadata, environment info, GPU stats, package versions, dataset stats, retry configuration to MLflow child run.
5. Add smoke integration test (`tests/integration/test_train_smoke.py`) running `cv.folds=2`, `max_steps=5` on toy data.
Deliverable: functioning multi-fold training CLI + tests.

## Phase 4 – User Story 2 (Aggregation & Reporting)
1. Implement `aggregate` subcommand (in `eval_engine.py`): fetch child runs via MLflow API, compute mean/std metrics, best fold details, threshold summary, produce Markdown/JSON artifacts.
2. Persist tuned thresholds per fold + global summary (`artifacts/thresholds.json`) and attach to parent run.
3. Generate optional HTML/Markdown table with metric comparisons + checkpoint paths.
4. Write unit tests mocking MLflow responses to validate aggregation math + artifact writing.
Deliverable: aggregation CLI + tests, documented in quickstart.

## Phase 5 – User Story 3 (Inference)
1. Implement `infer` subcommand: load checkpoint + tuned threshold, tokenize single pair or batch JSONL, output label (`yes`/`no`) + probability, optional MLflow logging.
2. Support deterministic seeds and ability to reuse cached tokenizer; provide `--checkpoint-run <run_id>` convenience flag.
3. Integrate CPU latency bench harness (≤256-token inputs) measuring p50/p95; log to MLflow metrics `latency_cpu_ms_p50/p95` and fail when p95 > 1000 ms (unless override tag set).
4. Update docs/quickstart with inference examples, latency expectations, and troubleshooting.
5. Add tests covering CLI argument parsing, single prediction, batch mode, optional logging, and bench harness wiring (mock timers).
Deliverable: inference CLI + docs/tests.

## Phase 6 – Polish & Release
- **Observability**: ensure logs capture fold index, epoch loss curves, learning rate schedule, GPU memory (nvidia-smi), Hydra configs, retry stats, and CPU latency bench results.
- **Reproducibility**: document deterministic flags (`torch.use_deterministic_algorithms`, cudnn settings) and store fold assignment artifacts with runs.
- **Deployment**: register best checkpoint in MLflow Model Registry, attach model signature + example IO, note version in README; publish CLI contract version reference.
- CI: add script or GitHub Action running lint/format/test; optional nightly smoke run with reduced data.

## Risks & Mitigations
| Risk | Impact | Mitigation |
| --- | --- | --- |
| GPU memory exhaustion despite DoRA | Training halts | Increase grad_accum, reduce batch size, lower DoRA rank, enable 8-bit optimizer, or switch to 4-bit adapters |
| Data leakage across folds | Inflated metrics | Enforce StratifiedGroupKFold with `post_id` groups; assert no duplicates; persist fold metadata |
| Threshold tuning instability | Poor inference decisions | Log PR curves, smooth thresholds via moving average, allow manual override for deployments |
| MLflow sqlite locks | Lost logs | Reduce concurrency, set `MLFLOW_SQLALCHEMY_POOL_SIZE=1`, close runs promptly, migrate to local server if needed |
| Tokenization latency | Slow training | Cache tokenized tensors per fold or leverage Hugging Face datasets with map-style caching |
| Dataset drift / mismatch | Model mismatch | Fail fast if counts ≠ expected unless override tag supplied; log dataset hash + manifest |
| External dependency outages | Run failures | Use shared retry/backoff utility with exponential backoff + jitter; surface errors with remediation guidance |
| CPU latency regressions | KPI miss | Maintain automated bench harness; integrate into CI and gating logic |

## Next Steps
1. Finalize `research.md` with dataset stats, prompt experiments, DoRA metrics, retry policy summary.
2. Scaffold Hydra configs + CLI contracts; review with team.
3. Implement data/model/training code with unit + integration tests, including retry utility + dataset validation logging.
4. Execute full 5-fold run, inspect MLflow results (incl. dataset metrics + CPU latency), tune thresholds, register best model.
5. Polish documentation (quickstart, README, contracts) and define backlog (batch inference API, monitoring, latency alerting).
