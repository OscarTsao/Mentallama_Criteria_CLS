# Tasks: Mentallam Binary Classifier

**Input**: spec.md, plan.md, research.md, data-model.md, quickstart.md
**Prerequisites**: Hydra config scaffold, MLflow environment variables, GPU access for smoke tests

## Phase 1: Setup & Shared Infrastructure
- [ ] T101 [P] [INF] Create Hydra config tree under `configs/` (base config.yaml + data/model/training/logging/inference nodes + overrides/debug). Add schema validation via dataclasses and ensure defaults list matches plan.
- [ ] T102 [INF] Extend `pyproject.toml` dependencies (peft, accelerate, bitsandbytes, hydra-core, mlflow, scikit-learn); update README quickstart.
- [ ] T103 [P] [INF] Add MLflow helper wrapper in `src/Project/SubProject/utils/mlflow_utils.py` (parent/child run context managers, nested logging, artifact helpers).
- [ ] T104 [INF] Wire `utils/set_seed` + `utils/get_logger` usage patterns into new engine modules; add docstrings/tests if missing.
- [ ] T105 [DOC] Produce CLI contract (`contracts/cli.md`) detailing Hydra commands (train/aggregate/infer), required overrides, env vars, ownership notes, and reference it from README/quickstart.
- [ ] T106 [INF] Implement shared retry/backoff utility (`src/infra/retries.py`) with Hydra-configured base/cap/max_attempts, exponential backoff + jitter, HTTP status allowlist. Add pytest coverage simulating transient HF/MLflow failures and ensuring non-retryable errors propagate.

- [ ] T201 [INF] Implement dataset loader (`data/dataset.py`) merging `data/redsm5` + `data/dsm5`, performing schema validation, label mapping, text cleaning, and computing cardinalities (posts, criteria, pairs). Fail fast when counts ≠ 1484/9/13356 unless override tag provided; log stats/manifest/metrics to MLflow.
- [ ] T202 [INF] Build StratifiedGroupKFold helper (`data/splits.py`) that persists `artifacts/folds/fold_{i}.json`, records group lists + hashes, stores dataset hash/override reason, and tags MLflow runs with dataset metadata.
- [ ] T203 [INF] Create prompt builder module (`models/prompt_builder.py`) with unit tests covering truncation, empty strings, unicode normalization.
- [ ] T204 [INF] Replace `models/model.py` with `MentallamClassifier` (loads `LlamaForSequenceClassification`, applies DoRA config, enables gradient checkpointing, exposes `forward` + `predict_proba`).
- [ ] T205 [INF] Add metrics & threshold utilities (`engine/metrics.py`) computing accuracy/precision/recall/F1, confusion matrix, PR curves, threshold sweeps.
- [ ] T206 [INF] Write pytest unit tests for loader, splitter, prompt builder, metrics module, and retry utility.

**Checkpoint**: Data/model/metrics infrastructure ready; tests pass.

## Phase 3: User Story 1 – 5-fold Training (P1)
- [ ] T301 [US1] Implement Hydra CLI `Project.SubProject.engine.train_engine` orchestrating parent MLflow run, fold loop, and config logging.
- [ ] T302 [US1] Implement fold runner: dataset/dataloader construction, accelerate Trainer/custom loop, DoRA optimizer, early stopping (patience=20).
- [ ] T303 [US1] Integrate evaluation + threshold sweep per fold; log metrics, curves, tuned threshold, confusion matrix, and checkpoint path.
- [ ] T304 [US1] Persist fold metadata + Hydra config snapshot to artifacts + MLflow; ensure graceful shutdown handling (KeyboardInterrupt).
- [ ] T305 [US1] Add integration smoke test (`tests/integration/test_train_smoke.py`) running 2-fold tiny dataset.

## Phase 4: User Story 2 – Aggregation & Reporting (P2)
- [ ] T401 [US2] Implement `aggregate` subcommand in `eval_engine.py`: fetch child runs via MLflow API, compute mean/std metrics, best fold, tune summaries.
- [ ] T402 [US2] Persist summary JSON/Markdown + thresholds file (`artifacts/thresholds.json`), attach to parent MLflow run.
- [ ] T403 [US2] Write unit tests mocking MLflow responses to verify aggregation math and artifact structure.

## Phase 5: User Story 3 – Inference CLI (P3)
- [ ] T501 [US3] Implement `infer` subcommand supporting single pair + batch JSONL input, tuned threshold usage, optional MLflow logging.
- [ ] T502 [US3] Provide CLI flags for checkpoint path or MLflow run reference; ensure reproducible seeding and logging of inference metadata.
- [ ] T503 [US3] Build CPU latency bench harness (≤256-token inputs) invoked via CLI/CI to compute latency_ms_p50/p95 with fail-fast if p95>1000 ms (unless override tag set). Log metrics + histogram to MLflow.
- [ ] T504 [US3] Add integration test for inference CLI covering single + batch predictions and bench harness wiring (mock timers for determinism).
- [ ] T505 [US3] Update quickstart/README with inference instructions, latency expectations, and how to interpret MLflow latency metrics.

## Phase 6: Polish & Cross-Cutting
- [ ] T601 [POL] Document Hydra overrides, MLflow workflow, and troubleshooting in `docs/` (link quickstart, add FAQ for OOM + sqlite locks).
- [ ] T602 [POL] Add lint/test automation (Makefile shortcut or GitHub Action) running ruff, black, pytest.
- [ ] T603 [POL] Validate quickstart commands end-to-end on reduced dataset; capture MLflow screenshots or logs for future reference.
- [ ] T604 [POL] Register best checkpoint in MLflow Model Registry (`mentallama-criteria-cls`) and record version in README + release notes.
