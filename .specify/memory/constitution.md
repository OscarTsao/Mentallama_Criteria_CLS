<!--
Sync Impact Report
- Version change: none → 1.0.0
- Modified principles: n/a (initial adoption)
- Added sections: Core Principles; Security & Data Management; Development Workflow & Quality Gates
- Templates requiring updates:
  ✅ .specify/templates/plan-template.md (aligned, no changes required)
  ✅ .specify/templates/spec-template.md (aligned, no changes required)
  ✅ .specify/templates/tasks-template.md (aligned, no changes required)
- Follow-up TODOs: none
-->

# Mentallama Criteria CLS Constitution

## Core Principles

### I. Model Architecture & Task (NON-NEGOTIABLE)
The system uses a Hugging Face Transformer encoder with a lightweight classification head to
perform binary classification: matched vs. unmatched. Inputs are a criterion description and a
post; the encoder produces a hidden representation (CLS/pooler), which feeds a linear classifier
to produce logits for {matched, unmatched}. All modeling code resides under
`src/Project/SubProject/models/`.

### II. Configuration via Hydra (NON-NEGOTIABLE)
All runtime parameters (model name, tokenizer, training hyperparameters, data paths, seeds,
MLflow settings) are managed with Hydra. Default YAMLs live in `configs/` with hierarchical
composition. Runs MUST be reproducible via a single CLI invocation with explicit overrides; the
resolved config MUST be logged with each run.

### III. Experiment Tracking with MLflow (NON-NEGOTIABLE)
MLflow is the system of record for experiments and model registration. The tracking backend uses
SQLite at `sqlite:///mlflow.db`; the artifact store is the local directory `mlruns/`. Every run MUST
log parameters, metrics, artifacts, and the resolved Hydra config. Prefer the utilities in
`src/Project/SubProject/utils/mlflow_utils.py` and enable autologging where applicable.

### IV. Testing & Code Quality
Use pytest for tests, Ruff for linting, Black for formatting (line length 100), and type hints where
feasible. Add unit tests for utilities and model shapes (e.g., classifier output dims). Determinism
tests SHOULD seed via `utils.set_seed`. CI MUST fail on lints/formatting/test failures.

### V. Reproducibility, Logging, and Observability
All experiments MUST set seeds, log environment/package versions, and emit structured logs via
`utils.get_logger`. Key training/eval metrics MUST be logged to MLflow. Randomness controls and
hardware‑specific flags SHOULD be set to favor reproducible results when practical.

## Security & Data Management
Do not commit secrets or large datasets. Store data under `data/`; write outputs to `outputs/` or
`artifacts/`. Respect data access controls; strip PII unless explicitly approved. The local
`mlflow.db` is for development; multi‑user or production requires a managed MLflow server with
network storage. Use environment variables for sensitive config and allow Hydra to read them.

## Development Workflow & Quality Gates
Preferred flow: `/speckit.specify` → `/speckit.clarify` → `/speckit.plan` → `/speckit.tasks` →
`/speckit.analyze` → implementation. Source lives under `src/`, tests under `tests/`. Pull requests
MUST include passing tests (where applicable), lint/format clean, and updated docs when behavior
changes. MLflow runs MUST be reproducible from the PR description with a copy‑pasteable command.

## Governance
This constitution supersedes ad‑hoc practices. Amendments require a PR, a rationale, and a version
bump (SemVer). Reviews MUST verify compliance with non‑negotiable principles (Hydra config,
MLflow tracking/registry, binary classifier framing). Violations require explicit justification in a
Complexity Tracking table (see plan template) and SHOULD be temporary.

**Version**: 1.0.0 | **Ratified**: 2025-11-12 | **Last Amended**: 2025-11-12
