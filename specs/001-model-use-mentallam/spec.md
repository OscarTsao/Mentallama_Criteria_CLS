# Feature Specification: Mentallam Binary Classifier with 5-Fold CV

**Feature Branch**: `001-model-use-mentallam`  
**Created**: 2025-11-12  
**Status**: Draft  
**Input**: User description: "the model should use mentallam from huggingface and the classification head provided by huggingface and run five fold training"

## Clarifications Resolved

1) Model ID: `klyang/MentaLLaMA-chat-7B`.

2) Input formatting: single prompt string
   `"post: {post}, criterion: {criterion} Does the post match the criterion description? Output yes or no"`.
   Tokenize this string with max_length=512, truncation=`longest_first`, padding to max length.

3) Cross‑validation: 5 folds, stratified by label and grouped so criteria from the same post stay in
   the same fold (use StratifiedGroupKFold with `groups=post_id`, shuffle=True, random_state=42).

4) Optimization: select best by validation F1; decision threshold tuned on validation to maximize F1
   (log tuned threshold per fold and at aggregate).

5) Training: up to 100 epochs with early stopping (patience=20). Use DoRA for parameter‑efficient
   training and enable AMP with bfloat16 if supported.

## Additional Clarifications Resolved

1) Classification head strategy for LLaMA chat model:
   Use `LlamaForSequenceClassification.from_pretrained("klyang/MentaLLaMA-chat-7B", num_labels=2)`
   with a freshly initialized head and fine‑tune via DoRA (accepted).

2) Dataset schema and location:
   Directories confirmed: `data/redsm5` (posts and groundtruth) and `data/dsm5` (criteria).
   Dataset size: 1484 posts × 9 DSM-5 criteria = 13,356 total samples.
   Expected schemas (to validate at implementation time):
   - Posts: `post_id, post` (1484 posts)
   - Groundtruth: `post_id, criterion_id, label` (yes/no → 1/0, 13,356 pairs)
   - Criteria: `criterion_id, criterion` (9 criteria)

3) DoRA configuration:
   Accepted defaults: `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`,
   `r=8`, `alpha=16`, `lora_dropout=0.05` (DoRA enabled).

4) Threshold tuning method:
   Accepted: sweep 0.00–1.00 (step 0.01) per fold to maximize F1; log tuned threshold and use for
   reporting.

5) Compute environment constraints:
   Accepted: enable gradient checkpointing with dynamic batch sizing (start batch_size=8, reduce on OOM);
   gradient accumulation adjusted to maintain effective batch of 32; use bf16 AMP when available.
   Consider smaller model for smoke tests if GPU memory is constrained.

## Clarifications

### Session 2025-11-12

- Q: Dependency version constraints? → A: Pin major versions, allow minor updates (e.g., transformers>=4.40,<5.0)
- Q: Training batch size strategy? → A: Dynamic batch size with gradient accumulation (start 8, reduce to 4/2/1 if OOM, adjust grad_accum to maintain effective batch=32)
- Q: Error handling for external dependencies? → A: Retry with exponential backoff (max 3 attempts), then fail gracefully with actionable error message
- Q: Expected dataset size range? → A: 1484 posts × 9 criteria = 13,356 samples
- Q: Metrics to log beyond accuracy and F1? → A: Precision, recall, F1, accuracy, ROC-AUC (comprehensive binary classification metrics)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train mentallam with 5-fold CV (Priority: P1)

As a researcher, I can train a binary classifier using the Hugging Face "mentallam" model with the
standard classification head to predict matched vs. unmatched for (criterion, post) pairs, running
5-fold cross-validation and logging metrics and artifacts.

**Why this priority**: Produces the core model and validates generalization across folds.

**Independent Test**: Run a single command to execute all 5 folds; each fold logs to MLflow with
params, metrics, artifacts, and resolved Hydra config.

**Acceptance Scenarios**:

1. Given a dataset of (criterion, post, label) and a fixed seed, when I run the training command
   with `cv.folds=5`, then MLflow records 5 child runs with per-fold metrics and saved checkpoints.
2. Given Hydra defaults in `configs/`, when I override `model.name=klyang/MentaLLaMA-chat-7B`, training resolves the
   config and logs it as an artifact.

---

### User Story 2 - Aggregate and report CV results (Priority: P2)

As a researcher, I can aggregate per-fold metrics (precision, recall, F1, accuracy, ROC-AUC) and log the
summary to MLflow, including mean, std, and best fold.

**Why this priority**: Enables model selection and reporting.

**Independent Test**: A single script computes aggregated metrics from MLflow child runs and logs a
parent summary run.

**Acceptance Scenarios**:

1. Given 5 completed fold runs, when I invoke the aggregation step, then the parent run has
   `metric_mean_*` and `metric_std_*` values and a link to each fold.

---

### User Story 3 - Inference for a single pair (Priority: P3)

As a user, I can run inference on a single (criterion, post) pair using a selected fold checkpoint or
the best model and receive a matched/unmatched label with score.

**Why this priority**: Validates end-to-end usability beyond training.

**Independent Test**: A CLI call performs tokenization, model forward, and outputs label + score.

**Acceptance Scenarios**:

1. Given a saved model, when I run the CLI with `criterion` and `post`, then it returns a label and
   probability and logs the inference config.

---

### Edge Cases

- Model ID fixed: `klyang/MentaLLaMA-chat-7B`.
- Long inputs: sequences exceeding max length require truncation/strategy (head-tail or sliding).
- Class imbalance: handle with weighted loss or sampling.
- Missing values: empty criterion/post should be filtered or flagged.
- Network failures: retry HuggingFace downloads and MLflow connections with exponential backoff (max 3 attempts), then fail with clear error message.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Use Hugging Face model `klyang/MentaLLaMA-chat-7B` with the built‑in classification
  head (`AutoModelForSequenceClassification`).
- **FR-002**: Build a single prompt string as
  `post: {post}, criterion: {criterion} Does the post match the criterion description? Output yes or no`,
  tokenize with `max_length=512`, `truncation=longest_first`, and pad to max.
- **FR-003**: Implement 5‑fold cross‑validation with StratifiedGroupKFold
  (`n_splits=5`, `shuffle=True`, `random_state=42`, `groups=post_id`).
- **FR-004**: Manage parameters via Hydra; defaults in `configs/`; allow CLI overrides. Persist the
  resolved config per run.
- **FR-005**: Optimize and select best by validation F1. Tune the decision threshold on validation to
  maximize F1; log the tuned threshold and use it for reporting.
- **FR-006**: Use MLflow with `sqlite:///mlflow.db` and `mlruns/` for artifacts; log params, metrics
  (precision, recall, F1, accuracy, ROC-AUC per fold and aggregated), resolved Hydra config, tuned
  threshold, and checkpoints per fold.
- **FR-007**: Training: up to 100 epochs with early stopping `patience=20`; use DoRA PEFT and AMP
  (bf16) when supported. Use dynamic batch sizing (start batch_size=8, reduce to 4/2/1 on OOM) with
  gradient accumulation adjusted to maintain effective batch size of 32.
- **FR-008**: Provide a CLI to run training (`cv.folds=5`) and inference for a single pair.
- **FR-009**: Set seeds via `utils.set_seed`; ensure reproducible splits and log env details.
- **FR-010**: Pin major versions of critical dependencies (transformers>=4.40,<5.0, torch>=2.2,<3.0) in
  `pyproject.toml` to allow minor updates and security patches while preventing breaking changes.
- **FR-011**: Implement retry logic with exponential backoff (max 3 attempts) for external dependencies
  (HuggingFace model downloads, MLflow connections). On persistent failure, exit gracefully with
  actionable error messages indicating the failure cause and remediation steps.
- **FR-012**: Validate dataset size on load: expect 1484 posts, 9 criteria, and 13,356 (post, criterion)
  pairs. Warn if counts differ and log actual dataset statistics to MLflow.

### Key Entities *(include if feature involves data)*

- **Sample**: {id, criterion_text, post_text, label} — 13,356 total samples (1484 posts × 9 criteria)
- **Post**: {post_id, post} — 1484 unique posts
- **Criterion**: {criterion_id, criterion} — 9 DSM-5 MDD criteria
- **CVSplit**: {fold_index, train_indices, val_indices, seed} — ~2,671 samples per fold
- **RunMetadata**: {fold_index, model_name, metrics, checkpoint_path}

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Complete 5-fold CV with each fold logging precision, recall, F1, accuracy, and ROC-AUC;
  aggregate mean and std for each metric reported in parent run.
- **SC-002**: All runs record resolved Hydra config and checkpoints in MLflow.
- **SC-003**: Re-run with same seed reproduces metrics within tolerance (<0.5% abs F1 change).
- **SC-004**: Inference CLI returns label+score under 1s on CPU for typical inputs.
