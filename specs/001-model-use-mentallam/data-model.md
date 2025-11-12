# Data Model – Mentallam Binary Classifier

## Entities

### Sample
- **Fields**: `sample_id` (uuid), `post_id`, `criterion_id`, `post_text`, `criterion_text`, `label`
  (`0=unmatched`, `1=matched`).
- **Rules**: Trim whitespace, normalize unicode, drop if either text empty. Map labels from `"yes"/"no"`.

### FoldAssignment
- **Fields**: `fold_index`, `train_indices`, `val_indices`, `groups` (list of `post_id`), `seed`,
  `tuned_threshold` (float), `label_distribution` (dict).
- **Rules**: Each `sample_id` appears exactly once across folds; groups must not overlap.

### RunArtifact
- **Fields**: `mlflow_run_id`, `fold_index`, `checkpoint_path`, `metrics` (dict), `threshold`,
  `confusion_matrix_path`, `pr_curve_path`, `hydra_config_path`.
- **Rules**: Artifact paths stored relative to repo root; metrics include accuracy, precision, recall,
  F1 (macro).

### HydraConfigSnapshot
- **Fields**: `config_name`, `yaml_blob`, `git_commit`, `timestamp`, `hydra_version`.
- **Purpose**: Persisted as MLflow artifact for auditing + reproducibility.

### ThresholdSummary
- **Fields**: `fold_index`, `threshold`, `f1`, `precision`, `recall`, `support`.
- **Usage**: Aggregation step produces list; stored as JSON for inference CLI.

## Relationships
- `Sample` -> aggregated tables (`data/redsm5`, `data/dsm5`).
- `FoldAssignment` references Sample indices and `post_id` groups.
- `RunArtifact` references FoldAssignment via `fold_index`; inference CLI loads by `mlflow_run_id`.
- `ThresholdSummary` aggregates across RunArtifacts.

## Storage Layout
- Raw data: `data/redsm5/*.csv`, `data/dsm5/*.csv`.
- Processed caches: optional `artifacts/cache/tokenized_fold_{i}.pt`.
- Fold metadata: `artifacts/folds/fold_{i}.json` (contains FoldAssignment + summary stats).
- MLflow artifacts: `mlruns/<experiment>/<run_id>/artifacts/` storing configs, checkpoints,
  metrics, curves.
- Threshold summaries: `artifacts/thresholds.json` and MLflow parent run artifact.

## Validation & Constraints
1. `label` must be in {0,1}; raise error otherwise.
2. `criterion_id` present in DSM5 lookup; `post_id` present in RedSM5 posts.
3. FoldAssignment JSON includes checksum/hash of source CSVs to detect drift.
4. RunArtifact `checkpoint_path` must exist before logging; add assertion.
5. HydraConfigSnapshot `git_commit` matches current repo state recorded in MLflow tags.
