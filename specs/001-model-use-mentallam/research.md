# Research Notes – Mentallam Binary Classifier

## Data Source Audit
- **RedSM5 posts/groundtruth**: located under `data/redsm5`. Columns observed in previous experiments:
  `post_id`, `post`, `criterion_id`, `label`. Confirm encoding (UTF-8) and handle stray HTML/entities.
- **DSM5 criteria**: located under `data/dsm5`. Columns: `criterion_id`, `criterion` (text). Ensure
  join keys align exactly with groundtruth table.
- **Action**: create notebook/script that prints dataset counts, label distribution per criterion, and
  number of posts per criterion. Save summary stats (CSV/Markdown) into `specs/001-.../artifacts/`.

## Prompt Formatting Experiments
- **Decision**: Use template `post: {post}, criterion: {criterion} Does the post match the criterion description? Output yes or no`.
- **Experiment**: Evaluate reversed order or question-first phrasing on a 200-sample validation slice;
  capture accuracy/F1 deltas. If difference <1%, retain original template for consistency.

## StratifiedGroupKFold Justification
- **Need**: Multiple criteria can map to the same post; naive stratification leaks post context.
- **Plan**: Use scikit-learn `StratifiedGroupKFold` with `groups=post_id`, `n_splits=5`, `shuffle=True`, `random_state=42`.
- **Action**: Validate fold generation script ensures each group appears in exactly one fold; store JSON
  metadata (fold -> group list) under `artifacts/folds/`.

## Threshold Tuning Approach
- **Decision**: Use `precision_recall_curve` to derive recall/precision arrays, then compute F1 for
  thresholds spaced 0.00–1.00 (step 0.01). Select argmax F1; log threshold, F1, precision, recall.
- **Validation**: Compare with manual `torch.linspace` sweep to ensure identical outcomes; document
  runtime difference to justify final choice.

## DoRA & Memory Feasibility
- **Base config**: r=8, alpha=16, dropout=0.05, target modules `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`, grad checkpointing on, grad_accum=4, bf16.
- **Action**: Run a single forward/backward pass on available GPU (A100 80GB target). Capture VRAM peak
  via `torch.cuda.max_memory_allocated()`. If VRAM > 60GB, consider reducing r or using 4-bit quantized
  base weights.

## Tokenization Strategy
- **Decision point**: whether to cache tokenized tensors per fold or tokenize on-the-fly.
- **Plan**: Benchmark both approaches on a 5k sample subset. Cache if it yields ≥15% speedup with
  manageable disk footprint (<5GB/fold). Document final decision in this file.

## Outstanding Questions / TODOs
1. Confirm actual column names + file formats in `data/redsm5` and `data/dsm5`; update loader docstrings.
2. Determine whether noise/PII needs masking before logging samples (if yes, document policy).
3. Evaluate whether to store fold assignments + tuned thresholds in the repository or MLflow artifacts only.
