# Project Specification: Mentallama_Criteria_CLS

## Overview

**Project Name**: Mentallama_Criteria_CLS
**Purpose**: Binary classification for DSM-5 Major Depressive Disorder (MDD) criteria detection
**Dataset**: ReDSM5 - Reddit posts annotated for depression symptoms
**Task Type**: Post-level (post, criterion) pair classification with explainability

---

## Clarifications

### Session 2025-11-12

- Q: Which base model should be used as the primary model for initial experiments? → A: mental/mental-bert-base-uncased
- Q: Which pooling strategy should be used to aggregate token representations? → A: last hidden state token (last_hidden_state[:, 0, :])
- Q: Which class weighting strategy should be used to handle class imbalance? → A: Inverse frequency weighting (pos_weight = neg_count / pos_count per class)
- Q: Which parameter-efficient fine-tuning strategy should be used? → A: LoRA with r=8, alpha=16
- Q: What is the prediction granularity and input format? → A: Post-level predictions with input format (post, criterion); groundtruth in status column

---

## 1. Problem Statement

Develop an ML system to automatically detect the presence or absence of DSM-5 Major Depressive Disorder criteria in social media text (Reddit posts). The system should:

- Classify (post, criterion) pairs at the post level for 9 DSM-5 criteria + 1 special case
- Input format: (post_text, criterion_description)
- Output: Binary prediction (status: 1 = criterion present, 0 = criterion absent)
- Provide explainable predictions with clinical rationale
- Handle class imbalance and hard negative examples

---

## 2. Data Specification

### 2.1 Dataset: ReDSM5

**Source**: Reddit posts annotated by licensed psychologists
**Size**: 1,484 posts with 1,547 expert annotations
**Access**: Gated dataset requiring user agreement (https://www.irlab.org/ReDSM5_agreement.odt)
**Public Sample**: 25 paraphrased entries available at `irlab-udc/redsm5-sample`

**Files**:
- `redsm5_posts.csv`: Contains `post_id` and full post `text`
- `data/DSM5/MDD_Criteira.json`: Contains criterion definitions with `criterion_id` and `criterion` text
- `groundtruth.csv`: Contains post-level annotations with columns:
  - `post_id`: Unique post identifier
  - `criterion_id`: DSM-5 criterion identifier (A.1 through A.9 + SPECIAL_CASE)
  - `status`: Binary label (1 = criterion present in post, 0 = criterion absent)

**Input Format**: Each training sample is a (post, criterion) pair:
- `post`: Full Reddit post text
- `criterion`: DSM-5 criterion description text
- `label`: Binary status (0 or 1)

### 2.2 Target Labels (10 Classes)

Based on DSM-5 criteria for Major Depressive Episode (Criterion A):

| ID | Symptom | DSM-5 Code | Training Examples | Description |
|----|---------|------------|-------------------|-------------|
| 1 | DEPRESSED_MOOD | A.1 | 328 | Depressed mood most of the day, nearly every day |
| 2 | ANHEDONIA | A.2 | 124 | Diminished interest/pleasure in activities |
| 3 | APPETITE_CHANGE | A.3 | 44 | Significant weight/appetite change |
| 4 | SLEEP_ISSUES | A.4 | 102 | Insomnia or hypersomnia nearly every day |
| 5 | PSYCHOMOTOR | A.5 | 35 | Psychomotor agitation or retardation |
| 6 | FATIGUE | A.6 | 124 | Fatigue or loss of energy nearly every day |
| 7 | WORTHLESSNESS | A.7 | 311 | Feelings of worthlessness or guilt |
| 8 | COGNITIVE_ISSUES | A.8 | 59 | Diminished ability to think/concentrate |
| 9 | SUICIDAL_THOUGHTS | A.9 | 165 | Recurrent thoughts of death/suicide |
| 10 | SPECIAL_CASE | N/A | 92 | Non-DSM-5 clinical discriminations |

**Class Imbalance Notes**:
- PSYCHOMOTOR is the rarest symptom (35 examples)
- WORTHLESSNESS and DEPRESSED_MOOD are most common
- 392 posts are hard negatives (no symptoms present)

### 2.3 Dataset Statistics

- **Total posts**: 1,484
- **Average post length**: 294.7 words (range: 2-6,990)
- **Average symptoms per post**: 1.39 (for status=1)
- **Average annotations per post**: 1.04
- **Hard negatives**: 392 posts

---

## 3. Model Architecture

### 3.1 Core Components

```
Input: Tokenized (post, criterion) pair
  Format: [CLS] post_text [SEP] criterion_text [SEP]
  ↓
Transformer Encoder (AutoModel)
  ↓
Pooled Output: [CLS] token from last hidden state (last_hidden_state[:, 0, :])
  ↓
Classification Head (Linear layer)
  ↓
Output: Binary logit (criterion present/absent)
```

**Implementation**: `src/Project/SubProject/models/model.py:11-21`

**Input Format**: Concatenate post and criterion with special tokens for pair classification
**Pooling Strategy**: Use the [CLS] token representation from the last hidden state (`last_hidden_state[:, 0, :]`) as the pair-level embedding for binary classification.

### 3.2 Model Variants to Explore

1. **Primary Base Model**:
   - `mental/mental-bert-base-uncased` (clinical mental health domain) - **Selected for initial experiments**

2. **Alternative Base Models** (for comparison/benchmarking):
   - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (biomedical literature)
   - `bert-base-uncased` (general domain baseline)
   - `roberta-base` (general domain, better architecture)
   - `mental/mental-roberta-base` (clinical RoBERTa variant)

3. **Parameter-Efficient Fine-Tuning**:
   - **LoRA (Low-Rank Adaptation)** - **Selected strategy**
     - Configuration: r=8, alpha=16, dropout=0.1
     - Target modules: query, key, value projection layers
     - Reduces trainable parameters by ~99% while maintaining performance

### 3.3 Classification Head

- **Binary classification head**: `Linear(hidden_dim, 1)` with sigmoid activation
- Output: Single logit indicating criterion presence probability
- Training: Binary Cross Entropy Loss with inverse frequency weighting

---

## 4. Training Specification

### 4.1 Loss Function

**Binary Cross Entropy with Inverse Frequency Weighting**:
```python
# Calculate inverse frequency weight for positive class
# pos_weight = (n_negative_samples) / (n_positive_samples)
n_pos = (labels == 1).sum()
n_neg = (labels == 0).sum()
pos_weight = torch.tensor([n_neg / n_pos])
loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Rationale**: Inverse frequency weighting automatically adjusts for class imbalance in (post, criterion) pairs (e.g., rare criteria like PSYCHOMOTOR: 35 positive examples vs common criteria like WORTHLESSNESS: 311 positive examples). When treating each (post, criterion) as a separate sample, this helps the model learn from underrepresented criteria.

### 4.2 Optimization Strategy

**Optimizer**: Fused AdamW or 8-bit AdamW (bitsandbytes)
- Learning rate: 1e-5 to 5e-5 (tune with Optuna)
- Weight decay: 0.01
- Warmup steps: 10% of total steps
- LR scheduler: Linear decay with warmup

**Mixed Precision Training**:
- bf16 (preferred if GPU supports) or fp16 with GradScaler
- TF32 enabled for Ampere+ GPUs

**Memory Optimizations**:
- Gradient checkpointing (saves ~30% VRAM)
- Gradient accumulation steps: 2-4
- Max sequence length: 512 tokens

**LoRA Configuration**:
- Rank (r): 8
- Alpha: 16
- Dropout: 0.1
- Target modules: ["query", "key", "value"] (attention projection layers)
- Trainable parameters: ~1% of total model parameters

**Attention Implementation**:
- SDPA (Scaled Dot-Product Attention) with Flash backend
- Or FlashAttention-2 if installed

### 4.3 Training Parameters

```python
batch_size = 8-16 (per device)
gradient_accumulation_steps = 2-4
learning_rate = 1e-5 to 5e-5
num_epochs = 3-10
max_seq_length = 512
warmup_ratio = 0.1
weight_decay = 0.01
max_grad_norm = 1.0
```

### 4.4 Data Handling

- **Input tokenization**: Tokenize (post, criterion) pairs as `[CLS] post [SEP] criterion [SEP]`
- **Max sequence length**: 512 tokens (truncate long posts if needed)
- **Padding**: Pad to max_length for batch processing
- **Class imbalance**: Handled via inverse frequency weighting in loss function
- **Hard negatives**: Include (post, criterion) pairs with status=0 (criterion absent)
- **Data augmentation**: Each post paired with all 10 criteria creates multiple training samples

---

## 5. Evaluation Metrics

### 5.1 Binary Classification Metrics (Overall)

For all (post, criterion) pairs:
- **Accuracy**: Fraction of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve

### 5.2 Per-Criterion Metrics

For each of 10 criteria (group by criterion_id):
- Criterion-specific Precision, Recall, F1-score
- Support (number of positive examples for that criterion)
- Helps identify which criteria are harder to detect

### 5.3 Aggregate Metrics

- **Macro F1**: Average F1 across all 10 criteria (equal weight)
- **Micro F1**: Global F1 across all (post, criterion) pairs
- **Weighted F1**: F1 weighted by criterion support

### 5.4 Clinical Metrics

- **Per-criterion confusion matrices**: Identify false positives/negatives per criterion
- **False positive rate**: Critical for clinical safety (avoid false alarms)
- **False negative rate**: Critical for SUICIDAL_THOUGHTS (must not miss)
- **Hard negative accuracy**: Performance on (post, criterion) pairs where status=0

### 5.5 Validation Strategy

- **5-fold cross-validation** with stratified splits
- **Stratification**: By criterion_id and status to ensure balanced representation
- **Grouping**: Keep all (post, criterion) pairs for same post_id in same fold
- **Hold-out test set**: 15-20% of posts (all their criteria pairs)
- **Validation set**: 10-15% for hyperparameter tuning

---

## 6. Experiment Tracking

### 6.1 MLflow Integration

**Tracking URI**: `file:./mlruns` (local) or remote server
**Experiment Name**: `dsm5_classification`

**Logged Parameters**:
- model_name (e.g., "mental-bert-base")
- learning_rate
- batch_size
- num_epochs
- max_seq_length
- optimizer_choice
- use_lora
- gradient_checkpointing

**Logged Metrics**:
- train_loss, val_loss per epoch
- Per-symptom F1, Precision, Recall
- Macro/Micro/Weighted F1
- Hard negative accuracy

**Logged Artifacts**:
- Best model checkpoint
- Confusion matrices (per symptom)
- Training curves (loss, metrics)
- Hyperparameter config

### 6.2 Optuna Hyperparameter Optimization

**Objective**: Maximize macro F1 on validation set

**Search Space**:
```python
learning_rate: [1e-6, 5e-5] (log scale)
batch_size: [8, 16, 32]
num_epochs: [3, 10]
weight_decay: [0.0, 0.1]
warmup_ratio: [0.0, 0.2]
dropout_prob: [0.1, 0.3]
use_class_weights: [True, False]
```

**Optimization**:
- Algorithm: TPE (Tree-structured Parzen Estimator)
- Trials: 50-100
- Pruning: MedianPruner (stop bad trials early)
- Storage: `sqlite:///optuna.db`

---

## 7. Performance Optimization

### 7.1 GPU Acceleration (per Optimization_List)

| Feature | Benefit | Implementation |
|---------|---------|----------------|
| bf16 AMP | Fast & stable mixed precision | `torch.cuda.amp.autocast(dtype=torch.bfloat16)` |
| TF32 | 2× faster FP32 matmuls | `torch.backends.cuda.matmul.allow_tf32 = True` |
| SDPA Flash | Memory-efficient attention | `model.config.attn_implementation = "sdpa"` |
| Gradient Checkpointing | 30% VRAM savings | `model.gradient_checkpointing_enable()` |
| Fused AdamW | Faster optimizer | `optimizer = "adamw_torch_fused"` |
| torch.compile | 5-25% speedup | `model = torch.compile(model)` |
| Sequence Packing | Higher throughput | Custom collator |

### 7.2 Data Pipeline Optimization

- **Pinned memory**: `DataLoader(pin_memory=True)`
- **Multiple workers**: `num_workers=4-8`
- **Persistent workers**: `persistent_workers=True`
- **Pre-tokenization**: Cache tokenized data to disk
- **Length bucketing**: Reduce padding waste

---

## 8. Implementation Roadmap

### Phase 1: Data Pipeline ✅ To Implement
- [ ] Load ReDSM5 CSV files
- [ ] Implement Dataset class (`dataset.py`)
- [ ] Tokenization with truncation/padding
- [ ] Multi-label encoding
- [ ] Sequence packing collator
- [ ] Train/val/test split

### Phase 2: Model & Training ✅ To Implement
- [x] Basic model architecture (done in `model.py`)
- [ ] LoRA integration (r=8, alpha=16, target: q/k/v projections)
- [ ] Multi-label loss with inverse frequency weighting
- [ ] Training loop (`train_engine.py`)
- [ ] Evaluation loop (`eval_engine.py`)
- [ ] Gradient checkpointing
- [ ] Mixed precision training (bf16)

### Phase 3: Optimization
- [ ] MLflow experiment tracking
- [ ] Optuna hyperparameter tuning
- [ ] Hard negative mining

### Phase 4: Evaluation & Analysis
- [ ] Per-symptom metrics
- [ ] Confusion matrices
- [ ] Error analysis
- [ ] Clinical rationale comparison
- [ ] Model explainability (attention visualization)

### Phase 5: Production
- [ ] Model serialization & loading
- [ ] Inference pipeline
- [ ] API endpoint (optional)
- [ ] Clinical safety checks
- [ ] Documentation & paper

---

## 9. Key Technical Decisions

### 9.1 Resolved

1. **Task type**: Binary classification for (post, criterion) pairs
2. **Granularity**: Post-level with (post, criterion) as input; output is status (0/1)
3. **Input format**: [CLS] post_text [SEP] criterion_text [SEP]
4. **Model base**: Transformer encoders (BERT family)
5. **Framework**: PyTorch + Hugging Face Transformers
6. **Tracking**: MLflow for experiments, Optuna for HPO
7. **Primary base model**: mental/mental-bert-base-uncased (clinical mental health domain)
8. **Pooling strategy**: [CLS] token from last hidden state (last_hidden_state[:, 0, :])
9. **Class weighting**: Inverse frequency weighting (pos_weight = neg_count / pos_count)
10. **Fine-tuning strategy**: LoRA with r=8, alpha=16, dropout=0.1
11. **Output**: Single binary logit per (post, criterion) pair

### 9.2 To Decide

1. **Sequence length**: 512 tokens sufficient or need 1024 for long posts?
2. **Truncation strategy**: Head-only, tail-only, or head+tail for posts exceeding 512 tokens?
3. **Threshold tuning**: Use 0.5 or tune threshold on validation set for optimal F1?

---

## 10. Clinical & Ethical Considerations

### 10.1 Safety Requirements

- **No diagnostic claims**: Model outputs are for research, not clinical diagnosis
- **False negative awareness**: Missing suicidal thoughts is critical
- **Privacy**: Reddit usernames removed, posts anonymized
- **Bias monitoring**: Check for demographic/linguistic biases

### 10.2 Evaluation Standards

- **Psychologist agreement**: Compare model predictions to expert rationales
- **Clinical validity**: Predictions should align with DSM-5 definitions
- **Interpretability**: Attention weights and saliency maps for transparency

### 10.3 Dataset Licensing

- ReDSM5 is Apache 2.0 licensed
- Requires user agreement for full dataset access
- Public sample available for prototyping

---

## 11. References

**Dataset Paper**:
```
Eliseo Bao, Anxo Pérez, Javier Parapar (2025)
ReDSM5: A Reddit Dataset for DSM-5 Depression Detection
Accepted at CIKM 2025
arXiv:2508.03399
```

**Dataset Location**:
- Full dataset: `data/redsm5/` (after access approval)
- Criteria reference: `data/data/DSM5/MDD_Criteira.json`

**Key Files**:
- Model: `src/Project/SubProject/models/model.py`
- Dataset: `src/Project/SubProject/data/dataset.py`
- Training: `src/Project/SubProject/engine/train_engine.py`
- Evaluation: `src/Project/SubProject/engine/eval_engine.py`
- Optimization guide: `Optimization_List`, `Optimization_Examples`

---

## 12. Success Criteria

### 12.1 Minimum Viable Product (MVP)

- ✅ Load and preprocess ReDSM5 data
- ✅ Train a baseline BERT model
- ✅ Achieve >0.60 macro F1 on validation set
- ✅ Log experiments to MLflow
- ✅ Generate per-symptom evaluation report

### 12.2 Production-Ready System

- ✅ Macro F1 >0.75 on test set
- ✅ Per-symptom F1 >0.60 for all symptoms
- ✅ False negative rate <10% for SUICIDAL_THOUGHTS
- ✅ Optimized inference (<100ms per sentence)
- ✅ Explainable predictions with attention visualization

### 12.3 Research Contributions

- ✅ Benchmark multiple clinical BERT variants
- ✅ Ablation study on optimization techniques
- ✅ Error analysis on hard cases
- ✅ Comparison with human psychologist agreement
- ✅ Publication-ready results and reproducible code

---

## Appendix A: DSM-5 Criteria Reference

Full criteria stored in: `data/data/DSM5/MDD_Criteira.json`

**Criterion A**: Five (or more) of the following symptoms during the same 2-week period, representing a change from previous functioning; at least one symptom is either (1) depressed mood or (2) loss of interest or pleasure.

1. **A.1 (DEPRESSED_MOOD)**: Depressed mood most of the day, nearly every day
2. **A.2 (ANHEDONIA)**: Markedly diminished interest/pleasure in activities
3. **A.3 (APPETITE_CHANGE)**: Significant weight/appetite change
4. **A.4 (SLEEP_ISSUES)**: Insomnia or hypersomnia nearly every day
5. **A.5 (PSYCHOMOTOR)**: Psychomotor agitation or retardation
6. **A.6 (FATIGUE)**: Fatigue or loss of energy nearly every day
7. **A.7 (WORTHLESSNESS)**: Feelings of worthlessness or inappropriate guilt
8. **A.8 (COGNITIVE_ISSUES)**: Diminished ability to think or concentrate
9. **A.9 (SUICIDAL_THOUGHTS)**: Recurrent thoughts of death or suicide

---

**Document Version**: 1.0
**Last Updated**: 2025-11-12
**Status**: Complete specification for implementation
