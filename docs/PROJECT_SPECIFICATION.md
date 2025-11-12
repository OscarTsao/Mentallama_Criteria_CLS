# Project Specification: Mentallama_Criteria_CLS

## Overview

**Project Name**: Mentallama_Criteria_CLS
**Purpose**: Multi-label text classification for DSM-5 Major Depressive Disorder (MDD) criteria detection
**Dataset**: ReDSM5 - Reddit posts annotated for depression symptoms
**Task Type**: Sentence-level clinical NLP with explainability

---

## 1. Problem Statement

Develop an ML system to automatically detect the presence or absence of DSM-5 Major Depressive Disorder symptoms in social media text (Reddit posts). The system should:

- Classify text at the sentence level for 9 DSM-5 symptoms + 1 special case
- Provide explainable predictions with clinical rationale
- Handle class imbalance and hard negative examples
- Support multi-label classification (posts can have 0-9 symptoms)

---

## 2. Data Specification

### 2.1 Dataset: ReDSM5

**Source**: Reddit posts annotated by licensed psychologists
**Size**: 1,484 posts with 1,547 expert annotations
**Access**: Gated dataset requiring user agreement (https://www.irlab.org/ReDSM5_agreement.odt)
**Public Sample**: 25 paraphrased entries available at `irlab-udc/redsm5-sample`

**Files**:
- `redsm5_posts.csv`: Contains `post_id` and full post `text`
- `redsm5_annotations.csv`: Contains sentence-level annotations with columns:
  - `post_id`: Unique post identifier
  - `sentence_id`: Unique sentence identifier
  - `sentence_text`: The annotated sentence
  - `DSM5_symptom`: The symptom category (see 2.2)
  - `status`: 1 (present) or 0 (absent)
  - `explanation`: Clinical rationale from psychologist

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
Input: Tokenized sentence text
  ↓
Transformer Encoder (AutoModel)
  ↓
Pooled Output ([CLS] token or mean pooling)
  ↓
Classification Head (Linear layer)
  ↓
Output: Logits for 10 classes
```

**Implementation**: `src/Project/SubProject/models/model.py:11-21`

### 3.2 Model Variants to Explore

1. **Base Models** (Hugging Face):
   - `mental/mental-bert-base-uncased` (clinical domain)
   - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
   - `bert-base-uncased` (baseline)
   - `roberta-base`
   - `mental/mental-roberta-base`

2. **Parameter-Efficient Fine-Tuning**:
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA for larger models)

### 3.3 Classification Head Options

- **Single linear layer** (current): `Linear(hidden_dim, 10)`
- **Multi-layer MLP**: Linear → ReLU → Dropout → Linear
- **Separate heads per symptom**: 10 binary classifiers

---

## 4. Training Specification

### 4.1 Loss Function

**Multi-label Binary Cross Entropy**:
```python
loss = torch.nn.BCEWithLogitsLoss()
# or with class weights for imbalance:
loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
```

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

- **Sequence packing**: Pack multiple short sentences into 512-token sequences
- **Length bucketing**: Group similar-length samples to reduce padding
- **Oversampling**: Upsample rare symptoms (PSYCHOMOTOR, APPETITE_CHANGE)
- **Hard negative mining**: Include explicit negatives in training

---

## 5. Evaluation Metrics

### 5.1 Per-Symptom Metrics

For each of 10 classes:
- Precision
- Recall
- F1-score
- Support (number of true positives)

### 5.2 Aggregate Metrics

- **Macro F1**: Average F1 across all classes (equal weight)
- **Micro F1**: Global F1 across all predictions
- **Weighted F1**: F1 weighted by class support
- **Hamming Loss**: Fraction of incorrect labels
- **Exact Match Ratio**: Percentage of exactly correct predictions

### 5.3 Clinical Metrics

- **Per-symptom confusion matrices**
- **False positive rate** (critical for clinical safety)
- **False negative rate** (missing symptoms)
- **Hard negative accuracy**: Performance on posts with no symptoms

### 5.4 Validation Strategy

- **K-fold cross-validation** (5-fold recommended)
- **Stratified split** by symptom presence
- **Hold-out test set**: 15-20% of data
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
- [ ] Multi-label loss implementation
- [ ] Training loop (`train_engine.py`)
- [ ] Evaluation loop (`eval_engine.py`)
- [ ] Gradient checkpointing
- [ ] Mixed precision training

### Phase 3: Optimization
- [ ] MLflow experiment tracking
- [ ] Optuna hyperparameter tuning
- [ ] Class weighting for imbalance
- [ ] Hard negative mining
- [ ] LoRA/QLoRA integration

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

1. **Task type**: Multi-label classification (not multi-class)
2. **Granularity**: Sentence-level (as per ReDSM5 annotations)
3. **Model base**: Transformer encoders (BERT family)
4. **Framework**: PyTorch + Hugging Face Transformers
5. **Tracking**: MLflow for experiments, Optuna for HPO

### 9.2 To Decide

1. **Pooling strategy**: [CLS] token vs mean pooling vs max pooling?
2. **Sequence length**: 512 tokens sufficient or need 1024?
3. **Class weighting**: Inverse frequency vs manual tuning?
4. **Hard negative handling**: Separate class or part of multi-label?
5. **Base model selection**: Mental-BERT vs PubMedBERT vs standard BERT?
6. **LoRA usage**: Full fine-tuning vs LoRA vs QLoRA?
7. **Post-level vs sentence-level**: Aggregate sentence predictions to post?

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
