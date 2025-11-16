# MentalLLaMA Encoder-Style NLI Implementation Verification Report

**Date**: 2025-11-16
**Repository**: Mentallama_Criteria_CLS
**Reviewer**: Claude Code Verification Bot
**Task**: Verify decoder→encoder (Gemma Encoder style) implementation with MentalLLaMA

---

## Executive Summary

This repository is **NOT YET COMPLIANT** with the decoder→encoder method described in the cited paper. The codebase contains specifications and planning documents but lacks the actual encoder-style implementation. This report provides:

- ✅ **9/9 critical checks performed**
- ❌ **8/9 checks FAILED**
- ✅ **1/9 checks PASSED** (data structure exists)
- 📦 **Complete patches provided** for all failures
- 🧪 **Unit tests generated** for verification
- 📝 **Run instructions included**

---

## Detailed Check Results

### ✅ CHECK 1: Decoder LM (MentalLLaMA) as Backbone - **FAIL**

**Status**: ❌ **FAIL**

**Expected**: Load `klyang/MentaLLaMA-chat-7B` using encoder-style wrapper, not causal LM.

**Found**:
- `src/Project/SubProject/models/model.py:14` uses generic `transformers.AutoModel.from_pretrained(model_name)`
- No explicit MentalLLaMA model loading
- No encoder-style configuration

**Evidence**:
```python
# Current implementation (model.py:11-15)
class Model(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(Model, self).__init__()
        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, num_labels)
```

**Issue**: Uses generic `AutoModel` which may load causal LM configuration. Should explicitly use encoder-only configuration.

---

### ✅ CHECK 2: Attention Masking (Causal → Bidirectional) - **FAIL**

**Status**: ❌ **FAIL**

**Expected**:
- Override LLaMA's default causal attention mask
- Implement full bidirectional attention (no triangular mask)
- Use attention_mask (1=token, 0=pad) without causal restriction

**Found**:
- No attention mask modification in codebase
- No custom attention implementation
- Relies on default HuggingFace behavior (which is causal for LLaMA)

**Evidence**:
```bash
$ rg "causal|causal_mask|lower_triangular|tril|torch.triu|make_causal" -n
# No results found in Python files
```

**Issue**: LLaMA models use causal attention by default. Without explicit override, the model will use triangular masking, preventing encoder-style bidirectional attention.

---

### ✅ CHECK 3: Classifier Head (Pooler + MLP) - **PARTIAL FAIL**

**Status**: ⚠️ **PARTIAL FAIL**

**Expected**:
- Pooling mechanism (first token or mean pooling)
- Dropout layer (≈0.1)
- Linear classifier to num_labels
- No generate() or text parsing

**Found**:
```python
# Current implementation (model.py:17-21)
def forward(self, input_ids, attention_mask):
    outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs[1]  # ❌ WRONG: LLaMA has no pooler output
    logits = self.classifier(pooled_output)
    return logits
```

**Issues**:
1. ❌ Assumes `outputs[1]` exists (LLaMA doesn't have pooler_output)
2. ❌ No dropout before classifier
3. ✅ Has linear classifier (good)
4. ✅ No generate() usage (good)

---

### ✅ CHECK 4: Dropout Placement & Rates - **FAIL**

**Status**: ❌ **FAIL**

**Expected**:
- Attention softmax output dropout ≈ 0.1
- FFN output dropout ≈ 0.1
- Classifier head dropout ≈ 0.1

**Found**:
```bash
$ rg "Dropout\(|dropout_rate|attn_dropout|ffn_dropout" -n
# No matches in implementation files
```

**Evidence**:
- `classification_head` class has `dropout_prob` parameter but doesn't use it
- No dropout in `Model.forward()`
- No custom attention layers with dropout

**Issue**: Missing dropout regularization critical for preventing overfitting.

---

### ✅ CHECK 5: Right-Padding & Attention_Mask Semantics - **FAIL**

**Status**: ❌ **FAIL**

**Expected**:
- Tokenizer configured with `padding_side="right"`
- attention_mask: 1=token, 0=pad
- Proper collate function for batching

**Found**:
- No tokenizer initialization code in repository
- No data collator implementation
- Dataset file is essentially empty (1 line)

**Evidence**:
```bash
$ wc -l src/Project/SubProject/data/dataset.py
1 src/Project/SubProject/data/dataset.py
```

---

### ✅ CHECK 6: MentalLLaMA Tokenizer/Embeddings/Config - **FAIL**

**Status**: ❌ **FAIL**

**Expected**:
- `AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-7B")`
- Embedding dimensions match MentalLLaMA (4096)
- Config confirms LLaMA architecture
- No Gemma or other model artifacts

**Found**:
- Only reference in documentation: `CLAUDE.md:114`
- No actual tokenizer loading code
- Only one tokenizer import in `scripts/register_model.py` (not used for training)

**Search Results**:
```bash
$ rg "AutoTokenizer" --type py
scripts/register_model.py:24:from transformers import AutoTokenizer
# No usage in core training code
```

---

### ✅ CHECK 7: Supervised Classification Loss (CrossEntropy) - **FAIL**

**Status**: ❌ **FAIL**

**Expected**:
- `nn.CrossEntropyLoss()` or `nn.BCEWithLogitsLoss()` for binary classification
- Training loop computing loss from logits and labels
- NO LM loss (no next-token prediction)

**Found**:
- Train engine file is empty: `1 line`
- No loss function implementation
- Only documentation reference to `BCEWithLogitsLoss` in specs

**Evidence**:
```bash
$ wc -l src/Project/SubProject/engine/train_engine.py
1 src/Project/SubProject/engine/train_engine.py
```

---

### ✅ CHECK 8: ReDSM5 → NLI Data Conversion - **PARTIAL PASS**

**Status**: ⚠️ **PARTIAL PASS**

**Expected**:
- Convert ReDSM5 format to (premise, hypothesis, label) pairs
- Premise = sentence_text
- Hypothesis = DSM-5 criterion text
- Label = 1 if status='1', else 0

**Found**:
- ✅ ReDSM5 data files exist: `redsm5_posts.csv`, `redsm5_annotations.csv`
- ✅ DSM-5 criteria JSON exists: `data/DSM5/MDD_Criteira.json`
- ❌ No data preprocessing script
- ❌ No dataset loader implementation

**Data Structure** (✅ Correct):
```csv
# redsm5_annotations.csv format:
post_id,sentence_id,sentence_text,DSM5_symptom,status,explanation

# Available criteria in MDD_Criteira.json:
A.1: Depressed mood
A.2: Anhedonia
A.3: Weight/appetite changes
A.4: Sleep issues
A.5: Psychomotor issues
A.6: Fatigue
A.7: Worthlessness/guilt
A.8: Cognitive issues
A.9: Suicidal ideation
```

---

### ✅ CHECK 9: Unit Tests & Example Scripts - **FAIL**

**Status**: ❌ **FAIL**

**Expected**:
- Unit tests for model forward pass
- Shape tests (deterministic outputs)
- Attention mask tests
- Tokenization tests
- Example inference script

**Found**:
```bash
$ ls tests/
.gitkeep
# No test files
```

---

## Summary Table

| Check | Component | Status | Issue |
|-------|-----------|--------|-------|
| 1 | MentalLLaMA Backbone | ❌ FAIL | Generic AutoModel, no encoder config |
| 2 | Bidirectional Attention | ❌ FAIL | No attention mask override |
| 3 | Classifier Head | ⚠️ PARTIAL | Wrong pooling (outputs[1]), no dropout |
| 4 | Dropout | ❌ FAIL | No dropout layers implemented |
| 5 | Padding/Masking | ❌ FAIL | No tokenizer/collator implementation |
| 6 | MentalLLaMA Config | ❌ FAIL | No tokenizer loading code |
| 7 | Classification Loss | ❌ FAIL | No training loop implemented |
| 8 | NLI Data Conversion | ⚠️ PARTIAL | Data exists, no loader |
| 9 | Tests & Examples | ❌ FAIL | No tests or examples |

**Overall Compliance**: ❌ **NOT COMPLIANT** (1 partial pass, 8 failures)

---

## Root Cause Analysis

The repository is in **early planning stage**:
- ✅ Comprehensive specifications exist (`specs/001-model-use-mentallam/`)
- ✅ Data files present
- ❌ Implementation phase not started
- ❌ Core files are placeholders (1 line each)

**Key Missing Components**:
1. Encoder-style attention implementation
2. MentalLLaMA-specific model wrapper
3. Proper pooling for non-pooler models
4. Dropout regularization
5. Tokenizer & data pipeline
6. Training loop with classification loss
7. Unit tests

---

## Patches & Solutions

All patches are provided in the following files:
- `PATCH_01_encoder_model.py` - Complete encoder-style model
- `PATCH_02_data_pipeline.py` - NLI dataset loader
- `PATCH_03_train_engine.py` - Training loop with CrossEntropyLoss
- `PATCH_04_tests.py` - Comprehensive unit tests
- `PATCH_05_inference_example.py` - Deterministic inference example

See detailed patches in subsequent files.

---

## Recommendations

### Immediate Actions (P0)
1. ✅ **Implement encoder-style attention** using custom LlamaModel wrapper
2. ✅ **Fix pooling** to use `last_hidden_state[:, 0, :]` or mean pooling
3. ✅ **Add dropout layers** (0.1) before classifier
4. ✅ **Implement tokenizer** with right-padding
5. ✅ **Create data loader** for NLI conversion

### High Priority (P1)
6. ✅ **Implement training loop** with CrossEntropyLoss
7. ✅ **Add unit tests** for all components
8. ✅ **Create inference script** with deterministic output

### Medium Priority (P2)
9. Validate on small dataset split
10. Profile memory usage
11. Add integration tests

---

## Next Steps

1. **Apply patches** in order (01→05)
2. **Install dependencies**: `pip install -e '.[dev]'`
3. **Run tests**: `pytest tests/ -v`
4. **Verify inference**: `python examples/inference_example.py`
5. **Run training**: Use fixed train_engine.py

---

## Appendix: Search Commands Summary

```bash
# Model/backbone search
rg "from_pretrained|MentalLLaMA|MentaLLaMA|klyang|LlamaModel" -n
# ✅ Found references in docs, ❌ not in implementation

# Attention masking search
rg "causal|causal_mask|lower_triangular|tril|torch.triu" -n
# ❌ No results (missing bidirectional attention)

# Loss function search
rg "CrossEntropyLoss|BCEWithLogitsLoss|nn.CrossEntropyLoss" -n
# ❌ Only in documentation

# Dropout search
rg "Dropout\(|dropout_rate|attn_dropout|ffn_dropout" -n
# ❌ No implementation

# Data conversion search
rg "redsm5|ReDSM5|DSM5|criterion|entailment" -n
# ✅ Data files exist, ❌ no loader
```

---

**Report End**
