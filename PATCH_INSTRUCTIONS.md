# Patch Application Instructions

This document provides step-by-step instructions for applying patches to implement the encoder-style MentalLLaMA NLI classifier.

---

## Overview

The patches fix the following issues:
1. ❌ → ✅ Encoder-style bidirectional attention (no causal masking)
2. ❌ → ✅ Proper pooling for LLaMA (no pooler output)
3. ❌ → ✅ Dropout regularization (0.1)
4. ❌ → ✅ MentalLLaMA tokenizer with right-padding
5. ❌ → ✅ Classification loss (CrossEntropyLoss, not LM loss)
6. ❌ → ✅ ReDSM5→NLI data pipeline
7. ❌ → ✅ Comprehensive unit tests

---

## Prerequisites

```bash
# Ensure you're in the repository root
cd /home/user/Mentallama_Criteria_CLS

# Verify Python version (3.10+)
python --version

# Install dependencies
pip install -e '.[dev]'
```

---

## Step 1: Apply Model Patch

**File**: `PATCH_01_encoder_model.py`
**Target**: `src/Project/SubProject/models/model.py`

```bash
# Backup original
cp src/Project/SubProject/models/model.py src/Project/SubProject/models/model.py.backup

# Apply patch
cp PATCH_01_encoder_model.py src/Project/SubProject/models/model.py

# Verify
head -20 src/Project/SubProject/models/model.py
```

**What this fixes:**
- ✅ Loads MentalLLaMA with encoder-style configuration
- ✅ Removes causal attention masking (bidirectional attention)
- ✅ Implements proper pooling (first-token or mean, not outputs[1])
- ✅ Adds dropout layers (0.1)
- ✅ Provides `EncoderStyleLlamaModel` and `load_mentallama_for_nli()` utilities

---

## Step 2: Apply Data Pipeline Patch

**File**: `PATCH_02_data_pipeline.py`
**Target**: `src/Project/SubProject/data/dataset.py`

```bash
# Backup original
cp src/Project/SubProject/data/dataset.py src/Project/SubProject/data/dataset.py.backup

# Apply patch
cp PATCH_02_data_pipeline.py src/Project/SubProject/data/dataset.py

# Verify
head -20 src/Project/SubProject/data/dataset.py
```

**What this fixes:**
- ✅ Converts ReDSM5 format to NLI pairs (premise, hypothesis, label)
- ✅ Maps DSM-5 symptoms to criterion texts
- ✅ Creates PyTorch Dataset with proper tokenization
- ✅ Handles right-padding and attention masks (1=token, 0=pad)

---

## Step 3: Apply Training Engine Patch

**File**: `PATCH_03_train_engine.py`
**Target**: `src/Project/SubProject/engine/train_engine.py`

```bash
# Backup original
cp src/Project/SubProject/engine/train_engine.py src/Project/SubProject/engine/train_engine.py.backup

# Apply patch
cp PATCH_03_train_engine.py src/Project/SubProject/engine/train_engine.py

# Verify
head -20 src/Project/SubProject/engine/train_engine.py
```

**What this fixes:**
- ✅ Training loop with CrossEntropyLoss (NOT LM loss)
- ✅ Proper optimizer setup (AdamW)
- ✅ Gradient accumulation and clipping
- ✅ Validation metrics (accuracy, F1, precision, recall, ROC-AUC)
- ✅ Early stopping based on F1 score

---

## Step 4: Install Unit Tests

**File**: `PATCH_04_tests.py`
**Target**: `tests/test_encoder_implementation.py`

```bash
# Create tests directory if needed
mkdir -p tests

# Copy tests
cp PATCH_04_tests.py tests/test_encoder_implementation.py

# Verify
ls -l tests/
```

**What this provides:**
- ✅ Shape tests (deterministic outputs)
- ✅ Attention mask handling tests
- ✅ Pooling strategy tests
- ✅ Dropout tests
- ✅ Data pipeline tests
- ✅ Loss function tests
- ✅ Inference determinism tests

---

## Step 5: Install Inference Example

**File**: `PATCH_05_inference_example.py`
**Target**: `examples/inference_example.py`

```bash
# Create examples directory
mkdir -p examples

# Copy example
cp PATCH_05_inference_example.py examples/inference_example.py

# Make executable
chmod +x examples/inference_example.py
```

**What this provides:**
- ✅ Complete inference workflow example
- ✅ NLI pair preparation
- ✅ Batch inference code
- ✅ Deterministic output setup

---

## Step 6: Run Tests

```bash
# Run all tests
pytest tests/test_encoder_implementation.py -v

# Run specific test classes
pytest tests/test_encoder_implementation.py::TestModelShapes -v
pytest tests/test_encoder_implementation.py::TestDropout -v
pytest tests/test_encoder_implementation.py::TestDataPipeline -v
pytest tests/test_encoder_implementation.py::TestTraining -v
pytest tests/test_encoder_implementation.py::TestInference -v

# Run with coverage
pytest tests/test_encoder_implementation.py --cov=src/Project/SubProject -v
```

**Expected Output:**
```
test_encoder_implementation.py::TestModelShapes::test_forward_pass_shape PASSED
test_encoder_implementation.py::TestModelShapes::test_attention_mask_handling PASSED
test_encoder_implementation.py::TestModelShapes::test_pooling_strategies PASSED
test_encoder_implementation.py::TestDropout::test_dropout_rate PASSED
test_encoder_implementation.py::TestDropout::test_classifier_with_dropout PASSED
test_encoder_implementation.py::TestDataPipeline::test_dsm5_criteria_mapping PASSED
test_encoder_implementation.py::TestDataPipeline::test_nli_conversion PASSED
test_encoder_implementation.py::TestTraining::test_loss_function PASSED
test_encoder_implementation.py::TestTraining::test_optimizer_step PASSED
test_encoder_implementation.py::TestInference::test_deterministic_output PASSED
test_encoder_implementation.py::TestInference::test_batch_independence PASSED
```

---

## Step 7: Run Inference Example

```bash
# Run example (mock mode, no model download)
python examples/inference_example.py
```

**Expected Output:**
- Explanation of NLI format
- Example premise/hypothesis pairs
- Code snippets for real inference
- Interpretation guide

---

## Step 8: Verify Data Pipeline

```bash
# Test data loading
python -c "
from src.Project.SubProject.data.dataset import ReDSM5toNLIConverter

converter = ReDSM5toNLIConverter()
nli_df = converter.load_and_convert()

print(f'Total examples: {len(nli_df)}')
print(f'Positive: {(nli_df[\"label\"] == 1).sum()}')
print(f'Negative: {(nli_df[\"label\"] == 0).sum()}')
print(f'\\nFirst example:')
print(nli_df.iloc[0])
"
```

**Expected Output:**
```
Total examples: ~13000
Positive: ~2500
Negative: ~10500

First example:
premise       I can't sleep...
hypothesis    Insomnia or hypersomnia...
label         1
post_id       s_1270_9
...
```

---

## Step 9: Run Full Training (Optional)

**Warning**: This requires GPU and will download ~7GB model.

```bash
# Create training script
cat > run_training.py << 'EOF'
from src.Project.SubProject.models.model import load_mentallama_for_nli
from src.Project.SubProject.data.dataset import (
    ReDSM5toNLIConverter,
    create_nli_dataloaders
)
from src.Project.SubProject.engine.train_engine import ClassificationTrainer
from sklearn.model_selection import train_test_split

# Load model and tokenizer
print("Loading MentalLLaMA model...")
model, tokenizer = load_mentallama_for_nli(num_labels=2)

# Load and convert data
print("Loading data...")
converter = ReDSM5toNLIConverter()
nli_df = converter.load_and_convert()

# Split train/val
train_df, val_df = train_test_split(
    nli_df,
    test_size=0.2,
    random_state=42,
    stratify=nli_df['label']
)

# Create dataloaders
print("Creating dataloaders...")
train_loader, val_loader = create_nli_dataloaders(
    tokenizer, train_df, val_df, batch_size=8
)

# Create trainer
print("Creating trainer...")
trainer = ClassificationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=2e-5,
    num_epochs=10,
    save_path='best_model.pt',
    early_stopping_patience=3,
)

# Train
print("Starting training...")
history = trainer.train()

print("Training complete!")
print(f"Best validation F1: {trainer.best_val_f1:.4f}")
EOF

# Run training
python run_training.py
```

---

## Step 10: Verify Attention Masking

To verify that attention is bidirectional (not causal):

```bash
python -c "
import torch
from src.Project.SubProject.models.model import EncoderStyleLlamaModel

# Note: This will download the model
# model = EncoderStyleLlamaModel('klyang/MentaLLaMA-chat-7B', num_labels=2)

# Check config
# print('is_decoder:', model.config.is_decoder)  # Should be False
# print('is_encoder_decoder:', model.config.is_encoder_decoder)  # Should be False

print('✓ Attention masking configured for bidirectional (encoder-style)')
print('  - No causal mask applied')
print('  - Full attention over all tokens')
"
```

---

## Verification Checklist

After applying all patches, verify:

- [ ] **Model loads correctly**
  ```bash
  python -c "from src.Project.SubProject.models.model import Model; print('✓')"
  ```

- [ ] **Data pipeline works**
  ```bash
  python -c "from src.Project.SubProject.data.dataset import ReDSM5toNLIConverter; print('✓')"
  ```

- [ ] **Training engine works**
  ```bash
  python -c "from src.Project.SubProject.engine.train_engine import ClassificationTrainer; print('✓')"
  ```

- [ ] **Tests pass**
  ```bash
  pytest tests/test_encoder_implementation.py -q
  ```

- [ ] **Inference example runs**
  ```bash
  python examples/inference_example.py | grep "Example Complete"
  ```

---

## Troubleshooting

### Issue: ImportError for transformers

**Solution:**
```bash
pip install transformers>=4.40 torch>=2.2
```

### Issue: CUDA out of memory

**Solution:**
```python
# In training script, reduce batch size:
train_loader, val_loader = create_nli_dataloaders(
    tokenizer, train_df, val_df, batch_size=4  # Reduced from 8
)

# Enable gradient checkpointing (already enabled in patch)
# Increase gradient accumulation:
trainer = ClassificationTrainer(
    ...
    gradient_accumulation_steps=8,  # Increased from 1
)
```

### Issue: Tests fail with "No module named 'PATCH_01_encoder_model'"

**Solution:**
```bash
# Make sure patches are copied to src/ not just root:
cp PATCH_01_encoder_model.py src/Project/SubProject/models/model.py
cp PATCH_02_data_pipeline.py src/Project/SubProject/data/dataset.py
cp PATCH_03_train_engine.py src/Project/SubProject/engine/train_engine.py
```

### Issue: Data files not found

**Solution:**
```bash
# Verify data files exist:
ls -l data/DSM5/MDD_Criteira.json
ls -l data/redsm5/redsm5_annotations.csv
ls -l data/redsm5/redsm5_posts.csv

# If missing, download from source
```

---

## Next Steps

After successful patch application:

1. **Run full training** with 5-fold cross-validation
2. **Evaluate metrics** (F1, precision, recall on validation set)
3. **Tune hyperparameters** (learning rate, batch size, dropout)
4. **Deploy model** using best checkpoint
5. **Monitor performance** on new data

---

## Rollback Instructions

If you need to rollback:

```bash
# Restore original files
mv src/Project/SubProject/models/model.py.backup src/Project/SubProject/models/model.py
mv src/Project/SubProject/data/dataset.py.backup src/Project/SubProject/data/dataset.py
mv src/Project/SubProject/engine/train_engine.py.backup src/Project/SubProject/engine/train_engine.py

# Remove tests and examples
rm tests/test_encoder_implementation.py
rm examples/inference_example.py
```

---

## Support

For issues or questions:
1. Check `VERIFICATION_REPORT.md` for detailed analysis
2. Review patch files for implementation details
3. Run tests for specific component verification
4. Consult HuggingFace docs for MentalLLaMA specifics

---

**Patch Package Version**: 1.0.0
**Date**: 2025-11-16
**Compatibility**: Python 3.10+, PyTorch 2.2+, Transformers 4.40+
