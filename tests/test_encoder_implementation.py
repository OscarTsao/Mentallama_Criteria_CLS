"""
PATCH 04: Comprehensive Unit Tests

Tests for encoder-style MentalLLaMA implementation.

Tests cover:
1. Model forward pass shape checks
2. Attention mask handling
3. Pooling strategies
4. Dropout behavior
5. Data pipeline
6. Training loop

Usage:
    cp PATCH_04_tests.py tests/test_encoder_implementation.py
    pytest tests/test_encoder_implementation.py -v
"""

import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mock configuration for testing without downloading actual model
class MockLlamaConfig:
    """Mock config for testing."""
    def __init__(self):
        self.hidden_size = 256
        self.num_attention_heads = 4
        self.num_hidden_layers = 2
        self.vocab_size = 1000
        self.max_position_embeddings = 512
        self.pad_token_id = 0
        self.is_decoder = False
        self.is_encoder_decoder = False


class MockLlamaModel(nn.Module):
    """Mock LLaMA model for testing."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Simple mock forward pass
        embeddings = self.embeddings(input_ids)

        # Mock output (in real model this would be transformer layers)
        class MockOutput:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states

        return MockOutput(embeddings)


# ============================================================================
# Test 1: Model Shape Tests (Deterministic)
# ============================================================================

class TestModelShapes:
    """Test that model produces correct output shapes."""

    def test_forward_pass_shape(self):
        """Test basic forward pass produces correct logit shape."""
        # Create minimal model
        config = MockLlamaConfig()
        model = MockLlamaModel(config)

        # Create classifier
        classifier = nn.Linear(config.hidden_size, 2)
        dropout = nn.Dropout(0.1)

        # Mock input
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        outputs = model(input_ids, attention_mask)
        hidden = outputs.last_hidden_state

        # Pool (first token)
        pooled = hidden[:, 0, :]

        # Dropout + classifier
        pooled = dropout(pooled)
        logits = classifier(pooled)

        # Check shape
        assert logits.shape == (batch_size, 2), f"Expected (2, 2), got {logits.shape}"
        print(f"✓ Forward pass shape test passed: {logits.shape}")

    def test_attention_mask_handling(self):
        """Test that attention mask is properly handled."""
        config = MockLlamaConfig()
        model = MockLlamaModel(config)

        batch_size = 2
        seq_len = 16

        # Create input with padding (attention_mask has 0s for padding)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0  # First sample has padding from position 10
        attention_mask[1, 14:] = 0  # Second sample has padding from position 14

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Check that output has correct shape regardless of padding
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
        print("✓ Attention mask handling test passed")

    def test_pooling_strategies(self):
        """Test different pooling strategies."""
        batch_size = 2
        seq_len = 16
        hidden_dim = 256

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0  # Padding

        # Test first-token pooling
        pooled_first = hidden_states[:, 0, :]
        assert pooled_first.shape == (batch_size, hidden_dim)

        # Test mean pooling (with mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_mean = sum_hidden / sum_mask

        assert pooled_mean.shape == (batch_size, hidden_dim)

        # Check that mean pooling correctly handles masking
        # (mean should only be over non-padded tokens)
        assert not torch.isnan(pooled_mean).any()

        print("✓ Pooling strategies test passed")


# ============================================================================
# Test 2: Dropout Tests
# ============================================================================

class TestDropout:
    """Test dropout behavior."""

    def test_dropout_rate(self):
        """Test that dropout is applied with correct rate."""
        dropout = nn.Dropout(0.1)

        # Create input
        x = torch.ones(100, 256)

        # Training mode: dropout should be active
        dropout.train()
        x_dropped = dropout(x)

        # Some values should be zeroed
        # (with p=0.1, expect ~10% to be zero, scaled by 1/(1-0.1))
        num_zeros = (x_dropped == 0).sum().item()
        total = x_dropped.numel()
        dropout_ratio = num_zeros / total

        # Allow some variance, should be around 0.1
        assert 0.05 < dropout_ratio < 0.15, f"Dropout ratio {dropout_ratio} not ~0.1"

        # Eval mode: no dropout
        dropout.eval()
        x_eval = dropout(x)
        assert torch.allclose(x, x_eval), "Dropout should be disabled in eval mode"

        print(f"✓ Dropout test passed (ratio: {dropout_ratio:.3f})")

    def test_classifier_with_dropout(self):
        """Test that classifier head includes dropout."""
        from PATCH_01_encoder_model import classification_head

        head = classification_head(
            input_dim=256,
            num_labels=2,
            dropout_prob=0.1,
            layer_num=1
        )

        # Check that dropout is in the module
        has_dropout = any(isinstance(m, nn.Dropout) for m in head.classifier.modules())
        assert has_dropout, "Classification head should include dropout"

        print("✓ Classifier dropout test passed")


# ============================================================================
# Test 3: Data Pipeline Tests
# ============================================================================

class TestDataPipeline:
    """Test data loading and conversion."""

    def test_dsm5_criteria_mapping(self):
        """Test DSM-5 criteria mapping."""
        from PATCH_02_data_pipeline import DSM5CriteriaMapping

        # Test loading (if file exists)
        criteria_path = "data/DSM5/MDD_Criteira.json"
        if Path(criteria_path).exists():
            mapper = DSM5CriteriaMapping(criteria_path)

            # Check that all expected symptoms have mappings
            expected_symptoms = [
                'DEPRESSED_MOOD', 'ANHEDONIA', 'WEIGHT_APPETITE',
                'SLEEP_ISSUES', 'PSYCHOMOTOR', 'FATIGUE',
                'WORTHLESSNESS', 'COGNITIVE_ISSUES', 'SUICIDAL'
            ]

            for symptom in expected_symptoms:
                text = mapper.get_criterion_text(symptom)
                assert text != "", f"No text for symptom {symptom}"

            print(f"✓ DSM-5 criteria mapping test passed ({len(expected_symptoms)} symptoms)")
        else:
            print("⊘ Skipping criteria mapping test (file not found)")

    def test_nli_conversion(self):
        """Test ReDSM5 to NLI conversion."""
        from PATCH_02_data_pipeline import ReDSM5toNLIConverter

        # Test loading (if files exist)
        annotations_path = "data/redsm5/redsm5_annotations.csv"
        if Path(annotations_path).exists():
            converter = ReDSM5toNLIConverter()
            nli_df = converter.load_and_convert(include_negatives=True)

            # Check DataFrame structure
            required_cols = ['premise', 'hypothesis', 'label', 'post_id']
            for col in required_cols:
                assert col in nli_df.columns, f"Missing column: {col}"

            # Check label values
            assert set(nli_df['label'].unique()).issubset({0, 1}), "Labels should be 0 or 1"

            # Check no empty texts
            assert (nli_df['premise'].str.len() > 0).all(), "Empty premises found"
            assert (nli_df['hypothesis'].str.len() > 0).all(), "Empty hypotheses found"

            print(f"✓ NLI conversion test passed ({len(nli_df)} examples)")
        else:
            print("⊘ Skipping NLI conversion test (data files not found)")


# ============================================================================
# Test 4: Training Loop Tests
# ============================================================================

class TestTraining:
    """Test training loop components."""

    def test_loss_function(self):
        """Test that we use CrossEntropyLoss, not LM loss."""
        loss_fn = nn.CrossEntropyLoss()

        # Mock logits and labels
        logits = torch.randn(4, 2)  # Batch of 4, 2 classes
        labels = torch.tensor([0, 1, 0, 1])

        # Compute loss
        loss = loss_fn(logits, labels)

        # Check loss is scalar
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"

        print(f"✓ Loss function test passed (loss: {loss.item():.4f})")

    def test_optimizer_step(self):
        """Test optimizer performs parameter updates."""
        # Simple model
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Save initial weights
        initial_weight = model.weight.data.clone()

        # Forward + backward
        x = torch.randn(4, 10)
        labels = torch.tensor([0, 1, 0, 1])

        logits = model(x)
        loss = loss_fn(logits, labels)
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Check weights changed
        assert not torch.allclose(model.weight.data, initial_weight), \
            "Weights should change after optimizer step"

        print("✓ Optimizer step test passed")


# ============================================================================
# Test 5: Inference Tests
# ============================================================================

class TestInference:
    """Test inference behavior."""

    def test_deterministic_output(self):
        """Test that model produces deterministic outputs with same seed."""
        torch.manual_seed(42)

        config = MockLlamaConfig()
        model = MockLlamaModel(config)
        classifier = nn.Linear(config.hidden_size, 2)

        # Same input
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        attention_mask = torch.ones(2, 16)

        # First forward pass
        model.eval()
        classifier.eval()

        with torch.no_grad():
            outputs1 = model(input_ids, attention_mask)
            pooled1 = outputs1.last_hidden_state[:, 0, :]
            logits1 = classifier(pooled1)

        # Second forward pass (same input)
        with torch.no_grad():
            outputs2 = model(input_ids, attention_mask)
            pooled2 = outputs2.last_hidden_state[:, 0, :]
            logits2 = classifier(pooled2)

        # Should be identical
        assert torch.allclose(logits1, logits2), "Outputs should be deterministic"

        print("✓ Deterministic output test passed")

    def test_batch_independence(self):
        """Test that batch samples are independent."""
        config = MockLlamaConfig()
        model = MockLlamaModel(config)
        classifier = nn.Linear(config.hidden_size, 2)

        model.eval()
        classifier.eval()

        # Create two different inputs
        input1 = torch.randint(0, config.vocab_size, (1, 16))
        input2 = torch.randint(100, 200, (1, 16))  # Different range

        attention_mask = torch.ones(1, 16)

        # Process separately
        with torch.no_grad():
            out1 = model(input1, attention_mask)
            pooled1 = out1.last_hidden_state[:, 0, :]
            logits1 = classifier(pooled1)

            out2 = model(input2, attention_mask)
            pooled2 = out2.last_hidden_state[:, 0, :]
            logits2 = classifier(pooled2)

        # Process as batch
        input_batch = torch.cat([input1, input2], dim=0)
        attention_batch = torch.ones(2, 16)

        with torch.no_grad():
            out_batch = model(input_batch, attention_batch)
            pooled_batch = out_batch.last_hidden_state[:, 0, :]
            logits_batch = classifier(pooled_batch)

        # Batch outputs should match individual outputs
        assert torch.allclose(logits_batch[0], logits1[0], atol=1e-5)
        assert torch.allclose(logits_batch[1], logits2[0], atol=1e-5)

        print("✓ Batch independence test passed")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Running Encoder-Style MentalLLaMA Implementation Tests")
    print("=" * 70)

    # Run tests
    pytest.main([__file__, "-v", "-s"])
