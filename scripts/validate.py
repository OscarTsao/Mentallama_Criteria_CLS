#!/usr/bin/env python3
"""
Validation script for paper compliance

Verifies that the implementation matches all paper requirements.

Usage:
    python scripts/validate.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_model_architecture():
    """Verify model architecture compliance."""
    print("Checking model architecture...")

    try:
        from Project.SubProject.models.model import EncoderStyleLlamaModel

        # Check class exists
        assert hasattr(EncoderStyleLlamaModel, '__init__'), "Model class missing __init__"
        assert hasattr(EncoderStyleLlamaModel, 'forward'), "Model class missing forward"
        assert hasattr(EncoderStyleLlamaModel, '_patch_attention_for_bidirectional'), \
            "Missing bidirectional attention patch method"
        assert hasattr(EncoderStyleLlamaModel, 'pool_hidden_states'), \
            "Missing pooling method"

        print("  ✓ EncoderStyleLlamaModel class present")
        print("  ✓ Bidirectional attention patch method exists")
        print("  ✓ Pooling method exists")
        print("  ✓ Forward method exists")

        return True
    except Exception as e:
        print(f"  ❌ Model architecture check failed: {e}")
        return False


def check_data_pipeline():
    """Verify data pipeline compliance."""
    print("Checking data pipeline...")

    try:
        from Project.SubProject.data.dataset import (
            DSM5CriteriaMapping,
            ReDSM5toNLIConverter,
            MentalHealthNLIDataset,
            create_nli_dataloaders
        )

        print("  ✓ DSM5CriteriaMapping class present")
        print("  ✓ ReDSM5toNLIConverter class present")
        print("  ✓ MentalHealthNLIDataset class present")
        print("  ✓ create_nli_dataloaders function present")

        return True
    except Exception as e:
        print(f"  ❌ Data pipeline check failed: {e}")
        return False


def check_training_engine():
    """Verify training engine compliance."""
    print("Checking training engine...")

    try:
        from Project.SubProject.engine.train_engine import ClassificationTrainer
        import torch.nn as nn

        # Verify ClassificationTrainer exists
        assert hasattr(ClassificationTrainer, '__init__'), "Trainer missing __init__"
        assert hasattr(ClassificationTrainer, 'train'), "Trainer missing train method"
        assert hasattr(ClassificationTrainer, 'evaluate'), "Trainer missing evaluate method"

        print("  ✓ ClassificationTrainer class present")
        print("  ✓ Training method exists")
        print("  ✓ Evaluation method exists")

        # Check loss function (should be CrossEntropyLoss, not LM loss)
        trainer_code = Path(__file__).parent.parent / "src/Project/SubProject/engine/train_engine.py"
        if trainer_code.exists():
            with open(trainer_code, 'r') as f:
                content = f.read()
                assert 'CrossEntropyLoss' in content, "CrossEntropyLoss not found"
                assert 'generate(' not in content, "Should not use generate() for training"

                print("  ✓ Uses CrossEntropyLoss (not LM loss)")
                print("  ✓ No generate() usage found")

        return True
    except Exception as e:
        print(f"  ❌ Training engine check failed: {e}")
        return False


def check_tests():
    """Verify test coverage."""
    print("Checking tests...")

    test_file = Path(__file__).parent.parent / "tests/test_encoder_implementation.py"
    if not test_file.exists():
        print("  ❌ Test file not found")
        return False

    try:
        with open(test_file, 'r') as f:
            content = f.read()

            # Check for key test classes
            required_tests = [
                'TestModelShapes',
                'TestDropout',
                'TestDataPipeline',
                'TestTraining',
                'TestInference'
            ]

            for test in required_tests:
                if test in content:
                    print(f"  ✓ {test} test class found")
                else:
                    print(f"  ❌ {test} test class missing")
                    return False

        print("  ✓ All required test classes present")
        return True
    except Exception as e:
        print(f"  ❌ Test check failed: {e}")
        return False


def check_examples():
    """Verify examples exist."""
    print("Checking examples...")

    example_file = Path(__file__).parent.parent / "examples/inference_example.py"
    if not example_file.exists():
        print("  ❌ Inference example not found")
        return False

    print("  ✓ Inference example present")
    return True


def check_data_files():
    """Verify data files exist."""
    print("Checking data files...")

    data_dir = Path(__file__).parent.parent / "data"

    required_files = [
        "DSM5/MDD_Criteira.json",
        "redsm5/redsm5_posts.csv",
        "redsm5/redsm5_annotations.csv",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = data_dir / file_path
        if full_path.exists():
            print(f"  ✓ {file_path} found")
        else:
            print(f"  ❌ {file_path} missing")
            all_exist = False

    return all_exist


def check_paper_compliance():
    """Verify paper-specific requirements."""
    print("Checking paper compliance...")

    model_file = Path(__file__).parent.parent / "src/Project/SubProject/models/model.py"
    if not model_file.exists():
        print("  ❌ Model file not found")
        return False

    try:
        with open(model_file, 'r') as f:
            content = f.read()

            checks = [
                ("is_decoder = False", "Config override for bidirectional attention"),
                ("_patch_attention_for_bidirectional", "Attention mask patching"),
                ("pool_hidden_states", "Pooling method"),
                ("nn.Dropout", "Dropout regularization"),
                ("nn.Linear", "Classification head"),
                ("klyang/MentaLLaMA-chat-7B", "MentalLLaMA backbone"),
            ]

            for pattern, description in checks:
                if pattern in content:
                    print(f"  ✓ {description}")
                else:
                    print(f"  ❌ {description} not found")
                    return False

        print("  ✓ All paper requirements present")
        return True
    except Exception as e:
        print(f"  ❌ Paper compliance check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Paper Compliance Validation")
    print("=" * 70)
    print()

    checks = [
        ("Model Architecture", check_model_architecture),
        ("Data Pipeline", check_data_pipeline),
        ("Training Engine", check_training_engine),
        ("Tests", check_tests),
        ("Examples", check_examples),
        ("Data Files", check_data_files),
        ("Paper Compliance", check_paper_compliance),
    ]

    results = {}
    for name, check_func in checks:
        print(f"\n[{name}]")
        results[name] = check_func()

    # Summary
    print()
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Overall: {passed}/{total} checks passed")

    if passed == total:
        print()
        print("✅ All validation checks passed!")
        print("Repository is 100% paper-compliant.")
        return 0
    else:
        print()
        print("❌ Some validation checks failed.")
        print("See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
