"""
Basic import tests

Tests that all modules can be imported without errors.
"""

import pytest


def test_utils_imports():
    """Test utils module imports"""
    from Project.SubProject.utils import get_logger, set_seed
    from Project.SubProject.utils.mlflow_utils import configure_mlflow, mlflow_run

    assert get_logger is not None
    assert set_seed is not None
    assert configure_mlflow is not None
    assert mlflow_run is not None


def test_data_imports():
    """Test data module imports"""
    from Project.SubProject.data import (
        MentalHealthDataset,
        Sample,
        create_folds,
        load_fold,
        FoldAssignment,
    )

    assert MentalHealthDataset is not None
    assert Sample is not None
    assert create_folds is not None


def test_models_imports():
    """Test models module imports"""
    from Project.SubProject.models import (
        MentallamClassifier,
        Model,
        classification_head,
        build_prompt,
        normalize_text,
        truncate_for_tokenizer,
    )

    assert MentallamClassifier is not None
    assert build_prompt is not None
    assert normalize_text is not None


def test_engine_imports():
    """Test engine module imports"""
    from Project.SubProject.engine import (
        compute_metrics,
        compute_confusion_matrix,
        tune_threshold,
        compute_pr_curve,
        compute_roc_curve,
        threshold_sweep_analysis,
        MetricsTracker,
    )

    assert compute_metrics is not None
    assert tune_threshold is not None
    assert MetricsTracker is not None


def test_prompt_builder():
    """Test prompt builder functionality"""
    from Project.SubProject.models import build_prompt, normalize_text

    # Test basic prompt
    prompt = build_prompt("I feel sad", "Depressed mood")
    assert "post:" in prompt
    assert "criterion:" in prompt
    assert "I feel sad" in prompt
    assert "Depressed mood" in prompt

    # Test normalization
    text = normalize_text("  Hello   World  ")
    assert text == "Hello World"

    # Test unicode normalization
    text = normalize_text("café")
    assert text == "café"


def test_metrics():
    """Test metrics computation"""
    import numpy as np
    from Project.SubProject.engine import compute_metrics, MetricsTracker

    # Test metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = compute_metrics(y_true, y_pred)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics

    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['f1'] <= 1.0

    # Test tracker
    tracker = MetricsTracker()
    tracker.update(0, {'val_f1': 0.8, 'val_loss': 0.5})
    tracker.update(1, {'val_f1': 0.85, 'val_loss': 0.4})

    best_epoch, best_metrics = tracker.get_best()
    assert best_epoch == 1
    assert best_metrics['val_f1'] == 0.85


def test_logger():
    """Test logger functionality"""
    from Project.SubProject.utils import get_logger

    logger = get_logger('test')
    assert logger is not None

    # Should not raise
    logger.info("Test message")
    logger.debug("Debug message")
    logger.warning("Warning message")


def test_seed():
    """Test seed setting"""
    from Project.SubProject.utils import set_seed
    import random
    import numpy as np

    # Set seed
    set_seed(42)
    r1 = random.random()
    n1 = np.random.rand()

    # Reset seed
    set_seed(42)
    r2 = random.random()
    n2 = np.random.rand()

    # Should be reproducible
    assert r1 == r2
    assert n1 == n2
