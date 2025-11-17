"""
Metrics and threshold tuning utilities

Provides evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
and threshold optimization for binary classification.
"""


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from Project.SubProject.utils.log import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute classification metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities for positive class (optional)

    Returns:
        Dictionary of metric values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    # Add ROC-AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """
    Compute confusion matrix

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, dict[str, float]]:
    """
    Find optimal classification threshold

    Args:
        y_true: Ground truth labels
        y_proba: Prediction probabilities for positive class
        metric: Metric to optimize ('f1', 'precision', 'recall')
        thresholds: Thresholds to try (default: 0.00 to 1.00 step 0.01)

    Returns:
        Tuple of (best_threshold, best_score, metrics_at_best)
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # Compute metrics
        current_metrics = compute_metrics(y_true, y_pred, y_proba)

        # Get score for optimization metric
        score = current_metrics.get(metric, 0.0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = current_metrics

    logger.info(
        f"Best threshold: {best_threshold:.3f} "
        f"({metric}={best_score:.4f})"
    )

    return best_threshold, best_score, best_metrics


def compute_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute precision-recall curve

    Args:
        y_true: Ground truth labels
        y_proba: Prediction probabilities for positive class

    Returns:
        Dictionary with 'precision', 'recall', 'thresholds' arrays
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute ROC curve

    Args:
        y_true: Ground truth labels
        y_proba: Prediction probabilities for positive class

    Returns:
        Dictionary with 'fpr', 'tpr', 'thresholds' arrays
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }


def threshold_sweep_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """
    Perform comprehensive threshold sweep

    Args:
        y_true: Ground truth labels
        y_proba: Prediction probabilities for positive class
        thresholds: Thresholds to try (default: 0.00 to 1.00 step 0.01)

    Returns:
        List of dictionaries with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)
        cm = compute_confusion_matrix(y_true, y_pred)

        result = {
            'threshold': float(threshold),
            **metrics,
            **cm,
        }
        results.append(result)

    return results


class MetricsTracker:
    """Track metrics across training steps"""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_roc_auc': [],
        }
        self.best_metrics = {}
        self.best_epoch = 0

    def update(self, epoch: int, metrics: dict[str, float]):
        """Update metrics for current epoch"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        # Track best F1
        if 'val_f1' in metrics:
            if not self.best_metrics or metrics['val_f1'] > self.best_metrics.get('val_f1', 0):
                self.best_metrics = metrics.copy()
                self.best_epoch = epoch

    def get_best(self) -> tuple[int, dict[str, float]]:
        """Get best epoch and metrics"""
        return self.best_epoch, self.best_metrics

    def should_stop(self, patience: int = 20) -> bool:
        """Check if early stopping criteria met"""
        if len(self.history['val_f1']) < patience:
            return False

        current_epoch = len(self.history['val_f1']) - 1
        return (current_epoch - self.best_epoch) >= patience

    def get_summary(self) -> dict:
        """Get summary statistics"""
        summary = {
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.history.get('val_f1', [])),
        }

        # Add final metrics
        for key, values in self.history.items():
            if values:
                summary[f'final_{key}'] = values[-1]

        return summary
