"""
Cross-validation fold generation

Creates StratifiedGroupKFold splits ensuring post_ids don't overlap
between train and validation sets.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from Project.SubProject.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class FoldAssignment:
    """Fold assignment metadata"""
    fold_index: int
    train_indices: list[int]
    val_indices: list[int]
    train_groups: list[str]  # post_ids in train
    val_groups: list[str]    # post_ids in val
    seed: int
    dataset_hash: str
    label_distribution: dict[str, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'fold_index': self.fold_index,
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'train_groups': self.train_groups,
            'val_groups': self.val_groups,
            'seed': self.seed,
            'dataset_hash': self.dataset_hash,
            'label_distribution': self.label_distribution,
            'n_train': len(self.train_indices),
            'n_val': len(self.val_indices),
        }


def create_folds(
    dataset,
    n_folds: int = 5,
    seed: int = 42,
    output_dir: str | None = None,
) -> list[FoldAssignment]:
    """
    Create stratified group k-fold splits

    Args:
        dataset: MentalHealthDataset instance
        n_folds: Number of folds
        seed: Random seed
        output_dir: Directory to save fold JSON files (optional)

    Returns:
        List of FoldAssignment objects
    """
    logger.info(f"Creating {n_folds}-fold splits with seed={seed}")

    # Get labels and groups
    labels = np.array([sample.label for sample in dataset.samples])
    groups = np.array([sample.post_id for sample in dataset.samples])

    # Create splitter
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Generate folds
    fold_assignments = []
    dataset_hash = dataset.get_dataset_hash()

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(labels, labels, groups)):
        # Get groups for this fold
        train_groups = list(set(groups[train_idx]))
        val_groups = list(set(groups[val_idx]))

        # Verify no overlap
        overlap = set(train_groups) & set(val_groups)
        if overlap:
            raise ValueError(f"Fold {fold_idx}: Groups overlap! {overlap}")

        # Compute label distribution
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        label_dist = {
            'train_positive': int(train_labels.sum()),
            'train_negative': int((train_labels == 0).sum()),
            'train_total': int(len(train_labels)),
            'val_positive': int(val_labels.sum()),
            'val_negative': int((val_labels == 0).sum()),
            'val_total': int(len(val_labels)),
        }

        # Create fold assignment
        fold = FoldAssignment(
            fold_index=fold_idx,
            train_indices=train_idx.tolist(),
            val_indices=val_idx.tolist(),
            train_groups=train_groups,
            val_groups=val_groups,
            seed=seed,
            dataset_hash=dataset_hash,
            label_distribution=label_dist,
        )

        fold_assignments.append(fold)

        # Log statistics
        logger.info(
            f"Fold {fold_idx}: "
            f"train={len(train_idx)} ({len(train_groups)} posts, "
            f"{label_dist['train_positive']}/{label_dist['train_total']} pos), "
            f"val={len(val_idx)} ({len(val_groups)} posts, "
            f"{label_dist['val_positive']}/{label_dist['val_total']} pos)"
        )

    # Save to files if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for fold in fold_assignments:
            fold_file = output_path / f"fold_{fold.fold_index}.json"
            with open(fold_file, 'w') as f:
                json.dump(fold.to_dict(), f, indent=2)
            logger.info(f"Saved fold {fold.fold_index} to {fold_file}")

    return fold_assignments


def load_fold(fold_file: str) -> FoldAssignment:
    """Load fold assignment from JSON file"""
    with open(fold_file) as f:
        data = json.load(f)

    return FoldAssignment(
        fold_index=data['fold_index'],
        train_indices=data['train_indices'],
        val_indices=data['val_indices'],
        train_groups=data['train_groups'],
        val_groups=data['val_groups'],
        seed=data['seed'],
        dataset_hash=data['dataset_hash'],
        label_distribution=data['label_distribution'],
    )
