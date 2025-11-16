#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training Script for MentalLLaMA Encoder-Style NLI Classifier

Paper: "Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks"

Implements spec-compliant 5-fold StratifiedGroupKFold cross-validation:
- Grouped by post_id (prevents data leakage across folds)
- Stratified by label (maintains class balance)
- Paper-aligned hyperparameters

Usage:
    python scripts/train_5fold_cv.py --batch-size 8 --epochs 100
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.SubProject.models.model import load_mentallama_for_nli
from Project.SubProject.data.dataset import (
    ReDSM5toNLIConverter,
    create_nli_dataloaders
)
from Project.SubProject.engine.train_engine import ClassificationTrainer
from Project.SubProject.utils.seed import set_seed
from Project.SubProject.utils.terminal_viz import TrainingVisualizer, print_model_info
from sklearn.model_selection import StratifiedGroupKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MentalLLaMA Encoder-Style NLI with 5-Fold CV'
    )

    # Model args
    parser.add_argument(
        '--model-name',
        type=str,
        default='klyang/MentaLLaMA-chat-7B',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--num-labels',
        type=int,
        default=2,
        help='Number of classification labels'
    )

    # Data args
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )

    # Training args (paper-aligned defaults)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size'
    )
    parser.add_argument(
        '--grad-accum',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum training epochs (paper: up to 100)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (paper: 20)'
    )

    # Cross-validation args
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds (paper: 5)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='Train single fold (0-4), or None for all folds'
    )

    # Output args
    parser.add_argument(
        '--output-dir',
        type=str,
        default='cv_results',
        help='Directory to save CV results'
    )

    # Other args
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default=None,
        choices=['bf16', 'fp16', 'fp32', None],
        help='Mixed precision mode (None=auto-detect, bf16=A100/H100, fp16=V100/T4, fp32=CPU/debug)'
    )

    return parser.parse_args()


def create_cv_splits(nli_df: pd.DataFrame, n_folds: int, seed: int):
    """
    Create 5-fold StratifiedGroupKFold splits.

    Args:
        nli_df: DataFrame with columns [premise, hypothesis, label, post_id, ...]
        n_folds: Number of folds
        seed: Random seed

    Returns:
        list of (train_indices, val_indices) tuples
    """
    # Extract labels and groups
    labels = nli_df['label'].values
    groups = nli_df['post_id'].values

    # Initialize StratifiedGroupKFold
    # This ensures:
    # 1. All examples from same post_id are in same fold (no leakage)
    # 2. Class distribution is balanced across folds
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []
    for train_idx, val_idx in sgkf.split(nli_df, y=labels, groups=groups):
        splits.append((train_idx, val_idx))

    logger.info(f"Created {n_folds}-fold StratifiedGroupKFold splits")
    logger.info(f"Total examples: {len(nli_df)}")
    logger.info(f"Unique posts: {nli_df['post_id'].nunique()}")

    # Log split statistics
    for i, (train_idx, val_idx) in enumerate(splits):
        train_df = nli_df.iloc[train_idx]
        val_df = nli_df.iloc[val_idx]

        logger.info(f"\nFold {i}:")
        logger.info(f"  Train: {len(train_df)} examples, "
                   f"{train_df['post_id'].nunique()} posts, "
                   f"{(train_df['label'] == 1).sum()} positive")
        logger.info(f"  Val: {len(val_df)} examples, "
                   f"{val_df['post_id'].nunique()} posts, "
                   f"{(val_df['label'] == 1).sum()} positive")

        # Verify no post_id overlap
        train_posts = set(train_df['post_id'].unique())
        val_posts = set(val_df['post_id'].unique())
        overlap = train_posts & val_posts
        assert len(overlap) == 0, f"Fold {i}: Found {len(overlap)} overlapping posts!"

    return splits


def train_single_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args,
    viz: TrainingVisualizer,
):
    """Train a single fold."""
    viz.display_info(f"Training Fold {fold_idx + 1}/{args.n_folds}")

    # Load model and tokenizer (fresh for each fold)
    viz.display_info(f"Loading model: {args.model_name}")
    model, tokenizer = load_mentallama_for_nli(
        model_name=args.model_name,
        num_labels=args.num_labels,
        device=args.device
    )
    viz.display_success("Model loaded")

    # Create dataloaders
    viz.display_info("Creating dataloaders...")
    train_loader, val_loader = create_nli_dataloaders(
        tokenizer,
        train_df,
        val_df,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    viz.display_success(f"{len(train_loader)} train batches, {len(val_loader)} val batches")

    # Create output directory for this fold
    fold_output_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(fold_output_dir / "best_model.pt")

    # Create trainer
    viz.display_info("Creating trainer...")
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        gradient_accumulation_steps=args.grad_accum,
        early_stopping_patience=args.patience,
        save_path=save_path,
        precision=args.precision,
    )
    viz.display_success("Trainer initialized")

    # Train
    print()  # Spacing
    history = trainer.train()

    # Save training history
    history_path = fold_output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)

    # Display results
    viz.display_training_complete(
        best_f1=trainer.best_val_f1,
        total_epochs=len(history['train_loss']),
        save_path=save_path
    )

    # Return fold results
    return {
        'fold': fold_idx,
        'best_val_f1': trainer.best_val_f1,
        'best_val_accuracy': max(history.get('val_accuracy', [0])),
        'best_val_loss': min(history['val_loss']),
        'final_train_loss': history['train_loss'][-1],
        'total_epochs': len(history['train_loss']),
        'history': history,
    }


def main():
    args = parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create visualizer
    viz = TrainingVisualizer(use_rich=True, use_plots=True)

    # Display header
    viz.print_header()

    # Set seed
    viz.display_info(f"Setting random seed: {args.seed}")
    set_seed(args.seed)

    # Load and convert data
    viz.display_info("Loading ReDSM5 data...")
    converter = ReDSM5toNLIConverter(
        posts_csv=f"{args.data_dir}/redsm5/redsm5_posts.csv",
        annotations_csv=f"{args.data_dir}/redsm5/redsm5_annotations.csv",
        criteria_json=f"{args.data_dir}/DSM5/MDD_Criteira.json",
    )
    nli_df = converter.load_and_convert(include_negatives=True)

    # Display data statistics
    data_stats = {
        'Total Examples': len(nli_df),
        'Positive (Entailment)': int((nli_df['label'] == 1).sum()),
        'Negative (Neutral)': int((nli_df['label'] == 0).sum()),
        'Unique Posts': int(nli_df['post_id'].nunique()),
    }
    viz.display_data_stats(data_stats)

    # Create CV splits
    viz.display_info(f"Creating {args.n_folds}-fold StratifiedGroupKFold splits...")
    splits = create_cv_splits(nli_df, n_folds=args.n_folds, seed=args.seed)
    viz.display_success(f"{args.n_folds} folds created (grouped by post_id, stratified by label)")

    # Display training configuration
    training_config = {
        'Model': args.model_name,
        'Learning Rate': args.lr,
        'Batch Size': args.batch_size,
        'Gradient Accumulation': args.grad_accum,
        'Effective Batch Size': args.batch_size * args.grad_accum,
        'Max Epochs': args.epochs,
        'Early Stopping Patience': args.patience,
        'Device': args.device,
        'Precision': args.precision if args.precision else 'auto-detect',
        'Max Sequence Length': args.max_length,
        'CV Folds': args.n_folds,
        'Output Directory': args.output_dir,
    }
    viz.display_config(training_config)

    # Train folds
    fold_results = []

    if args.fold is not None:
        # Train single fold
        train_idx, val_idx = splits[args.fold]
        train_df = nli_df.iloc[train_idx]
        val_df = nli_df.iloc[val_idx]

        result = train_single_fold(args.fold, train_df, val_df, args, viz)
        fold_results.append(result)
    else:
        # Train all folds
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print("\n" + "=" * 70)
            viz.display_info(f"Starting Fold {fold_idx + 1}/{args.n_folds}")
            print("=" * 70)

            train_df = nli_df.iloc[train_idx]
            val_df = nli_df.iloc[val_idx]

            result = train_single_fold(fold_idx, train_df, val_df, args, viz)
            fold_results.append(result)

    # Aggregate results
    print("\n" + "=" * 70)
    viz.display_info("5-Fold Cross-Validation Results")
    print("=" * 70)

    f1_scores = [r['best_val_f1'] for r in fold_results]
    acc_scores = [r['best_val_accuracy'] for r in fold_results]

    cv_results = {
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'mean_accuracy': float(np.mean(acc_scores)),
        'std_accuracy': float(np.std(acc_scores)),
        'fold_results': fold_results,
    }

    # Display summary
    summary_stats = {
        'Mean Val F1': f"{cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}",
        'Mean Val Accuracy': f"{cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}",
        'Best Fold F1': f"{max(f1_scores):.4f}",
        'Worst Fold F1': f"{min(f1_scores):.4f}",
    }
    viz.display_data_stats(summary_stats)

    # Save aggregated results
    results_path = Path(args.output_dir) / "cv_results.json"
    with open(results_path, 'w') as f:
        # Remove history from serialization (too large)
        cv_results_serializable = {
            'mean_f1': cv_results['mean_f1'],
            'std_f1': cv_results['std_f1'],
            'mean_accuracy': cv_results['mean_accuracy'],
            'std_accuracy': cv_results['std_accuracy'],
            'fold_results': [
                {k: v for k, v in r.items() if k != 'history'}
                for r in cv_results['fold_results']
            ]
        }
        json.dump(cv_results_serializable, f, indent=2)

    viz.display_success(f"Results saved to {results_path}")

    print("\n" + "=" * 70)
    print("✓ 5-Fold Cross-Validation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
