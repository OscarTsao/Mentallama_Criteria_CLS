#!/usr/bin/env python3
"""
Training script for MentalLLaMA Encoder-Style NLI Classifier

Paper: "Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks"

Usage:
    python scripts/train.py --batch-size 8 --epochs 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.SubProject.models.model import load_mentallama_for_nli
from Project.SubProject.data.dataset import (
    ReDSM5toNLIConverter,
    create_nli_dataloaders
)
from Project.SubProject.engine.train_engine import ClassificationTrainer
from Project.SubProject.utils.seed import set_seed
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MentalLLaMA Encoder-Style NLI Classifier'
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
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )

    # Training args
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
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='best_model.pt',
        help='Path to save best model'
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

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("MentalLLaMA Encoder-Style NLI Classifier Training")
    logger.info("Paper: Adapting Decoder-Based LMs for Encoder Tasks")
    logger.info("=" * 70)

    # Set seed for reproducibility
    logger.info(f"Setting random seed: {args.seed}")
    set_seed(args.seed)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = load_mentallama_for_nli(
        model_name=args.model_name,
        num_labels=args.num_labels,
        device=args.device
    )
    logger.info(f"✓ Model loaded successfully")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Num labels: {args.num_labels}")

    # Load and convert data
    logger.info("Loading ReDSM5 data...")
    converter = ReDSM5toNLIConverter(
        posts_csv=f"{args.data_dir}/redsm5/redsm5_posts.csv",
        annotations_csv=f"{args.data_dir}/redsm5/redsm5_annotations.csv",
        criteria_json=f"{args.data_dir}/DSM5/MDD_Criteira.json",
    )
    nli_df = converter.load_and_convert(include_negatives=True)

    logger.info(f"✓ Loaded {len(nli_df)} NLI examples")
    logger.info(f"  Positive (entailment): {(nli_df['label'] == 1).sum()}")
    logger.info(f"  Negative (neutral): {(nli_df['label'] == 0).sum()}")

    # Split train/validation
    logger.info(f"Splitting data (test_size={args.test_size})...")
    train_df, val_df = train_test_split(
        nli_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=nli_df['label']
    )
    logger.info(f"✓ Train: {len(train_df)}, Val: {len(val_df)}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_nli_dataloaders(
        tokenizer,
        train_df,
        val_df,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    logger.info(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        gradient_accumulation_steps=args.grad_accum,
        early_stopping_patience=args.patience,
        save_path=args.save_path,
    )

    logger.info("Training configuration:")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.grad_accum}")
    logger.info(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info(f"  Max epochs: {args.epochs}")
    logger.info(f"  Early stopping patience: {args.patience}")
    logger.info(f"  Save path: {args.save_path}")

    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    history = trainer.train()

    # Final results
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)
    logger.info(f"Best validation F1: {trainer.best_val_f1:.4f}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    logger.info(f"Model saved to: {args.save_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
