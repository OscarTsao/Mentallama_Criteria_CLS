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
from Project.SubProject.utils.terminal_viz import TrainingVisualizer, print_model_info
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
        default=100,
        help='Number of training epochs (paper: up to 100)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (paper: 20 epochs)'
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
    parser.add_argument(
        '--precision',
        type=str,
        default=None,
        choices=['bf16', 'fp16', 'fp32', None],
        help='Mixed precision mode (None=auto-detect, bf16=A100/H100, fp16=V100/T4, fp32=CPU/debug)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create visualizer
    viz = TrainingVisualizer(use_rich=True, use_plots=True)

    # Display header
    viz.print_header()

    # Set seed for reproducibility
    viz.display_info(f"Setting random seed: {args.seed}")
    set_seed(args.seed)

    # Load model and tokenizer
    viz.display_info(f"Loading model: {args.model_name}")
    model, tokenizer = load_mentallama_for_nli(
        model_name=args.model_name,
        num_labels=args.num_labels,
        device=args.device
    )
    viz.display_success(f"Model loaded successfully")

    # Display model info
    print_model_info(model)

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
    }
    viz.display_data_stats(data_stats)

    # Split train/validation
    viz.display_info(f"Splitting data (test_size={args.test_size})...")
    train_df, val_df = train_test_split(
        nli_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=nli_df['label']
    )

    # Display split statistics
    split_stats = {
        'Training Examples': len(train_df),
        'Validation Examples': len(val_df),
        'Train Positive': int((train_df['label'] == 1).sum()),
        'Train Negative': int((train_df['label'] == 0).sum()),
        'Val Positive': int((val_df['label'] == 1).sum()),
        'Val Negative': int((val_df['label'] == 0).sum()),
    }
    viz.display_data_stats(split_stats)

    # Create dataloaders
    viz.display_info("Creating dataloaders...")
    train_loader, val_loader = create_nli_dataloaders(
        tokenizer,
        train_df,
        val_df,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    viz.display_success(f"Dataloaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Display training configuration
    training_config = {
        'Learning Rate': args.lr,
        'Batch Size': args.batch_size,
        'Gradient Accumulation': args.grad_accum,
        'Effective Batch Size': args.batch_size * args.grad_accum,
        'Max Epochs': args.epochs,
        'Early Stopping Patience': args.patience,
        'Device': args.device,
        'Precision': args.precision if args.precision else 'auto-detect',
        'Max Sequence Length': args.max_length,
        'Save Path': args.save_path,
    }
    viz.display_config(training_config)

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
        save_path=args.save_path,
        precision=args.precision,
    )
    viz.display_success("Trainer initialized")

    # Train
    print()  # Add spacing before training starts

    history = trainer.train()

    # Display training curves
    viz.plot_training_curves(history)

    # Display final results
    viz.display_training_complete(
        best_f1=trainer.best_val_f1,
        total_epochs=len(history['train_loss']),
        save_path=args.save_path
    )

    # Display final metrics
    final_metrics = {
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
        'val_accuracy': history['val_accuracy'][-1],
        'val_f1': history['val_f1'][-1] if 'val_f1' in history else trainer.best_val_f1,
    }
    viz.display_epoch_metrics(epoch=len(history['train_loss']) - 1, metrics=final_metrics)


if __name__ == "__main__":
    main()
