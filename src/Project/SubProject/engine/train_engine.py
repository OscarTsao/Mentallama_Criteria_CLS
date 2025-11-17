"""
Training Engine

Orchestrates cross-validation training with MLflow tracking.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import sys

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import mlflow

from Project.SubProject.data import MentalHealthDataset, create_folds
from Project.SubProject.models import MentallamClassifier, build_prompt
from Project.SubProject.engine.metrics import (
    compute_metrics,
    tune_threshold,
    MetricsTracker,
    compute_confusion_matrix,
)
from Project.SubProject.utils import get_logger, set_seed, configure_mlflow, mlflow_run

logger = get_logger(__name__)


class TrainingCollator:
    """Collate function for training"""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """Collate batch of samples"""
        prompts = []
        labels = []

        for item in batch:
            prompt = build_prompt(item['post'], item['criterion'])
            prompts.append(prompt)
            labels.append(item['label'])

        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]")
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate model"""
    model.eval()

    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs['loss']
        probs = torch.softmax(outputs['logits'], dim=-1)

        # Collect predictions
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

        total_loss += loss.item()
        num_batches += 1

    # Convert to arrays
    y_true = np.array(all_labels)
    y_proba = np.array(all_probs)

    # Compute metrics with default threshold
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0

    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_proba': y_proba,
    }


def train_fold(
    fold_idx: int,
    dataset: MentalHealthDataset,
    train_indices: list,
    val_indices: list,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """Train single fold"""
    logger.info(f"Training fold {fold_idx}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = MentallamClassifier(
        model_name=config.get('model_name', 'klyang/MentaLLaMA-chat-7B'),
        num_labels=2,
        use_peft=config.get('use_peft', True),
        gradient_checkpointing=config.get('gradient_checkpointing', True),
        device_map='auto' if torch.cuda.is_available() else None,
    )

    # Create datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create dataloaders
    collator = TrainingCollator(model.tokenizer, max_length=config.get('max_length', 512))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
    )

    # Training loop
    tracker = MetricsTracker()
    best_checkpoint_path = output_dir / f"fold_{fold_idx}_best.pt"
    best_f1 = 0.0

    num_epochs = config.get('num_epochs', 10)
    patience = config.get('patience', 20)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        eval_results = evaluate(model, val_loader, device)
        val_metrics = eval_results['metrics']

        # Log metrics
        logger.info(
            f"Fold {fold_idx} Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        # Track metrics
        tracker.update(epoch, {
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_roc_auc': val_metrics.get('roc_auc', 0.0),
        })

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, best_checkpoint_path)
            logger.info(f"Saved best checkpoint (F1={best_f1:.4f})")

        # Early stopping
        if tracker.should_stop(patience):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Threshold tuning on validation set
    logger.info("Tuning threshold on validation set...")
    eval_results = evaluate(model, val_loader, device)
    tuned_threshold, tuned_score, tuned_metrics = tune_threshold(
        eval_results['y_true'],
        eval_results['y_proba'],
        metric='f1',
    )

    # Return results
    return {
        'fold_index': fold_idx,
        'best_checkpoint': str(best_checkpoint_path),
        'tuned_threshold': tuned_threshold,
        'tuned_metrics': tuned_metrics,
        'training_summary': tracker.get_summary(),
    }


def main():
    """Main training orchestration"""
    parser = argparse.ArgumentParser(description='Train MentaLLaMA classifier')
    parser.add_argument('--data-dir', default='data/redsm5', help='RedSM5 data directory')
    parser.add_argument('--dsm5-dir', default='data/DSM5', help='DSM5 data directory')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment-name', default='mentallama-training', help='MLflow experiment name')
    parser.add_argument('--tracking-uri', default='sqlite:///mlflow.db', help='MLflow tracking URI')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Configure MLflow
    configure_mlflow(tracking_uri=args.tracking_uri, experiment=args.experiment_name)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = MentalHealthDataset(
        redsm5_path=args.data_dir,
        dsm5_path=args.dsm5_dir,
    )

    # Create folds
    logger.info(f"Creating {args.n_folds} folds...")
    folds = create_folds(
        dataset,
        n_folds=args.n_folds,
        seed=args.seed,
        output_dir=str(output_dir / 'folds'),
    )

    # Training configuration
    config = {
        'model_name': 'klyang/MentaLLaMA-chat-7B',
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': 512,
        'use_peft': True,
        'gradient_checkpointing': True,
        'patience': 20,
    }

    # Start parent MLflow run
    with mlflow_run('cv-training', tags={'stage': 'training'}, params=config):
        all_results = []

        # Train each fold
        for fold in folds:
            fold_results = train_fold(
                fold_idx=fold.fold_index,
                dataset=dataset,
                train_indices=fold.train_indices,
                val_indices=fold.val_indices,
                config=config,
                output_dir=output_dir,
            )

            all_results.append(fold_results)

            # Log fold results to MLflow
            mlflow.log_metrics({
                f'fold_{fold.fold_index}_f1': fold_results['tuned_metrics']['f1'],
                f'fold_{fold.fold_index}_threshold': fold_results['tuned_threshold'],
            })

        # Aggregate results
        avg_f1 = np.mean([r['tuned_metrics']['f1'] for r in all_results])
        logger.info(f"Average F1 across folds: {avg_f1:.4f}")

        # Save results
        results_file = output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'config': config,
                'folds': all_results,
                'average_f1': float(avg_f1),
            }, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
