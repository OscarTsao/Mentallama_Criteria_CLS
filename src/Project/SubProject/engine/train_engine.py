"""
PATCH 03: Training Engine with Classification Loss

This patch provides a complete training loop using CrossEntropyLoss
(NOT LM loss) for supervised binary classification.

Key features:
- CrossEntropyLoss for classification (not next-token prediction)
- Proper optimizer setup
- Training/validation loops
- Metric tracking
- Early stopping

Usage:
    cp PATCH_03_train_engine.py src/Project/SubProject/engine/train_engine.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, List, Literal
import logging
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def create_experiment_dir(base_dir: str = "experiments", run_name: Optional[str] = None) -> Path:
    """
    Create timestamped experiment directory.

    Args:
        base_dir: Base experiments directory
        run_name: Optional run name (default: timestamp)

    Returns:
        Path to experiment directory

    Structure:
        experiments/
        └── YYYY-MM-DD_HH-MM-SS_{run_name}/
            ├── config.json
            ├── best_model.pt
            ├── metrics.json
            └── training_history.json
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        exp_name = f"{timestamp}_{run_name}"
    else:
        exp_name = timestamp

    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def get_optimal_precision() -> Literal["bf16", "fp16", "fp32"]:
    """
    Detect optimal precision for current hardware.

    Returns:
        "bf16" if GPU supports bfloat16 (A100, H100, etc.)
        "fp16" if GPU supports float16 (V100, T4, etc.)
        "fp32" if CPU or old GPU
    """
    if not torch.cuda.is_available():
        return "fp32"

    # Check for bf16 support (Ampere and newer: A100, RTX 30xx, etc.)
    if torch.cuda.is_bf16_supported():
        return "bf16"

    # Fallback to fp16 for older GPUs
    return "fp16"


class ClassificationTrainer:
    """
    Trainer for binary classification using CrossEntropyLoss.

    This is NOT a language modeling trainer - we use classification loss,
    not next-token prediction loss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 2e-5,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3,
        save_path: Optional[str] = None,
        precision: Optional[Literal["bf16", "fp16", "fp32"]] = None,
        experiment_dir: Optional[Path] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize trainer with mixed precision support and experiment tracking.

        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            optimizer: Optimizer (if None, creates AdamW)
            lr: Learning rate
            num_epochs: Number of epochs
            device: Device to train on
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model (DEPRECATED: use experiment_dir instead)
            precision: Mixed precision mode ("bf16", "fp16", "fp32", or None for auto-detect)
            experiment_dir: Directory to save experiment artifacts (checkpoint, config, metrics)
            config: Training configuration dict to save alongside checkpoint
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience

        # Experiment tracking
        self.experiment_dir = experiment_dir
        self.config = config or {}

        # Backward compatibility: if save_path provided but no experiment_dir, use save_path
        if save_path and experiment_dir is None:
            self.save_path = save_path
        elif experiment_dir:
            self.save_path = str(experiment_dir / "best_model.pt")
        else:
            self.save_path = None

        # Mixed precision configuration (from CLAUDE.md:124)
        # "Compute: gradient checkpointing, grad_accum=4, bf16 AMP when available"
        if precision is None:
            precision = get_optimal_precision()
        self.precision = precision

        # Configure mixed precision
        self.use_amp = precision in ["bf16", "fp16"] and device == "cuda"
        if self.use_amp:
            self.autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            # GradScaler only needed for fp16 (bf16 doesn't need it)
            self.scaler = GradScaler() if precision == "fp16" else None
            logger.info(f"Mixed precision training enabled: {precision}")
            if precision == "bf16":
                logger.info("  Using bfloat16 (no GradScaler needed)")
            else:
                logger.info("  Using float16 (with GradScaler)")
        else:
            self.autocast_dtype = None
            self.scaler = None
            logger.info(f"Training in {precision} (no mixed precision)")

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

        # Loss function: CrossEntropyLoss for classification
        # NOT LM loss!
        self.loss_fn = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
        }

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision (if enabled)
            if self.use_amp:
                with autocast(dtype=self.autocast_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    # Get loss
                    if isinstance(outputs, dict):
                        loss = outputs['loss']
                    else:
                        logits = outputs[1] if isinstance(outputs, tuple) else outputs
                        loss = self.loss_fn(logits, labels)

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
            else:
                # FP32 path (no autocast)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Get loss
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    logits = outputs[1] if isinstance(outputs, tuple) else outputs
                    loss = self.loss_fn(logits, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass (with or without scaler)
            if self.scaler is not None:
                # FP16 path with GradScaler
                self.scaler.scale(loss).backward()
            else:
                # BF16 or FP32 path (no scaler needed)
                loss.backward()

            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # FP16 path with GradScaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # BF16 or FP32 path
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': total_loss / num_batches})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            dataloader: Dataloader to evaluate on (defaults to self.val_loader)

        Returns:
            Dictionary of metrics
        """
        if dataloader is None:
            dataloader = self.val_loader

        if dataloader is None:
            return {}

        self.model.eval()

        all_preds = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision (if enabled)
            if self.use_amp:
                with autocast(dtype=self.autocast_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            # Get logits and loss
            if isinstance(outputs, dict):
                logits = outputs['logits']
                loss = outputs.get('loss', None)
            else:
                if isinstance(outputs, tuple):
                    loss, logits = outputs
                else:
                    logits = outputs
                    loss = self.loss_fn(logits, labels)

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average='binary',
            zero_division=0,
        )

        # ROC AUC (use probability of positive class)
        try:
            probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
            roc_auc = roc_auc_score(all_labels, probs[:, 1])
        except Exception:
            roc_auc = 0.0

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
        }

        return metrics

    def save_experiment_artifacts(self, epoch: int, val_metrics: Dict[str, float]):
        """
        Save best model checkpoint, config, and metrics to experiment directory.

        Saves:
            - best_model.pt: Model state dict
            - config.json: Training configuration
            - metrics.json: Best epoch metrics (loss, F1, accuracy, etc.)
            - training_history.json: Full training history

        Args:
            epoch: Current epoch number
            val_metrics: Validation metrics for this epoch
        """
        if self.experiment_dir is None:
            return

        # Save model checkpoint
        model_path = self.experiment_dir / "best_model.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Saved model checkpoint: {model_path}")

        # Save configuration
        config_path = self.experiment_dir / "config.json"
        config_to_save = {
            **self.config,  # User-provided config
            'training': {
                'lr': self.optimizer.param_groups[0]['lr'],
                'num_epochs': self.num_epochs,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_grad_norm': self.max_grad_norm,
                'early_stopping_patience': self.early_stopping_patience,
                'precision': self.precision,
                'device': str(self.device),
            },
            'model': {
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        logger.info(f"Saved configuration: {config_path}")

        # Save best epoch metrics
        metrics_path = self.experiment_dir / "metrics.json"
        metrics_to_save = {
            'epoch': epoch + 1,
            'best_val_f1': float(self.best_val_f1),
            'val_loss': float(val_metrics['loss']),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_precision': float(val_metrics['precision']),
            'val_recall': float(val_metrics['recall']),
            'val_roc_auc': float(val_metrics['roc_auc']),
            'confusion_matrix': val_metrics['confusion_matrix'].tolist(),
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")

        # Save full training history
        history_path = self.experiment_dir / "training_history.json"
        history_to_save = {
            k: [float(v) for v in vals] for k, vals in self.history.items()
        }
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        logger.info(f"Saved training history: {history_path}")

    def train(self) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"  Train loss: {train_loss:.4f}")

            # Evaluate
            if self.val_loader is not None:
                val_metrics = self.evaluate()

                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_recall'].append(val_metrics['recall'])

                logger.info(f"  Val loss: {val_metrics['loss']:.4f}")
                logger.info(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
                logger.info(f"  Val F1: {val_metrics['f1']:.4f}")
                logger.info(f"  Val precision: {val_metrics['precision']:.4f}")
                logger.info(f"  Val recall: {val_metrics['recall']:.4f}")
                logger.info(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
                logger.info(f"  Confusion matrix:\n{val_metrics['confusion_matrix']}")

                # Early stopping check (based on best F1 score)
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.epochs_without_improvement = 0

                    # Save all experiment artifacts (checkpoint, config, metrics, history)
                    if self.experiment_dir:
                        self.save_experiment_artifacts(epoch, val_metrics)
                    # Backward compatibility: also save to save_path if provided
                    elif self.save_path is not None:
                        torch.save(self.model.state_dict(), self.save_path)
                        logger.info(f"  Saved best model to {self.save_path}")

                    logger.info(f"  ✓ New best F1: {self.best_val_f1:.4f}")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(patience={self.early_stopping_patience})"
                    )
                    break

        logger.info("Training complete!")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")

        return self.history


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Training engine implementation complete.")
    print("\nUsage example:")
    print("""
    from PATCH_01_encoder_model import load_mentallama_for_nli
    from PATCH_02_data_pipeline import create_nli_dataloaders, ReDSM5toNLIConverter
    from PATCH_03_train_engine import ClassificationTrainer

    # Load model and tokenizer
    model, tokenizer = load_mentallama_for_nli(num_labels=2)

    # Load and convert data
    converter = ReDSM5toNLIConverter()
    nli_df = converter.load_and_convert()

    # Split train/val (simplified - use proper CV in production)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(nli_df, test_size=0.2, random_state=42)

    # Create dataloaders
    train_loader, val_loader = create_nli_dataloaders(
        tokenizer, train_df, val_df, batch_size=8
    )

    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=2e-5,
        num_epochs=10,
        save_path='best_model.pt'
    )

    # Train
    history = trainer.train()
    """)
