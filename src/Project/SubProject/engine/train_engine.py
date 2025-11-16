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
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize trainer.

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
            save_path: Path to save best model
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.save_path = save_path

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

            # Forward pass
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

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
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

            # Forward pass
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

                # Early stopping check
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.epochs_without_improvement = 0

                    # Save best model
                    if self.save_path is not None:
                        torch.save(self.model.state_dict(), self.save_path)
                        logger.info(f"  Saved best model to {self.save_path}")
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
