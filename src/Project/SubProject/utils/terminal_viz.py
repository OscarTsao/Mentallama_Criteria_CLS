"""
Terminal Visualization Utilities for MentalLLaMA Training

Provides rich terminal output with:
- Live training metrics
- Progress bars
- Terminal plots
- Colored logging
- Summary tables

Usage:
    from Project.SubProject.utils.terminal_viz import TrainingVisualizer

    viz = TrainingVisualizer()
    viz.display_training_start(config)

    for epoch in range(epochs):
        with viz.epoch_progress(total_steps) as progress:
            for step in steps:
                metrics = train_step()
                progress.update(metrics)

        viz.display_epoch_summary(epoch, metrics)

    viz.display_training_complete(history)
"""

import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")

try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    print("Warning: 'plotext' library not available. Install with: pip install plotext")


class TrainingVisualizer:
    """Rich terminal visualization for training."""

    def __init__(self, use_rich: bool = True, use_plots: bool = True):
        """
        Initialize visualizer.

        Args:
            use_rich: Use rich library for formatting
            use_plots: Use plotext for terminal plots
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.use_plots = use_plots and PLOTEXT_AVAILABLE

        if self.use_rich:
            self.console = Console()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': [],
        }

    def print_header(self):
        """Print training header."""
        if self.use_rich:
            header = Panel(
                "[bold cyan]MentalLLaMA Encoder-Style NLI Classifier[/bold cyan]\n"
                "[dim]Paper: Adapting Decoder-Based LMs for Encoder Tasks[/dim]\n"
                "[dim]Model: klyang/MentaLLaMA-chat-7B[/dim]",
                border_style="cyan",
                box=box.DOUBLE,
            )
            self.console.print(header)
        else:
            print("=" * 70)
            print("MentalLLaMA Encoder-Style NLI Classifier")
            print("Paper: Adapting Decoder-Based LMs for Encoder Tasks")
            print("Model: klyang/MentaLLaMA-chat-7B")
            print("=" * 70)

    def display_config(self, config: Dict[str, Any]):
        """Display training configuration."""
        if self.use_rich:
            table = Table(title="Training Configuration", box=box.ROUNDED)
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")

            for key, value in config.items():
                table.add_row(str(key), str(value))

            self.console.print(table)
        else:
            print("\nTraining Configuration:")
            print("-" * 70)
            for key, value in config.items():
                print(f"  {key:30s}: {value}")
            print("-" * 70)

    def display_data_stats(self, stats: Dict[str, int]):
        """Display dataset statistics."""
        if self.use_rich:
            table = Table(title="Dataset Statistics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="yellow", justify="right")

            for key, value in stats.items():
                table.add_row(key, f"{value:,}")

            self.console.print(table)
        else:
            print("\nDataset Statistics:")
            print("-" * 70)
            for key, value in stats.items():
                print(f"  {key:30s}: {value:,}")
            print("-" * 70)

    def display_epoch_start(self, epoch: int, total_epochs: int):
        """Display epoch start."""
        if self.use_rich:
            self.console.print(
                f"\n[bold magenta]Epoch {epoch + 1}/{total_epochs}[/bold magenta]",
                style="bold"
            )
        else:
            print(f"\nEpoch {epoch + 1}/{total_epochs}")
            print("-" * 70)

    def create_progress_bar(self, total: int, desc: str = "Training"):
        """Create a progress bar for training."""
        if self.use_rich:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
        else:
            # Fallback to simple counter
            class SimpleProgress:
                def __init__(self, total, desc):
                    self.total = total
                    self.desc = desc
                    self.current = 0

                def update(self, advance=1):
                    self.current += advance
                    pct = (self.current / self.total) * 100
                    print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.1f}%)", end="")

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    print()  # New line

            return SimpleProgress(total, desc)

    def display_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Display epoch metrics."""
        if self.use_rich:
            table = Table(box=box.SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")

            metric_order = ['train_loss', 'val_loss', 'val_accuracy',
                          'val_precision', 'val_recall', 'val_f1', 'val_roc_auc']

            for key in metric_order:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, float):
                        table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")

            self.console.print(table)
        else:
            print("\nEpoch Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.4f}")

    def display_confusion_matrix(self, cm):
        """Display confusion matrix."""
        if self.use_rich:
            table = Table(title="Confusion Matrix", box=box.ROUNDED)
            table.add_column("", style="cyan")
            table.add_column("Pred: 0", style="yellow", justify="center")
            table.add_column("Pred: 1", style="yellow", justify="center")

            table.add_row("True: 0", str(cm[0, 0]), str(cm[0, 1]))
            table.add_row("True: 1", str(cm[1, 0]), str(cm[1, 1]))

            self.console.print(table)
        else:
            print("\nConfusion Matrix:")
            print(f"              Pred: 0    Pred: 1")
            print(f"  True: 0  {cm[0, 0]:8d}  {cm[0, 1]:8d}")
            print(f"  True: 1  {cm[1, 0]:8d}  {cm[1, 1]:8d}")

    def plot_training_curves(self, history: Dict[str, List[float]]):
        """Plot training curves in terminal."""
        if not self.use_plots:
            return

        epochs = list(range(1, len(history.get('train_loss', [])) + 1))

        if not epochs:
            return

        # Plot loss curves
        if self.use_rich:
            self.console.print("\n[bold cyan]Training Curves[/bold cyan]")
        else:
            print("\nTraining Curves:")

        plt.clf()
        plt.title("Loss Curves")

        if 'train_loss' in history and history['train_loss']:
            plt.plot(epochs, history['train_loss'], label="Train Loss", marker="braille")

        if 'val_loss' in history and history['val_loss']:
            plt.plot(epochs, history['val_loss'], label="Val Loss", marker="braille")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.theme("dark")
        plt.show()

        # Plot F1 score
        if 'val_f1' in history and history['val_f1']:
            plt.clf()
            plt.title("Validation F1 Score")
            plt.plot(epochs, history['val_f1'], label="Val F1", marker="braille")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.ylim(0, 1)
            plt.theme("dark")
            plt.show()

    def display_training_complete(
        self,
        best_f1: float,
        total_epochs: int,
        save_path: Optional[str] = None
    ):
        """Display training completion summary."""
        if self.use_rich:
            summary = Panel(
                f"[bold green]✓ Training Complete![/bold green]\n\n"
                f"[cyan]Total Epochs:[/cyan] {total_epochs}\n"
                f"[cyan]Best Val F1:[/cyan] {best_f1:.4f}\n" +
                (f"[cyan]Model Saved:[/cyan] {save_path}" if save_path else ""),
                border_style="green",
                box=box.DOUBLE,
            )
            self.console.print(summary)
        else:
            print("\n" + "=" * 70)
            print("✓ Training Complete!")
            print("=" * 70)
            print(f"Total Epochs: {total_epochs}")
            print(f"Best Val F1: {best_f1:.4f}")
            if save_path:
                print(f"Model Saved: {save_path}")
            print("=" * 70)

    def display_error(self, message: str):
        """Display error message."""
        if self.use_rich:
            self.console.print(f"[bold red]Error:[/bold red] {message}")
        else:
            print(f"Error: {message}")

    def display_warning(self, message: str):
        """Display warning message."""
        if self.use_rich:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
        else:
            print(f"Warning: {message}")

    def display_info(self, message: str):
        """Display info message."""
        if self.use_rich:
            self.console.print(f"[bold blue]Info:[/bold blue] {message}")
        else:
            print(f"Info: {message}")

    def display_success(self, message: str):
        """Display success message."""
        if self.use_rich:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")


class LiveTrainingDisplay:
    """Live display of training metrics."""

    def __init__(self):
        """Initialize live display."""
        self.use_rich = RICH_AVAILABLE

        if self.use_rich:
            self.console = Console()
            self.layout = Layout()

            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="metrics", size=10),
                Layout(name="progress", size=3),
            )

    def update(self, epoch: int, step: int, total_steps: int, metrics: Dict[str, float]):
        """Update live display."""
        if not self.use_rich:
            print(f"\rEpoch {epoch} - Step {step}/{total_steps} - Loss: {metrics.get('loss', 0):.4f}", end="")
            return

        # Header
        header_text = Text(f"Epoch {epoch} - Training", style="bold cyan")
        self.layout["header"].update(Panel(header_text, border_style="cyan"))

        # Metrics table
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")

        self.layout["metrics"].update(Panel(table, border_style="green"))

        # Progress
        progress = f"{step}/{total_steps} ({(step/total_steps)*100:.1f}%)"
        self.layout["progress"].update(Panel(progress, border_style="yellow"))

        self.console.print(self.layout)


# Convenience functions
def create_visualizer(use_rich: bool = True, use_plots: bool = True) -> TrainingVisualizer:
    """Create a training visualizer."""
    return TrainingVisualizer(use_rich=use_rich, use_plots=use_plots)


def print_model_info(model):
    """Print model information."""
    viz = TrainingVisualizer()

    if RICH_AVAILABLE:
        table = Table(title="Model Information", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="green")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Model Type", model.__class__.__name__)

        if hasattr(model, 'config'):
            table.add_row("Hidden Size", str(model.config.hidden_size))
            table.add_row("Num Labels", str(getattr(model, 'num_labels', 'N/A')))

        viz.console.print(table)
    else:
        print("\nModel Information:")
        print("-" * 70)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Type: {model.__class__.__name__}")
        print("-" * 70)


if __name__ == "__main__":
    # Demo
    print("Terminal Visualization Demo\n")

    viz = create_visualizer()

    viz.print_header()

    config = {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'epochs': 10,
        'device': 'cuda',
    }
    viz.display_config(config)

    stats = {
        'Total Examples': 13000,
        'Train Examples': 10400,
        'Val Examples': 2600,
        'Positive Examples': 3250,
        'Negative Examples': 9750,
    }
    viz.display_data_stats(stats)

    # Simulate training
    for epoch in range(2):
        viz.display_epoch_start(epoch, 2)

        metrics = {
            'train_loss': 0.35 - epoch * 0.05,
            'val_loss': 0.40 - epoch * 0.05,
            'val_accuracy': 0.80 + epoch * 0.02,
            'val_precision': 0.78 + epoch * 0.02,
            'val_recall': 0.82 + epoch * 0.02,
            'val_f1': 0.80 + epoch * 0.02,
            'val_roc_auc': 0.85 + epoch * 0.02,
        }

        viz.display_epoch_metrics(epoch, metrics)

    viz.display_training_complete(best_f1=0.82, total_epochs=2, save_path='best_model.pt')

    print("\n✓ Demo complete")
