#!/usr/bin/env python3
"""
Model Registration Script

Registers a trained model in the MLflow Model Registry with proper
metadata, signatures, and example inputs.

Usage:
    python scripts/register_model.py \
        --run-id <mlflow_run_id> \
        --model-name mentallama-criteria-cls \
        --stage Production
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import torch
from mlflow.models import infer_signature
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint_info(run_id: str) -> Dict[str, Any]:
    """
    Load checkpoint and metadata from MLflow run.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary containing checkpoint path, metrics, and metadata
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Extract metrics
    metrics = {
        'accuracy': run.data.metrics.get('val_accuracy', 0.0),
        'precision': run.data.metrics.get('val_precision', 0.0),
        'recall': run.data.metrics.get('val_recall', 0.0),
        'f1': run.data.metrics.get('val_f1', 0.0),
        'roc_auc': run.data.metrics.get('val_roc_auc', 0.0),
    }

    # Extract tags
    tags = run.data.tags
    fold_index = tags.get('fold_index', 'unknown')
    threshold = run.data.metrics.get('tuned_threshold', 0.5)

    # Get checkpoint artifact path
    artifact_uri = run.info.artifact_uri
    checkpoint_path = f"{artifact_uri}/checkpoints/best.pt"

    info = {
        'run_id': run_id,
        'fold_index': fold_index,
        'checkpoint_path': checkpoint_path,
        'metrics': metrics,
        'threshold': threshold,
        'tags': tags,
    }

    logger.info(f"Loaded checkpoint info from run {run_id}")
    logger.info(f"  Fold: {fold_index}")
    logger.info(f"  F1: {metrics['f1']:.3f}")
    logger.info(f"  Threshold: {threshold:.3f}")

    return info


def create_model_signature() -> mlflow.models.ModelSignature:
    """
    Create MLflow model signature defining input/output schema.

    Returns:
        Model signature
    """
    from mlflow.types.schema import Schema, ColSpec

    input_schema = Schema([
        ColSpec("string", "post"),
        ColSpec("string", "criterion"),
    ])

    output_schema = Schema([
        ColSpec("string", "prediction"),
        ColSpec("double", "probability"),
        ColSpec("double", "threshold"),
    ])

    signature = mlflow.models.ModelSignature(
        inputs=input_schema,
        outputs=output_schema
    )

    return signature


def create_example_input() -> Dict[str, Any]:
    """
    Create example input for model inference.

    Returns:
        Example input dictionary
    """
    return {
        "post": "I have trouble sleeping and feel tired all day",
        "criterion": "Insomnia or hypersomnia nearly every day"
    }


def register_model(
    run_id: str,
    model_name: str,
    stage: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Register model in MLflow Model Registry.

    Args:
        run_id: MLflow run ID containing the checkpoint
        model_name: Name for the registered model
        stage: Optional stage (Staging, Production, Archived)
        description: Optional model description
        tags: Optional additional tags

    Returns:
        Model version string
    """
    logger.info(f"Registering model from run {run_id}")

    # Load checkpoint info
    checkpoint_info = load_checkpoint_info(run_id)

    # Get MLflow client
    client = mlflow.tracking.MlflowClient()

    # Create model signature
    signature = create_model_signature()
    logger.info("Created model signature")

    # Create example input
    example_input = create_example_input()
    logger.info("Created example input")

    # Build model description
    if description is None:
        description = (
            f"MentaLLaMA Binary Classifier for DSM-5 Criteria Matching\n\n"
            f"Metrics:\n"
            f"  - F1: {checkpoint_info['metrics']['f1']:.3f}\n"
            f"  - Accuracy: {checkpoint_info['metrics']['accuracy']:.3f}\n"
            f"  - Precision: {checkpoint_info['metrics']['precision']:.3f}\n"
            f"  - Recall: {checkpoint_info['metrics']['recall']:.3f}\n"
            f"  - ROC AUC: {checkpoint_info['metrics']['roc_auc']:.3f}\n\n"
            f"Tuned threshold: {checkpoint_info['threshold']:.3f}\n"
            f"Fold: {checkpoint_info['fold_index']}\n"
            f"Source run: {run_id}"
        )

    # Build tags
    model_tags = {
        'task': 'binary_classification',
        'domain': 'mental_health',
        'base_model': 'klyang/MentaLLaMA-chat-7B',
        'peft_method': 'dora',
        'fold_index': str(checkpoint_info['fold_index']),
        'f1_score': f"{checkpoint_info['metrics']['f1']:.3f}",
        'threshold': f"{checkpoint_info['threshold']:.3f}",
        'source_run_id': run_id,
    }

    if tags:
        model_tags.update(tags)

    # Register model
    logger.info(f"Registering model '{model_name}'...")

    # First, log the model to the run if not already logged
    artifact_path = "model"

    # Register the model
    model_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        # Check if model already exists
        client.get_registered_model(model_name)
        logger.info(f"Model '{model_name}' already exists, creating new version")
    except mlflow.exceptions.RestException:
        # Model doesn't exist, will be created automatically
        logger.info(f"Creating new model '{model_name}'")

    # Create model version
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags=model_tags
    )

    version_number = model_version.version
    logger.info(f"Model registered as version {version_number}")

    # Update model version description
    client.update_model_version(
        name=model_name,
        version=version_number,
        description=description
    )

    # Transition to stage if specified
    if stage:
        logger.info(f"Transitioning model to {stage} stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Model transitioned to {stage}")

    # Save model info to file
    model_info = {
        'model_name': model_name,
        'version': version_number,
        'stage': stage,
        'run_id': run_id,
        'metrics': checkpoint_info['metrics'],
        'threshold': checkpoint_info['threshold'],
        'fold_index': checkpoint_info['fold_index'],
    }

    output_file = Path("model_registry_info.json")
    with open(output_file, 'w') as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Model info saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("Model Registration Summary")
    print("="*60)
    print(f"Model Name: {model_name}")
    print(f"Version: {version_number}")
    print(f"Stage: {stage or 'None'}")
    print(f"Run ID: {run_id}")
    print(f"F1 Score: {checkpoint_info['metrics']['f1']:.3f}")
    print(f"Threshold: {checkpoint_info['threshold']:.3f}")
    print(f"Model URI: models:/{model_name}/{version_number}")
    print("="*60)

    return f"{version_number}"


def main():
    parser = argparse.ArgumentParser(
        description="Register trained model in MLflow Model Registry"
    )
    parser.add_argument(
        '--run-id',
        required=True,
        help='MLflow run ID containing the checkpoint'
    )
    parser.add_argument(
        '--model-name',
        default='mentallama-criteria-cls',
        help='Name for the registered model (default: mentallama-criteria-cls)'
    )
    parser.add_argument(
        '--stage',
        choices=['Staging', 'Production', 'Archived', None],
        default=None,
        help='Stage to transition the model to'
    )
    parser.add_argument(
        '--description',
        default=None,
        help='Optional model description'
    )
    parser.add_argument(
        '--tracking-uri',
        default='sqlite:///mlflow.db',
        help='MLflow tracking URI (default: sqlite:///mlflow.db)'
    )
    parser.add_argument(
        '--tag',
        action='append',
        help='Additional tag in format key=value (can be repeated)'
    )

    args = parser.parse_args()

    # Set tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)
    logger.info(f"MLflow tracking URI: {args.tracking_uri}")

    # Parse tags
    tags = {}
    if args.tag:
        for tag in args.tag:
            key, value = tag.split('=', 1)
            tags[key] = value

    # Register model
    try:
        version = register_model(
            run_id=args.run_id,
            model_name=args.model_name,
            stage=args.stage,
            description=args.description,
            tags=tags
        )
        logger.info(f"Registration complete: {args.model_name} v{version}")
    except Exception as e:
        logger.error(f"Failed to register model: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
