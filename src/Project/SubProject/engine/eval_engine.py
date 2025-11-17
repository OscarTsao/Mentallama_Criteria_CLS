"""
Evaluation Engine

Provides inference and evaluation capabilities.
"""

import argparse
import json
from pathlib import Path

import torch

from Project.SubProject.models import MentallamClassifier, build_prompt
from Project.SubProject.utils import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Inference engine for trained models"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
    ):
        """
        Initialize inference engine

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model from {checkpoint_path}")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> MentallamClassifier:
        """Load model from checkpoint"""
        # For now, create a new model and load state dict
        # In production, you'd load the full PEFT model
        model = MentallamClassifier(
            model_name='klyang/MentaLLaMA-chat-7B',
            num_labels=2,
            use_peft=True,
            gradient_checkpointing=False,
            device_map='auto' if self.device.type == 'cuda' else None,
        )

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    @torch.no_grad()
    def predict(
        self,
        post: str,
        criterion: str,
        threshold: float = 0.5,
    ) -> dict:
        """
        Make prediction for a single (post, criterion) pair

        Args:
            post: Social media post text
            criterion: DSM-5 criterion text
            threshold: Classification threshold

        Returns:
            Dictionary with prediction results
        """
        # Build prompt
        prompt = build_prompt(post, criterion)

        # Tokenize
        encoded = self.model.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get predictions
        outputs = self.model(input_ids, attention_mask)
        probs = torch.softmax(outputs['logits'], dim=-1)

        # Get probability for positive class
        prob_positive = probs[0, 1].item()
        prediction = 1 if prob_positive >= threshold else 0

        return {
            'prediction': 'yes' if prediction == 1 else 'no',
            'probability': prob_positive,
            'threshold': threshold,
            'label': prediction,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        pairs: list[dict[str, str]],
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> list[dict]:
        """
        Make predictions for multiple (post, criterion) pairs

        Args:
            pairs: List of dictionaries with 'post' and 'criterion' keys
            threshold: Classification threshold
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]

            # Build prompts
            prompts = [build_prompt(p['post'], p['criterion']) for p in batch_pairs]

            # Tokenize
            encoded = self.model.tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt',
            )

            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Get predictions
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs['logits'], dim=-1)

            # Process results
            for j, prob in enumerate(probs[:, 1].cpu().numpy()):
                prediction = 1 if prob >= threshold else 0
                results.append({
                    'prediction': 'yes' if prediction == 1 else 'no',
                    'probability': float(prob),
                    'threshold': threshold,
                    'label': prediction,
                })

        return results


def main():
    """Main inference CLI"""
    parser = argparse.ArgumentParser(description='Run inference with trained model')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Single prediction
    predict_parser = subparsers.add_parser('predict', help='Predict single pair')
    predict_parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    predict_parser.add_argument('--post', required=True, help='Post text')
    predict_parser.add_argument('--criterion', required=True, help='Criterion text')
    predict_parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')

    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Predict batch from JSONL')
    batch_parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    batch_parser.add_argument('--input', required=True, help='Input JSONL file')
    batch_parser.add_argument('--output', required=True, help='Output JSONL file')
    batch_parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    batch_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')

    args = parser.parse_args()

    if args.command == 'predict':
        # Single prediction
        engine = InferenceEngine(args.checkpoint)

        result = engine.predict(
            post=args.post,
            criterion=args.criterion,
            threshold=args.threshold,
        )

        print(json.dumps(result, indent=2))

    elif args.command == 'batch':
        # Batch prediction
        engine = InferenceEngine(args.checkpoint)

        # Load input
        logger.info(f"Loading input from {args.input}")
        pairs = []
        with open(args.input) as f:
            for line in f:
                pairs.append(json.loads(line))

        # Run inference
        logger.info(f"Running inference on {len(pairs)} pairs...")
        results = engine.predict_batch(
            pairs,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )

        # Save output
        logger.info(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            for pair, result in zip(pairs, results):
                output = {**pair, **result}
                f.write(json.dumps(output) + '\n')

        logger.info(f"Saved {len(results)} predictions")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
