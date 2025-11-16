"""
PATCH 05: Deterministic Inference Example

Example script demonstrating encoder-style inference with MentalLLaMA.

Shows:
1. Loading model and tokenizer
2. Preparing NLI input (premise, hypothesis)
3. Running inference
4. Interpreting results

Usage:
    python PATCH_05_inference_example.py
"""

from typing import Dict, List, Tuple
import json


# Mock implementation for demonstration (without downloading model)
class MockInferenceExample:
    """Mock inference example that works without downloading actual model."""

    def __init__(self):
        print("=" * 70)
        print("MentalLLaMA Encoder-Style NLI Inference Example")
        print("=" * 70)
        print()

    def load_dsm5_criteria(self) -> Dict[str, str]:
        """Load DSM-5 criteria texts."""
        # In real usage, load from data/DSM5/MDD_Criteira.json
        criteria = {
            'A.1': "Depressed mood most of the day, nearly every day",
            'A.2': "Markedly diminished interest or pleasure in activities",
            'A.4': "Insomnia or hypersomnia nearly every day",
            'A.6': "Fatigue or loss of energy nearly every day",
            'A.7': "Feelings of worthlessness or excessive guilt",
        }
        return criteria

    def prepare_nli_input(
        self,
        premise: str,
        hypothesis: str,
    ) -> str:
        """
        Prepare input in NLI format.

        Args:
            premise: The sentence/post text
            hypothesis: The DSM-5 criterion text

        Returns:
            Formatted input string
        """
        # For MentalLLaMA, we concatenate premise and hypothesis
        # The tokenizer will handle special tokens
        return f"{premise} {hypothesis}"

    def run_inference_example(self):
        """Run complete inference example."""
        print("1. Loading DSM-5 Criteria")
        print("-" * 70)
        criteria = self.load_dsm5_criteria()
        for criterion_id, text in list(criteria.items())[:3]:
            print(f"  {criterion_id}: {text[:60]}...")
        print()

        print("2. Example Social Media Posts")
        print("-" * 70)
        posts = [
            "I can't sleep at all anymore. Been awake for days.",
            "Feeling really happy and energized today!",
            "I feel so worthless and guilty about everything.",
        ]
        for i, post in enumerate(posts):
            print(f"  Post {i+1}: {post}")
        print()

        print("3. Creating NLI Pairs")
        print("-" * 70)
        print("Format: (premise=post, hypothesis=criterion)")
        print()

        # Example 1: Positive case (should match)
        premise1 = posts[0]
        hypothesis1 = criteria['A.4']  # Sleep issues
        print(f"Pair 1:")
        print(f"  Premise: '{premise1}'")
        print(f"  Hypothesis: '{hypothesis1}'")
        print(f"  Expected: ENTAILMENT (1) - Post mentions sleep problems")
        print()

        # Example 2: Negative case (should not match)
        premise2 = posts[1]
        hypothesis2 = criteria['A.7']  # Worthlessness
        print(f"Pair 2:")
        print(f"  Premise: '{premise2}'")
        print(f"  Hypothesis: '{hypothesis2}'")
        print(f"  Expected: NEUTRAL (0) - Post is positive, not about worthlessness")
        print()

        # Example 3: Positive case
        premise3 = posts[2]
        hypothesis3 = criteria['A.7']  # Worthlessness
        print(f"Pair 3:")
        print(f"  Premise: '{premise3}'")
        print(f"  Hypothesis: '{hypothesis3}'")
        print(f"  Expected: ENTAILMENT (1) - Post mentions worthlessness/guilt")
        print()

        print("4. Model Inference (Simulated)")
        print("-" * 70)
        print("NOTE: This is a mock example. For actual inference:")
        print()
        print("```python")
        print("from PATCH_01_encoder_model import load_mentallama_for_nli")
        print()
        print("# Load model and tokenizer")
        print("model, tokenizer = load_mentallama_for_nli(")
        print("    model_name='klyang/MentaLLaMA-chat-7B',")
        print("    num_labels=2,")
        print("    device='cuda'")
        print(")")
        print()
        print("# Prepare input")
        print("inputs = tokenizer(")
        print("    premise1,")
        print("    hypothesis1,")
        print("    max_length=512,")
        print("    padding='max_length',")
        print("    truncation='longest_first',")
        print("    return_tensors='pt'")
        print(").to('cuda')")
        print()
        print("# Inference")
        print("model.eval()")
        print("with torch.no_grad():")
        print("    outputs = model(")
        print("        input_ids=inputs['input_ids'],")
        print("        attention_mask=inputs['attention_mask']")
        print("    )")
        print("    logits = outputs['logits']")
        print("    probs = F.softmax(logits, dim=1)")
        print("    prediction = torch.argmax(logits, dim=1)")
        print()
        print("# Results")
        print("print(f'Logits: {logits}')")
        print("print(f'Probabilities: {probs}')")
        print("print(f'Prediction: {prediction.item()}')  # 0 or 1")
        print("print(f'Label: {\"ENTAILMENT\" if prediction.item() == 1 else \"NEUTRAL\"}')")
        print("```")
        print()

        print("5. Interpreting Results")
        print("-" * 70)
        print("Model outputs:")
        print("  - logits: Raw scores for each class [NEUTRAL, ENTAILMENT]")
        print("  - probabilities: Softmax of logits (sum to 1.0)")
        print("  - prediction: argmax of logits (0 or 1)")
        print()
        print("Label mapping:")
        print("  - 0 = NEUTRAL/CONTRADICTION (post does NOT match criterion)")
        print("  - 1 = ENTAILMENT (post DOES match criterion)")
        print()

        print("6. Batch Inference")
        print("-" * 70)
        print("For multiple pairs, use batch processing:")
        print()
        print("```python")
        print("# Prepare batch")
        print("premises = [post1, post2, post3]")
        print("hypotheses = [criterion1, criterion2, criterion3]")
        print()
        print("inputs = tokenizer(")
        print("    premises,")
        print("    hypotheses,")
        print("    max_length=512,")
        print("    padding=True,")
        print("    truncation='longest_first',")
        print("    return_tensors='pt'")
        print(").to('cuda')")
        print()
        print("# Batch inference")
        print("with torch.no_grad():")
        print("    outputs = model(**inputs)")
        print("    predictions = torch.argmax(outputs['logits'], dim=1)")
        print()
        print("# Results for each pair")
        print("for i, pred in enumerate(predictions):")
        print("    print(f'Pair {i+1}: {pred.item()}')")
        print("```")
        print()

        print("7. Deterministic Results")
        print("-" * 70)
        print("To ensure reproducible results:")
        print()
        print("```python")
        print("import torch")
        print("import numpy as np")
        print("import random")
        print()
        print("def set_seed(seed=42):")
        print("    random.seed(seed)")
        print("    np.random.seed(seed)")
        print("    torch.manual_seed(seed)")
        print("    torch.cuda.manual_seed_all(seed)")
        print("    torch.backends.cudnn.deterministic = True")
        print("    torch.backends.cudnn.benchmark = False")
        print()
        print("set_seed(42)")
        print("# Now run inference - results will be reproducible")
        print("```")
        print()

        print("=" * 70)
        print("Example Complete")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Apply patches 01-03 to implement full model")
        print("2. Install dependencies: pip install -e '.[dev]'")
        print("3. Run this script with actual model")
        print("4. Test on your own data")


class RealInferenceExample:
    """
    Real inference example (requires model download).
    Only runs if transformers is available.
    """

    @staticmethod
    def run():
        try:
            import torch
            import torch.nn.functional as F
            from PATCH_01_encoder_model import load_mentallama_for_nli

            print("Loading MentalLLaMA model...")
            print("(This may take a few minutes for first download)")

            model, tokenizer = load_mentallama_for_nli(
                model_name='klyang/MentaLLaMA-chat-7B',
                num_labels=2,
            )

            print("Model loaded successfully!")
            print()

            # Example inference
            premise = "I can't sleep at all anymore. Been awake for days."
            hypothesis = "Insomnia or hypersomnia nearly every day"

            print(f"Premise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print()

            # Tokenize
            inputs = tokenizer(
                premise,
                hypothesis,
                max_length=512,
                padding='max_length',
                truncation='longest_first',
                return_tensors='pt'
            )

            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1)

            print(f"Logits: {logits[0].cpu().numpy()}")
            print(f"Probabilities: {probs[0].cpu().numpy()}")
            print(f"Prediction: {prediction.item()}")
            print(f"Label: {'ENTAILMENT' if prediction.item() == 1 else 'NEUTRAL'}")

        except ImportError:
            print("Error: Required packages not installed")
            print("Run: pip install transformers torch")
        except Exception as e:
            print(f"Error during inference: {e}")
            print("Falling back to mock example...")
            MockInferenceExample().run_inference_example()


if __name__ == "__main__":
    # Use mock example (works without dependencies)
    # For real inference, uncomment RealInferenceExample.run()
    MockInferenceExample().run_inference_example()
