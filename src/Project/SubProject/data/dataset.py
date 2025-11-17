"""
PATCH 02: ReDSM5 → NLI Data Pipeline

This patch provides data loading and NLI conversion for ReDSM5 dataset.

Converts:
    ReDSM5 (post, sentence, symptom, status)
    → NLI (premise, hypothesis, label)

Where:
    - premise = sentence_text (from redsm5_annotations.csv)
    - hypothesis = DSM-5 criterion text (from MDD_Criteira.json)
    - label = 1 if status==1 (sentence matches symptom), else 0

Usage:
    cp PATCH_02_data_pipeline.py src/Project/SubProject/data/dataset.py
"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DSM5CriteriaMapping:
    """Maps DSM-5 symptom names to criterion text."""

    def __init__(self, criteria_json_path: str = "data/DSM5/MDD_Criteira.json"):
        """
        Load DSM-5 criteria from JSON file.

        Expected format:
        {
            "diagnosis": "Major Depressive Disorder",
            "criteria": [
                {"id": "A.1", "text": "Depressed mood..."},
                ...
            ]
        }
        """
        with open(criteria_json_path, 'r') as f:
            data = json.load(f)

        # Build mapping from criterion ID to text
        self.id_to_text = {
            item['id']: item['text']
            for item in data['criteria']
        }

        # Build mapping from symptom name (in annotations) to criterion text
        # Based on the ReDSM5 annotation format
        self.symptom_to_criterion = {
            'DEPRESSED_MOOD': self.id_to_text.get('A.1', ''),
            'ANHEDONIA': self.id_to_text.get('A.2', ''),
            'WEIGHT_APPETITE': self.id_to_text.get('A.3', ''),
            'SLEEP_ISSUES': self.id_to_text.get('A.4', ''),
            'PSYCHOMOTOR': self.id_to_text.get('A.5', ''),
            'FATIGUE': self.id_to_text.get('A.6', ''),
            'WORTHLESSNESS': self.id_to_text.get('A.7', ''),
            'COGNITIVE_ISSUES': self.id_to_text.get('A.8', ''),
            'SUICIDAL': self.id_to_text.get('A.9', ''),
        }

    def get_criterion_text(self, symptom_name: str) -> str:
        """Get DSM-5 criterion text for a given symptom name."""
        return self.symptom_to_criterion.get(symptom_name, "")

    def get_all_symptoms(self) -> List[str]:
        """Get all symptom names."""
        return list(self.symptom_to_criterion.keys())

    def get_all_criteria(self) -> Dict[str, str]:
        """Get all 9 DSM-5 criteria as {symptom_name: criterion_text}."""
        return self.symptom_to_criterion.copy()


class ReDSM5toNLIConverter:
    """Converts ReDSM5 dataset to NLI format."""

    def __init__(
        self,
        posts_csv: str = "data/redsm5/redsm5_posts.csv",
        annotations_csv: str = "data/redsm5/redsm5_annotations.csv",
        criteria_json: str = "data/DSM5/MDD_Criteira.json",
    ):
        """
        Initialize converter with data file paths.

        Args:
            posts_csv: Path to posts CSV (post_id, text)
            annotations_csv: Path to annotations CSV (post_id, sentence_id,
                sentence_text, DSM5_symptom, status, explanation)
            criteria_json: Path to DSM-5 criteria JSON
        """
        self.posts_csv = posts_csv
        self.annotations_csv = annotations_csv

        # Load DSM-5 criteria mapping
        self.criteria_map = DSM5CriteriaMapping(criteria_json)

        logger.info(f"Loading data from {annotations_csv}")

    def load_and_convert(
        self,
        include_negatives: bool = True,
        negative_sampling_ratio: float = 1.0,
        exhaustive_pairing: bool = True,
    ) -> pd.DataFrame:
        """
        Load ReDSM5 data and convert to NLI format.

        Args:
            include_negatives: Whether to include negative examples (status=0)
            negative_sampling_ratio: Ratio of negatives to positives (if < 1.0,
                randomly sample negatives)
            exhaustive_pairing: If True, create ALL (sentence, criterion) pairs using
                Cartesian product. If False, only use pairs from annotations.csv.

        Returns:
            DataFrame with columns:
                - premise (sentence_text)
                - hypothesis (DSM-5 criterion text)
                - label (1=entailment, 0=neutral/contradiction)
                - post_id (for grouping in cross-validation)
                - sentence_id (for tracking)
                - symptom (original symptom name)
        """
        # Load annotations
        df = pd.read_csv(self.annotations_csv)

        logger.info(f"Loaded {len(df)} annotations")
        logger.info(f"Unique posts: {df['post_id'].nunique()}")
        logger.info(f"Unique sentences: {df['sentence_text'].nunique()}")
        logger.info(f"Unique symptoms: {df['DSM5_symptom'].nunique()}")

        if exhaustive_pairing:
            # EXHAUSTIVE PAIRING APPROACH (Cartesian Product)
            # Create ALL (sentence, criterion) pairs from posts × criteria
            # Use annotations.csv as ground truth lookup
            logger.info("Using exhaustive pairing: ALL sentences × ALL 9 criteria")

            # Get all unique sentences with metadata
            sentence_info = df[['post_id', 'sentence_id', 'sentence_text']].drop_duplicates()
            logger.info(f"Found {len(sentence_info)} unique sentences")

            # Get all 9 DSM-5 criteria
            all_criteria = self.criteria_map.get_all_criteria()
            logger.info(f"Found {len(all_criteria)} DSM-5 criteria")

            # Create ground truth lookup: (sentence_text, symptom) → status
            ground_truth = {}
            for _, row in df.iterrows():
                key = (row['sentence_text'], row['DSM5_symptom'])
                ground_truth[key] = row['status']

            logger.info(f"Built ground truth lookup with {len(ground_truth)} entries")

            # Create Cartesian product: all_sentences × all_criteria
            nli_data = []
            for _, sent_row in sentence_info.iterrows():
                sentence_text = sent_row['sentence_text']
                post_id = sent_row['post_id']
                sentence_id = sent_row['sentence_id']

                for symptom, criterion_text in all_criteria.items():
                    # Lookup ground truth
                    key = (sentence_text, symptom)
                    status = ground_truth.get(key, 0)  # Default to 0 if not annotated

                    # Convert status to binary label
                    # status=1 → label=1 (entailment)
                    # status=0 or not in annotations → label=0 (neutral)
                    label = 1 if status == 1 else 0

                    # Skip negatives if requested
                    if label == 0 and not include_negatives:
                        continue

                    nli_data.append({
                        'premise': sentence_text,
                        'hypothesis': criterion_text,
                        'label': label,
                        'post_id': post_id,
                        'sentence_id': sentence_id,
                        'symptom': symptom,
                    })

            logger.info(f"Created {len(nli_data)} NLI pairs via Cartesian product")
            logger.info(f"  Total possible pairs: {len(sentence_info)} × {len(all_criteria)} = {len(sentence_info) * len(all_criteria)}")

        else:
            # SPARSE PAIRING APPROACH (Original)
            # Only create pairs that exist in annotations.csv
            logger.info("Using sparse pairing: only annotated pairs from annotations.csv")

            nli_data = []

            for idx, row in df.iterrows():
                sentence_text = row['sentence_text']
                symptom = row['DSM5_symptom']
                status = row['status']
                post_id = row['post_id']
                sentence_id = row['sentence_id']

                # Get DSM-5 criterion text for this symptom
                criterion_text = self.criteria_map.get_criterion_text(symptom)

                if not criterion_text:
                    logger.warning(f"No criterion text for symptom: {symptom}")
                    continue

                # Convert status to binary label
                label = 1 if status == 1 else 0

                # Skip negatives if requested
                if label == 0 and not include_negatives:
                    continue

                nli_data.append({
                    'premise': sentence_text,
                    'hypothesis': criterion_text,
                    'label': label,
                    'post_id': post_id,
                    'sentence_id': sentence_id,
                    'symptom': symptom,
                })

        nli_df = pd.DataFrame(nli_data)

        # Sample negatives if ratio < 1.0
        if negative_sampling_ratio < 1.0 and include_negatives:
            pos = nli_df[nli_df['label'] == 1]
            neg = nli_df[nli_df['label'] == 0]

            n_pos = len(pos)
            n_neg_target = int(n_pos * negative_sampling_ratio)

            if n_neg_target < len(neg):
                neg = neg.sample(n=n_neg_target, random_state=42)

            nli_df = pd.concat([pos, neg]).reset_index(drop=True)

        logger.info(f"Created {len(nli_df)} NLI examples")
        logger.info(f"Positive examples: {(nli_df['label'] == 1).sum()}")
        logger.info(f"Negative examples: {(nli_df['label'] == 0).sum()}")

        return nli_df


class MentalHealthNLIDataset(Dataset):
    """PyTorch Dataset for Mental Health NLI task."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        premise_col: str = 'premise',
        hypothesis_col: str = 'hypothesis',
        label_col: str = 'label',
    ):
        """
        Initialize dataset.

        Args:
            dataframe: DataFrame with NLI data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            premise_col: Name of premise column
            hypothesis_col: Name of hypothesis column
            label_col: Name of label column
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.premise_col = premise_col
        self.hypothesis_col = hypothesis_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Returns:
            dict with keys: input_ids, attention_mask, labels
        """
        row = self.data.iloc[idx]

        premise = str(row[self.premise_col])
        hypothesis = str(row[self.hypothesis_col])
        label = int(row[self.label_col])

        # CRITICAL: Use paper-specified input format template
        # From CLAUDE.md: "post: {post}, criterion: {criterion} Does the post match the criterion description? Output yes or no"
        # This is the EXACT format required by the spec for MentalLLaMA NLI
        formatted_input = (
            f"post: {premise}, criterion: {hypothesis} "
            f"Does the post match the criterion description? Output yes or no"
        )

        # Tokenize the formatted input as a single sequence
        encoding = self.tokenizer(
            formatted_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def create_nli_dataloaders(
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders with hardware optimizations.

    Args:
        tokenizer: HuggingFace tokenizer
        train_df: Training data DataFrame
        val_df: Validation data DataFrame (optional)
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers (default: 4 for faster I/O)

    Returns:
        (train_loader, val_loader) tuple

    Note:
        - pin_memory=True is enabled for faster GPU transfer
        - num_workers=4 provides good I/O parallelism without overhead
    """
    train_dataset = MentalHealthNLIDataset(
        train_df,
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_df is not None:
        val_dataset = MentalHealthNLIDataset(
            val_df,
            tokenizer,
            max_length=max_length,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing ReDSM5 to NLI conversion...")

    # Convert data
    converter = ReDSM5toNLIConverter()
    nli_df = converter.load_and_convert(include_negatives=True)

    print(f"\nDataset statistics:")
    print(f"Total examples: {len(nli_df)}")
    print(f"Positive: {(nli_df['label'] == 1).sum()}")
    print(f"Negative: {(nli_df['label'] == 0).sum()}")
    print(f"\nClass distribution by symptom:")
    print(nli_df.groupby(['symptom', 'label']).size().unstack(fill_value=0))

    print("\nFirst example:")
    print(nli_df.iloc[0])

    # Test dataset
    print("\n\nTesting dataset creation...")
    from transformers import AutoTokenizer

    # Note: For actual use, load MentalLLaMA tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-7B")
    # For testing without downloading:
    print("(Skipping tokenizer test - requires model download)")
    print("✓ Data pipeline implementation complete")
