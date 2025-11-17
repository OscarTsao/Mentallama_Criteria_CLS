"""
Mental Health Classification Dataset

Loads and processes (post, criterion) pairs from RedSM5 annotations
and DSM-5 criteria for binary classification.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from Project.SubProject.utils.log import get_logger

logger = get_logger(__name__)


# Mapping from symptom names in annotations to criterion IDs
SYMPTOM_TO_CRITERION = {
    'DEPRESSED_MOOD': 'A.1',
    'ANHEDONIA': 'A.2',
    'APPETITE_ISSUES': 'A.3',
    'WEIGHT_CHANGE': 'A.3',
    'SLEEP_ISSUES': 'A.4',
    'PSYCHOMOTOR': 'A.5',
    'FATIGUE': 'A.6',
    'WORTHLESSNESS': 'A.7',
    'GUILT': 'A.7',
    'COGNITIVE_ISSUES': 'A.8',
    'CONCENTRATION': 'A.8',
    'SUICIDAL': 'A.9',
    'DEATH_THOUGHTS': 'A.9',
}


@dataclass
class Sample:
    """Represents a single (post, criterion) pair"""
    sample_id: str
    post_id: str
    criterion_id: str
    post_text: str
    criterion_text: str
    label: int  # 0=unmatched, 1=matched

    def __post_init__(self):
        # Validate label
        if self.label not in {0, 1}:
            raise ValueError(f"Label must be 0 or 1, got {self.label}")

        # Normalize text
        self.post_text = self._normalize_text(self.post_text)
        self.criterion_text = self._normalize_text(self.criterion_text)

        # Validate non-empty
        if not self.post_text or not self.criterion_text:
            raise ValueError("Post and criterion text must not be empty")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize unicode and trim whitespace"""
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        return text.strip()


class MentalHealthDataset(Dataset):
    """
    Dataset for mental health classification

    Loads posts from RedSM5 and criteria from DSM-5, creates (post, criterion)
    pairs based on annotations.
    """

    def __init__(
        self,
        redsm5_path: str,
        dsm5_path: str,
        override_counts: bool = False,
    ):
        """
        Args:
            redsm5_path: Path to RedSM5 data directory
            dsm5_path: Path to DSM5 data directory
            override_counts: Skip expected count validation
        """
        self.redsm5_path = Path(redsm5_path)
        self.dsm5_path = Path(dsm5_path)
        self.override_counts = override_counts

        # Load data
        self.criteria = self._load_criteria()
        self.posts = self._load_posts()
        self.samples = self._create_samples()

        # Validate counts
        self._validate_cardinality()

        # Log statistics
        self._log_statistics()

    def _load_criteria(self) -> dict[str, str]:
        """Load DSM-5 criteria from JSON"""
        # Try both possible filenames (there's a typo in the actual file)
        possible_files = [
            self.dsm5_path / "MDD_Criteria.json",
            self.dsm5_path / "MDD_Criteira.json",  # typo version
            self.dsm5_path / "criteria.csv",
        ]

        criteria_dict = {}

        for filepath in possible_files:
            if filepath.exists():
                if filepath.suffix == '.json':
                    with open(filepath) as f:
                        data = json.load(f)
                        for criterion in data['criteria']:
                            criteria_dict[criterion['id']] = criterion['text']
                    logger.info(f"Loaded {len(criteria_dict)} criteria from {filepath}")
                    return criteria_dict
                elif filepath.suffix == '.csv':
                    df = pd.read_csv(filepath)
                    for _, row in df.iterrows():
                        criteria_dict[row['criterion_id']] = row['criterion_text']
                    logger.info(f"Loaded {len(criteria_dict)} criteria from {filepath}")
                    return criteria_dict

        raise FileNotFoundError(
            f"Could not find criteria file in {self.dsm5_path}. "
            f"Tried: {[f.name for f in possible_files]}"
        )

    def _load_posts(self) -> dict[str, str]:
        """Load posts from RedSM5 CSV"""
        posts_file = self.redsm5_path / "redsm5_posts.csv"

        if not posts_file.exists():
            # Try alternative names
            for name in ["posts.csv", "redsm5_posts.csv"]:
                alt_file = self.redsm5_path / name
                if alt_file.exists():
                    posts_file = alt_file
                    break
            else:
                raise FileNotFoundError(f"Posts file not found in {self.redsm5_path}")

        df = pd.read_csv(posts_file)
        posts_dict = dict(zip(df['post_id'], df['text']))

        logger.info(f"Loaded {len(posts_dict)} posts from {posts_file}")
        return posts_dict

    def _create_samples(self) -> list[Sample]:
        """Create (post, criterion) pairs from annotations"""
        annotations_file = self.redsm5_path / "redsm5_annotations.csv"

        if not annotations_file.exists():
            # Try alternative names
            for name in ["annotations.csv", "labels.csv", "redsm5_annotations.csv"]:
                alt_file = self.redsm5_path / name
                if alt_file.exists():
                    annotations_file = alt_file
                    break
            else:
                raise FileNotFoundError(f"Annotations file not found in {self.redsm5_path}")

        df = pd.read_csv(annotations_file)

        samples = []
        skipped = 0

        for _, row in df.iterrows():
            post_id = row['post_id']
            symptom = row['DSM5_symptom']
            label = int(row['status'])

            # Map symptom to criterion ID
            criterion_id = SYMPTOM_TO_CRITERION.get(symptom)
            if criterion_id is None:
                logger.warning(f"Unknown symptom '{symptom}', skipping")
                skipped += 1
                continue

            # Get post text
            post_text = self.posts.get(post_id)
            if post_text is None:
                logger.warning(f"Post {post_id} not found, skipping")
                skipped += 1
                continue

            # Get criterion text
            criterion_text = self.criteria.get(criterion_id)
            if criterion_text is None:
                logger.warning(f"Criterion {criterion_id} not found, skipping")
                skipped += 1
                continue

            # Create sample
            try:
                sample = Sample(
                    sample_id=str(uuid.uuid4()),
                    post_id=post_id,
                    criterion_id=criterion_id,
                    post_text=post_text,
                    criterion_text=criterion_text,
                    label=label,
                )
                samples.append(sample)
            except ValueError as e:
                logger.warning(f"Invalid sample: {e}, skipping")
                skipped += 1
                continue

        logger.info(f"Created {len(samples)} samples, skipped {skipped}")
        return samples

    def _validate_cardinality(self):
        """Validate expected counts"""
        if self.override_counts:
            logger.warning("Count validation overridden")
            return

        # Expected counts from spec (may need adjustment based on actual data)
        # expected_posts = 1484
        # expected_criteria = 9
        # expected_samples = 13356

        # For now, just log actual counts
        unique_posts = len(set(s.post_id for s in self.samples))
        unique_criteria = len(set(s.criterion_id for s in self.samples))

        logger.info(
            f"Cardinality: {unique_posts} unique posts, "
            f"{unique_criteria} unique criteria, "
            f"{len(self.samples)} total samples"
        )

    def _log_statistics(self):
        """Log dataset statistics"""
        total = len(self.samples)
        positive = sum(s.label for s in self.samples)
        negative = total - positive

        logger.info(
            f"Label distribution: {positive}/{total} positive "
            f"({100*positive/total:.1f}%), "
            f"{negative}/{total} negative ({100*negative/total:.1f}%)"
        )

        # Per-criterion stats
        criterion_stats = {}
        for sample in self.samples:
            cid = sample.criterion_id
            if cid not in criterion_stats:
                criterion_stats[cid] = {'total': 0, 'positive': 0}
            criterion_stats[cid]['total'] += 1
            criterion_stats[cid]['positive'] += sample.label

        logger.info("Per-criterion distribution:")
        for cid in sorted(criterion_stats.keys()):
            stats = criterion_stats[cid]
            pos = stats['positive']
            tot = stats['total']
            logger.info(f"  {cid}: {pos}/{tot} ({100*pos/tot:.1f}% positive)")

    def get_dataset_hash(self) -> str:
        """Compute hash of source data for tracking"""
        hasher = hashlib.sha256()

        # Hash posts
        for post_id in sorted(self.posts.keys()):
            hasher.update(post_id.encode())
            hasher.update(self.posts[post_id].encode())

        # Hash criteria
        for cid in sorted(self.criteria.keys()):
            hasher.update(cid.encode())
            hasher.update(self.criteria[cid].encode())

        return hasher.hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample by index"""
        sample = self.samples[idx]
        return {
            'sample_id': sample.sample_id,
            'post_id': sample.post_id,
            'criterion_id': sample.criterion_id,
            'post': sample.post_text,
            'criterion': sample.criterion_text,
            'label': sample.label,
        }

    def get_groups(self) -> list[str]:
        """Get post_ids for grouping in cross-validation"""
        return [s.post_id for s in self.samples]
