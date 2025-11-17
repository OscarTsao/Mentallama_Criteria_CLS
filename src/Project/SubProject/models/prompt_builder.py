"""
Prompt builder for MentaLLaMA classification

Formats (post, criterion) pairs into prompts for binary classification.
"""

import unicodedata
from typing import Optional


def build_prompt(post: str, criterion: str, max_length: Optional[int] = None) -> str:
    """
    Build classification prompt

    Args:
        post: Social media post text
        criterion: DSM-5 criterion description
        max_length: Optional maximum character length for truncation

    Returns:
        Formatted prompt string
    """
    # Normalize unicode
    post = normalize_text(post)
    criterion = normalize_text(criterion)

    # Validate non-empty
    if not post:
        raise ValueError("Post text cannot be empty")
    if not criterion:
        raise ValueError("Criterion text cannot be empty")

    # Build prompt
    prompt = (
        f"post: {post}, criterion: {criterion} "
        f"Does the post match the criterion description? Output yes or no"
    )

    # Truncate if needed
    if max_length and len(prompt) > max_length:
        # Truncate the post (longest component) while keeping structure
        available_for_post = max_length - len(
            f"post: , criterion: {criterion} "
            f"Does the post match the criterion description? Output yes or no"
        )
        if available_for_post > 50:  # Ensure we have reasonable space
            truncated_post = post[:available_for_post - 3] + "..."
            prompt = (
                f"post: {truncated_post}, criterion: {criterion} "
                f"Does the post match the criterion description? Output yes or no"
            )
        else:
            # If criterion is too long, truncate both
            half_length = max_length // 2 - 100
            truncated_post = post[:half_length] + "..."
            truncated_criterion = criterion[:half_length] + "..."
            prompt = (
                f"post: {truncated_post}, criterion: {truncated_criterion} "
                f"Does the post match the criterion description? Output yes or no"
            )

    return prompt


def normalize_text(text: str) -> str:
    """
    Normalize text: unicode normalization and whitespace cleanup

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalization (NFKC - compatibility composition)
    text = unicodedata.normalize('NFKC', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def truncate_for_tokenizer(
    post: str,
    criterion: str,
    max_tokens: int = 512,
    avg_chars_per_token: float = 4.0
) -> tuple[str, str]:
    """
    Estimate truncation based on approximate token count

    Args:
        post: Post text
        criterion: Criterion text
        max_tokens: Maximum number of tokens
        avg_chars_per_token: Average characters per token (rough estimate)

    Returns:
        Tuple of (truncated_post, truncated_criterion)
    """
    # Reserve tokens for prompt template
    # "post: , criterion:  Does the post match the criterion description? Output yes or no"
    template_tokens = 20  # rough estimate

    available_tokens = max_tokens - template_tokens

    # Estimate current token usage
    post_tokens_est = len(post) / avg_chars_per_token
    criterion_tokens_est = len(criterion) / avg_chars_per_token
    total_tokens_est = post_tokens_est + criterion_tokens_est

    if total_tokens_est <= available_tokens:
        # No truncation needed
        return post, criterion

    # Truncate proportionally, but favor keeping more of the post
    post_ratio = 0.7  # Keep 70% of tokens for post
    criterion_ratio = 0.3  # Keep 30% for criterion

    max_post_tokens = int(available_tokens * post_ratio)
    max_criterion_tokens = int(available_tokens * criterion_ratio)

    max_post_chars = int(max_post_tokens * avg_chars_per_token)
    max_criterion_chars = int(max_criterion_tokens * avg_chars_per_token)

    truncated_post = post
    truncated_criterion = criterion

    if len(post) > max_post_chars:
        truncated_post = post[:max_post_chars - 3] + "..."

    if len(criterion) > max_criterion_chars:
        truncated_criterion = criterion[:max_criterion_chars - 3] + "..."

    return truncated_post, truncated_criterion
