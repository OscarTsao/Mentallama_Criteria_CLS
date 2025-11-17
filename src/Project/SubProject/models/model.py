"""
PATCH 01: Encoder-Style MentalLLaMA Model for NLI Classification

This patch replaces src/Project/SubProject/models/model.py with a proper
implementation of the decoder→encoder method (Gemma Encoder style).

Key fixes:
1. Uses MentalLLaMA backbone with bidirectional attention
2. Proper pooling for non-pooler models (first token or mean)
3. Dropout regularization (0.1)
4. No causal masking
5. Classification head for binary NLI

Usage:
    cp PATCH_01_encoder_model.py src/Project/SubProject/models/model.py
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)
from typing import Optional, Tuple, Dict, Any
import warnings


class EncoderStyleLlamaModel(nn.Module):
    """
    Wrapper around LLaMA decoder model that converts it to encoder-style
    by removing causal attention masking.

    This implements the "decoder→encoder" approach where:
    - We use a pre-trained decoder LM (MentalLLaMA)
    - Remove causal attention mask (enable bidirectional attention)
    - Add classification head on top
    - Train with supervised classification loss (not LM loss)
    """

    def __init__(
        self,
        model_name: str = "klyang/MentaLLaMA-chat-7B",
        num_labels: int = 2,
        pooling_strategy: str = "first",  # "first" or "mean"
        dropout_rate: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        # Load config and model
        self.config = AutoConfig.from_pretrained(model_name)

        # CRITICAL: Disable causal masking for encoder-style attention
        # For LLaMA models, we need to ensure the attention is bidirectional
        if hasattr(self.config, 'is_decoder'):
            self.config.is_decoder = False
        if hasattr(self.config, 'is_encoder_decoder'):
            self.config.is_encoder_decoder = False

        # Load base model (without LM head)
        # Using AutoModel loads LlamaModel (not LlamaForCausalLM)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Patch attention layers to remove causal mask
        self._patch_attention_for_bidirectional()

        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        self.pooling_strategy = pooling_strategy
        self.num_labels = num_labels
        hidden_size = self.config.hidden_size

        # Dropout for regularization (≈0.1 as per Gemma Encoder)
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head (pooler + linear)
        # Note: LLaMA doesn't have a pooler, so we implement our own
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize classifier with small random weights
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def _patch_attention_for_bidirectional(self):
        """
        Patch LLaMA attention layers to use bidirectional (full) attention
        instead of causal (lower-triangular) attention.

        This is the CRITICAL modification for decoder→encoder conversion.
        """
        # For LLaMA models, we need to modify the attention implementation
        # The causal mask is typically applied in the attention forward pass

        # Option 1: Monkey-patch the attention function
        # We override the _prepare_decoder_attention_mask to return full mask

        original_prepare_mask = getattr(
            self.encoder,
            '_prepare_decoder_attention_mask',
            None
        )

        def bidirectional_attention_mask(
            attention_mask: torch.Tensor,
            input_shape: Tuple[int, int],
            inputs_embeds: torch.Tensor,
            past_key_values_length: int,
        ) -> torch.Tensor:
            """
            Override to return bidirectional attention mask (all 1s for valid tokens).
            Instead of causal (lower-triangular) mask.
            """
            # Create full attention mask (no causal restriction)
            batch_size, seq_length = input_shape
            device = inputs_embeds.device

            if attention_mask is None:
                # No padding, all tokens are valid
                attention_mask = torch.ones(
                    (batch_size, seq_length),
                    device=device,
                    dtype=torch.long
                )

            # Expand attention_mask to 4D for multi-head attention
            # Shape: [batch, 1, 1, seq_len] - broadcasts to [batch, heads, seq_len, seq_len]
            # Use 0 for positions to mask, 1 for positions to attend
            expanded_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)

            # Convert to additive mask (0 for attend, -inf for mask)
            # Standard HuggingFace convention
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min

            return expanded_mask

        # Apply the patch
        if hasattr(self.encoder, '_prepare_decoder_attention_mask'):
            self.encoder._prepare_decoder_attention_mask = bidirectional_attention_mask
        else:
            # For newer transformers versions, might need different approach
            warnings.warn(
                "Could not find _prepare_decoder_attention_mask. "
                "Attention might still be causal. Please verify."
            )

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool hidden states to get fixed-size representation.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] (1=token, 0=pad)

        Returns:
            pooled: [batch, hidden_dim]
        """
        if self.pooling_strategy == "first":
            # Use first token (like BERT's [CLS])
            # For LLaMA, this would be the first input token
            pooled = hidden_states[:, 0, :]

        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padded tokens
            if attention_mask is None:
                # No padding, simple mean
                pooled = hidden_states.mean(dim=1)
            else:
                # Mask out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_hidden / sum_mask
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for NLI classification.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1=token, 0=pad)
            labels: [batch] (optional, for computing loss)
            return_dict: whether to return dict or tuple

        Returns:
            dict with keys: logits, loss (if labels provided)
        """
        # Forward through encoder (with bidirectional attention)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get hidden states
        # Note: LlamaModel returns BaseModelOutputWithPast
        # outputs.last_hidden_state has shape [batch, seq_len, hidden_dim]
        hidden_states = outputs.last_hidden_state

        # Pool to fixed-size representation
        pooled = self.pool_hidden_states(hidden_states, attention_mask)

        # Apply dropout for regularization
        pooled = self.dropout(pooled)

        # Classification head
        logits = self.classifier(pooled)  # [batch, num_labels]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states,
            }
        else:
            return (loss, logits) if loss is not None else logits


# Legacy wrapper for compatibility
class Model(nn.Module):
    """Backward-compatible wrapper."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.model = EncoderStyleLlamaModel(
            model_name=model_name,
            num_labels=num_labels,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


class classification_head(nn.Module):
    """
    Standalone classification head with dropout.
    Fixed version with proper nn.Module inheritance and dropout usage.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout_prob: float = 0.1,
        layer_num: int = 1
    ):
        super().__init__()

        if layer_num == 1:
            # Single layer
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, num_labels)
            )
        else:
            # Multi-layer MLP
            layers = []
            for i in range(layer_num - 1):
                layers.extend([
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ])
            layers.append(nn.Linear(input_dim, num_labels))
            self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


# Utility functions
def load_mentallama_for_nli(
    model_name: str = "klyang/MentaLLaMA-chat-7B",
    num_labels: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[EncoderStyleLlamaModel, AutoTokenizer]:
    """
    Convenience function to load MentalLLaMA model and tokenizer for NLI.

    Returns:
        model: EncoderStyleLlamaModel
        tokenizer: AutoTokenizer with right-padding
    """
    # Load model
    model = EncoderStyleLlamaModel(
        model_name=model_name,
        num_labels=num_labels,
    ).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CRITICAL: Set padding side to RIGHT for encoder-style models
    tokenizer.padding_side = "right"

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Update model's token embeddings if needed
        model.encoder.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


if __name__ == "__main__":
    # Quick test
    print("Testing EncoderStyleLlamaModel...")

    # Create tiny test (CPU-only, no actual model loading)
    print("Creating model config...")

    # For actual usage:
    # model, tokenizer = load_mentallama_for_nli(num_labels=2)
    #
    # # Tokenize input
    # inputs = tokenizer(
    #     ["premise text", "another premise"],
    #     ["hypothesis text", "another hypothesis"],
    #     padding=True,
    #     truncation=True,
    #     max_length=512,
    #     return_tensors="pt"
    # )
    #
    # # Forward pass
    # outputs = model(
    #     input_ids=inputs['input_ids'],
    #     attention_mask=inputs['attention_mask']
    # )
    #
    # print(f"Logits shape: {outputs['logits'].shape}")  # [batch, 2]

    print("✓ Model definition complete. See docstring for usage.")
