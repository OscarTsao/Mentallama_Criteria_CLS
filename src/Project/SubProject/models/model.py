"""
MentaLLaMA Model for Binary Classification

Wrapper around LlamaForSequenceClassification with PEFT/DoRA support.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from Project.SubProject.utils.log import get_logger

logger = get_logger(__name__)


class MentallamClassifier(nn.Module):
    """
    MentaLLaMA-based binary classifier with DoRA fine-tuning

    Loads klyang/MentaLLaMA-chat-7B and adds a classification head
    for binary classification of (post, criterion) pairs.
    """

    def __init__(
        self,
        model_name: str = "klyang/MentaLLaMA-chat-7B",
        num_labels: int = 2,
        use_peft: bool = True,
        peft_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: bool = True,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_labels: Number of output labels (2 for binary)
            use_peft: Whether to use PEFT/DoRA
            peft_config: PEFT configuration dict
            gradient_checkpointing: Enable gradient checkpointing
            load_in_8bit: Load model in 8-bit mode
            device_map: Device map for multi-GPU
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        logger.info(f"Loading model: {model_name}")

        # Load base model
        self.model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            torch_dtype=torch.float16 if not load_in_8bit else None,
        )

        # Enable gradient checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Apply PEFT/DoRA
        if use_peft:
            self._apply_peft(peft_config or {})

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        logger.info(f"Model loaded with {self.count_parameters():,} trainable parameters")

    def _apply_peft(self, config: Dict[str, Any]):
        """Apply PEFT/DoRA configuration"""
        # Default DoRA config from spec
        default_config = {
            'r': 8,
            'lora_alpha': 16,
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            'lora_dropout': 0.05,
            'bias': 'none',
            'task_type': TaskType.SEQ_CLS,
            'use_dora': True,  # Enable DoRA
        }

        # Merge with provided config
        default_config.update(config)

        logger.info(f"Applying PEFT with config: {default_config}")

        peft_config = LoraConfig(**default_config)
        self.model = get_peft_model(self.model, peft_config)

        logger.info("PEFT/DoRA applied successfully")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)

        Returns:
            Dictionary with 'loss', 'logits', and 'predictions'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        result = {
            'logits': logits,
            'predictions': predictions,
        }

        if labels is not None:
            result['loss'] = outputs.loss

        return result

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get probability scores

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Probability tensor (batch_size, num_labels)
        """
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        return probs

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_pretrained(self, save_path: str):
        """Save model checkpoint"""
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def load_pretrained(cls, load_path: str, **kwargs):
        """Load model checkpoint"""
        logger.info(f"Loading model from {load_path}")

        # Create instance without loading base model
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Load PEFT model
        instance.model = PeftModel.from_pretrained(
            load_path,
            **kwargs
        )

        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        instance.model_name = load_path
        instance.num_labels = instance.model.config.num_labels

        return instance


# Legacy class for backwards compatibility
class classification_head(nn.Module):
    """Simple classification head (legacy)"""

    def __init__(self, input_dim: int, num_labels: int, dropout_prob: float = 0.1, layer_num: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(x)


# Legacy model class for backwards compatibility
class Model(nn.Module):
    """Generic transformer classifier (legacy)"""

    def __init__(self, model_name: str, num_labels: int):
        super(Model, self).__init__()
        from transformers import AutoModel
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        logits = self.classifier(pooled_output)
        return logits
