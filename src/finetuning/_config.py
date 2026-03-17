"""
Pydantic class for LoRA configuration

🫵 You shouldn't be here, no touching unless you want something to break!
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List

# Workshop restrictions on hyperparams
MAX_EPOCHS = 3
MAX_LORA_RANK = 32
MAX_GLOBAL_BATCH_SIZE = 4
ALLOWED_TARGET_LAYERS = frozenset(
    [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)

class LoRAConfig(BaseModel):
    """Pydantic config for LoRA."""
    num_epochs: int = Field(ge=1, le=MAX_EPOCHS)
    learning_rate: float = Field(ge=0)

    lora_rank: int = Field(ge=1, le=MAX_LORA_RANK)
    lora_alpha: int = Field(ge=1)

    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)

    target_layers: List[str]

    @field_validator("target_layers")
    @classmethod
    def validate_target_layers(cls, v):
        invalid = set(v) - ALLOWED_TARGET_LAYERS
        if invalid:
            raise ValueError(f"Invalid target layers: {invalid}")
        return v

    @model_validator(mode="after")
    def validate_global_batch_size(self):
        global_batch = self.micro_batch_size * self.gradient_accumulation_steps
        if global_batch > MAX_GLOBAL_BATCH_SIZE:
            raise ValueError(
                f"Global batch size {global_batch} exceeds limit {MAX_GLOBAL_BATCH_SIZE}"
            )
        return self

class DPOConfig(BaseModel):
    """Pydantic config for DPO."""
    num_epochs: int = Field(ge=1, le=MAX_EPOCHS)
    learning_rate: float = Field(ge=0)
    beta: float = Field(gt=0)

    lora_rank: int = Field(ge=1, le=MAX_LORA_RANK)
    lora_alpha: int = Field(ge=1)

    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)

    target_layers: List[str]

    @field_validator("target_layers")
    @classmethod
    def validate_target_layers(cls, v):
        invalid = set(v) - ALLOWED_TARGET_LAYERS
        if invalid:
            raise ValueError(f"Invalid target layers: {invalid}")
        return v

    @model_validator(mode="after")
    def validate_global_batch_size(self):
        global_batch = self.micro_batch_size * self.gradient_accumulation_steps
        if global_batch > MAX_GLOBAL_BATCH_SIZE:
            raise ValueError(
                f"Global batch size {global_batch} exceeds limit {MAX_GLOBAL_BATCH_SIZE}"
            )
        return self