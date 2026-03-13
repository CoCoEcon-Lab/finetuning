"""Configured defaults for fine-tuning workshop"""

from pathlib import Path
from dataclasses import dataclass

ROOT_DIR = Path(__file__).parents[2]

@dataclass(frozen=True)
class ModelDefaults:
    """Defaults for LLM configuration."""

    model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    model_dir = ROOT_DIR / 'models'

MODEL_DEFAULTS: ModelDefaults = ModelDefaults()

class DataDefaults:
    """Defaults for dataset configuration."""

    dataset = ''
    data_dir = ROOT_DIR / 'data'