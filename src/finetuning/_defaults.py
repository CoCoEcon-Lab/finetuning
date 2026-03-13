"""
Configured defaults for fine-tuning workshop

🫵 You shouldn't be here, no touching unless you want something to break!
"""

from pathlib import Path
from dataclasses import dataclass

ROOT_DIR = Path(__file__).parents[2]

@dataclass(frozen=True)
class WorkshopDefaults:
    """Defaults for LLM configuration."""

    model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    dataset = 'fingriffin/natural-questions-corporate-jargon'

    hf_dir = ROOT_DIR / 'hf'

WORKSHOP_DEFAULTS: WorkshopDefaults = WorkshopDefaults()