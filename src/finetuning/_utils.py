"""
Utilities for fine-tuning workshop.

🫵 You shouldn't be here, no touching unless you want something to break!
"""

import os
import datasets
from finetuning._defaults import WORKSHOP_DEFAULTS

def setup_hf() -> None:
    """Configure HF caches to local folders."""
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable not set!")
    os.environ["HF_HOME"] = str(WORKSHOP_DEFAULTS.hf_dir)
    os.environ["HF_HUB_CACHE"] = str(WORKSHOP_DEFAULTS.hf_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(WORKSHOP_DEFAULTS.hf_dir)
    os.environ["HF_DATASETS_CACHE"] = str(WORKSHOP_DEFAULTS.hf_dir)

def get_system_prompt(ds: datasets.Dataset) -> str:
    """Get system prompt from a dataset."""
    return ds["train"]["messages"][0][0]["content"]