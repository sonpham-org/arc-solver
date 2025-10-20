"""Model configuration registry for available LLM models.

This module exports a DEFAULT_MODEL string and a MODEL_CONFIGS dict where
each key is a model name and the value is a small dict of metadata. The
runner imports DEFAULT_MODEL to set the model unless overridden via CLI.
"""

from typing import Dict

# Default model to use when none is supplied via CLI
DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-06-17"

# A small registry of models we expect to use. Add entries here as
# we experiment with more models. The metadata fields are optional and
# are informative only.
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "gemini-2.5-flash-lite-preview-06-17": {
        "provider": "google",
        "description": "Gemini 2.5 flash lite preview",
    },
    # Example additional entries for future use
    # "gemini-1.0": {"provider": "google", "description": "Older model"},
}

def is_known_model(name: str) -> bool:
    """Return True if the model name is in the registry."""
    return name in MODEL_CONFIGS
