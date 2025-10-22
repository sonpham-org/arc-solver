"""Model configuration registry for available LLM models.

This module exports a DEFAULT_MODEL string and a MODEL_CONFIGS dict where
each key is a model name and the value is a small dict of metadata. The
runner imports DEFAULT_MODEL to set the model unless overridden via CLI.
"""

from typing import Dict, Optional

# Default model to use when none is supplied via CLI
DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-06-17"

# Pricing is expressed as USD per 1,000,000 tokens (per 1M). The cost
# calculator below assumes tokens are raw token counts (not words) and
# scales linearly. The user requested to assume prompts are <200k tokens
# so we use the short-context / base rates for models that have a long-
# context surcharge.

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    # Gemini 2.5 Pro (production "Pro" model)
    "gemini-2.5-pro": {
        "provider": "google",
        "description": "Gemini 2.5 Pro (flagship)",
        "pricing": {"input_per_m": 1.25, "output_per_m": 10.0},
    },
    # Gemini 2.5 Flash (fast)
    "gemini-2.5-flash": {
        "provider": "google",
        "description": "Gemini 2.5 Flash",
        "pricing": {"input_per_m": 0.30, "output_per_m": 2.50},
    },
    # Gemini 2.5 Flash-Lite (cost-optimized)
    "gemini-2.5-flash-lite": {
        "provider": "google",
        "description": "Gemini 2.5 Flash-Lite",
        "pricing": {"input_per_m": 0.10, "output_per_m": 0.40},
    },
    # Gemini 2.0 Flash
    "gemini-2.0-flash": {
        "provider": "google",
        "description": "Gemini 2.0 Flash",
        "pricing": {"input_per_m": 0.15, "output_per_m": 0.60},
    },
    # Gemini 2.0 Flash-Lite
    "gemini-2.0-flash-lite": {
        "provider": "google",
        "description": "Gemini 2.0 Flash-Lite",
        "pricing": {"input_per_m": 0.075, "output_per_m": 0.30},
    },
    # LearnLM 2.0 Flash (experimental / community-sourced)
    "learnlm-2.0-flash": {
        "provider": "learnlm",
        "description": "LearnLM 2.0 Flash (experimental)",
        "pricing": {"input_per_m": 0.08, "output_per_m": 0.30},
    },
    # Keep the original preview default as an alias to the flash-lite entry
    "gemini-2.5-flash-lite-preview-06-17": {
        "provider": "google",
        "description": "Gemini 2.5 flash lite preview",
        "alias_of": "gemini-2.5-flash-lite",
    },
}


def is_known_model(name: str) -> bool:
    """Return True if the model name is in the registry (or matches a known prefix)."""
    return find_model_key(name) is not None


def find_model_key(name: str) -> Optional[str]:
    """Find the canonical model key for the given model name.

    Matching strategy:
    - exact match
    - case-insensitive substring match (registry key in name or name in key)
    Returns the registry key or None if not found.
    """
    if not name:
        return None
    lname = name.lower()
    # 1) exact match
    if lname in MODEL_CONFIGS:
        return lname
    # 2) look for a registry key that's a substring of the provided name
    for key in MODEL_CONFIGS:
        if key in lname or lname in key:
            return key
    return None


def get_pricing_for(model_name: str) -> Optional[Dict[str, float]]:
    """Return the pricing dict for a model name or None if unknown.

    The returned dict has keys: input_per_m, output_per_m (USD per 1M tokens).
    """
    key = find_model_key(model_name)
    if not key:
        return None
    config = MODEL_CONFIGS.get(key, {})
    # resolve aliases
    if "alias_of" in config:
        config = MODEL_CONFIGS.get(config["alias_of"], config)
    pricing = config.get("pricing")
    return pricing


def estimate_cost(model_name: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Estimate cost in USD for the given token usage and model name.

    input_tokens and output_tokens are integer token counts. Pricing is
    per 1,000,000 tokens; cost is scaled linearly.
    If the model is unknown, raises KeyError.
    """
    pricing = get_pricing_for(model_name)
    if pricing is None:
        raise KeyError(f"Unknown model for pricing: {model_name}")
    in_rate = float(pricing.get("input_per_m", 0.0))
    out_rate = float(pricing.get("output_per_m", 0.0))
    cost = (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate
    return float(cost)
