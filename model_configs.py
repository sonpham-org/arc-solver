"""Model configuration registry for available LLM models.

This module exports a DEFAULT_MODEL string and a MODEL_CONFIGS dict where
each key is a model name and the value is a small dict of metadata. The
runner imports DEFAULT_MODEL to set the model unless overridden via CLI.
"""

from typing import Dict, Optional

# Default model to use when none is supplied via CLI
# Prefer the stable flash-lite model as the default (preview aliases remain available)
DEFAULT_MODEL = "gemini-2.5-flash-lite"

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
        # No public HuggingFace counterpart for Google Gemini; set to None
        "hf_model": None,
    },
    # Gemini 2.5 Flash (fast)
    "gemini-2.5-flash": {
        "provider": "google",
        "description": "Gemini 2.5 Flash",
        "pricing": {"input_per_m": 0.30, "output_per_m": 2.50},
        "hf_model": None,
    },
    # Gemini 2.5 Flash-Lite (cost-optimized)
    "gemini-2.5-flash-lite": {
        "provider": "google",
        "description": "Gemini 2.5 Flash-Lite",
        "pricing": {"input_per_m": 0.10, "output_per_m": 0.40},
        "hf_model": None,
    },
    # Gemini 3 Pro (newer generation)
    "gemini-3-pro": {
        "provider": "google",
        "description": "Gemini 3 Pro (powerful multimodal)",
        "pricing": {"input_per_m": 2.00, "output_per_m": 12.00},
        "hf_model": None,
    },
    # Gemini 2.0 Flash
    "gemini-2.0-flash": {
        "provider": "google",
        "description": "Gemini 2.0 Flash",
        "pricing": {"input_per_m": 0.15, "output_per_m": 0.60},
        "hf_model": None,
    },
    # Gemini 2.0 Flash-Lite
    "gemini-2.0-flash-lite": {
        "provider": "google",
        "description": "Gemini 2.0 Flash-Lite",
        "pricing": {"input_per_m": 0.075, "output_per_m": 0.30},
        "hf_model": None,
    },
    # LearnLM 2.0 Flash (experimental / community-sourced)
    "learnlm-2.0-flash": {
        "provider": "learnlm",
        "description": "LearnLM 2.0 Flash (experimental)",
        "pricing": {"input_per_m": 0.08, "output_per_m": 0.30},
        # If LearnLM publishes a HF repo, add the identifier here; unknown for now
        "hf_model": None,
    },
    # Keep the original preview default as an alias to the flash-lite entry
    "gemini-2.5-flash-lite-preview-06-17": {
        "provider": "google",
        "description": "Gemini 2.5 flash lite preview",
        "alias_of": "gemini-2.5-flash-lite",
        "hf_model": None,
    },
    # Additional preview aliases observed on the Gemini pricing page
    "gemini-2.5-flash-preview-09-2025": {
        "provider": "google",
        "description": "Gemini 2.5 Flash preview (Sept 2025)",
        "alias_of": "gemini-2.5-flash",
        "hf_model": None,
    },
    "gemini-2.5-flash-lite-preview-09-2025": {
        "provider": "google",
        "description": "Gemini 2.5 Flash-Lite preview (Sept 2025)",
        "alias_of": "gemini-2.5-flash-lite",
        "hf_model": None,
    },
    # Llama 3.1 via Ollama
    "llama3.1": {
        "provider": "ollama",
        "description": "Llama 3.1 via Ollama",
        "pricing": {"input_per_m": 0.0, "output_per_m": 0.0},
        "hf_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    # Qwen 2.5 32B via Ollama
    "qwen2.5:32b": {
        "provider": "ollama",
        "description": "Qwen 2.5 32B via Ollama",
        "pricing": {"input_per_m": 0.0, "output_per_m": 0.0},
        "hf_model": "Qwen/Qwen2.5-32B-Instruct",
    },
    # DeekSeek-R1 32B via Ollama
    "deepseek-r1:32b": {
        "provider": "ollama",
        "description": "DeepSeek-R1 32B via Ollama",
        "pricing": {"input_per_m": 0.0, "output_per_m": 0.0},
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    },
}


# Optional mapping from Ollama-like model names to HuggingFace repository IDs.
# Use this when you have an Ollama name (or similar) and want a recommended HF
# identifier for running the same model in vLLM / HF transformers.
OLLAMA_TO_HF: Dict[str, str] = {
    "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}


def resolve_hf_model(model_name: str) -> Optional[str]:
    """Return a HuggingFace/vLLM identifier for a given registry model name.

    Resolution strategy:
    - find canonical registry key via `find_model_key`
    - resolve `alias_of` if present
    - return `hf_model` from the resolved config if available
    - otherwise, fall back to `OLLAMA_TO_HF` by checking exact name matches
    - return None if no mapping is found
    """
    if not model_name:
        return None
    key = find_model_key(model_name)
    if not key:
        # try direct mapping lookup (use lower-case keys for OLLAMA_TO_HF)
        mm = OLLAMA_TO_HF.get(model_name)
        return mm

    cfg = MODEL_CONFIGS.get(key, {})
    if "alias_of" in cfg:
        cfg = MODEL_CONFIGS.get(cfg["alias_of"], cfg)

    hf = cfg.get("hf_model")
    if hf:
        return hf

    # fallback: try OLLAMA_TO_HF for the original provided name
    return OLLAMA_TO_HF.get(model_name) or OLLAMA_TO_HF.get(key)


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
