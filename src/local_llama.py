"""Quick local LLaMA test runner.

This script tries to load a local/remote LLaMA-family model using two possible backends:

- Hugging Face transformers + bitsandbytes (4-bit NF4) -- preferred on an RTX 4090.
- llama-cpp-python + a ggml .bin model file -- fallback if you have a local ggml quantized model.

It attempts to be helpful: prints CUDA info, catches OOM and gives install hints.

Usage (examples):
  python local_llama.py --backend hf --model meta-llama/Llama-2-7b-chat-hf
  python local_llama.py --backend llama_cpp --ggml /path/to/ggml-model-q4_0.bin

Notes:
- An RTX 4090 (24GB) should easily run 7B models in 4-bit. For HF 7B you should use bitsandbytes
  (load_in_4bit / NF4). If you hit OOM, try smaller model or check quantization.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Optional


def cuda_info() -> str:
    """Return a short string describing CUDA device info if available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA not available"

        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        total_gb = prop.total_memory / 1024 ** 3
        return f"CUDA device: {prop.name} (index {idx}), {total_gb:.2f} GB"
    except Exception:
        return "Unable to query CUDA (torch not installed or error)"


def try_transformers_4bit(model_name: str, prompt: str = "Write a short haiku about autumn.") -> None:
    """Try to load a HF model in 4-bit using bitsandbytes + transformers.

    This approach is suitable for an RTX 4090 when using a 7B model in 4-bit.
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
        )

        # BitsAndBytesConfig moved around across versions; import lazily
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            # Older/newer HF might not expose it; rely on kwargs fallback (may error)
            BitsAndBytesConfig = None

        print("[hf] torch version:", getattr(torch, "__version__", "<unknown>"))
        print("[hf] Checking CUDA:", cuda_info())

        bnb_config = None
        if BitsAndBytesConfig is not None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        print(f"[hf] Loading tokenizer for {model_name}...")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        print(f"[hf] Loading model {model_name} in 4-bit (this may download weights)...")
        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            # Some setups expect load_in_4bit directly (older helpers); add as best-effort
            load_kwargs["load_in_4bit"] = True

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        print("[hf] Creating generation pipeline...")
        gen = pipeline("text-generation", model=model, tokenizer=tok)
        print("[hf] Generating sample...")
        out = gen(prompt, max_new_tokens=128, do_sample=False)
        print("[hf] OUTPUT:\n", out[0]["generated_text"])

    except Exception as e:
        print("[hf] Error while trying transformers + bitsandbytes:")
        traceback.print_exc()
        # Provide helpful hints
        print("Hints:")
        print(" - Ensure you installed: pip install -U transformers accelerate bitsandbytes")
        print(" - For model access you may need to login with `huggingface-cli login` or use a local path")
        print(" - If you see CUDA OOM, try a smaller model or confirm quantization settings (4-bit NF4 recommended)")
        raise


def try_llama_cpp(ggml_path: str, prompt: str = "Write a short haiku about autumn.") -> None:
    """Try generating with llama-cpp-python using a ggml quantized model.

    This requires `llama-cpp-python` and a local ggml .bin file (q4_* recommended for a 4090).
    """
    try:
        from llama_cpp import Llama

        print("[llama_cpp] Using ggml model at:", ggml_path)
        llm = Llama(model_path=ggml_path)
        resp = llm(prompt, max_tokens=128)
        # llama-cpp-python returns a dict with 'choices'
        text = None
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                # Depending on version, content may be in 'text' or 'message' etc.
                text = choices[0].get("text") or choices[0].get("message") or str(choices[0])
        if not text:
            # fallback to repr
            text = str(resp)
        print("[llama_cpp] OUTPUT:\n", text)
    except Exception:
        print("[llama_cpp] Error while trying llama-cpp-python:")
        traceback.print_exc()
        print("Hints:")
        print(" - Install: pip install llama-cpp-python")
        print(" - Provide a path to a ggml q4_0/q4_1 model file with --ggml /path/to/model.bin")
        raise


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Quick local LLaMA model tester for RTX 4090")
    p.add_argument("--backend", choices=("hf", "llama_cpp"), default="hf", help="Backend to use")
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", help="Hugging Face model id or local path (for hf backend)")
    p.add_argument("--ggml", default=None, help="Path to ggml .bin file (for llama_cpp backend)")
    p.add_argument("--prompt", default="Write a short haiku about autumn.", help="Prompt to send to the model")
    args = p.parse_args(argv)

    print("Local LLaMA tester starting")
    print(cuda_info())

    try:
        if args.backend == "hf":
            print("Selecting Hugging Face + bitsandbytes backend")
            try_transformers_4bit(args.model, args.prompt)
        else:
            if not args.ggml:
                print("ERROR: backend 'llama_cpp' requires --ggml /path/to/ggml-model.bin")
                return 2
            try_llama_cpp(args.ggml, args.prompt)
    except Exception:
        print("Run failed; see trace above for details")
        return 1

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
