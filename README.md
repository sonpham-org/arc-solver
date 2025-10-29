# arc-solver — local LLaMA tester

This small helper demonstrates how to quickly test a local LLaMA-family model on an RTX 4090.

Files
- `src/local_llama.py`: small CLI that attempts to load a HF Llama model in 4-bit using bitsandbytes
  (recommended for 7B models on an RTX 4090) or fall back to `llama-cpp-python` with a ggml file.
- `tests/test_local_llama_smoke.py`: lightweight pytest that mocks heavy ML packages so you can run
  tests without installing CUDA wheels or large packages.

Quick start (recommended)

1. Create a Python environment (3.10+ recommended).
2. To actually run a real model with Hugging Face + 4-bit compression on an RTX 4090, install:

```bash
pip install -U pip
# install a torch wheel that matches your CUDA (example for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -U transformers accelerate bitsandbytes
```

Then run the tester:

```bash
python src/local_llama.py --backend hf --model meta-llama/Llama-2-7b-chat-hf
```

If you prefer a local ggml model and `llama-cpp-python`:

```bash
pip install llama-cpp-python
python src/local_llama.py --backend llama_cpp --ggml /path/to/ggml-model-q4_0.bin
```

Running the smoke tests (fast, no heavy deps)

```bash
pip install -U pytest
pytest -q
```

Notes
- These instructions assume an RTX 4090 (24 GB). A 4090 can run 7B models comfortably in 4-bit.
- The smoke tests mock ML packages so CI or local dev can run quickly. To test real models you must
  install the real packages and have enough disk space and GPU memory.
