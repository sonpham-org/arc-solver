# arc-solver — ARC Challenge Solver

This project provides tools to solve ARC (Abstraction and Reasoning Corpus) challenges using various LLMs, including local LLaMA models and Google Gemini API.

## Files

- `src/main.py`: Main script to run Gemini LLM on ARC challenges.
- `src/local_llama.py`: CLI to test local LLaMA models on RTX 4090.
- `tests/test_local_llama_smoke.py`: Lightweight pytest for local LLaMA testing.
- Other scripts: `count_challenges.py`, `investigate_scores.py`, `view_db.py` for analysis.

## Quick Start with Docker

1. Ensure you have Docker and Docker Compose installed.

2. Set your `GEMINI_API_KEY` environment variable:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. Build the Docker image (optional, or use docker-compose which builds automatically):

   ```bash
   ./scripts/build_docker.sh
   ```

4. Run with Docker Compose:

   ```bash
   docker-compose up
   ```

   This will run the main solver on training challenges.

   Or run directly with Docker:

   ```bash
   docker run --rm -it -e GEMINI_API_KEY=$GEMINI_API_KEY -v $(pwd)/output:/app/output arc-solver:latest python src/main.py --challenges data/arc-2024/arc-agi_training_challenges.json
   ```

## Manual Setup (Local LLaMA)

1. Create a Python environment (3.10+ recommended).

2. To run real models with Hugging Face + 4-bit compression on an RTX 4090, install:

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

## Running Gemini Solver

After installing dependencies:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key_here"
python src/main.py --challenges data/arc-2024/arc-agi_training_challenges.json --solutions data/arc-2024/arc-agi_training_solutions.json
```

## Running Tests

```bash
pip install -U pytest
pytest -q
```

## Notes

- The smoke tests mock ML packages so CI or local dev can run quickly. To test real models you must install the real packages and have enough disk space and GPU memory.
- For GPU support in Docker, you may need to use `--gpus all` with `docker run` if using local models.
