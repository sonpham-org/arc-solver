# ARC Solver - Installation Guide

## Quick Start

### 1. Install the package

```bash
# Standard installation
pip install -e .

# Or install with all dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Optional: Install local LLM support

For local LLM support (requires GPU):

```bash
# First install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install optional local-llm dependencies
pip install -e ".[local-llm]"
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```bash
# For Google Gemini models (primary)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: For other providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
LANGCHAIN_API_KEY=your_langchain_key_here
```

## Usage

After installation, you can run the main entry points:

```bash
# 1. Run baseline ARC solver
python run_arc_baseline.py

# 2. Run LangGraph agent
python run_langgraph_agent.py

# 4. Run ARC visualizer
python arc_visualizer.py
```

## Package Structure

With the `pyproject.toml`, you can now use cleaner imports:

```python
# Instead of relative imports, use absolute imports
from agent import ARCLangGraphAgent
from prompts import build_arc_prompt, build_apply_prompt
from utils import load_arc_tasks, calculate_results
```

## Development Installation

For development with optional dev tools:

```bash
pip install -e ".[dev]"
```

This installs pytest, black, and flake8 for testing and code formatting.
