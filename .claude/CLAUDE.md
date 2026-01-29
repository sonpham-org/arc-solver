# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARC Solver is a LangGraph-based AI agent for solving Abstract Reasoning Corpus (ARC) tasks. It uses an evolutionary multi-solution approach: generate multiple solution candidates, test them on training examples, then iteratively refine and fuse solutions across loops. Supports Gemini (default), OpenAI, Anthropic, and Ollama models.

## Commands

### Setup
```bash
pip install -e .                  # Editable install
pip install -r requirements.txt   # All dependencies
pip install -e ".[dev]"           # Dev tools (pytest, black, flake8)
pip install -e ".[local-llm]"    # Local LLM support (requires GPU)
```

### Running
```bash
python run_langgraph_agent.py     # Main agent (configure via top-of-file constants)
python run_arc_baseline.py        # Baseline solver (simpler, direct prompting)
python arc_visualizer.py          # Flask web UI at http://localhost:5000
```

### Formatting & Linting
```bash
black .                           # Format (120 char line length)
flake8 .                          # Lint
```

### Testing
```bash
pytest tests/                     # Run tests (test directory configured but sparse)
```

### Docker
```bash
bash bash_scripts/build_docker.sh # Build image
docker-compose up arc-solver      # Run container
```

### Helper Scripts
```bash
bash bash_scripts/run_main.sh     # Run main agent
bash bash_scripts/run_smoke.sh    # Quick smoke test
```

## Architecture

### Execution Flow

```
Initialize task → Generate seed solutions → [Loop N times] →
  Test on training data → Rank solutions → Refine failures →
  Fuse top solutions → Mutate/evolve → Finalize best solution
```

### Core Layers

**Agent** (`agentic/agent.py`): `ARCLangGraphAgent` orchestrates the workflow. Delegates to a graph builder (evolutionary or simple) to construct the LangGraph state machine.

**Graph Builders** (`agentic/graphs/`): Define the LangGraph workflow structure. `EvolutionaryGraphBuilder` is the main one — implements multi-loop refinement with routing predicates (`one_solution_succeeded`, `out_of_loops`, `decide_setup`).

**Nodes** (`agentic/nodes.py`): LangGraph node functions — `generate_code_node`, `test_code_node`, `evolve_code_node`, `finalize_node`, `save_state_node`, `verify_solutions_node`. This is the largest single file (~1200 lines).

**Actions** (`agentic/actions/`): Modular functions that nodes call. Key modules:
- `reasoning.py` — generate reasoning traces analyzing input/output patterns
- `code_generation.py` — produce Python transformation code from reasoning
- `code_execution.py` — execute code on grids, evaluate results
- `refine.py` — reflect on failures, generate improved solutions
- `fuse.py` / `fusion.py` — combine strengths of two solutions
- `rag.py` — vector store operations (Qdrant) for retrieval-augmented generation

**Schema** (`agentic/schema.py`): TypedDict definitions — `AgentState` (workflow state), `CodeSolution` (solution with code, reasoning, metrics), `ExampleResult` (per-example evaluation), `WorkflowOutput` (final results).

**Augmentation** (`agentic/augmentation.py`): Dihedral transforms (8 geometric transformations) and color permutations to generate synthetic training examples.

### Supporting Modules

**Prompts** (`prompts/`): Each file exports a prompt builder function (e.g., `build_arc_prompt`, `build_reflection_prompt`, `build_code_repair_prompt`).

**Utils** (`utils/`): Task loading (`task_utils`), grid operations & overlap calculation (`grid_utils`), JSON extraction from LLM responses (`json_utils`), grid sanitization (`sanitize_utils`), metrics computation (`calculation_utils`).

**Model Config** (`model_configs.py`): Registry of supported models with pricing, provider info, and fuzzy name matching via `find_model_key()`.

### Entry Points

- `run_langgraph_agent.py` — Main entry point. Configuration is via constants at the top of the file (MODEL, MODE, NUM_TASKS, NUM_WORKERS, NUM_LOOPS, RESUME_RUN, etc.). Supports single/batch mode and resuming previous runs.
- `run_arc_baseline.py` — Simpler baseline with CLI args (`--model`, `--number-of-tasks`, `--task-index`).
- `arc_visualizer.py` — Flask app for browsing results in `output/output_agent/`.

### Data

ARC task datasets live in `data/arc-2024/` and `data/arc-2025/` as JSON files. Results are saved to `output/output_agent/<timestamp>/<task_id>/`.

## Key Conventions

- Python 3.9+, Black formatting with 120-char lines
- LLM provider abstraction via LangChain — models are swapped by changing config strings
- State is tracked as immutable TypedDict (`AgentState`) flowing through LangGraph nodes
- Root-level `prompts.py` and `utils.py` are backward-compatibility wrappers that re-export from their respective packages
- Environment variables for API keys (`.env` file): `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
