# arc-solver — ARC Challenge Solver

This project provides tools to solve ARC (Abstraction and Reasoning Corpus) challenges using various LLMs, including local LLaMA models and Google Gemini API. The solver uses advanced prompting techniques with reflection, code repair, and smart routing to maximize ARC task performance.

## Code Architecture and Flow

### Core Components

#### Main Entry Points
- **`arc_prompt.py`**: Main execution script with comprehensive task processing, reflection, and retry logic
- **`arc_baseline.py`**: Simplified baseline solver for direct test output prediction
- **`model_configs.py`**: Model configuration and cost estimation
- **`utils.py`**: Core utilities for task sanitization, result calculation, and file operations

#### Prompt System (`prompts/` directory)
The prompt system has been modularized for better maintainability:
- **`arc_prompt.py`**: Main ARC prompt for comprehensive task solving with reasoning and JSON output
- **`apply_prompt.py`**: Strict rule application prompt for test inputs
- **`apply_prompt_2.py`**: Flexible rule application with interpretation guidance
- **`reflection_prompt.py`**: Rule refinement prompt based on failure analysis
- **`code_repair_prompt.py`**: Python code debugging and repair prompts
- **`arc_reflection_prompt.py`**: Task-level reflection for failed attempts
- **`baseline_prompt.py`**: Minimal prompt for direct test output prediction

#### Legacy Compatibility
- **`prompts.py`**: Legacy interface that imports from the new modular prompt system

### Execution Flow

#### 1. Task Processing Pipeline (`arc_prompt.py`)

```
Task Input → Sanitization (Optional) → Prompt Generation → Model Execution → JSON Extraction → Validation → Testing → Results
```

**Key Steps:**
1. **Task Loading**: Load ARC tasks from JSON files (training/evaluation)
2. **Task Selection**: Choose specific tasks by index, ID, or random sampling
3. **Sanitization** (Optional): Clean task data and IDs for consistent processing
4. **Prompt Generation**: Build comprehensive prompts with training examples and instructions
5. **Model Execution**: Send prompts to selected LLM (Ollama, Gemini, etc.)
6. **Smart Routing** (Optional): Enhance responses using reflection and continuation
7. **JSON Extraction**: Parse structured transformation rules from model responses
8. **Validation**: Verify JSON structure and completeness
9. **Code Testing**: Execute Python code on training examples
10. **Result Processing**: Test on evaluation examples and calculate metrics

#### 2. Advanced Features

**Smart Routing System:**
- **Reasoning Completion Detection**: Analyzes if model response contains complete reasoning
- **Automatic Continuation**: Requests continuation if reasoning appears incomplete
- **JSON Regeneration**: Asks model to regenerate JSON if extraction fails
- **Code Repair**: Fixes Python execution errors through targeted prompts
- **Reflection Cycles**: Iterative improvement through failure analysis

**Retry and Recovery Logic:**
- **Task Retries**: Retry failed tasks with different random seeds
- **Code Repair Attempts**: Fix execution errors while preserving JSON structure
- **Reflection Attempts**: Learn from failures and generate improved solutions
- **Continuation Support**: Resume interrupted runs from existing output directories

#### 3. Model Integration

**Supported Providers:**
- **Ollama**: Local model execution (llama3.1, qwen2.5, etc.)
- **Google Gemini**: Cloud API with various model variants
- **Other Models**: Extensible configuration system

**Model Configuration:**
```python
MODEL_CONFIGS = {
    "model-name": {
        "provider": "provider_name",
        "input_cost": cost_per_1k_tokens,
        "output_cost": cost_per_1k_tokens,
        "max_tokens": limit
    }
}
```

#### 4. Output Format and Results

**Per-Task Results:**
```json
{
  "trains": [
    {
      "input": ["grid_representation"],
      "expected": ["expected_output"],
      "predicted": ["model_prediction"],
      "correct": boolean,
      "overlap": percentage,
      "iou": percentage,
      "error": "error_message_if_any"
    }
  ],
  "tests": [...],  // Similar structure for test examples
  "transformations_json": {...},  // Extracted transformation rules
  "total_tokens": count,
  "estimated_cost": dollars
}
```

**Execution Summary:**
- Task completion rates and accuracy metrics
- Token usage and cost analysis
- Error categorization and debugging information
- Performance comparisons across different approaches

#### 5. Baseline Solver (`arc_baseline.py`)

Simplified pipeline for direct comparison:
```
Task Input → Minimal Prompt → Model Execution → Direct JSON Output → Evaluation
```

**Key Differences:**
- No reasoning or step-by-step transformations
- Direct test output prediction
- Faster execution with lower complexity
- Used for baseline performance measurement

### Advanced Prompt Engineering

#### Prompt Design Principles
1. **Comprehensive Context**: Include all training examples with clear input/output relationships
2. **Structured Guidelines**: Explicit rules for transformation inference and generalization
3. **Output Constraints**: Strict JSON format requirements with executable Python code
4. **Error Prevention**: Built-in validation and common pitfall warnings

#### Reflection and Learning System
1. **Failure Analysis**: Detailed examination of incorrect predictions vs. expected outputs
2. **Pattern Recognition**: Identification of missed transformation rules or edge cases
3. **Iterative Refinement**: Progressive improvement through multiple reflection cycles
4. **Code Debugging**: Systematic approach to Python execution error resolution

### Configuration and Customization

#### Key Configuration Variables
- **Task Selection**: Specific indices, random sampling, or complete dataset processing
- **Model Parameters**: Temperature, token limits, and provider-specific settings
- **Processing Options**: Sanitization, parallel execution, output verbosity
- **Advanced Features**: Smart routing, reflection cycles, retry attempts

#### Command Line Interface
Extensive CLI options for all configuration aspects, with sensible defaults and parameter validation.

## File Structure

```
arc-solver/
├── arc_prompt.py          # Main execution script
├── arc_baseline.py        # Baseline solver
├── model_configs.py       # Model configurations
├── utils.py              # Core utilities
├── prompts.py            # Legacy prompt interface
├── prompts/              # Modular prompt system
│   ├── __init__.py
│   ├── arc_prompt.py
│   ├── apply_prompt.py
│   ├── apply_prompt_2.py
│   ├── reflection_prompt.py
│   ├── code_repair_prompt.py
│   ├── arc_reflection_prompt.py
│   └── baseline_prompt.py
├── data/                 # ARC datasets
├── output/               # Results and logs
└── requirements.txt      # Dependencies
```

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
