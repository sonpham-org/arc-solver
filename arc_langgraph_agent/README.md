# ARC LangGraph Agent

A LangGraph-based agent for solving ARC (Abstraction and Reasoning Corpus) problems using large language models configured via `model_configs.py`.

## Overview

This agent uses a workflow-based approach to iteratively:
1. **Generate** Python code to solve ARC problems using an LLM
2. **Test** the code against training examples  
3. **Refine** the solution based on failures (up to max attempts)
4. **Output** the final solution with helper functions

## Installation

```bash
# Install core dependencies
pip install langgraph langchain-core

# Install LLM provider packages based on your model configs
pip install langchain-google-genai    # For Google Gemini models
pip install langchain-ollama         # For local Ollama models
```

## Quick Start

### 1. Using Model Configs (Recommended)

```python
from model_configs import DEFAULT_MODEL
from arc_langgraph_agent import ARCLangGraphAgent

# Initialize LLM from your model configs
llm = initialize_llm_from_config(DEFAULT_MODEL)  # Uses gemini-2.5-flash-lite-preview-06-17

# Create agent
agent = ARCLangGraphAgent(llm, max_attempts=3)

# Solve ARC task
result = agent.solve_task(task_id, task_data)
print(f"Success: {result['success']}")
print(f"Code: {result['final_solution']['main_code']}")
```

### 2. Available Models

Your `model_configs.py` defines these models:
- **gemini-2.5-pro**: Flagship Google model
- **gemini-2.5-flash**: Fast Google model  
- **gemini-2.5-flash-lite**: Cost-optimized Google model (default)
- **gemini-2.0-flash**: Latest Google model
- **learnlm-2.0-flash**: Experimental LearnLM model
- **llama3.1**: Local Llama model via Ollama
- **qwen2.5:32b**: Local Qwen model via Ollama

### 3. Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# GEMINI_API_KEY=your-google-api-key

# For Ollama models (install and run locally)
ollama pull llama3.1
ollama pull qwen2.5:32b
```

## Command Line Usage

```bash
# Use default model (gemini-2.5-flash-lite-preview-06-17)
python run_langgraph_agent.py --mode single

# Use specific models from your configs  
python run_langgraph_agent.py --mode single --model gemini-2.5-pro
python run_langgraph_agent.py --mode single --model llama3.1
python run_langgraph_agent.py --mode single --model qwen2.5:32b

# Batch testing
python run_langgraph_agent.py --mode batch --batch-size 5 --model gemini-2.0-flash

# Test specific task
python run_langgraph_agent.py --mode single --task-id "00d62c1b" --model gemini-2.0-flash
```

## Architecture

### State Management
- **AgentState**: Tracks workflow progress, solutions, test results
- **CodeSolution**: Stores generated code, helper functions, confidence
- **TestResult**: Records execution results on training examples

### Workflow Nodes
- **generate_code_node**: LLM generates Python transformation code
- **test_code_node**: Execute code on training examples
- **refinement_node**: LLM improves code based on failures  
- **finalize_node**: Create final prediction

### Helper Tools
The agent has access to 20+ helper functions for:
- Grid manipulation (copy, rotate, flip, crop)
- Pattern detection (rectangles, lines, shapes)
- Color operations (count, filter, map)
- Geometric transformations

## API Reference

### ARCLangGraphAgent

```python
class ARCLangGraphAgent:
    def __init__(self, llm, max_attempts: int = 5)
    
    def solve_task(self, task_id: str, task_data: Dict) -> WorkflowOutput
    def solve_multiple_tasks(self, tasks: Dict) -> Dict[str, WorkflowOutput]  
    def export_solution_to_json(self, result: WorkflowOutput) -> Dict
    def test_solution_on_examples(self, result: WorkflowOutput, task_data: Dict) -> Dict
```

### Example Task Format

```python
task_data = {
    "train": [
        {
            "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        }
    ],
    "test": [
        {"input": [[0, 1], [0, 0]]}
    ]
}
```

## Examples

See `example_llm_usage.py` for complete examples with different LLM providers.

Run the example:
```bash
python example_llm_usage.py
```

## Performance

The agent typically:
- Uses 1-3 LLM calls per task (generation + refinements)
- Runs in 10-30 seconds per task (depending on LLM speed)
- Generates executable Python code with helper functions
- Provides step-by-step transformation descriptions

## Troubleshooting

### Import Errors
```bash
# Missing LangGraph
pip install langgraph langchain-core

# Missing LLM provider
pip install langchain-openai  # or langchain-anthropic, langchain-ollama
```

### API Key Issues
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### LLM Connection
```python
# Test LLM directly
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello")
print(response.content)
```

## Contributing

The agent is designed to be extensible:
- Add new helper tools in `tools.py`
- Modify prompts in `nodes.py`
- Extend workflow in `workflow.py`
- Add new LLM providers in examples