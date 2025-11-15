# ARC Solver Prompts - Testing Guide

This folder contains prompt building functions for the ARC (Abstraction and Reasoning Corpus) solver, each now equipped with comprehensive testing functionality.

## 📁 Prompt Files

Each prompt file now includes a `__main__` section that allows you to test the prompt independently:

### Core Prompt Files
- **`arc_prompt.py`** - Main ARC prompt for rule inference from training examples
- **`apply_prompt.py`** - Exact step-by-step rule application prompt  
- **`apply_prompt_2.py`** - Flexible rule interpretation and application prompt
- **`baseline_prompt.py`** - Minimal prompt for direct output generation

### Reflection & Repair Prompts  
- **`arc_reflection_prompt.py`** - Analyzes failed attempts and generates improved solutions
- **`reflection_prompt.py`** - Refines transformation rules based on failure analysis
- **`code_repair_prompt.py`** - Fixes Python execution errors in valid JSON responses
- **`enhanced_code_repair_prompt.py`** - Advanced code repair with comprehensive error analysis

### Utility Prompts
- **`continuation_prompt.py`** - Continues incomplete reasoning analysis
- **`json_regeneration_prompt.py`** - Regenerates JSON from complete reasoning
- **`json_repair_prompt.py`** - Repairs malformed JSON syntax

## 🧪 Testing Features

Each prompt file supports **two types of tests**:

### 1. Direct API Testing
- **Gemini API**: Direct calls to Google's Gemini models
- **Ollama API**: Local LLM testing via Ollama server

### 2. LangChain Integration  
- Tests prompt construction through LangChain framework
- Uses ChatGoogleGenerativeAI for consistent interface

## 🚀 Quick Start

### Test Individual Prompts

```bash
# Test a specific prompt file
cd /path/to/arc-solver/prompts
python apply_prompt.py
python arc_prompt.py
python baseline_prompt.py
# ... etc for any prompt file
```

### Test All Prompts

```bash
# Run comprehensive testing suite
python test_all_prompts.py

# Test only a specific prompt via the suite
python test_all_prompts.py apply_prompt
python test_all_prompts.py arc_reflection_prompt
```

## ⚙️ Setup Requirements

### 1. Install Dependencies

```bash
pip install google-generativeai langchain-google-genai requests
```

### 2. Environment Variables

```bash
# Set your Gemini API key for testing
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Optional: Ollama Setup

For local LLM testing, install and run Ollama:

```bash
# Install Ollama (see https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., llama3.2)
ollama pull llama3.2

# Start Ollama server (runs on localhost:11434)
ollama serve
```

## 📋 Test Output Example

When you run a test, you'll see output like this:

```
============================================================
TESTING APPLY PROMPT
============================================================

Generated Prompt:
----------------------------------------
You are an expert ARC executor.

====================
Task test_001
====================

Rule Summary:
Copy the input grid and change all 0s to 1s
[... full prompt content ...]
----------------------------------------

1. Testing with Gemini API:
🔄 Testing with Gemini API...
✅ Gemini API Response:
[... model response ...]

2. Testing with Ollama API:
🔄 Testing with Ollama API (llama3.2)...
✅ Ollama API Response:
[... model response ...]

3. Testing with LangChain:
🔄 Testing with LangChain (Gemini)...
✅ LangChain Response:
[... model response ...]
```

## 🛠️ Customization

### Modify Test Data

Each prompt file contains sample test data that you can customize:

```python
# Example from apply_prompt.py
sample_final_json = {
    "rule_summary": "Your custom rule description",
    "step_by_step_rule": [
        "1) Your first step",
        "2) Your second step"
    ],
    "pseudocode": "Your pseudocode here"
}

sample_task = {
    "train": [
        {
            "input": [[0, 1], [1, 0]],
            "output": [[1, 0], [0, 1]]
        }
    ],
    "test": [
        {"input": [[1, 1], [0, 0]]}
    ]
}
```

### Change Model Settings

```python
# Use different Ollama models
test_with_ollama_api(test_prompt, model_name="llama3.1")
test_with_ollama_api(test_prompt, model_name="mistral")

# Use different Gemini models  
model = genai.GenerativeModel('gemini-1.5-pro')  # Instead of gemini-1.5-flash
```

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**: The lint warnings about unresolved imports are expected - the imports are wrapped in try/except blocks and will only be used if the packages are installed.

2. **Gemini API errors**: Make sure your `GEMINI_API_KEY` is set correctly:
   ```bash
   echo $GEMINI_API_KEY
   ```

3. **Ollama connection errors**: Ensure Ollama is running:
   ```bash
   curl http://localhost:11434/api/version
   ```

4. **LangChain errors**: Install the correct LangChain packages:
   ```bash
   pip install langchain-google-genai langchain-core
   ```

### Environment Check

The test suite includes an environment checker:

```bash
python test_all_prompts.py
# Will show:
# 🔍 Environment Check:
#    GEMINI_API_KEY: ✅ Set / ❌ Not set  
#    Ollama Server: ✅ Running / ❌ Not running
```

## 📊 Test Results

The testing suite provides detailed feedback:
- **Individual test results** for each API method
- **Success/failure indicators** with clear error messages  
- **Comprehensive summary** showing overall test statistics
- **Detailed error information** for troubleshooting

## 🎯 Use Cases

### Development & Debugging
- Test prompt changes immediately
- Verify prompt formatting and structure
- Compare responses across different models

### Model Evaluation  
- Compare Gemini vs Ollama performance
- Test prompt effectiveness with different models
- Validate LangChain integration

### CI/CD Integration
- Automated prompt testing in pipelines  
- Regression testing for prompt modifications
- Quality assurance for ARC solver components

---

**Happy Testing! 🚀**