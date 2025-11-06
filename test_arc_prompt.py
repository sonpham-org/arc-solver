#!/usr/bin/env python3
"""
Test script for ARC prompt generation and model execution.

This script loads an ARC task from the training challenges dataset,
generates a prompt using build_arc_prompt, and runs it with a selected model.
"""

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import google.generativeai as genai
from dotenv import load_dotenv

from model_configs import MODEL_CONFIGS, is_known_model
from prompts import build_arc_prompt
from utils import sanitize_task

# ANSI color codes for terminal output
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def print_prompt_in_blue(prompt):
    """Print the prompt in blue color."""
    if PRINT_INPUT:
        print(f"\n{BLUE}{'='*60}")
        print("PROMPT SENT TO MODEL:")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}{RESET}")
    else:
        print(f"{BLUE}[Prompt sent to model - hidden]{RESET}")


def count_tokens_simple(text):
    """Simple token counting approximation (words * 1.3 for rough estimate)."""
    if not text:
        return 0
    # Simple approximation: split by whitespace and multiply by 1.3
    words = len(text.split())
    return int(words * 1.3)


def print_token_count_in_blue(response):
    """Print the token count in blue color."""
    if response:
        token_count = count_tokens_simple(response)
        print(f"\n{BLUE}{'='*60}")
        print(f"GENERATED TOKENS: {token_count:,}")
        print(f"{'='*60}{RESET}")
    else:
        print(f"\n{BLUE}{'='*60}")
        print("GENERATED TOKENS: 0 (no response)")
        print(f"{'='*60}{RESET}")


def extract_json_from_response(response):
    """Extract JSON from response text, looking for ```json ``` blocks."""
    if not response:
        return None
    
    # Look for ```json ... ``` blocks
    import re
    json_pattern = r'```json\s*\n(.*?)\n```'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if not matches:
        return None
    
    # Try to parse the first JSON block found
    try:
        json_text = matches[0].strip()
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"{RED}Error parsing JSON: {e}{RESET}")
        return None


def validate_json_structure(data):
    """Validate that the JSON has the expected structure."""
    if not isinstance(data, dict):
        return False
    
    if "step_by_step_transformations" not in data:
        return False
    
    steps = data["step_by_step_transformations"]
    if not isinstance(steps, list):
        return False
    
    for step in steps:
        if not isinstance(step, dict):
            return False
        required_fields = ["step_number", "description", "python_code", "example_input", "example_output"]
        if not all(field in step for field in required_fields):
            return False
    
    return True


def execute_transformation_code(code_lines, input_grid):
    """Execute the transformation code on an input grid."""
    try:
        # Join code lines into a single script
        code = '\n'.join(code_lines)
        
        # Create a local namespace for execution
        local_namespace = {"input_grid": input_grid}
        
        # Execute the code
        exec(code, {"__builtins__": __builtins__, "np": __import__("numpy")}, local_namespace)
        
        # Look for the transform function and call it
        if 'transform' in local_namespace:
            transform_func = local_namespace['transform']
            result = transform_func(input_grid)
            return result, None  # Success: return result and no error
        else:
            error_msg = "No 'transform' function found in generated code"
            print(f"{RED}Error: {error_msg}{RESET}")
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Error executing transformation code: {str(e)}"
        print(f"{RED}{error_msg}{RESET}")
        return None, error_msg


def calculate_grid_overlap(predicted, expected):
    """Calculate percentage overlap between two grids."""
    if not predicted or not expected:
        return 0.0
    
    if len(predicted) != len(expected):
        return 0.0
    
    total_cells = 0
    matching_cells = 0
    
    for i, (pred_row, exp_row) in enumerate(zip(predicted, expected)):
        if len(pred_row) != len(exp_row):
            return 0.0
        
        for j, (pred_cell, exp_cell) in enumerate(zip(pred_row, exp_row)):
            total_cells += 1
            if pred_cell == exp_cell:
                matching_cells += 1
    
    return (matching_cells / total_cells) * 100.0 if total_cells > 0 else 0.0


def test_transformations_on_task(json_data, task_data):
    """Test the transformations from JSON on all training examples in the task."""
    if not validate_json_structure(json_data):
        print(f"{RED}Invalid JSON structure{RESET}")
        return False
    
    steps = json_data["step_by_step_transformations"]
    if not steps:
        print(f"{RED}No transformation steps found{RESET}")
        return False
    
    # Use the last step's python code (most complete transformation)
    last_step = steps[-1]
    python_code = last_step["python_code"]
    
    training_examples = task_data.get("train", [])
    if not training_examples:
        print(f"{RED}No training examples found in task{RESET}")
        return False
    
    print(f"\n{GREEN}Testing transformation on {len(training_examples)} training examples...{RESET}")
    
    all_successful = True
    total_overlap = 0.0
    
    for i, example in enumerate(training_examples):
        input_grid = example["input"]
        expected_output = example["output"]
        
        print(f"\nTesting example {i+1}:")
        
        # Execute transformation
        predicted_output, error = execute_transformation_code(python_code, input_grid)
        
        if predicted_output is None:
            print(f"{RED}Failed to execute transformation on example {i+1}{RESET}")
            all_successful = False
            # Store error details for potential refinement
            if not hasattr(test_transformations_on_task, 'first_error'):
                test_transformations_on_task.first_error = {
                    'error': error,
                    'example_idx': i,
                    'input_grid': input_grid,
                    'expected_output': expected_output
                }
            continue
        
        # Calculate overlap
        overlap = calculate_grid_overlap(predicted_output, expected_output)
        total_overlap += overlap
        
        print(f"  Overlap: {overlap:.1f}%")
        if overlap == 100.0:
            print(f"  {GREEN}✓ Perfect match!{RESET}")
        elif overlap >= 90.0:
            print(f"  {GREEN}✓ Very close match{RESET}")
        elif overlap >= 70.0:
            print(f"  {BLUE}~ Good match{RESET}")
        else:
            print(f"  {RED}✗ Poor match{RESET}")
    
    if all_successful:
        avg_overlap = total_overlap / len(training_examples)
        print(f"\n{GREEN}Successfully transformed all examples!{RESET}")
        print(f"{GREEN}Average overlap: {avg_overlap:.1f}%{RESET}")
    else:
        print(f"\n{RED}Some transformations failed{RESET}")
    
    return all_successful


def test_on_test_examples(json_data, task_data, task_id, solutions_data):
    """Test the transformations on test examples and compare with solutions."""
    if not validate_json_structure(json_data):
        print(f"{RED}Invalid JSON structure{RESET}")
        return False
    
    steps = json_data["step_by_step_transformations"]
    if not steps:
        print(f"{RED}No transformation steps found{RESET}")
        return False
    
    # Use the last step's python code (most complete transformation)
    last_step = steps[-1]
    python_code = last_step["python_code"]
    
    test_examples = task_data.get("test", [])
    if not test_examples:
        print(f"{RED}No test examples found in task{RESET}")
        return False
    
    # Get solutions for this task
    task_solutions = solutions_data.get(task_id, [])
    if not task_solutions:
        print(f"{RED}No solutions found for task {task_id}{RESET}")
        return False
    
    if len(test_examples) != len(task_solutions):
        print(f"{RED}Mismatch: {len(test_examples)} test examples but {len(task_solutions)} solutions{RESET}")
        return False
    
    print(f"\n{GREEN}Testing on {len(test_examples)} test examples...{RESET}")
    
    all_successful = True
    total_overlap = 0.0
    
    for i, (test_example, expected_output) in enumerate(zip(test_examples, task_solutions)):
        input_grid = test_example["input"]
        
        print(f"\nTesting test example {i+1}:")
        
        # Execute transformation
        predicted_output, error = execute_transformation_code(python_code, input_grid)
        
        if predicted_output is None:
            print(f"{RED}Failed to execute transformation on test example {i+1}{RESET}")
            all_successful = False
            # Store error details for potential refinement
            if not hasattr(test_on_test_examples, 'first_error'):
                test_on_test_examples.first_error = {
                    'error': error,
                    'example_idx': i,
                    'input_grid': input_grid,
                    'expected_output': expected_output
                }
            continue
        
        # Calculate overlap
        overlap = calculate_grid_overlap(predicted_output, expected_output)
        total_overlap += overlap
        
        print(f"  Overlap: {overlap:.1f}%")
        if overlap == 100.0:
            print(f"  {GREEN}✓ Perfect match!{RESET}")
        elif overlap >= 90.0:
            print(f"  {GREEN}✓ Very close match{RESET}")
        elif overlap >= 70.0:
            print(f"  {BLUE}~ Good match{RESET}")
        else:
            print(f"  {RED}✗ Poor match{RESET}")
    
    if all_successful:
        avg_overlap = total_overlap / len(test_examples)
        print(f"\n{GREEN}Successfully transformed all test examples!{RESET}")
        print(f"{GREEN}Average test overlap: {avg_overlap:.1f}%{RESET}")
    else:
        print(f"\n{RED}Some test transformations failed{RESET}")
    
    return all_successful


def create_error_refinement_prompt(original_json, error_details, failed_example_idx, input_grid, expected_output):
    """Create a follow-up prompt to refine the JSON based on errors."""
    
    prompt = f"""The previous transformation code failed with the following error:

ERROR DETAILS:
{error_details}

FAILED ON EXAMPLE {failed_example_idx + 1}:
Input grid:
{json.dumps(input_grid, indent=2)}

Expected output:
{json.dumps(expected_output, indent=2)}

ORIGINAL JSON RESPONSE:
```json
{json.dumps(original_json, indent=2)}
```

Please analyze the error and provide a CORRECTED JSON response that fixes the issue. The corrected code should:

1. Handle the specific error that occurred
2. Work correctly on the failed example
3. Maintain the same structure with "step_by_step_transformations"
4. Include working Python code that can execute without errors

Please provide the complete corrected JSON in a ```json ``` block."""

    return prompt


def ask_model_for_refinement(model, provider, original_json, error_details, failed_example_idx, input_grid, expected_output):
    """Ask the model to refine the JSON based on the error."""
    
    print(f"\n{BLUE}{'='*60}")
    print("REQUESTING MODEL REFINEMENT...")
    print(f"{'='*60}{RESET}")
    
    refinement_prompt = create_error_refinement_prompt(
        original_json, error_details, failed_example_idx, input_grid, expected_output
    )
    
    print_prompt_in_blue(refinement_prompt)
    
    # Call the appropriate model function
    if model == "llama3.1" or provider == "ollama":
        response = run_with_ollama(refinement_prompt, model=model)
    elif provider == "google" or provider == "learnlm":
        response = run_with_gemini(refinement_prompt, model)
    else:
        response = run_with_other_model(refinement_prompt, model)
    
    return response


def run_single_task(task_id, task_data, solutions, model, provider):
    """Run a single task and return results."""
    print(f"\n{'='*80}")
    print(f"PROCESSING TASK: {task_id}")
    print(f"{'='*80}")
    
    # Apply sanitization if enabled
    if SANITIZE_TASK:
        print(f"\nSanitizing task {task_id}...")
        sanitized_task_data, sanitized_task_id = sanitize_task(task_data, task_id)
        working_task_data = sanitized_task_data
        
        if SANITIZE_ID:
            working_task_id = sanitized_task_id
            print(f"Using sanitized task ID: {working_task_id}")
        else:
            working_task_id = task_id
            print(f"Using original task ID: {working_task_id}")
        
        print("Numbers have been mapped to random characters")
    else:
        print(f"\nUsing original task representation...")
        working_task_data, working_task_id = task_data, task_id
    
    # Generate prompt
    print(f"\nGenerating prompt for task {working_task_id}...")
    prompt = build_arc_prompt({working_task_id: working_task_data}, working_task_id)
    
    print(f"Task details (ID: {task_id}):")
    print(f"  - Training examples: {len(task_data.get('train', []))}")
    print(f"  - Test examples: {len(task_data.get('test', []))}")
    
    # Display the prompt that will be sent to the model
    print_prompt_in_blue(prompt)
    
    # Run with selected model
    if model == "llama3.1" or provider == "ollama":
        response = run_with_ollama(prompt, model=model)
    elif provider == "google" or provider == "learnlm":
        response = run_with_gemini(prompt, model)
    else:
        response = run_with_other_model(prompt, model)
    
    if not response:
        print("No response received from model.")
        return {
            'task_id': task_id,
            'success': False,
            'training_success': False,
            'test_success': False,
            'test_overlap': 0.0,
            'error': 'No model response'
        }
    
    if PRINT_OUTPUT:
        print(f"\n{'='*60}")
        print("MODEL RESPONSE:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}")
    else:
        print(f"{GREEN}[Model response received - hidden]{RESET}")
    
    # Display token count in blue
    print_token_count_in_blue(response)
    
    # Extract and validate JSON
    print(f"\n{'='*60}")
    print("JSON EXTRACTION AND VALIDATION:")
    print(f"{'='*60}")
    
    extracted_json = extract_json_from_response(response)
    
    if not extracted_json:
        print(f"{RED}Failed to extract JSON from response{RESET}")
        return {
            'task_id': task_id,
            'success': False,
            'training_success': False,
            'test_success': False,
            'test_overlap': 0.0,
            'error': 'Failed to extract JSON'
        }
    
    print(f"{GREEN}Successfully extracted JSON!{RESET}")
    if PRINT_OUTPUT:
        print(json.dumps(extracted_json, indent=2))
    else:
        print(f"{BLUE}[JSON content hidden]{RESET}")
    
    # Test transformations on training examples
    print(f"\n{'='*60}")
    print("TESTING TRANSFORMATIONS ON TRAINING EXAMPLES:")
    print(f"{'='*60}")
    
    # Clear any previous error state
    if hasattr(test_transformations_on_task, 'first_error'):
        delattr(test_transformations_on_task, 'first_error')
    
    training_success = test_transformations_on_task(extracted_json, task_data)
    
    # Check if training failed and we should ask for refinement
    if not training_success and hasattr(test_transformations_on_task, 'first_error'):
        error_info = test_transformations_on_task.first_error
        
        print(f"\n{RED}Training failed. Asking model for refinement...{RESET}")
        
        refinement_response = ask_model_for_refinement(
            model, provider,
            extracted_json, 
            error_info['error'],
            error_info['example_idx'],
            error_info['input_grid'],
            error_info['expected_output']
        )
        
        if refinement_response:
            refined_json = extract_json_from_response(refinement_response)
            if refined_json:
                print(f"\n{GREEN}Extracted refined JSON! Testing again...{RESET}")
                
                # Clear previous error
                if hasattr(test_transformations_on_task, 'first_error'):
                    delattr(test_transformations_on_task, 'first_error')
                
                print(f"\n{'='*60}")
                print("RETESTING WITH REFINED SOLUTION:")
                print(f"{'='*60}")
                
                training_success = test_transformations_on_task(refined_json, task_data)
                extracted_json = refined_json  # Use refined version for test examples
    
    # Test on test examples with solutions
    print(f"\n{'='*60}")
    print("TESTING TRANSFORMATIONS ON TEST EXAMPLES:")
    print(f"{'='*60}")
    
    # Clear any previous error state for test examples
    if hasattr(test_on_test_examples, 'first_error'):
        delattr(test_on_test_examples, 'first_error')
    
    test_success = test_on_test_examples(extracted_json, task_data, task_id, solutions)
    
    # Calculate test overlap for statistics
    test_overlap = 0.0
    if test_success:
        # Get detailed test overlap
        steps = extracted_json.get("step_by_step_transformations", [])
        if steps:
            last_step = steps[-1]
            python_code = last_step["python_code"]
            test_examples = task_data.get("test", [])
            task_solutions = solutions.get(task_id, [])
            
            if test_examples and task_solutions and len(test_examples) == len(task_solutions):
                total_overlap = 0.0
                for test_example, expected_output in zip(test_examples, task_solutions):
                    predicted_output, _ = execute_transformation_code(python_code, test_example["input"])
                    if predicted_output is not None:
                        overlap = calculate_grid_overlap(predicted_output, expected_output)
                        total_overlap += overlap
                test_overlap = total_overlap / len(test_examples)
    
    # Overall success summary for this task
    print(f"\n{'='*60}")
    print(f"TASK {task_id} RESULTS:")
    print(f"{'='*60}")
    
    overall_success = training_success and test_success
    
    if overall_success:
        print(f"{GREEN}🎉 TASK PASSED! Solution works on both training and test examples.{RESET}")
    elif training_success:
        print(f"{BLUE}⚠️  Training examples passed, but test examples failed.{RESET}")
    elif test_success:
        print(f"{BLUE}⚠️  Test examples passed, but training examples failed.{RESET}")
    else:
        print(f"{RED}❌ Both training and test examples failed.{RESET}")
    
    return {
        'task_id': task_id,
        'success': overall_success,
        'training_success': training_success,
        'test_success': test_success,
        'test_overlap': test_overlap,
        'error': None
    }


def run_single_task_wrapper(args):
    """Wrapper function for parallel execution."""
    task_id, task_data, solutions, model, provider = args
    return run_single_task(task_id, task_data, solutions, model, provider)

# ============================================================================
# CONFIGURATION VARIABLES - Modify these as needed
# ============================================================================

# Task selection: Set to specific index (0-399) or None for random selection
TASK_INDEX = None  # Example: 0, 42, 150, or None

# Number of tasks to run (if TASK_INDEX is None, will run this many random tasks)
NUMBER_OF_TASKS = 1  # Set to 1 for single task, or higher for multiple tasks

# Model selection: Choose from available models in model_configs.py
MODEL = "gemini-2.5-flash-lite-preview-06-17"  # Example: "llama3.1", "gemini-2.5-flash", etc.

# Path to the ARC challenges and solutions JSON files
CHALLENGES_PATH = "data/arc-2024/arc-agi_training_challenges.json"
SOLUTIONS_PATH = "data/arc-2024/arc-agi_training_solutions.json"

# Random seed for reproducible task selection (set to None for random)
RANDOM_SEED = None  # Example: 42, 123, or None

# Model parameters (for Ollama models)
TEMPERATURE = 0.6  # Controls randomness (0.0 = deterministic, 1.0 = very random)
NUM_PREDICT = -1  # Maximum number of tokens to generate (-1 = no limit)

# Task sanitization: Whether to use sanitized task data and ID
SANITIZE_TASK = False  # Set to False to use original task representation

# Sanitized ID: Whether to use sanitized task ID (only applies when SANITIZE_TASK is True)
SANITIZE_ID = True  # Set to False to keep original task ID even when using sanitized data

# Execution settings
PARALLEL = True  # Set to True to run tasks in parallel (faster but less readable output)
PRINT_INPUT = False  # Set to False to hide prompts sent to the language model
PRINT_OUTPUT = False  # Set to False to hide model responses and detailed output


def load_arc_tasks(filepath):
    """Load ARC tasks from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_arc_solutions(filepath):
    """Load ARC solutions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_task_by_index(tasks, index=None):
    """Get a task by index or randomly select one if index is None."""
    task_ids = list(tasks.keys())
    
    if index is None:
        # Random selection
        selected_id = random.choice(task_ids)
        print(f"Randomly selected task: {selected_id}")
    else:
        # Specific index
        if index < 0 or index >= len(task_ids):
            raise ValueError(f"Index {index} out of range. Available tasks: 0-{len(task_ids)-1}")
        selected_id = task_ids[index]
        print(f"Selected task by index {index}: {selected_id}")
    
    return selected_id, tasks[selected_id]


def get_multiple_tasks(tasks, num_tasks, specific_index=None):
    """Get multiple tasks for batch processing."""
    task_ids = list(tasks.keys())
    selected_tasks = []
    
    if specific_index is not None:
        # Single specific task
        if specific_index < 0 or specific_index >= len(task_ids):
            raise ValueError(f"Index {specific_index} out of range. Available tasks: 0-{len(task_ids)-1}")
        selected_id = task_ids[specific_index]
        selected_tasks.append((selected_id, tasks[selected_id]))
        print(f"Selected specific task: {selected_id}")
    else:
        # Random selection
        if num_tasks > len(task_ids):
            num_tasks = len(task_ids)
            print(f"Limiting to {num_tasks} tasks (max available)")
        
        selected_ids = random.sample(task_ids, num_tasks)
        for task_id in selected_ids:
            selected_tasks.append((task_id, tasks[task_id]))
        
        print(f"Randomly selected {len(selected_tasks)} tasks: {[tid for tid, _ in selected_tasks]}")
    
    return selected_tasks


def run_with_ollama(prompt, model="llama3.1"):
    """Run the prompt with Ollama using API if available, otherwise fallback to CLI."""
    
    print(f"\n{'='*60}")
    print(f"Running with Ollama model: {model}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max tokens: {'unlimited' if NUM_PREDICT == -1 else NUM_PREDICT}")
    print(f"{'='*60}")
    
    try:
        # Use Ollama API
        api_url = "http://localhost:11434/api/generate"
        
        # Prepare options - only include num_predict if it's not -1
        options = {"temperature": TEMPERATURE}
        if NUM_PREDICT != -1:
            options["num_predict"] = NUM_PREDICT
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        response = requests.post(api_url, json=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama API. Make sure Ollama is running.")
        print("You can start it with: ollama serve")
        print("Falling back to CLI...")
    except Exception as e:
        print(f"Error with Ollama API: {e}")
        print("Falling back to CLI...")
    
    # Fallback to simple ollama run without parameters
    try:
        print("Using simple ollama run command (parameters not supported)...")
        cmd = ["ollama", "run", model]
        
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: Ollama not found. Please install Ollama first.")
        return None


def run_with_gemini(prompt, model):
    """Run the prompt with Google Gemini API."""
    
    print(f"\n{'='*60}")
    print(f"Running with Gemini model: {model}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max tokens: {'unlimited' if NUM_PREDICT == -1 else NUM_PREDICT}")
    print(f"{'='*60}")
    

    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return None
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Map model names to their actual Gemini model IDs
        model_mapping = {
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-2.5-flash-lite-preview-06-17": "gemini-2.5-flash-lite",
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
            "learnlm-2.0-flash": "learnlm-2.0-flash"
        }
        
        actual_model = model_mapping.get(model, model)
        
        # Create the model with generation config
        generation_config = {"temperature": TEMPERATURE}
        if NUM_PREDICT != -1:
            generation_config["max_output_tokens"] = NUM_PREDICT
        
        model_instance = genai.GenerativeModel(
            model_name=actual_model,
            generation_config=generation_config
        )
        
        # Generate response
        response = model_instance.generate_content(prompt)
        
        if response.text:
            return response.text
        else:
            print("Error: No response text received from Gemini")
            if hasattr(response, 'prompt_feedback'):
                print(f"Prompt feedback: {response.prompt_feedback}")
            return None
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None


def run_with_other_model(prompt, model):
    """Placeholder for other model integrations."""
    print(f"\n{'='*60}")
    print(f"Model '{model}' integration not implemented yet.")
    print(f"This would use the model configuration:")
    print(f"Provider: {MODEL_CONFIGS[model].get('provider', 'unknown')}")
    print(f"Description: {MODEL_CONFIGS[model].get('description', 'No description')}")
    print(f"{'='*60}")
    
    # For now, just print the prompt
    print("\nGenerated prompt:")
    print(prompt)
    
    return "Model integration not implemented yet."


def main():
    # Load environment variables from .env file
    load_dotenv()
    print("Loaded environment variables from .env file")
    
    # Debug: Check if API key is loaded
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"✓ GEMINI_API_KEY loaded (first 10 chars: {gemini_key[:10]}...)")
    else:
        print("✗ GEMINI_API_KEY not found in environment")
    
    # Set random seed if provided
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"Using random seed: {RANDOM_SEED}")
    
    # Check if model is known
    if not is_known_model(MODEL):
        print(f"Error: Unknown model '{MODEL}'")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    # Load ARC challenges
    challenges_path = Path(CHALLENGES_PATH)
    if not challenges_path.exists():
        print(f"Error: Challenges file not found: {challenges_path}")
        sys.exit(1)
    
    try:
        tasks = load_arc_tasks(challenges_path)
        print(f"Loaded {len(tasks)} ARC tasks from {challenges_path}")
    except Exception as e:
        print(f"Error loading tasks: {e}")
        sys.exit(1)
    
    # Load ARC solutions
    solutions_path = Path(SOLUTIONS_PATH)
    if not solutions_path.exists():
        print(f"Error: Solutions file not found: {solutions_path}")
        sys.exit(1)
    
    try:
        solutions = load_arc_solutions(solutions_path)
        print(f"Loaded solutions for {len(solutions)} tasks from {solutions_path}")
    except Exception as e:
        print(f"Error loading solutions: {e}")
        sys.exit(1)
    
    # Get model configuration
    model_config = MODEL_CONFIGS.get(MODEL, {})
    provider = model_config.get('provider', 'unknown')
    
    # Select tasks to run
    try:
        selected_tasks = get_multiple_tasks(tasks, NUMBER_OF_TASKS, TASK_INDEX)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*100}")
    print(f"RUNNING {len(selected_tasks)} TASK(S) WITH MODEL: {MODEL}")
    print(f"{'='*100}")
    
    # Process all tasks
    all_results = []
    successful_tasks = 0
    total_test_overlap = 0.0
    
    if PARALLEL and len(selected_tasks) > 1:
        print(f"\n{BLUE}Running {len(selected_tasks)} tasks in parallel...{RESET}")
        
        # Prepare arguments for parallel execution
        task_args = [(task_id, task_data, solutions, MODEL, provider) 
                    for task_id, task_data in selected_tasks]
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=min(len(selected_tasks), 4)) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_single_task_wrapper, args): args[0] 
                            for args in task_args}
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_task)):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    if result['test_success']:
                        successful_tasks += 1
                        total_test_overlap += result['test_overlap']
                    
                    # Show progress
                    status = "✓" if result['test_success'] else "✗"
                    print(f"{status} Completed {task_id} ({i+1}/{len(selected_tasks)})")
                    
                except Exception as e:
                    print(f"{RED}✗ Error processing {task_id}: {e}{RESET}")
                    all_results.append({
                        'task_id': task_id,
                        'success': False,
                        'training_success': False,
                        'test_success': False,
                        'test_overlap': 0.0,
                        'error': str(e)
                    })
        
        # Sort results by task_id for consistent output
        all_results.sort(key=lambda x: x['task_id'])
        
    else:
        # Sequential processing
        for i, (task_id, task_data) in enumerate(selected_tasks):
            print(f"\n{'='*100}")
            print(f"TASK {i+1}/{len(selected_tasks)}")
            print(f"{'='*100}")
            
            result = run_single_task(task_id, task_data, solutions, MODEL, provider)
            all_results.append(result)
            
            if result['test_success']:
                successful_tasks += 1
                total_test_overlap += result['test_overlap']
    
    # Print final statistics
    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    print(f"\n{BLUE}TASK PERFORMANCE:{RESET}")
    print(f"  Total tasks processed: {len(selected_tasks)}")
    print(f"  Tasks with successful test predictions: {successful_tasks}")
    print(f"  Success rate: {(successful_tasks / len(selected_tasks) * 100):.1f}%")
    
    if successful_tasks > 0:
        avg_test_overlap = total_test_overlap / successful_tasks
        print(f"  Average test overlap (successful tasks): {avg_test_overlap:.1f}%")
    
    overall_test_overlap = total_test_overlap / len(selected_tasks) if selected_tasks else 0
    print(f"  Overall average test overlap: {overall_test_overlap:.1f}%")
    
    print(f"\n{BLUE}DETAILED BREAKDOWN:{RESET}")
    training_successes = sum(1 for r in all_results if r['training_success'])
    test_successes = sum(1 for r in all_results if r['test_success'])
    
    print(f"  Training examples passed: {training_successes}/{len(selected_tasks)} ({training_successes/len(selected_tasks)*100:.1f}%)")
    print(f"  Test examples passed: {test_successes}/{len(selected_tasks)} ({test_successes/len(selected_tasks)*100:.1f}%)")
    
    print(f"\n{BLUE}TASK RESULTS:{RESET}")
    for result in all_results:
        status = "✓" if result['test_success'] else "✗"
        overlap_str = f"({result['test_overlap']:.1f}%)" if result['test_success'] else "(Failed)"
        print(f"  {status} {result['task_id']} {overlap_str}")
    
    # Overall success message
    if successful_tasks == len(selected_tasks):
        print(f"\n{GREEN}🎉 ALL TASKS PASSED! Perfect success rate!{RESET}")
    elif successful_tasks > len(selected_tasks) * 0.7:
        print(f"\n{GREEN}🟢 EXCELLENT PERFORMANCE! {successful_tasks}/{len(selected_tasks)} tasks passed.{RESET}")
    elif successful_tasks > len(selected_tasks) * 0.3:
        print(f"\n{BLUE}🟡 MODERATE PERFORMANCE. {successful_tasks}/{len(selected_tasks)} tasks passed.{RESET}")
    else:
        print(f"\n{RED}🔴 POOR PERFORMANCE. Only {successful_tasks}/{len(selected_tasks)} tasks passed.{RESET}")
    
    print(f"\n{'='*100}")
    print("RUN COMPLETED")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()