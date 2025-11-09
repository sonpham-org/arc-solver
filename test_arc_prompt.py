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
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv

from model_configs import MODEL_CONFIGS, is_known_model, estimate_cost
from prompts import build_arc_prompt
from utils import sanitize_task, calculate_results

# ============================================================================
# CONFIGURATION VARIABLES - Modify these as needed
# ============================================================================

# Task selection: Set to specific index (0-399), task ID string, or None for random selection
TASK_INDEX = None  # Example: 0, 42, 150, "00d62c1b", or None

# Number of tasks to run (if TASK_INDEX is None, will run this many random tasks)
NUMBER_OF_TASKS = -1  # Set to 1 for single task, or higher for multiple tasks

# Model selection: Choose from available models in model_configs.py
MODEL = "gemini-2.5-flash-lite-preview-06-17"  # The other is "qwen2.5:32b"

# Path to the ARC challenges and solutions JSON files
YEAR = 2025
if YEAR == 2024:
    TRAINING_CHALLENGES_PATH = "data/arc-2024/arc-agi_training_challenges.json"
    TRAINING_SOLUTIONS_PATH = "data/arc-2024/arc-agi_training_solutions.json"
    EVALUATION_CHALLENGES_PATH = "data/arc-2024/arc-agi_evaluation_challenges.json"
    EVALUATION_SOLUTIONS_PATH = "data/arc-2024/arc-agi_evaluation_solutions.json"
else:
    TRAINING_CHALLENGES_PATH = "data/arc-2025/arc-agi_training_challenges.json"
    TRAINING_SOLUTIONS_PATH = "data/arc-2025/arc-agi_training_solutions.json"
    EVALUATION_CHALLENGES_PATH = "data/arc-2025/arc-agi_evaluation_challenges.json"
    EVALUATION_SOLUTIONS_PATH = "data/arc-2025/arc-agi_evaluation_solutions.json"

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

# Enhanced processing settings
USE_SMART_ROUTER = True  # Set to True to use smart router for response enhancement with any model
MAX_REFLECTIONS = 5  # Maximum number of reflection cycles for complete enhancement

# ANSI color codes for terminal output
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def parse_arguments():
    """Parse command line arguments with defaults from ALL_CAPS variables."""
    parser = argparse.ArgumentParser(description="ARC Test Script")
    
    parser.add_argument("--task-index", type=str, default=TASK_INDEX, 
                        help=f"Index (0-399) or task ID string to run (default: {TASK_INDEX})")
    parser.add_argument("--number-of-tasks", type=int, default=NUMBER_OF_TASKS,
                        help=f"Number of tasks to run (default: {NUMBER_OF_TASKS})")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Model to use for inference (default: {MODEL})")
    parser.add_argument("--training-challenges-path", type=str, default=TRAINING_CHALLENGES_PATH,
                        help=f"Path to training challenges JSON file (default: {TRAINING_CHALLENGES_PATH})")
    parser.add_argument("--training-solutions-path", type=str, default=TRAINING_SOLUTIONS_PATH,
                        help=f"Path to training solutions JSON file (default: {TRAINING_SOLUTIONS_PATH})")
    parser.add_argument("--evaluation-challenges-path", type=str, default=EVALUATION_CHALLENGES_PATH,
                        help=f"Path to evaluation challenges JSON file (default: {EVALUATION_CHALLENGES_PATH})")
    parser.add_argument("--evaluation-solutions-path", type=str, default=EVALUATION_SOLUTIONS_PATH,
                        help=f"Path to evaluation solutions JSON file (default: {EVALUATION_SOLUTIONS_PATH})")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for task selection (default: {RANDOM_SEED})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Temperature for model generation (default: {TEMPERATURE})")
    parser.add_argument("--num-predict", type=int, default=NUM_PREDICT,
                        help=f"Number of tokens to predict (-1 for default) (default: {NUM_PREDICT})")
    parser.add_argument("--sanitize-task", action="store_true", default=SANITIZE_TASK,
                        help=f"Use sanitized task representation (default: {SANITIZE_TASK})")
    parser.add_argument("--sanitize-id", action="store_true", default=SANITIZE_ID,
                        help=f"Use sanitized task ID (default: {SANITIZE_ID})")
    parser.add_argument("--parallel", action="store_true", default=PARALLEL,
                        help=f"Run tasks in parallel (default: {PARALLEL})")
    parser.add_argument("--print-input", action="store_true", default=PRINT_INPUT,
                        help=f"Print input prompts (default: {PRINT_INPUT})")
    parser.add_argument("--print-output", action="store_true", default=PRINT_OUTPUT,
                        help=f"Print model outputs (default: {PRINT_OUTPUT})")
    parser.add_argument("--use-smart-router", action="store_true", default=USE_SMART_ROUTER,
                        help=f"Use smart router for response enhancement with any model (default: {USE_SMART_ROUTER})")
    parser.add_argument("--max-reflections", type=int, default=MAX_REFLECTIONS,
                        help=f"Maximum number of reflection cycles for complete enhancement (default: {MAX_REFLECTIONS})")
    
    return parser.parse_args()

# ============================================================================
# HELPERS
# ============================================================================

def print_prompt_in_blue(prompt, print_input=False):
    """Print the prompt in blue color."""
    if print_input:
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


def grid_to_string_lines(grid):
    """Convert grid to array of line-separated strings."""
    if not grid:
        return []
    return [''.join(map(str, row)) for row in grid]


def is_valid_prediction(predicted):
    """Check if prediction is a valid 2D grid."""
    if not predicted:
        return False
    if not isinstance(predicted, list):
        return False
    if not all(isinstance(row, list) for row in predicted):
        return False
    if not predicted:
        return False
    row_length = len(predicted[0])
    return all(len(row) == row_length for row in predicted)


def calculate_grid_iou(predicted, expected):
    """Calculate intersection over union of grid dimensions."""
    if not predicted or not expected:
        return 0.0
    
    pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
    exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
    
    if pred_h == 0 or pred_w == 0 or exp_h == 0 or exp_w == 0:
        return 0.0
    
    # Calculate intersection and union of dimensions
    intersection_h = min(pred_h, exp_h)
    intersection_w = min(pred_w, exp_w)
    union_h = max(pred_h, exp_h)
    union_w = max(pred_w, exp_w)
    
    intersection_area = intersection_h * intersection_w
    union_area = union_h * union_w
    
    return (intersection_area / union_area) * 100.0 if union_area > 0 else 0.0


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


def execute_transformation_code(code_lines, input_grid, helper_functions=None):
    """Execute the transformation code on an input grid, optionally with helper functions."""
    try:
        # Start with helper functions if provided
        all_code_lines = []
        if helper_functions:
            if isinstance(helper_functions, list):
                all_code_lines.extend(helper_functions)
            else:
                all_code_lines.append(helper_functions)
            all_code_lines.append("")  # Add blank line for separation
        
        # Add the main transformation code
        if isinstance(code_lines, list):
            all_code_lines.extend(code_lines)
        else:
            all_code_lines.append(code_lines)
        
        # Join all code lines into a single script
        code = '\n'.join(all_code_lines)
        
        # Create a single namespace for execution
        namespace = {
            "__builtins__": __builtins__, 
            "np": __import__("numpy"),
            "input_grid": input_grid
        }
        
        # Execute the code in the single namespace
        exec(code, namespace)
        
        # Look for the transform function and call it
        if 'transform' in namespace:
            transform_func = namespace['transform']
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
    
    # Get helper functions from JSON
    helper_functions = json_data.get("helper_python_functions", [])
    
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
        predicted_output, error = execute_transformation_code(python_code, input_grid, helper_functions)
        
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
    
    # Get helper functions from JSON
    helper_functions = json_data.get("helper_python_functions", [])
    
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
        predicted_output, error = execute_transformation_code(python_code, input_grid, helper_functions)
        
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


def ask_model_for_refinement(model, provider, original_json, error_details, failed_example_idx, input_grid, expected_output, print_input=False):
    """Ask the model to refine the JSON based on the error."""
    
    print(f"\n{BLUE}{'='*60}")
    print("REQUESTING MODEL REFINEMENT...")
    print(f"{'='*60}{RESET}")
    
    refinement_prompt = create_error_refinement_prompt(
        original_json, error_details, failed_example_idx, input_grid, expected_output
    )
    
    print_prompt_in_blue(refinement_prompt, print_input)
    
    # Call the appropriate model function
    if model == "llama3.1" or provider == "ollama":
        response = run_with_ollama(refinement_prompt, model=model)
    elif provider == "google" or provider == "learnlm":
        response = run_with_gemini(refinement_prompt, model)
    else:
        response = run_with_other_model(refinement_prompt, model)
    
    return response


def process_training_examples(task_data, extracted_json):
    """Process training examples and return results in the new format."""
    training_examples = task_data.get("train", [])
    train_results = []
    
    if not validate_json_structure(extracted_json):
        # If JSON is invalid, mark all training examples as failed
        for i, example in enumerate(training_examples):
            train_entry = {
                'input': grid_to_string_lines(example["input"]),
                'output': grid_to_string_lines(example["output"]),
                'valid_predict': False,
                'predict': [],
                'iou': 0.0,
                'overlap': 0.0,
                'correct': False,
                'transformation_succeed': [False]
            }
            train_results.append(train_entry)
        return train_results
    
    steps = extracted_json["step_by_step_transformations"]
    if not steps:
        # No steps found, mark all as failed
        for i, example in enumerate(training_examples):
            train_entry = {
                'input': grid_to_string_lines(example["input"]),
                'output': grid_to_string_lines(example["output"]),
                'valid_predict': False,
                'predict': [],
                'iou': 0.0,
                'overlap': 0.0,
                'correct': False,
                'transformation_succeed': [False]
            }
            train_results.append(train_entry)
        return train_results
    
    # Get helper functions from JSON
    helper_functions = extracted_json.get("helper_python_functions", [])
    
    # Test each step's transformation on all training examples
    transformation_successes = []
    for step_idx, step in enumerate(steps):
        python_code = step.get("python_code", [])
        step_successes = []
        
        for example in training_examples:
            predicted_output, error = execute_transformation_code(python_code, example["input"], helper_functions)
            success = predicted_output is not None
            step_successes.append(success)
        
        transformation_successes.append(step_successes)
    
    # Use the last step for final predictions
    last_step = steps[-1]
    python_code = last_step.get("python_code", [])
    
    for i, example in enumerate(training_examples):
        input_grid = example["input"]
        expected_output = example["output"]
        
        predicted_output, error = execute_transformation_code(python_code, input_grid, helper_functions)
        
        valid_predict = predicted_output is not None
        if valid_predict:
            iou = calculate_grid_iou(predicted_output, expected_output)
            overlap = calculate_grid_overlap(predicted_output, expected_output)
            correct = (iou == 100.0 and overlap == 100.0)
            predict_lines = grid_to_string_lines(predicted_output)
        else:
            iou = 0.0
            overlap = 0.0
            correct = False
            predict_lines = []
        
        # Get transformation success for this example across all steps
        example_transform_successes = [step_successes[i] for step_successes in transformation_successes]
        
        train_entry = {
            'input': grid_to_string_lines(input_grid),
            'output': grid_to_string_lines(expected_output),
            'valid_predict': valid_predict,
            'predict': predict_lines,
            'iou': iou,
            'overlap': overlap,
            'correct': correct,
            'transformation_succeed': example_transform_successes
        }
        train_results.append(train_entry)
    
    return train_results


def process_test_examples(task_data, task_solutions, extracted_json):
    """Process test examples and return results in the new format."""
    test_examples = task_data.get("test", [])
    test_results = []
    
    if not validate_json_structure(extracted_json) or not task_solutions:
        # If JSON is invalid or no solutions, mark all test examples as failed
        for i, example in enumerate(test_examples):
            expected_output = task_solutions[i] if i < len(task_solutions) else []
            test_entry = {
                'input': grid_to_string_lines(example["input"]),
                'output': grid_to_string_lines(expected_output),
                'valid_predict': False,
                'predict': [],
                'iou': 0.0,
                'overlap': 0.0,
                'correct': False,
                'transformation_succeed': [False]
            }
            test_results.append(test_entry)
        return test_results
    
    steps = extracted_json["step_by_step_transformations"]
    if not steps:
        # No steps found, mark all as failed
        for i, example in enumerate(test_examples):
            expected_output = task_solutions[i] if i < len(task_solutions) else []
            test_entry = {
                'input': grid_to_string_lines(example["input"]),
                'output': grid_to_string_lines(expected_output),
                'valid_predict': False,
                'predict': [],
                'iou': 0.0,
                'overlap': 0.0,
                'correct': False,
                'transformation_succeed': [False]
            }
            test_results.append(test_entry)
        return test_results
    
    # Get helper functions from JSON
    helper_functions = extracted_json.get("helper_python_functions", [])
    
    # Test each step's transformation on all test examples
    transformation_successes = []
    for step_idx, step in enumerate(steps):
        python_code = step.get("python_code", [])
        step_successes = []
        
        for example in test_examples:
            predicted_output, error = execute_transformation_code(python_code, example["input"], helper_functions)
            success = predicted_output is not None
            step_successes.append(success)
        
        transformation_successes.append(step_successes)
    
    # Use the last step for final predictions
    last_step = steps[-1]
    python_code = last_step.get("python_code", [])
    
    for i, example in enumerate(test_examples):
        input_grid = example["input"]
        expected_output = task_solutions[i] if i < len(task_solutions) else []
        
        predicted_output, error = execute_transformation_code(python_code, input_grid, helper_functions)
        
        valid_predict = predicted_output is not None
        if valid_predict and expected_output:
            iou = calculate_grid_iou(predicted_output, expected_output)
            overlap = calculate_grid_overlap(predicted_output, expected_output)
            correct = (iou == 100.0 and overlap == 100.0)
            predict_lines = grid_to_string_lines(predicted_output)
        else:
            iou = 0.0
            overlap = 0.0
            correct = False
            predict_lines = grid_to_string_lines(predicted_output) if valid_predict else []
        
        # Get transformation success for this example across all steps
        example_transform_successes = [step_successes[i] for step_successes in transformation_successes]
        
        test_entry = {
            'input': grid_to_string_lines(input_grid),
            'output': grid_to_string_lines(expected_output),
            'valid_predict': valid_predict,
            'predict': predict_lines,
            'iou': iou,
            'overlap': overlap,
            'correct': correct,
            'transformation_succeed': example_transform_successes
        }
        test_results.append(test_entry)
    
    return test_results


def save_new_format_results(task_id, result_data, output_dir):
    """Save results in the new JSON format."""
    output_file = Path(output_dir) / f"{task_id}.json"
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    return output_file


def run_single_task(task_id, task_data, solutions, model, provider, output_dir=None, 
                   use_sanitized_task=False, use_sanitized_id=True, print_input=False, print_output=False,
                   use_smart_router=True, max_reflections=5):
    """Run a single task and return results."""
    print(f"\n{'='*80}")
    print(f"PROCESSING TASK: {task_id}")
    print(f"{'='*80}")
    
    # Apply sanitization if enabled
    if use_sanitized_task:
        print(f"\nSanitizing task {task_id}...")
        sanitized_task_data, sanitized_task_id = sanitize_task(task_data, task_id)
        working_task_data = sanitized_task_data
        
        if use_sanitized_id:
            working_task_id = sanitized_task_id
            print(f"Using sanitized task ID: {working_task_id}")
        else:
            working_task_id = task_id
            print(f"Using original task ID: {working_task_id}")
        
        print("Numbers have been mapped to random characters")
    else:
        print(f"\nUsing original task representation...")
        working_task_data, working_task_id = task_data, task_id
    
    # Determine the provider if not specified
    if not provider:
        from model_configs import MODEL_CONFIGS
        if model in MODEL_CONFIGS:
            provider = MODEL_CONFIGS[model].get("provider", "unknown")
        elif model == "llama3.1":
            provider = "ollama"
        else:
            provider = "unknown"
    
    # Generate prompt
    print(f"\nGenerating prompt for task {working_task_id}...")
    prompt = build_arc_prompt({working_task_id: working_task_data}, working_task_id)
    
    print(f"Task details (ID: {task_id}):")
    print(f"  - Training examples: {len(task_data.get('train', []))}")
    print(f"  - Test examples: {len(task_data.get('test', []))}")
    
    # Display the prompt that will be sent to the model
    print_prompt_in_blue(prompt, print_input)
    
    # Count input tokens
    input_tokens = count_tokens_simple(prompt)
    
    # Run with selected model
    if model == "llama3.1" or provider == "ollama":
        response = run_with_ollama(prompt, model=model)
    elif provider == "google" or provider == "learnlm":
        response = run_with_gemini(prompt, model)
    else:
        response = run_with_other_model(prompt, model)
    
    # Initialize result structure
    result = {
        'total_tokens': 0,
        'input_tokens': input_tokens,
        'output_tokens': 0,
        'estimated_cost': 0.0,
        'transformations_json_generated': False,
        'transformations_json': None,
        'trains': [],
        'tests': []
    }
    
    if not response:
        print("No response received from model.")
        return result
    
    # Check if we should enhance the response with smart routing
    initial_json_extracted = extract_json_from_response(response) is not None
    
    # Count output tokens from initial response
    initial_output_tokens = count_tokens_simple(response)
    
    if response and use_smart_router:
        print(f"\n{BLUE}Checking if response enhancement is needed...{RESET}")
        print(f"Input tokens: {input_tokens}, Output tokens: {initial_output_tokens}, Initial JSON extracted: {'✓' if initial_json_extracted else '✗'}")
        
        # Use smart routing to enhance response with any model
        enhanced_response, additional_tokens = enhance_response_with_smart_routing(
            original_response=response,
            input_tokens=input_tokens,
            output_tokens=initial_output_tokens,
            json_extracted=initial_json_extracted,
            task_id=task_id,
            model=model,
            provider=provider,
            print_input=print_input,
            max_reflections=max_reflections,
            task_data=task_data
        )
        
        # Use the enhanced response
        response = enhanced_response
    else:
        # No enhancement used, no additional tokens
        additional_tokens = 0
    
    # Count output tokens and calculate totals including enhancement tokens
    output_tokens = count_tokens_simple(response)
    total_tokens = input_tokens + output_tokens + additional_tokens
    
    # Calculate estimated cost including enhancement tokens
    try:
        estimated_cost = estimate_cost(model, input_tokens, output_tokens + additional_tokens)
    except KeyError:
        estimated_cost = 0.0  # Unknown model or free model
    
    result.update({
        'total_tokens': total_tokens,
        'output_tokens': output_tokens,
        'estimated_cost': estimated_cost
    })
    
    if print_output:
        print(f"\n{'='*60}")
        print("MODEL RESPONSE:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}")
    else:
        print(f"{GREEN}[Model response received - hidden]{RESET}")
    
    # Display token count in blue
    print_token_count_in_blue(response)
    if additional_tokens > 0:
        print(f"Task {task_id}: Input tokens: {input_tokens}, Output tokens: {output_tokens}, Enhancement tokens: +{additional_tokens}, Total: {total_tokens}, Estimated cost: ${estimated_cost:.6f}")
    else:
        print(f"Task {task_id}: Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total: {total_tokens}, Estimated cost: ${estimated_cost:.6f}")
    
    # Extract and validate JSON
    print(f"\n{'='*60}")
    print("JSON EXTRACTION AND VALIDATION:")
    print(f"{'='*60}")
    
    extracted_json = extract_json_from_response(response)
    
    if not extracted_json:
        print(f"{RED}Failed to extract JSON from response{RESET}")
        
        # Show analysis for failed JSON extraction
        print(f"\n{'='*60}")
        print("JSON ANALYSIS:")
        print(f"{'='*60}")
        print(f"1. JSON Generated: {RED}✗ NO - Failed to extract JSON from response{RESET}")
        print(f"2. JSON Valid: {RED}✗ NO - No JSON to validate{RESET}")
        print(f"3. Transformation Steps: {RED}Cannot analyze - No JSON{RESET}")
        print(f"4. Steps Analysis: {RED}Cannot analyze - No JSON{RESET}")
        
        # Check if smart router should have been used but wasn't
        if use_smart_router:
            print(f"\n{BLUE}Note: Smart router was used but still couldn't generate valid JSON{RESET}")
        elif not use_smart_router:
            print(f"\n{BLUE}Suggestion: Enable smart router (--use-smart-router) to attempt JSON recovery{RESET}")
        
        result['transformations_json_generated'] = False
        return result
    
    print(f"{GREEN}Successfully extracted JSON!{RESET}")
    result['transformations_json_generated'] = True
    result['transformations_json'] = extracted_json
    
    if print_output:
        print(json.dumps(extracted_json, indent=2))
    else:
        print(f"{BLUE}[JSON content hidden]{RESET}")
    
    # Analyze JSON structure and transformation steps
    print(f"\n{'='*60}")
    print("JSON ANALYSIS:")
    print(f"{'='*60}")
    
    # 1. Is JSON generated?
    print(f"1. JSON Generated: {GREEN}✓ YES{RESET}")
    
    # 2. Is it valid?
    is_valid = validate_json_structure(extracted_json)
    if is_valid:
        print(f"2. JSON Valid: {GREEN}✓ YES{RESET}")
    else:
        print(f"2. JSON Valid: {RED}✗ NO - Missing required structure{RESET}")
    
    # 3. How many transformation steps
    if is_valid:
        steps = extracted_json.get("step_by_step_transformations", [])
        helper_funcs = extracted_json.get("helper_python_functions", [])
        
        num_steps = len(steps)
        num_helpers = len(helper_funcs) if helper_funcs else 0
        
        print(f"3. Transformation Steps: {BLUE}{num_steps} steps{RESET}")
        if num_helpers > 0:
            print(f"   Helper Functions: {BLUE}{num_helpers} functions{RESET}")
        
        # 4. What transformation steps were successful?
        if num_steps > 0:
            print(f"4. Steps Analysis:")
            
            # Test each step on training examples to see which ones work
            training_examples = task_data.get("train", [])
            if training_examples:
                for step_idx, step in enumerate(steps):
                    step_num = step_idx + 1
                    python_code = step.get("python_code", [])
                    step_desc = step.get("description", "No description")
                    
                    # Test this step on all training examples
                    step_successes = []
                    for example in training_examples:
                        predicted_output, error = execute_transformation_code(python_code, example["input"], helper_funcs)
                        success = predicted_output is not None
                        step_successes.append(success)
                    
                    success_count = sum(step_successes)
                    total_examples = len(training_examples)
                    success_rate = (success_count / total_examples * 100) if total_examples > 0 else 0
                    
                    if success_count == total_examples:
                        status_color = GREEN
                        status_symbol = "✓"
                    elif success_count > 0:
                        status_color = BLUE
                        status_symbol = "~"
                    else:
                        status_color = RED
                        status_symbol = "✗"
                    
                    # Truncate description if too long
                    if isinstance(step_desc, list):
                        desc_text = " ".join(step_desc)
                    else:
                        desc_text = str(step_desc)
                    
                    if len(desc_text) > 60:
                        desc_text = desc_text[:57] + "..."
                    
                    print(f"   Step {step_num}: {status_color}{status_symbol} {success_count}/{total_examples} ({success_rate:.1f}%) - {desc_text}{RESET}")
            else:
                print(f"   {RED}No training examples to test steps{RESET}")
        else:
            print(f"4. Steps Analysis: {RED}No transformation steps found{RESET}")
    else:
        print(f"3. Transformation Steps: {RED}Cannot analyze - Invalid JSON structure{RESET}")
        print(f"4. Steps Analysis: {RED}Cannot analyze - Invalid JSON structure{RESET}")
    
    # Process training examples
    result['trains'] = process_training_examples(task_data, extracted_json)
    
    # Process test examples  
    result['tests'] = process_test_examples(task_data, solutions.get(task_id, []), extracted_json)
    
    # Save results to output directory if specified
    if output_dir:
        try:
            save_new_format_results(task_id, result, output_dir)
            print(f"{GREEN}Results saved to: {output_dir}/{task_id}.json{RESET}")
        except Exception as e:
            print(f"{RED}Failed to save results: {e}{RESET}")
    
    return result


def run_single_task_wrapper(args):
    """Wrapper function for parallel execution."""
    task_id, task_data, solutions, model, provider, output_dir, use_sanitized_task, use_sanitized_id, print_input, print_output, use_smart_router, max_reflections = args
    return run_single_task(task_id, task_data, solutions, model, provider, output_dir, 
                          use_sanitized_task, use_sanitized_id, print_input, print_output, 
                          use_smart_router, max_reflections)


def save_task_results(task_id, task_data, solutions, predicted_outputs, output_dir):
    """Save test inputs, expected outputs, and predicted outputs for a task."""
    task_output_dir = Path(output_dir) / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    test_examples = task_data.get("test", [])
    task_solutions = solutions.get(task_id, [])
    
    # Save each test example
    for i, (test_example, expected_output) in enumerate(zip(test_examples, task_solutions)):
        test_input = test_example["input"]
        predicted_output = predicted_outputs[i] if i < len(predicted_outputs) else None
        
        # Create result data
        result_data = {
            "task_id": task_id,
            "test_index": i,
            "test_input": test_input,
            "test_output": expected_output,
            "produced_output": predicted_output,
            "input_size": [len(test_input), len(test_input[0]) if test_input else 0],
            "expected_size": [len(expected_output), len(expected_output[0]) if expected_output else 0],
            "produced_size": [len(predicted_output), len(predicted_output[0]) if predicted_output else 0] if predicted_output else None
        }
        
        # Save to JSON file
        output_file = task_output_dir / f"test_{i}.json"
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    return task_output_dir


def calculate_matching_criteria(predicted, expected):
    """Calculate various matching criteria between predicted and expected outputs."""
    if not predicted or not expected:
        return {
            'overlap_percent': 0.0,
            'exact_match': False,
            'size_match': False,
            'shape_match': False
        }
    
    # Size matching
    pred_height, pred_width = len(predicted), len(predicted[0]) if predicted else 0
    exp_height, exp_width = len(expected), len(expected[0]) if expected else 0
    size_match = (pred_height == exp_height and pred_width == exp_width)
    shape_match = size_match  # Same as size match for 2D grids
    
    if not size_match:
        return {
            'overlap_percent': 0.0,
            'exact_match': False,
            'size_match': False,
            'shape_match': False
        }
    
    # Calculate overlap
    total_cells = pred_height * pred_width
    matching_cells = 0
    
    for i in range(pred_height):
        for j in range(pred_width):
            if predicted[i][j] == expected[i][j]:
                matching_cells += 1
    
    overlap_percent = (matching_cells / total_cells) * 100.0 if total_cells > 0 else 0.0
    exact_match = overlap_percent >= 100.0
    
    return {
        'overlap_percent': overlap_percent,
        'exact_match': exact_match,
        'size_match': size_match,
        'shape_match': shape_match
    }


def test_summary(output_dir):
    """Generate summary from saved test results."""
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"{RED}Output directory not found: {output_dir}{RESET}")
        return
    
    print(f"\n{'='*100}")
    print(f"TEST SUMMARY FROM: {output_dir}")
    print(f"{'='*100}")
    
    # Find all task directories
    task_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    if not task_dirs:
        print(f"{RED}No task directories found in {output_dir}{RESET}")
        return
    
    all_results = []
    
    for task_dir in task_dirs:
        task_id = task_dir.name
        test_files = sorted(task_dir.glob("test_*.json"))
        
        task_results = {
            'task_id': task_id,
            'test_results': [],
            'overall_exact_match': True,
            'overall_size_match': True,
            'overall_overlap': 0.0
        }
        
        total_overlap = 0.0
        
        for test_file in test_files:
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            predicted = test_data.get('produced_output')
            expected = test_data.get('test_output')
            
            criteria = calculate_matching_criteria(predicted, expected)
            test_data['matching_criteria'] = criteria
            task_results['test_results'].append(test_data)
            
            if not criteria['exact_match']:
                task_results['overall_exact_match'] = False
            if not criteria['size_match']:
                task_results['overall_size_match'] = False
            
            total_overlap += criteria['overlap_percent']
        
        if test_files:
            task_results['overall_overlap'] = total_overlap / len(test_files)
        
        all_results.append(task_results)
    
    # Calculate overall statistics
    total_tasks = len(all_results)
    exact_match_tasks = sum(1 for r in all_results if r['overall_exact_match'])
    size_match_tasks = sum(1 for r in all_results if r['overall_size_match'])
    completed_tasks = sum(1 for r in all_results if any(tr.get('produced_output') for tr in r['test_results']))
    
    total_overlap_sum = sum(r['overall_overlap'] for r in all_results)
    overall_avg_overlap = total_overlap_sum / total_tasks if total_tasks > 0 else 0.0
    
    # Display summary
    print(f"\n{BLUE}SUMMARY STATISTICS:{RESET}")
    print(f"  Total tasks analyzed: {total_tasks}")
    print(f"  Tasks with completed predictions: {completed_tasks}")
    print(f"  Completion rate: {(completed_tasks / total_tasks * 100):.1f}%")
    print(f"  Tasks with exact matches (100%): {exact_match_tasks}")
    print(f"  Exact match rate: {(exact_match_tasks / total_tasks * 100):.1f}%")
    print(f"  Tasks with matching sizes: {size_match_tasks}")
    print(f"  Size match rate: {(size_match_tasks / total_tasks * 100):.1f}%")
    print(f"  Overall average overlap: {overall_avg_overlap:.1f}%")
    
    # Detailed task results
    print(f"\n{BLUE}DETAILED TASK RESULTS:{RESET}")
    for result in all_results:
        status_exact = "✓" if result['overall_exact_match'] else "✗"
        status_size = "✓" if result['overall_size_match'] else "✗"
        overlap = result['overall_overlap']
        
        print(f"  {status_exact} {result['task_id']} - Exact:{status_exact} Size:{status_size} Overlap:{overlap:.1f}%")
    
    return all_results


def load_arc_tasks(filepath):
    """Load ARC tasks from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_arc_solutions(filepath):
    """Load ARC solutions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_task_by_index(tasks, index=None):
    """Get a task by index, task ID string, or randomly select one if index is None."""
    task_ids = list(tasks.keys())
    
    if index is None:
        # Random selection
        selected_id = random.choice(task_ids)
        print(f"Randomly selected task: {selected_id}")
    elif isinstance(index, str) and not (index.isdigit() and len(index) <= 4):
        # Task ID string (not a simple numeric index)
        if index in tasks:
            selected_id = index
            print(f"Selected task by ID: {selected_id}")
        else:
            raise ValueError(f"Task ID '{index}' not found. Available tasks: {len(task_ids)} total")
    else:
        # Numeric index (either int or string that represents a reasonable index)
        numeric_index = int(index)
        if numeric_index < 0 or numeric_index >= len(task_ids):
            raise ValueError(f"Index {numeric_index} out of range. Available tasks: 0-{len(task_ids)-1}")
        selected_id = task_ids[numeric_index]
        print(f"Selected task by index {numeric_index}: {selected_id}")
    
    return selected_id, tasks[selected_id]


def get_multiple_tasks(tasks, num_tasks, specific_index=None):
    """Get multiple tasks for batch processing."""
    task_ids = list(tasks.keys())
    selected_tasks = []
    
    if specific_index is not None:
        # Single specific task (either index or task ID)
        if isinstance(specific_index, str) and not (specific_index.isdigit() and len(specific_index) <= 4):
            # Task ID string (not a simple numeric index)
            if specific_index in tasks:
                selected_id = specific_index
                print(f"Selected specific task by ID: {selected_id}")
            else:
                raise ValueError(f"Task ID '{specific_index}' not found. Available tasks: {len(task_ids)} total")
        else:
            # Numeric index
            numeric_index = int(specific_index)
            if numeric_index < 0 or numeric_index >= len(task_ids):
                raise ValueError(f"Index {numeric_index} out of range. Available tasks: 0-{len(task_ids)-1}")
            selected_id = task_ids[numeric_index]
            print(f"Selected specific task by index {numeric_index}: {selected_id}")
        
        selected_tasks.append((selected_id, tasks[selected_id]))
    else:
        # Random selection or all tasks
        if num_tasks == -1:
            # Run all tasks
            num_tasks = len(task_ids)
            selected_ids = task_ids
            print(f"Running ALL {num_tasks} tasks")
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
        # Import genai locally to avoid ALTS warnings at module level
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Create the model with generation config
        generation_config = {"temperature": TEMPERATURE}
        if NUM_PREDICT != -1:
            generation_config["max_output_tokens"] = NUM_PREDICT
        
        model_instance = genai.GenerativeModel(
            model_name=model,
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


def create_continuation_prompt(previous_response):
    """Create a prompt to continue reasoning when it appears incomplete."""
    return f"""Continue your analysis from where you left off. Your previous response was:

{previous_response}

Please continue your reasoning process and complete the analysis. Make sure to:

1. Continue from exactly where you stopped
2. Complete your step-by-step analysis of the patterns
3. Identify the core transformation rule(s)
4. Provide the final JSON response in the required format

Continue your reasoning:"""


def create_json_regeneration_prompt(previous_response):
    """Create a prompt to regenerate JSON when reasoning is complete but JSON is invalid."""
    return f"""Your reasoning analysis is complete, but the JSON output needs to be regenerated in the correct format.

Your previous analysis:
{previous_response}

Based on your analysis above, please provide ONLY the final JSON response in the exact required format:

```json
{{
  "helper_python_functions": [
    "..."
  ],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [
        "..."
      ],
      "python_code": [
        "def transform(grid):",
        "    # COMPLETE transformation implementation",
        "    # This function must be fully executable on its own",
        "    # and return a complete transformed grid",
        "    return processed_grid"
      ],
      "example_input": [...],
      "example_output": [...]
  }}],
  "confidence": "high|medium|low"
}}
```

CRITICAL REQUIREMENTS:
- Each step MUST have a complete, standalone `def transform(grid):` function
- DO NOT split transformation logic across multiple steps
- If a step has no actual transformation, return the input grid: `return grid`
- Each transform function must be fully executable and return a complete grid"""


def create_json_repair_prompt(invalid_json_text, json_error):
    """Create a prompt to repair malformed JSON."""
    return f"""The JSON response you generated has a formatting error and cannot be parsed.

INVALID JSON:
```json
{invalid_json_text}
```

JSON PARSING ERROR:
{json_error}

Please provide a CORRECTED version of this JSON that:
1. Fixes the specific parsing error mentioned above
2. Maintains all the content and logic from the original
3. Follows proper JSON syntax with correct quotes, brackets, and commas
4. Uses the exact required structure for ARC transformations
5. ENSURES each step has a complete, standalone `def transform(grid):` function

Corrected JSON:
```json
{{
  "helper_python_functions": [...],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [...],
      "python_code": [
        "def transform(grid):",
        "    # COMPLETE transformation implementation",
        "    # This function must be fully executable on its own", 
        "    return processed_grid"
      ],
      "example_input": [...],
      "example_output": [...]
  }}],
  "confidence": "high|medium|low"
}}
```

CRITICAL: DO NOT split transformation logic across multiple steps. Each transform function must be complete and executable."""


def create_code_repair_prompt(valid_json, execution_error, failed_example=None, all_failures=None):
    """Create a prompt to fix code execution errors in valid JSON with comprehensive failure details."""
    
    # Build detailed error context
    error_context = ""
    if failed_example:
        error_context = f"""
PRIMARY FAILED EXAMPLE (Example {failed_example.get('example_idx', 0) + 1}):
Input Grid: {failed_example.get('input', 'N/A')}
Expected Output: {failed_example.get('expected_output', 'N/A')}
Error: {failed_example.get('error', 'N/A')}
"""
    
    # Add additional failure context if available
    if all_failures and len(all_failures) > 1:
        error_context += f"\nADDITIONAL FAILURES ({len(all_failures) - 1} more examples):\n"
        for i, failure in enumerate(all_failures[1:6], 1):  # Show up to 5 additional failures
            error_context += f"Example {failure.get('example_idx', 0) + 1}: {failure.get('error', 'Unknown error')}\n"
            error_context += f"  Input: {failure.get('input', 'N/A')}\n"
            error_context += f"  Expected: {failure.get('expected_output', 'N/A')}\n"
        
        if len(all_failures) > 6:
            error_context += f"... and {len(all_failures) - 6} more failures\n"

    return f"""The JSON response is valid but the Python code fails to execute on multiple training examples.

CURRENT VALID JSON:
```json
{json.dumps(valid_json, indent=2)}
```

CODE EXECUTION ANALYSIS:
Total Training Examples: {len(all_failures) if all_failures else 1}
Failed Examples: {len(all_failures) if all_failures else 1}
Success Rate: 0%

DETAILED FAILURE INFORMATION:
{error_context}

CRITICAL ISSUE DETECTED:
The transformation logic appears to be split across multiple steps with incomplete transform functions.
This violates the requirement that each step must have a complete, standalone transform(grid) function.

REQUIREMENTS FOR REPAIR:
1. Each step MUST have a complete, standalone `def transform(grid):` function
2. DO NOT split transformation logic across multiple steps  
3. If you need multiple steps, each should perform a complete transformation on its own
4. Fix the specific execution errors mentioned above
5. Ensure the transform() function is properly defined and callable
6. Handle edge cases that cause failures across different input grids
7. Use proper error handling for robust execution
8. If a step has no actual transformation logic, return the input unchanged: `return grid`

Please provide a CORRECTED JSON with properly structured, complete transform functions:

```json
{{
  "helper_python_functions": [...],
  "step_by_step_transformations": [{{
      "step_number": 1,
      "description": [...],
      "python_code": [
        "def transform(grid):",
        "    # COMPLETE transformation implementation",
        "    # This function must be fully executable on its own",
        "    # Include ALL necessary logic, imports, and processing",
        "    # Handle the specific errors identified above",
        "    return processed_grid"
      ],
      "example_input": [...],
      "example_output": [...]
  }}],
  "confidence": "high|medium|low"
}}
```

REMEMBER: Each transform function must be complete and executable independently!"""


def test_json_execution_comprehensive(extracted_json, task_data):
    """Test if the JSON can execute successfully on ALL training examples with detailed error reporting."""
    if not validate_json_structure(extracted_json):
        return False, "Invalid JSON structure", None
    
    steps = extracted_json.get("step_by_step_transformations", [])
    if not steps:
        return False, "No transformation steps found", None
    
    # Get helper functions and last step
    helper_functions = extracted_json.get("helper_python_functions", [])
    last_step = steps[-1]
    python_code = last_step.get("python_code", [])
    
    # Test on ALL training examples
    training_examples = task_data.get("train", [])
    if not training_examples:
        return False, "No training examples found", None
    
    print(f"\n{BLUE}🧪 COMPREHENSIVE CODE TESTING{RESET}")
    print(f"Testing code on {len(training_examples)} training examples...")
    
    execution_results = []
    
    for i, example in enumerate(training_examples):
        input_grid = example["input"]
        expected_output = example["output"]
        
        print(f"\n  📝 Training Example {i+1}:")
        print(f"     Input size: {len(input_grid)}x{len(input_grid[0]) if input_grid else 0}")
        print(f"     Expected output size: {len(expected_output)}x{len(expected_output[0]) if expected_output else 0}")
        print(f"     Input grid: {input_grid}")
        
        predicted_output, error = execute_transformation_code(python_code, input_grid, helper_functions)
        
        if predicted_output is None:
            print(f"     {RED}❌ EXECUTION FAILED{RESET}")
            print(f"     {RED}Error: {error}{RESET}")
            execution_results.append({
                'example_idx': i,
                'input': input_grid,
                'expected_output': expected_output,
                'success': False,
                'error': error,
                'predicted_output': None
            })
        else:
            overlap = calculate_grid_overlap(predicted_output, expected_output)
            print(f"     {GREEN}✅ EXECUTION SUCCESS{RESET}")
            print(f"     Predicted size: {len(predicted_output)}x{len(predicted_output[0]) if predicted_output else 0}")
            print(f"     Overlap: {overlap:.1f}%")
            print(f"     Predicted grid: {predicted_output}")
            
            execution_results.append({
                'example_idx': i,
                'input': input_grid,
                'expected_output': expected_output,
                'success': True,
                'error': None,
                'predicted_output': predicted_output,
                'overlap': overlap
            })
    
    # Check if any failed
    failed_examples = [r for r in execution_results if not r['success']]
    if failed_examples:
        print(f"\n{RED}💥 CODE EXECUTION FAILURES: {len(failed_examples)}/{len(training_examples)} examples failed{RESET}")
        return False, failed_examples[0]['error'], failed_examples[0]  # Return first failure for repair
    else:
        successful_count = len([r for r in execution_results if r['success']])
        avg_overlap = sum(r.get('overlap', 0) for r in execution_results if r['success']) / successful_count if successful_count > 0 else 0
        print(f"\n{GREEN}🎯 ALL TRAINING EXAMPLES PASSED!{RESET}")
        print(f"   Success rate: {successful_count}/{len(training_examples)} ({successful_count/len(training_examples)*100:.1f}%)")
        print(f"   Average overlap: {avg_overlap:.1f}%")
        return True, None, None


def is_reasoning_complete(response_text):
    """Check if the reasoning appears to be complete based on text patterns."""
    if not response_text:
        return False
    
    # Convert to lowercase for pattern matching
    text = response_text.lower()
    
    # Check for completion indicators
    completion_patterns = [
        'in conclusion',
        'final rule',
        'final transformation',
        'therefore, the rule',
        'so the pattern',
        'the transformation rule is',
        'applying this rule',
        'final json',
        'json response',
        '```json',
        'step_by_step_transformations'
    ]
    
    has_completion_indicator = any(pattern in text for pattern in completion_patterns)
    
    # Check if it ends abruptly (incomplete)
    incomplete_endings = [
        'the pattern seems to',
        'we can see that',
        'this suggests',
        'looking at',
        'analyzing',
        'examining',
        'considering',
        'based on this'
    ]
    
    # Get the last few words to check for incomplete endings
    words = text.split()
    if len(words) > 10:
        last_portion = ' '.join(words[-10:])
        ends_incomplete = any(ending in last_portion for ending in incomplete_endings)
    else:
        ends_incomplete = False
    
    # Reasoning is complete if it has completion indicators and doesn't end abruptly
    return has_completion_indicator and not ends_incomplete


def call_model(prompt, model, provider=None):
    """
    Generic function to call any model with the appropriate provider.
    
    Args:
        prompt (str): The prompt to send to the model
        model (str): The model name
        provider (str, optional): The provider name, if not auto-detected
        
    Returns:
        str: The model response, or None if failed
    """
    # Auto-detect provider if not specified
    if provider is None:
        from model_configs import MODEL_CONFIGS
        if model in MODEL_CONFIGS:
            provider = MODEL_CONFIGS[model].get("provider", "unknown")
        elif model == "llama3.1":
            provider = "ollama"
        else:
            provider = "unknown"
    
    # Route to the appropriate model function
    if provider == "ollama" or model == "llama3.1":
        return run_with_ollama(prompt, model)
    elif provider == "google" or provider == "learnlm":
        return run_with_gemini(prompt, model)
    else:
        return run_with_other_model(prompt, model)


def enhance_response_with_smart_routing(original_response, input_tokens, output_tokens, json_extracted, task_id, model, provider=None, print_input=False, max_reflections=5, task_data=None):
    """
    Enhance the response by continuing reasoning or fixing JSON/code issues using any model with recursive reflection.
    
    Args:
        original_response (str): The original model response
        input_tokens (int): Number of input tokens in the original prompt
        output_tokens (int): Number of output tokens in the original response
        json_extracted (bool): Whether valid JSON was extracted from the response
        task_id (str): Task identifier for logging
        model (str): The model to use for enhancement
        provider (str, optional): The provider name, if not auto-detected
        print_input (bool): Whether to print prompts
        max_reflections (int): Maximum number of reflection cycles for complete enhancement
        task_data (dict, optional): Task data for testing code execution
        
    Returns:
        tuple: (Enhanced response with fixed JSON and code, additional tokens consumed)
    """
    print(f"\n{BLUE}{'='*60}")
    print("SMART RESPONSE ENHANCEMENT WITH RECURSIVE REFLECTION")
    print(f"Using model: {model} (provider: {provider or 'auto-detect'})")
    print(f"Max reflection cycles: {max_reflections}")
    print(f"{'='*60}{RESET}")
    
    print(f"Initial conditions:")
    print(f"  - Input tokens: {input_tokens}")
    print(f"  - Output tokens: {output_tokens}")
    print(f"  - JSON initially extracted: {'✓' if json_extracted else '✗'}")
    print(f"  - Reasoning complete: {'✓' if is_reasoning_complete(original_response) else '✗'}")
    
    current_response = original_response
    reflection_cycle = 0
    enhancement_actions = []
    additional_tokens = 0  # Track additional tokens consumed during enhancement
    
    # Main reflection loop - continue until we have working code or hit max reflections
    while reflection_cycle < max_reflections:
        reflection_cycle += 1
        print(f"\n{BLUE}🔄 REFLECTION CYCLE {reflection_cycle}/{max_reflections}{RESET}")
        
        cycle_enhanced = False
        
        # STEP 1: Handle incomplete reasoning (reasoning continuation)
        # Stop reasoning continuation 2 cycles before max reflections to force JSON
        max_continuation_cycles = max_reflections - 2
        needs_continuation = (output_tokens > 500 and not is_reasoning_complete(current_response) and reflection_cycle <= max_continuation_cycles)
        if needs_continuation:
            print(f"\n{BLUE}� REASONING CONTINUATION NEEDED{RESET}")
            print(f"Reason: High output token count ({output_tokens}) + incomplete reasoning (cycle {reflection_cycle}/{max_continuation_cycles})")
            enhancement_actions.append(f"Reasoning Continuation (Cycle {reflection_cycle})")
            
            continuation_prompt = create_continuation_prompt(current_response)
            print_prompt_in_blue(continuation_prompt, print_input)
            
            continuation_response = call_model(continuation_prompt, model, provider)
            
            if continuation_response:
                # Count tokens for this additional call
                continuation_input_tokens = count_tokens_simple(continuation_prompt)
                continuation_output_tokens = count_tokens_simple(continuation_response)
                additional_tokens += continuation_input_tokens + continuation_output_tokens
                
                current_response += "\n\n" + continuation_response
                cycle_enhanced = True
                print(f"  {GREEN}✅ Reasoning continued successfully (+{continuation_input_tokens + continuation_output_tokens} tokens){RESET}")
            else:
                print(f"  {RED}❌ Failed to get continuation response{RESET}")
        
        # STEP 2: Handle JSON extraction and validation issues
        extracted_json = extract_json_from_response(current_response)
        
        if is_reasoning_complete(current_response) and not extracted_json:
            # SCENARIO 1: Reasoning complete but no JSON found
            print(f"\n{BLUE}🔧 JSON GENERATION NEEDED{RESET}")
            print("Reason: Reasoning complete but no JSON found")
            enhancement_actions.append(f"JSON Generation (Cycle {reflection_cycle})")
            
            json_regen_prompt = create_json_regeneration_prompt(current_response)
            print_prompt_in_blue(json_regen_prompt, print_input)
            
            json_response = call_model(json_regen_prompt, model, provider)
            
            if json_response:
                # Count tokens for this additional call
                json_input_tokens = count_tokens_simple(json_regen_prompt)
                json_output_tokens = count_tokens_simple(json_response)
                additional_tokens += json_input_tokens + json_output_tokens
                
                current_response += "\n\n" + json_response
                cycle_enhanced = True
                extracted_json = extract_json_from_response(current_response)
                print(f"  {GREEN}✅ JSON generation attempt completed (+{json_input_tokens + json_output_tokens} tokens){RESET}")
            else:
                print(f"  {RED}❌ Failed to get JSON generation response{RESET}")
        
        # SCENARIO 2: JSON found but invalid/malformed
        if extracted_json is None and '```json' in current_response:
            # Extract the raw JSON text to see what's wrong
            import re
            json_pattern = r'```json\s*\n(.*?)\n```'
            matches = re.findall(json_pattern, current_response, re.DOTALL)
            if matches:
                raw_json = matches[-1].strip()  # Get the last JSON block
                try:
                    json.loads(raw_json)  # This should fail
                except json.JSONDecodeError as e:
                    print(f"\n{BLUE}🔧 JSON REPAIR NEEDED{RESET}")
                    print(f"Reason: JSON syntax error - {str(e)}")
                    enhancement_actions.append(f"JSON Repair (Cycle {reflection_cycle})")
                    
                    json_repair_prompt = create_json_repair_prompt(raw_json, str(e))
                    print_prompt_in_blue(json_repair_prompt, print_input)
                    
                    repair_response = call_model(json_repair_prompt, model, provider)
                    
                    if repair_response:
                        # Count tokens for this additional call
                        repair_input_tokens = count_tokens_simple(json_repair_prompt)
                        repair_output_tokens = count_tokens_simple(repair_response)
                        additional_tokens += repair_input_tokens + repair_output_tokens
                        
                        current_response += "\n\n" + repair_response
                        cycle_enhanced = True
                        extracted_json = extract_json_from_response(current_response)
                        print(f"  {GREEN}✅ JSON repair attempt completed (+{repair_input_tokens + repair_output_tokens} tokens){RESET}")
                    else:
                        print(f"  {RED}❌ Failed to get JSON repair response{RESET}")
        
        # SCENARIO 3: Valid JSON but code execution fails
        if extracted_json and task_data:
            print(f"\n{BLUE}🧪 TESTING JSON CODE EXECUTION{RESET}")
            can_execute, execution_error, failed_example = test_json_execution_comprehensive(extracted_json, task_data)
            
            if not can_execute:
                print(f"\n{BLUE}🔧 CODE REPAIR NEEDED{RESET}")
                print(f"Reason: Code execution failed")
                enhancement_actions.append(f"Code Repair (Cycle {reflection_cycle})")
                
                # Get all failure details if available
                all_failures = []
                if isinstance(execution_error, list):  # Multiple failures
                    all_failures = execution_error
                    primary_error = execution_error[0]['error'] if execution_error else "Multiple execution failures"
                    failed_example = execution_error[0] if execution_error else failed_example
                else:
                    primary_error = execution_error
                    if failed_example:
                        all_failures = [failed_example]
                
                code_repair_prompt = create_code_repair_prompt(extracted_json, primary_error, failed_example, all_failures)
                print_prompt_in_blue(code_repair_prompt, print_input)
                
                repair_response = call_model(code_repair_prompt, model, provider)
                
                if repair_response:
                    # Count tokens for this additional call
                    code_repair_input_tokens = count_tokens_simple(code_repair_prompt)
                    code_repair_output_tokens = count_tokens_simple(repair_response)
                    additional_tokens += code_repair_input_tokens + code_repair_output_tokens
                    
                    current_response += "\n\n" + repair_response
                    cycle_enhanced = True
                    extracted_json = extract_json_from_response(current_response)
                    print(f"  {GREEN}✅ Code repair attempt completed (+{code_repair_input_tokens + code_repair_output_tokens} tokens){RESET}")
                    
                    # Test the repaired code
                    if extracted_json:
                        can_execute_after, _, _ = test_json_execution_comprehensive(extracted_json, task_data)
                        if can_execute_after:
                            print(f"  {GREEN}🎯 CODE REPAIR SUCCESSFUL! All examples now pass{RESET}")
                            break  # Success! Exit reflection loop
                        else:
                            print(f"  {RED}🔄 Code still has issues, will continue reflection...{RESET}")
                            # Add additional guidance for the next iteration
                            if reflection_cycle < max_reflections:
                                print(f"  {BLUE}💡 Tip: Ensure each step has a complete transform(grid) function{RESET}")
                else:
                    print(f"  {RED}❌ Failed to get code repair response{RESET}")
            else:
                print(f"  {GREEN}🎯 CODE EXECUTION SUCCESSFUL! All examples pass{RESET}")
                break  # Success! Exit reflection loop
        
        # Check if we made any progress this cycle
        if not cycle_enhanced:
            print(f"\n{BLUE}🏁 No enhancement needed or possible in this cycle{RESET}")
            break
    
    # Final status report
    final_json_extracted = extract_json_from_response(current_response) is not None
    final_code_working = False
    
    if final_json_extracted and task_data:
        final_extracted_json = extract_json_from_response(current_response)
        can_execute_final, _, _ = test_json_execution_comprehensive(final_extracted_json, task_data)
        final_code_working = can_execute_final
    
    print(f"\n{BLUE}{'='*60}")
    print("ENHANCEMENT SUMMARY")
    print(f"{'='*60}{RESET}")
    print(f"Reflection cycles completed: {reflection_cycle}/{max_reflections}")
    print(f"Enhancement actions taken: {', '.join(enhancement_actions) if enhancement_actions else 'None'}")
    print(f"Final JSON extracted: {'✓' if final_json_extracted else '✗'}")
    print(f"Final code working: {'✓' if final_code_working else '✗'}")
    
    if final_code_working:
        print(f"{GREEN}🎉 ENHANCEMENT COMPLETE: Fully working solution achieved!{RESET}")
    elif final_json_extracted:
        print(f"{BLUE}📝 PARTIAL SUCCESS: JSON extracted but code may have issues{RESET}")
    else:
        print(f"{RED}❌ ENHANCEMENT INCOMPLETE: Could not achieve working solution{RESET}")
    
    print(f"Total additional tokens consumed during enhancement: {additional_tokens}")
    
    return current_response, additional_tokens

# ============================================================================
# CORE LOGIC
# ============================================================================


def main():
    # Parse command line arguments - they override the ALL_CAPS defaults
    args = parse_arguments()
    
    # Use argparse values (which default to ALL_CAPS variables)
    task_index = args.task_index
    # Convert task_index to appropriate type (int, str, or None)
    if task_index is not None:
        if isinstance(task_index, str):
            if task_index.lower() == 'none':
                task_index = None
            elif task_index.isdigit() and len(task_index) <= 4:
                # Only convert to int if it's a reasonable index (<=4 digits)
                # ARC task IDs are typically 8 character hex strings
                task_index = int(task_index)
            # else keep as string (task ID)
    
    number_of_tasks = args.number_of_tasks
    model = args.model
    training_challenges_path = args.training_challenges_path
    training_solutions_path = args.training_solutions_path
    evaluation_challenges_path = args.evaluation_challenges_path
    evaluation_solutions_path = args.evaluation_solutions_path
    random_seed = args.random_seed
    temperature = args.temperature
    num_predict = args.num_predict
    sanitize_task = args.sanitize_task
    sanitize_id = args.sanitize_id
    parallel = args.parallel
    print_input = args.print_input
    print_output = args.print_output
    use_smart_router = args.use_smart_router
    max_reflections = args.max_reflections
    
    # Load environment variables from .env file
    load_dotenv()
    print("Loaded environment variables from .env file")
    
    # Create output directory with timestamp
    timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    # Save parameters to params.json
    params = {
        'script': 'test_arc_prompt.py',
        'task_index': task_index,
        'number_of_tasks': number_of_tasks,
        'model': model,
        'training_challenges_path': training_challenges_path,
        'training_solutions_path': training_solutions_path,
        'evaluation_challenges_path': evaluation_challenges_path,
        'evaluation_solutions_path': evaluation_solutions_path,
        'random_seed': random_seed,
        'temperature': temperature,
        'num_predict': num_predict,
        'sanitize_task': sanitize_task,
        'sanitize_id': sanitize_id,
        'parallel': parallel,
        'print_input': print_input,
        'print_output': print_output,
        'use_smart_router': use_smart_router,
        'max_reflections': max_reflections,
        'timestamp': timestamp
    }
    
    params_file = output_dir / 'params.json'
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {params_file}")
    
    # Debug: Check if API key is loaded
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"✓ GEMINI_API_KEY loaded (first 10 chars: {gemini_key[:10]}...)")
    else:
        print("✗ GEMINI_API_KEY not found in environment")
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    
    # Check if model is known
    if not is_known_model(model):
        print(f"Error: Unknown model '{model}'")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    # Load ARC challenges
    training_challenges_path = Path(training_challenges_path)
    if not training_challenges_path.exists():
        print(f"Error: Training challenges file not found: {training_challenges_path}")
        sys.exit(1)
    
    evaluation_challenges_path = Path(evaluation_challenges_path)
    if not evaluation_challenges_path.exists():
        print(f"Error: Evaluation challenges file not found: {evaluation_challenges_path}")
        sys.exit(1)

    try:
        training_tasks = load_arc_tasks(training_challenges_path)
        print(f"Loaded {len(training_tasks)} ARC training tasks from {training_challenges_path}")
        evaluation_tasks = load_arc_tasks(evaluation_challenges_path)
        print(f"Loaded {len(evaluation_tasks)} ARC evaluation tasks from {evaluation_challenges_path}")
        
        # Combine tasks for processing
        tasks = {**training_tasks, **evaluation_tasks}
        print(f"Total tasks available: {len(tasks)}")
    except Exception as e:
        print(f"Error loading tasks: {e}")
        sys.exit(1)
    
    # Load ARC solutions
    training_solutions_path = Path(training_solutions_path)
    if not training_solutions_path.exists():
        print(f"Error: Training solutions file not found: {training_solutions_path}")
        sys.exit(1)
        
    evaluation_solutions_path = Path(evaluation_solutions_path)
    if not evaluation_solutions_path.exists():
        print(f"Error: Evaluation solutions file not found: {evaluation_solutions_path}")
        sys.exit(1)

    try:
        training_solutions = load_arc_solutions(training_solutions_path)
        print(f"Loaded training solutions for {len(training_solutions)} tasks from {training_solutions_path}")
        evaluation_solutions = load_arc_solutions(evaluation_solutions_path)
        print(f"Loaded evaluation solutions for {len(evaluation_solutions)} tasks from {evaluation_solutions_path}")
        
        # Combine solutions for processing
        solutions = {**training_solutions, **evaluation_solutions}
        print(f"Total solutions available: {len(solutions)}")
    except Exception as e:
        print(f"Error loading solutions: {e}")
        sys.exit(1)
    
    # Get model configuration
    model_config = MODEL_CONFIGS.get(model, {})
    provider = model_config.get('provider', 'unknown')
    
    # Select tasks to run
    try:
        selected_tasks = get_multiple_tasks(tasks, number_of_tasks, task_index)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*100}")
    print(f"RUNNING {len(selected_tasks)} TASK(S) WITH MODEL: {model}")
    print(f"{'='*100}")
    
    # Process all tasks
    all_results = []
    total_tokens = 0
    total_estimated_cost = 0.0
    successful_json_generations = 0
    
    # Check if we should disable parallelization for local models
    if parallel and len(selected_tasks) > 1:
        # Disable parallel processing for local models (ollama, qwen, etc.)
        if provider == "ollama" or model in ["llama3.1", "qwen2.5:32b"] or "qwen" in model.lower() or "llama" in model.lower():
            print(f"\n{BLUE}Disabling parallel processing for local model: {model}{RESET}")
            print(f"{BLUE}Local models work better with sequential processing{RESET}")
            parallel = False
    
    if parallel and len(selected_tasks) > 1:
        print(f"\n{BLUE}Running {len(selected_tasks)} tasks in parallel...{RESET}")
        
        # Prepare arguments for parallel execution
        task_args = [(task_id, task_data, solutions, model, provider, output_dir, sanitize_task, sanitize_id, print_input, print_output, use_smart_router, max_reflections) 
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
                    
                    total_tokens += result['total_tokens']
                    total_estimated_cost += result['estimated_cost']
                    if result['transformations_json_generated']:
                        successful_json_generations += 1
                    
                    # Show progress
                    status = "✓" if result['transformations_json_generated'] else "✗"
                    print(f"{status} Completed {task_id} ({i+1}/{len(selected_tasks)})")
                    
                except Exception as e:
                    print(f"{RED}✗ Error processing {task_id}: {e}{RESET}")
                    # Create empty result for failed task
                    error_result = {
                        'total_tokens': 0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'estimated_cost': 0.0,
                        'transformations_json_generated': False,
                        'transformations_json': None,
                        'trains': [],
                        'tests': []
                    }
                    all_results.append(error_result)
        
        # Sort results by processing order (we don't have task_id in new format)
        # Results are already in completion order which is fine
        
    else:
        # Sequential processing
        for i, (task_id, task_data) in enumerate(selected_tasks):
            print(f"\n{'='*100}")
            print(f"TASK {i+1}/{len(selected_tasks)}")
            print(f"{'='*100}")
            
            result = run_single_task(task_id, task_data, solutions, model, provider, output_dir, sanitize_task, sanitize_id, print_input, print_output, use_smart_router, max_reflections)
            all_results.append(result)
            
            total_tokens += result['total_tokens']
            total_estimated_cost += result['estimated_cost']
            if result['transformations_json_generated']:
                successful_json_generations += 1
    
    # Print final statistics
    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Calculate statistics from new format
    total_correct_tests = 0
    total_correct_trains = 0
    total_test_count = 0
    total_train_count = 0
    avg_test_overlap = 0.0
    avg_train_overlap = 0.0
    
    for result in all_results:
        # Count test results
        for test_result in result.get('tests', []):
            total_test_count += 1
            if test_result.get('correct', False):
                total_correct_tests += 1
            avg_test_overlap += test_result.get('overlap', 0.0)
        
        # Count train results
        for train_result in result.get('trains', []):
            total_train_count += 1
            if train_result.get('correct', False):
                total_correct_trains += 1
            avg_train_overlap += train_result.get('overlap', 0.0)
    
    # Calculate averages
    if total_test_count > 0:
        avg_test_overlap /= total_test_count
        test_accuracy = (total_correct_tests / total_test_count) * 100
    else:
        test_accuracy = 0.0
        avg_test_overlap = 0.0
    
    if total_train_count > 0:
        avg_train_overlap /= total_train_count
        train_accuracy = (total_correct_trains / total_train_count) * 100
    else:
        train_accuracy = 0.0
        avg_train_overlap = 0.0

    print(f"\n{BLUE}TASK PERFORMANCE:{RESET}")
    print(f"  Total tasks processed: {len(selected_tasks)}")
    print(f"  Successful JSON generations: {successful_json_generations}/{len(selected_tasks)} ({successful_json_generations/len(selected_tasks)*100:.1f}%)")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Total estimated cost: ${total_estimated_cost:.6f}")
    print(f"  Average cost per task: ${total_estimated_cost/len(selected_tasks):.6f}")
    
    print(f"\n{BLUE}TEST PERFORMANCE:{RESET}")
    print(f"  Total test examples: {total_test_count}")
    print(f"  Correct test predictions: {total_correct_tests}")
    print(f"  Test accuracy: {test_accuracy:.1f}%")
    print(f"  Average test overlap: {avg_test_overlap:.1f}%")
    
    print(f"\n{BLUE}TRAINING PERFORMANCE:{RESET}")
    print(f"  Total training examples: {total_train_count}")
    print(f"  Correct training predictions: {total_correct_trains}")
    print(f"  Training accuracy: {train_accuracy:.1f}%")
    print(f"  Average training overlap: {avg_train_overlap:.1f}%")
    
    print(f"\n{BLUE}DETAILED BREAKDOWN:{RESET}")
    for i, result in enumerate(all_results):
        task_id = selected_tasks[i][0] if i < len(selected_tasks) else f"task_{i}"
        json_status = "✓" if result['transformations_json_generated'] else "✗"
        cost = result['estimated_cost']
        tokens = result['total_tokens']
        print(f"  {json_status} {task_id} - Tokens: {tokens:,}, Cost: ${cost:.6f}")
    
    # Overall performance message
    if successful_json_generations == len(selected_tasks):
        print(f"\n{GREEN}🎉 ALL TASKS GENERATED JSON! {successful_json_generations}/{len(selected_tasks)} successful generations.{RESET}")
    elif successful_json_generations > len(selected_tasks) * 0.7:
        print(f"\n{GREEN}🟢 EXCELLENT PERFORMANCE! {successful_json_generations}/{len(selected_tasks)} successful JSON generations.{RESET}")
    elif successful_json_generations > len(selected_tasks) * 0.3:
        print(f"\n{BLUE}🟡 MODERATE PERFORMANCE. {successful_json_generations}/{len(selected_tasks)} successful JSON generations.{RESET}")
    else:
        print(f"\n{RED}🔴 POOR PERFORMANCE. Only {successful_json_generations}/{len(selected_tasks)} successful JSON generations.{RESET}")
    
    # Generate summary statistics
    print(f"\nGenerating summary statistics...")
    try:
        calculate_results(str(output_dir))
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    # Generate task ID files
    print(f"\nGenerating task ID files...")
    try:
        training_task_ids = list(training_tasks.keys())
        evaluation_task_ids = list(evaluation_tasks.keys())
        
        # Write training task IDs
        training_ids_file = output_dir / "training_task_ids.txt"
        with open(training_ids_file, 'w') as f:
            f.write('\n'.join(training_task_ids))
        print(f"Training task IDs saved to: {training_ids_file}")
        
        # Write evaluation task IDs
        evaluation_ids_file = output_dir / "evaluation_task_ids.txt"
        with open(evaluation_ids_file, 'w') as f:
            f.write('\n'.join(evaluation_task_ids))
        print(f"Evaluation task IDs saved to: {evaluation_ids_file}")
        
    except Exception as e:
        print(f"Error generating task ID files: {e}")

    print(f"\n{'='*100}")
    print("RUN COMPLETED")
    print(f"{'='*100}")
    print(f"Results saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()