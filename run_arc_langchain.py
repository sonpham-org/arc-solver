#!/usr/bin/env python3
"""
Simple LangChain agent for ARC puzzle solving.

This script creates a simple agent that uses LangChain to generate JSON solutions 
for ARC (Abstraction and Reasoning Corpus) puzzles using the arc_prompt system.
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
import os

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")

# Import the ARC prompt builder
from prompts.arc_prompt import build_arc_prompt

# Import utilities from our organized utils package
from utils import (
    load_arc_tasks, get_task_by_index,
    extract_json_from_response, validate_json_structure,
    print_colored, run_with_langchain
)
from utils.display_utils import BLUE, GREEN, RED, YELLOW, RESET, BOLD

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

# Task selection
TASK_INDEX = None  # Set to specific index (0-399), task ID string, or None for random
MODEL = "gemini-2.5-flash"  # LangChain model to use
YEAR = 2024  # ARC dataset year
RANDOM_SEED = None  # Random seed for reproducible task selection

# Data paths
if YEAR == 2024:
    TRAINING_CHALLENGES_PATH = "data/arc-2024/arc-agi_training_challenges.json"
    TRAINING_SOLUTIONS_PATH = "data/arc-2024/arc-agi_training_solutions.json"
else:
    TRAINING_CHALLENGES_PATH = "data/arc-2025/arc-agi_training_challenges.json"
    TRAINING_SOLUTIONS_PATH = "data/arc-2025/arc-agi_training_solutions.json"

# Output settings
PRINT_INPUT = True
PRINT_OUTPUT = True
PRINT_JSON = True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple ARC LangChain Agent")
    
    parser.add_argument("--task-index", type=str, default=TASK_INDEX, 
                        help=f"Index (0-399) or task ID string to run (default: {TASK_INDEX})")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"LangChain model to use (default: {MODEL})")
    parser.add_argument("--training-challenges-path", type=str, default=TRAINING_CHALLENGES_PATH,
                        help=f"Path to training challenges JSON file (default: {TRAINING_CHALLENGES_PATH})")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for task selection (default: {RANDOM_SEED})")
    parser.add_argument("--print-input", action="store_true", default=PRINT_INPUT,
                        help=f"Print input prompts (default: {PRINT_INPUT})")
    parser.add_argument("--print-output", action="store_true", default=PRINT_OUTPUT,
                        help=f"Print model outputs (default: {PRINT_OUTPUT})")
    parser.add_argument("--print-json", action="store_true", default=PRINT_JSON,
                        help=f"Print extracted JSON (default: {PRINT_JSON})")
    
    return parser.parse_args()

def run_arc_task(task_id, task_data, model_name, print_input=True, print_output=True, print_json=True):
    """Run a single ARC task using LangChain."""
    
    print_colored(f"\n{'='*60}", GREEN)
    print_colored(f"RUNNING TASK: {task_id}", GREEN, bold=True)
    print_colored(f"{'='*60}", GREEN)
    
    print(f"Task details:")
    print(f"  - Training examples: {len(task_data.get('train', []))}")
    print(f"  - Test examples: {len(task_data.get('test', []))}")
    
    # Generate the prompt
    print_colored("\n🔄 Generating ARC prompt...", BLUE)
    prompt = build_arc_prompt({task_id: task_data}, task_id)
    
    if print_input:
        print_colored(f"\n{'='*60}", BLUE)
        print_colored("PROMPT SENT TO LLM:", BLUE, bold=True)
        print_colored(f"{'='*60}", BLUE)
        print_colored(prompt, BLUE)
        print_colored(f"{'='*60}", BLUE)
    
    # Run with LangChain
    print_colored(f"\n🤖 Running with LangChain model: {model_name}", GREEN)
    response = run_with_langchain(prompt, model_name)
    
    if not response:
        print_colored("❌ Failed to get response from model", RED)
        return None
    
    if print_output:
        print_colored(f"\n{'='*60}", GREEN)
        print_colored("MODEL RESPONSE:", GREEN, bold=True)
        print_colored(f"{'='*60}", GREEN)
        print_colored(response, GREEN)
        print_colored(f"{'='*60}", GREEN)
    
    # Extract and validate JSON
    print_colored("\n📋 Extracting JSON from response...", BLUE)
    extracted_json = extract_json_from_response(response)
    
    if not extracted_json:
        print_colored("❌ No valid JSON found in response", RED)
        return {
            'task_id': task_id,
            'success': False,
            'error': 'No JSON extracted',
            'response': response
        }
    
    print_colored("✅ JSON successfully extracted!", GREEN)
    
    # Validate JSON structure
    is_valid, validation_message = validate_json_structure(extracted_json)
    
    if is_valid:
        print_colored(f"✅ JSON structure is valid: {validation_message}", GREEN)
    else:
        print_colored(f"⚠️  JSON structure issue: {validation_message}", YELLOW)
    
    if print_json:
        print_colored(f"\n{'='*60}", YELLOW)
        print_colored("EXTRACTED JSON:", YELLOW, bold=True)
        print_colored(f"{'='*60}", YELLOW)
        print_colored(json.dumps(extracted_json, indent=2), YELLOW)
        print_colored(f"{'='*60}", YELLOW)
    
    # Analyze the JSON structure
    print_colored("\n📊 JSON Analysis:", BLUE, bold=True)
    
    if "helper_python_functions" in extracted_json:
        helper_funcs = extracted_json["helper_python_functions"]
        print(f"  - Helper functions: {len(helper_funcs) if isinstance(helper_funcs, list) else 'Invalid format'}")
    
    if "step_by_step_transformations" in extracted_json:
        steps = extracted_json["step_by_step_transformations"]
        if isinstance(steps, list):
            print(f"  - Transformation steps: {len(steps)}")
            for i, step in enumerate(steps, 1):
                if isinstance(step, dict) and "description" in step:
                    desc = step["description"]
                    if isinstance(desc, list) and desc:
                        print(f"    Step {i}: {desc[0][:60]}{'...' if len(desc[0]) > 60 else ''}")
        else:
            print(f"  - Transformation steps: Invalid format")
    
    if "python_code" in extracted_json:
        code = extracted_json["python_code"]
        if isinstance(code, list):
            print(f"  - Python code lines: {len(code)}")
            # Show first few lines of code
            for i, line in enumerate(code[:3]):
                print(f"    {i+1}: {line}")
            if len(code) > 3:
                print(f"    ... ({len(code) - 3} more lines)")
        else:
            print(f"  - Python code: Invalid format")
    
    return {
        'task_id': task_id,
        'success': True,
        'json_valid': is_valid,
        'validation_message': validation_message,
        'extracted_json': extracted_json,
        'response': response
    }


def main():
    """Main function to run the LangChain ARC agent."""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.random_seed is not None:
        random.seed(args.random_seed)
        print_colored(f"🎲 Random seed set to: {args.random_seed}", BLUE)
    
    print_colored(f"\n{'='*80}", GREEN)
    print_colored("ARC LANGCHAIN AGENT", GREEN, bold=True)
    print_colored(f"{'='*80}", GREEN)
    print_colored(f"Model: {args.model}", GREEN)
    print_colored(f"Dataset: ARC {YEAR}", GREEN)
    print_colored(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", GREEN)
    
    # Load tasks
    print_colored(f"\n📁 Loading tasks from: {args.training_challenges_path}", BLUE)
    
    try:
        tasks = load_arc_tasks(args.training_challenges_path)
        print_colored(f"✅ Loaded {len(tasks)} tasks", GREEN)
    except FileNotFoundError:
        print_colored(f"❌ File not found: {args.training_challenges_path}", RED)
        return
    except Exception as e:
        print_colored(f"❌ Error loading tasks: {e}", RED)
        return
    
    # Select task
    try:
        if args.task_index and args.task_index.isdigit():
            task_index = int(args.task_index)
        else:
            task_index = args.task_index
            
        task_id, selected_index, task_data = get_task_by_index(tasks, task_index)
        print_colored(f"📋 Selected task: {task_id} (index: {selected_index})", GREEN)
        
    except (ValueError, IndexError) as e:
        print_colored(f"❌ Task selection error: {e}", RED)
        return
    
    # Run the task
    result = run_arc_task(
        task_id=task_id,
        task_data=task_data,
        model_name=args.model,
        print_input=args.print_input,
        print_output=args.print_output,
        print_json=args.print_json
    )
    
    # Summary
    print_colored(f"\n{'='*80}", GREEN)
    print_colored("SUMMARY", GREEN, bold=True)
    print_colored(f"{'='*80}", GREEN)
    
    if result and result['success']:
        print_colored(f"✅ Task {task_id} completed successfully", GREEN)
        print_colored(f"✅ JSON extracted: {'Yes' if result.get('extracted_json') else 'No'}", GREEN)
        print_colored(f"✅ JSON structure valid: {'Yes' if result.get('json_valid') else 'No'}", GREEN if result.get('json_valid') else YELLOW)
        if result.get('validation_message'):
            print_colored(f"   Validation: {result['validation_message']}", GREEN if result.get('json_valid') else YELLOW)
    else:
        print_colored(f"❌ Task {task_id} failed", RED)
        if result and result.get('error'):
            print_colored(f"   Error: {result['error']}", RED)
    
    print_colored(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", BLUE)


if __name__ == "__main__":
    main()