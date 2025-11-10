#!/usr/bin/env python3
"""
Baseline test script for ARC: sends a minimal prompt asking the model to output
the expected test output grid as a JSON 2D array and compares directly to the
known solution for the same task id.

This is adapted from test_arc_prompt.py but simplified to only request the
final test output in a JSON array and compute a simple overlap/score.
"""

import json
import os
import random
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

from model_configs import MODEL_CONFIGS, is_known_model, estimate_cost
from prompts import build_arc_prompt, build_arc_baseline_prompt
from utils import calculate_results

# Configuration variables - edit these directly or use command line args
TASK_INDEX = None
NUMBER_OF_TASKS = -1  # Set to -1 to run ALL tasks
MODEL = "gemini-2.5-flash-lite-preview-06-17"  # The other is "qwen2.5:32b"
# Path to the ARC challenges and solutions JSON files
YEAR = 2024
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
RANDOM_SEED = 42
TEMPERATURE = 0.70
NUM_PREDICT = -1
PARALLEL = True
PRINT_INPUT = False
PRINT_OUTPUT = False

def parse_arguments():
    """Parse command line arguments with defaults from ALL_CAPS variables."""
    parser = argparse.ArgumentParser(description="ARC Baseline Test Script")
    
    parser.add_argument("--task-index", type=int, default=TASK_INDEX, 
                        help=f"Index of specific task to run (default: {TASK_INDEX})")
    parser.add_argument("--number-of-tasks", type=int, default=NUMBER_OF_TASKS,
                        help=f"Number of tasks to run (-1 for ALL tasks) (default: {NUMBER_OF_TASKS})")
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
    parser.add_argument("--parallel", action="store_true", default=PARALLEL,
                        help=f"Run tasks in parallel (default: {PARALLEL})")
    parser.add_argument("--print-input", action="store_true", default=PRINT_INPUT,
                        help=f"Print input prompts (default: {PRINT_INPUT})")
    parser.add_argument("--print-output", action="store_true", default=PRINT_OUTPUT,
                        help=f"Print model outputs (default: {PRINT_OUTPUT})")
    
    return parser.parse_args()

def count_tokens_simple(text):
    """Simple token counting approximation (words * 1.3 for rough estimate)."""
    if not text:
        return 0
    # Simple approximation: split by whitespace and multiply by 1.3
    words = len(text.split())
    return int(words * 1.3)

def run_with_ollama(prompt, model="llama3.1", temperature=0.7, num_predict=-1):
    # Reuse the simple Ollama runner from test_arc_prompt.py
    api_url = "http://localhost:11434/api/generate"
    options = {"temperature": temperature}
    if num_predict != -1:
        options["num_predict"] = num_predict

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options
    }

    try:
        resp = requests.post(api_url, json=data, timeout=300)
        if resp.status_code == 200:
            return resp.json().get('response', '')
        else:
            print(f"Ollama HTTP error: {resp.status_code}")
            return None
    except Exception:
        # Fallback to CLI if API not available
        try:
            result = subprocess.run(["ollama", "run", model], input=prompt, text=True, capture_output=True, check=True)
            return result.stdout
        except Exception as e:
            print(f"Ollama error: {e}")
            return None


def extract_json_from_response(response: str):
    if not response:
        return None
    import re
    pattern = r'```json\s*\n(.*?)\n```'
    m = re.findall(pattern, response, re.DOTALL)
    if not m:
        # Try to parse entire response as JSON
        try:
            return json.loads(response)
        except Exception:
            return None
    try:
        return json.loads(m[0])
    except Exception:
        return None


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


def calculate_grid_overlap(predicted, expected):
    if predicted is None or expected is None:
        return 0.0
    # If predicted is a single grid but expected is a single grid, compare directly
    if isinstance(predicted, list) and isinstance(expected, list) and all(isinstance(r, list) for r in predicted) and all(isinstance(r, list) for r in expected):
        # 2D comparison
        if len(predicted) != len(expected):
            return 0.0
        total = 0
        match = 0
        for pr, er in zip(predicted, expected):
            if len(pr) != len(er):
                return 0.0
            for a, b in zip(pr, er):
                total += 1
                if a == b:
                    match += 1
        return (match / total) * 100.0 if total else 0.0
    else:
        return 0.0


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


def main():
    # Parse command line arguments - they override the ALL_CAPS defaults
    args = parse_arguments()
    
    # Use argparse values (which default to ALL_CAPS variables)
    task_index = args.task_index
    number_of_tasks = args.number_of_tasks
    model = args.model
    training_challenges_path = args.training_challenges_path
    training_solutions_path = args.training_solutions_path
    evaluation_challenges_path = args.evaluation_challenges_path
    evaluation_solutions_path = args.evaluation_solutions_path
    random_seed = args.random_seed
    temperature = args.temperature
    num_predict = args.num_predict
    parallel = args.parallel
    print_input = args.print_input
    print_output = args.print_output
    
    load_dotenv()
    timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')
    output_dir = Path('output') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters to params.json
    params = {
        'script': 'test_arc_baseline.py',
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
        'parallel': parallel,
        'print_input': print_input,
        'print_output': print_output,
        'timestamp': timestamp
    }
    
    params_file = output_dir / 'params.json'
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {params_file}")

    # Basic checks
    if not is_known_model(model):
        print(f"Unknown model: {model}")
        sys.exit(1)

    training_challenges_path = Path(training_challenges_path)
    training_solutions_path = Path(training_solutions_path)
    evaluation_challenges_path = Path(evaluation_challenges_path)
    evaluation_solutions_path = Path(evaluation_solutions_path)
    
    if not training_challenges_path.exists() or not training_solutions_path.exists():
        print("Missing training challenges or solutions JSON files. Update TRAINING_CHALLENGES_PATH / TRAINING_SOLUTIONS_PATH")
        sys.exit(1)
    
    if not evaluation_challenges_path.exists() or not evaluation_solutions_path.exists():
        print("Missing evaluation challenges or solutions JSON files. Update EVALUATION_CHALLENGES_PATH / EVALUATION_SOLUTIONS_PATH")
        sys.exit(1)

    training_tasks = json.load(open(training_challenges_path))
    training_solutions = json.load(open(training_solutions_path))
    evaluation_tasks = json.load(open(evaluation_challenges_path))
    evaluation_solutions = json.load(open(evaluation_solutions_path))
    
    # Combine tasks and solutions for processing
    tasks = {**training_tasks, **evaluation_tasks}
    solutions = {**training_solutions, **evaluation_solutions}

    # Select multiple tasks for baseline using number_of_tasks (or a specific task_index)
    if random_seed is not None:
        random.seed(random_seed)

    all_task_ids = list(tasks.keys())

    if task_index is None:
        if number_of_tasks == -1:
            # Use all tasks when number_of_tasks is -1
            selected_ids = all_task_ids
            print(f"Running ALL tasks ({len(all_task_ids)} tasks)")
        else:
            n = min(number_of_tasks, len(all_task_ids))
            selected_ids = random.sample(all_task_ids, n)
    else:
        # If task_index provided, run that single task (keeps backward compatibility)
        if task_index < 0 or task_index >= len(all_task_ids):
            print(f"task_index {task_index} out of range")
            sys.exit(1)
        selected_ids = [all_task_ids[task_index]]

    print(f"Running {len(selected_ids)} task(s): {selected_ids}")

    overall_scores = []

    model_config = MODEL_CONFIGS.get(model, {})
    provider = model_config.get('provider', 'ollama')

    for task_id in selected_ids:
        task = tasks[task_id]
        
        # Process training examples first (don't send to model, just save ground truth)
        train_results = []
        train_data = task.get('train', [])
        
        for train_example in train_data:
            train_input = train_example['input']
            train_output = train_example['output']
            
            # For training data, we have the correct output so predict is same as output
            iou = 100.0  # Perfect match since we're using ground truth
            overlap = 100.0  # Perfect match
            correct = True
            valid_predict = True
            
            train_entry = {
                'input': grid_to_string_lines(train_input),
                'output': grid_to_string_lines(train_output), 
                'valid_predict': valid_predict,
                'predict': grid_to_string_lines(train_output),
                'iou': iou,
                'overlap': overlap,
                'correct': correct
            }
            train_results.append(train_entry)
        
        # Now process test examples with model predictions
        prompt = build_arc_baseline_prompt({task_id: task}, task_id)

        if print_input:
            print(prompt)
        else:
            print(f"Sending prompt for task {task_id} (hidden)")

        # Count input tokens
        input_tokens = count_tokens_simple(prompt)

        # Call model
        if provider == 'google' or provider == 'learnlm':
            response_text = None
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("GEMINI_API_KEY not set; cannot use Gemini")
                sys.exit(1)
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model_instance = genai.GenerativeModel(model_name=model, generation_config={"temperature": temperature})
                    response = model_instance.generate_content(prompt)
                    response_text = response.text
                except Exception as e:
                    print(f"Gemini call failed: {e}")
                    sys.exit(1)
        else:
            response_text = run_with_ollama(prompt, model=model, temperature=temperature, num_predict=num_predict)

        if not response_text:
            print(f"No response from model for task {task_id}.")
            overall_scores.append(0.0)
            continue

        # Count output tokens
        output_tokens = count_tokens_simple(response_text)
        total_tokens = input_tokens + output_tokens

        # Calculate estimated cost
        try:
            estimated_cost = estimate_cost(model, input_tokens, output_tokens)
        except KeyError:
            estimated_cost = 0.0  # Unknown model or free model

        print(f"Task {task_id}: Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total: {total_tokens}, Estimated cost: ${estimated_cost:.6f}")

        if print_output:
            print(response_text)

        extracted = extract_json_from_response(response_text)
        if extracted is None:
            print(f"Failed to extract JSON from model response for task {task_id}")
            overall_scores.append(0.0)
            continue

        # Normalize extracted to list of grids
        if isinstance(extracted, list) and extracted and all(isinstance(r, list) for r in extracted[0]):
            predicted_list = extracted
        elif isinstance(extracted, list) and all(isinstance(r, list) for r in extracted):
            predicted_list = [extracted]
        else:
            predicted_list = [extracted]

        # Get solution(s) for this task
        task_solutions = solutions.get(task_id)
        if not task_solutions:
            print(f"No solutions found for task {task_id}")
            overall_scores.append(0.0)
            continue

        expected_list = task_solutions

        # Process test results with new format
        test_results = []
        scores = []
        
        for i, expected in enumerate(expected_list):
            predicted = predicted_list[i] if i < len(predicted_list) else None
            test_input = task['test'][i]['input'] if i < len(task.get('test', [])) else None
            
            valid_predict = is_valid_prediction(predicted)
            iou = calculate_grid_iou(predicted, expected) if valid_predict else 0.0
            overlap = calculate_grid_overlap(predicted, expected) if valid_predict else 0.0
            correct = (iou == 100.0 and overlap == 100.0) if valid_predict else False
            
            scores.append(overlap)
            print(f"{task_id} - Test {i}: IOU={iou:.1f}%, Overlap={overlap:.1f}%, Correct={correct}")
            
            test_entry = {
                'input': grid_to_string_lines(test_input) if test_input else [],
                'output': grid_to_string_lines(expected),
                'predict': grid_to_string_lines(predicted) if predicted else [],
                'iou': iou,
                'overlap': overlap,
                'correct': correct
            }
            test_results.append(test_entry)

        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average overlap for task {task_id}: {avg:.1f}%")
        overall_scores.append(avg)

        # Save per-task results with new format
        try:
            task_result = {
                'trains': train_results,
                'test': test_results,
                'total_tokens': total_tokens,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'estimated_cost': estimated_cost
            }

            out_file = output_dir / f"{task_id}.json"
            with open(out_file, 'w') as f:
                json.dump(task_result, f, indent=2)

            print(f"Saved results for {task_id} -> {out_file}")
        except Exception as e:
            print(f"Failed to save results for {task_id}: {e}")

    # Overall summary
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"\nOverall average overlap across {len(overall_scores)} task(s): {overall_avg:.1f}%")
    
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


if __name__ == '__main__':
    main()
