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
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

from model_configs import MODEL_CONFIGS, is_known_model
from prompts import build_arc_prompt, build_arc_baseline_prompt

# We reuse many configuration variables from test_arc_prompt.py for consistency
TASK_INDEX = None
NUMBER_OF_TASKS = -1  # Set to -1 to run ALL tasks
MODEL = "qwen2.5:32b"
CHALLENGES_PATH = "data/arc-2025/arc-agi_training_challenges.json"
SOLUTIONS_PATH = "data/arc-2025/arc-agi_training_solutions.json"
RANDOM_SEED = 42
TEMPERATURE = 0.70
NUM_PREDICT = -1
PARALLEL = False
PRINT_INPUT = False
PRINT_OUTPUT = False

BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
 
def run_with_ollama(prompt, model="llama3.1"):
    # Reuse the simple Ollama runner from test_arc_prompt.py
    api_url = "http://localhost:11434/api/generate"
    options = {"temperature": TEMPERATURE}
    if NUM_PREDICT != -1:
        options["num_predict"] = NUM_PREDICT

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


def main():
    load_dotenv()
    timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')
    output_dir = Path('output') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Basic checks
    if not is_known_model(MODEL):
        print(f"Unknown model: {MODEL}")
        sys.exit(1)

    challenges_path = Path(CHALLENGES_PATH)
    solutions_path = Path(SOLUTIONS_PATH)
    if not challenges_path.exists() or not solutions_path.exists():
        print("Missing challenges or solutions JSON files. Update CHALLENGES_PATH / SOLUTIONS_PATH")
        sys.exit(1)

    tasks = json.load(open(challenges_path))
    solutions = json.load(open(solutions_path))

    # Select multiple tasks for baseline using NUMBER_OF_TASKS (or a specific TASK_INDEX)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    all_task_ids = list(tasks.keys())

    if TASK_INDEX is None:
        if NUMBER_OF_TASKS == -1:
            # Use all tasks when NUMBER_OF_TASKS is -1
            selected_ids = all_task_ids
            print(f"Running ALL tasks ({len(all_task_ids)} tasks)")
        else:
            n = min(NUMBER_OF_TASKS, len(all_task_ids))
            selected_ids = random.sample(all_task_ids, n)
    else:
        # If TASK_INDEX provided, run that single task (keeps backward compatibility)
        if TASK_INDEX < 0 or TASK_INDEX >= len(all_task_ids):
            print(f"TASK_INDEX {TASK_INDEX} out of range")
            sys.exit(1)
        selected_ids = [all_task_ids[TASK_INDEX]]

    print(f"Running {len(selected_ids)} task(s): {selected_ids}")

    overall_scores = []

    model_config = MODEL_CONFIGS.get(MODEL, {})
    provider = model_config.get('provider', 'ollama')

    for task_id in selected_ids:
        task = tasks[task_id]
        prompt = build_arc_baseline_prompt({task_id: task}, task_id)

        if PRINT_INPUT:
            print(prompt)
        else:
            print(f"Sending prompt for task {task_id} (hidden)")

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
                    model_instance = genai.GenerativeModel(model_name=MODEL, generation_config={"temperature": TEMPERATURE})
                    response = model_instance.generate_content(prompt)
                    response_text = response.text
                except Exception as e:
                    print(f"Gemini call failed: {e}")
                    sys.exit(1)
        else:
            response_text = run_with_ollama(prompt, model=MODEL)

        if not response_text:
            print(f"No response from model for task {task_id}.")
            overall_scores.append(0.0)
            continue

        if PRINT_OUTPUT:
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

        # Compare predicted_list and expected_list elementwise (by index)
        scores = []
        for i, expected in enumerate(expected_list):
            predicted = predicted_list[i] if i < len(predicted_list) else None
            overlap = calculate_grid_overlap(predicted, expected)
            scores.append(overlap)
            print(f"{task_id} - Test {i}: Overlap = {overlap:.1f}%")

        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average overlap for task {task_id}: {avg:.1f}%")
        overall_scores.append(avg)

        # Save per-task results immediately for later scoring
        try:
            task_result = {
                task_id: {
                    'test': []
                }
            }

            for i, expected in enumerate(expected_list):
                predicted = predicted_list[i] if i < len(predicted_list) else None
                entry = {
                    'input': task['test'][i]['input'] if i < len(task.get('test', [])) else None,
                    'output': expected,
                    'produce': predicted
                }
                task_result[task_id]['test'].append(entry)

            out_file = output_dir / f"{task_id}.json"
            with open(out_file, 'w') as f:
                json.dump(task_result, f, indent=2)

            print(f"Saved results for {task_id} -> {out_file}")
        except Exception as e:
            print(f"Failed to save results for {task_id}: {e}")

    # Overall summary
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"\nOverall average overlap across {len(overall_scores)} task(s): {overall_avg:.1f}%")


if __name__ == '__main__':
    main()
