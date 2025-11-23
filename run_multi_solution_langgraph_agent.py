#!/usr/bin/env python3
"""
LangGraph multi-solution agent runner for ARC tasks.
This script mirrors `run_langgraph_agent.py` but uses the
`MultiSolutionARCLangGraphAgent` to collect multiple solutions per task.
"""
# Load environment variables first, before any other imports
from dotenv import load_dotenv
load_dotenv()

# Suppress Google Cloud warnings immediately after loading .env
import os
import sys
import warnings

# Set environment variables to suppress Google Cloud warnings
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2') 
os.environ.setdefault('GRPC_LOG_SEVERITY_LEVEL', 'ERROR')
os.environ.setdefault('GOOGLE_CLOUD_DISABLE_GRPC_LOGS', 'true')
os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', '')
os.environ.setdefault('GOOGLE_CLOUD_PROJECT', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Suppress Python warnings
warnings.filterwarnings("ignore", message=".*ALTS.*")
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
import random
import datetime
import sys as _sys
import time
import concurrent.futures
import threading
from typing import Dict, List, Any, Tuple, Optional

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the MultiSolution LangGraph agent
from arc_langgraph_multi_solution_agents.agent import MultiSolutionARCLangGraphAgent

# Import model configurations and utilities
from model_configs import MODEL_CONFIGS, find_model_key


# ==========================================================================
# CONFIGURATION VARIABLES - Modify these as needed
# ==========================================================================

# Model selection: Choose from available models in model_configs.py
MODEL = "gemini-2.5-flash-lite"

# Test mode configuration
MODE = "batch"  # "single" or "batch"
NUM_WORKERS = 8  # Number of parallel workers for batch mode

# Task selection for single mode
TASK_ID = None  # Specific task ID to test (for single mode)
TASK_INDEX = None  # Task index to test (for single mode)

# Batch mode configuration
NUM_TASKS = 100  # Number of tasks for batch mode

# Processing configuration
MAX_ATTEMPTS = 3  # Maximum attempts per task
RANDOM_SEED = 42  # Random seed for reproducibility

ENABLE_CODE_PREDICT = True  # Whether to enable code-predicted outputs during testing
ENABLE_LLM_PREDICT = False # Whether to enable LLM-predicted outputs during testing
ENABLE_VISUAL_CUE = False  # When True, generate and pass input/output images to the LLM

NUM_INITIAL_SOLUTIONS = 10
NUM_LOOPS = 3
NUM_SEED_SOLUTIONS = 10
NUM_REFINEMENTS = 5
NUM_SOLUTIONS_PER_REFINEMENT = 3
NUM_FUSIONS = 5
NUM_SOLUTIONS_PER_FUSION = 3

# Year selection for ARC dataset directory (change to 2025 if using 2025 data)
YEAR = 2025

# Default ARC JSON paths (will be exposed as argparse defaults)
TRAINING_TASKS_JSON = f"data/arc-{YEAR}/arc-agi_training_challenges.json"
TRAINING_SOLUTIONS_JSON = f"data/arc-{YEAR}/arc-agi_training_solutions.json"
EVALUATION_TASKS_JSON = f"data/arc-{YEAR}/arc-agi_evaluation_challenges.json"
EVALUATION_SOLUTIONS_JSON = f"data/arc-{YEAR}/arc-agi_evaluation_solutions.json"


def load_arc_tasks(file_path: str) -> Dict[str, Dict]:
    """Load ARC tasks from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_arc_solutions(file_path: str) -> Dict[str, List[List[List[int]]]]:
    """Load ARC solutions from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_task_by_index(
    tasks: Dict[str, Dict],
    solutions: Optional[Dict[str, List[List[List[int]]]]] = None,
    index: Optional[int] = None
) -> Tuple[str, Dict, Optional[Any]]:
    """Get task by index or random task if index is None.

    Also return the associated solution from `solutions` when available.
    """
    task_ids = list(tasks.keys())
    if index is not None:
        if 0 <= index < len(task_ids):
            task_id = task_ids[index]
        else:
            raise IndexError(f"Task index {index} out of range [0, {len(task_ids)-1}]")
    else:
        task_id = random.choice(task_ids)

    task_data = tasks[task_id]
    task_solution = solutions.get(task_id) if solutions is not None else None
    return task_id, task_data, task_solution


def create_output_directory() -> str:
    """Create a timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_params(output_dir: str, args: argparse.Namespace, model: str) -> None:
    """Save run parameters to params.json."""
    params = {
        "model": model,
        "mode": args.mode,
        "max_attempts": args.max_attempts,
        "random_seed": args.random_seed,
        "timestamp": datetime.datetime.now().isoformat()
    }
    # Include year and file paths when available
    if hasattr(args, 'YEAR') and args.YEAR is not None:
        params['year'] = args.YEAR
    for key in ('TRAINING_TASKS_JSON', 'TRAINING_SOLUTIONS_JSON', 'EVALUATION_TASKS_JSON', 'EVALUATION_SOLUTIONS_JSON'):
        if hasattr(args, key) and getattr(args, key):
            params[key.lower()] = getattr(args, key)
    if args.mode == "single":
        if args.task_id:
            params["task_id"] = args.task_id
        elif args.task_index is not None:
            params["task_index"] = args.task_index
    elif args.mode == "batch":
        params["num_tasks"] = args.num_tasks

    with open(os.path.join(output_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=2)


def save_task_ids(output_dir: str, training_tasks: Dict, evaluation_tasks: Dict) -> None:
    """Save task ID lists to separate files."""
    with open(os.path.join(output_dir, "training_task_ids.txt"), 'w') as f:
        for task_id in sorted(training_tasks.keys()):
            f.write(f"{task_id}\n")
    
    with open(os.path.join(output_dir, "evaluation_task_ids.txt"), 'w') as f:
        for task_id in sorted(evaluation_tasks.keys()):
            f.write(f"{task_id}\n")


def save_task_result(output_dir: str, task_id: str, result: Dict[str, Any]) -> None:
    """Save individual task result to JSON file."""
    def grid_to_string_lines(grid: Optional[List[List[Any]]]) -> List[str]:
        """Convert a 2D grid of values into a list of string lines for easy viewing."""
        if not grid:
            return []
        try:
            return ["".join(str(c) for c in row) for row in grid]
        except Exception:
            # Fallback: stringify each row
            return [str(row) for row in grid]

    # Normalize training_results and testing_results to include
    # `predicted_output` and `expected_output` as list-of-strings views
    for si, solution in enumerate(result.get("solutions_list", [])):
        for key in ("training_results", "testing_results"):
            entries = solution.get(key)
            if not entries:
                continue
            for entry in entries:
                # Set list-of-strings fields (override or add)
                entry["expected_output"] = grid_to_string_lines(entry["expected_output"])
                entry["predicted_output"] = grid_to_string_lines(entry["predicted_output"])
                entry["llm_predicted_output"] = grid_to_string_lines(entry.get("llm_predicted_output"))
                entry["input"] = grid_to_string_lines(entry["input"])
        # Also check for any visual cues attached to the solution's transformation
        # (actions may attach `_visual_cues` to the step_by_step_transformation)
        try:
            trans = solution.get('step_by_step_transformation')
            visual_files = []
            if isinstance(trans, dict) and trans.get('_visual_cues'):
                import base64
                # Ensure per-task folder exists
                task_folder = os.path.join(output_dir, task_id)
                try:
                    os.makedirs(task_folder, exist_ok=True)
                except Exception:
                    task_folder = output_dir

                for vc in trans.get('_visual_cues'):
                    ex_idx = vc.get('example_index')
                    in_b64 = vc.get('input_b64')
                    out_b64 = vc.get('output_b64')
                    if in_b64:
                        try:
                            img_bytes = base64.b64decode(in_b64)
                            fname_in = f"{task_id}_sol{si}_ex{ex_idx}_input.png"
                            fpath_in = os.path.join(task_folder, fname_in)
                            with open(fpath_in, 'wb') as imgf:
                                imgf.write(img_bytes)
                            visual_files.append(os.path.join(task_id, fname_in) if task_folder != output_dir else fname_in)
                        except Exception:
                            pass
                    if out_b64:
                        try:
                            img_bytes = base64.b64decode(out_b64)
                            fname_out = f"{task_id}_sol{si}_ex{ex_idx}_expected.png"
                            fpath_out = os.path.join(task_folder, fname_out)
                            with open(fpath_out, 'wb') as imgf:
                                imgf.write(img_bytes)
                            visual_files.append(os.path.join(task_id, fname_out) if task_folder != output_dir else fname_out)
                        except Exception:
                            pass
            if visual_files:
                solution['_visual_cue_files'] = visual_files
        except Exception:
            pass

    # Ensure the JSON is saved inside the per-task folder for easier inspection by the visualizer
    task_folder = os.path.join(output_dir, task_id)
    try:
        os.makedirs(task_folder, exist_ok=True)
        out_path = os.path.join(task_folder, f"{task_id}.json")
    except Exception:
        out_path = os.path.join(output_dir, f"{task_id}.json")

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)


def shorten_error(msg: Optional[str], max_len: int = 120) -> str:
    """Return a shortened, whitespace-normalized version of an error message.

    If `msg` is falsy returns empty string.
    """
    if not msg:
        return ""
    s = " ".join(str(msg).split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def calculate_average_score(result: Dict[str, Any]) -> None:
    """Calculate priority and average scores for training and testing results.

    Adds the following fields to `result` in-place:
      - `training_priority_score`: percentage of training examples that are fully correct (0-100)
      - `training_average_score`: average of `overlap_percentage` across training examples
        where `matching_size` is truthy (0-100). If no matching_size entries, value is 0.0.
      - `testing_priority_score` and `testing_average_score`: same as above for testing results.
    """
    def compute_scores(entries: Optional[List[Dict[str, Any]]]) -> Tuple[float, float]:
        """Compute Priority (percent fully correct) and Overlap (avg overlap for matching_size).

        Returns (priority_pct, overlap_avg)
        """
        if not entries:
            return 0.0, 0.0

        total = len(entries)
        # Priority: percentage fully correct
        num_correct = sum(1 for e in entries if bool(e.get('code_success', False)))
        priority = (num_correct / total) * 100.0 if total > 0 else 0.0

        # Overlap: average overlap_percentage for entries with matching_size truthy
        overlaps = [float(e.get('overlap_percentage', 0.0)) if e.get('matching_size') is True else 0.0 for e in entries]
        average = sum(overlaps) / len(overlaps)

        return priority, average

    training_entries = result.get('training_results') or []
    testing_entries = result.get('testing_results') or []

    # If the workflow produced multiple candidate solutions, compute per-solution scores.
    # Each solution is expected to be a dict with `training_results` and `testing_results`.
    solutions = result.get('solutions_list') or []
    if solutions:
        best_idx = None
        best_priority = -1.0
        best_overlap = -1.0
        best_testing_priority = -1.0
        best_testing_overlap = -1.0

        for idx, sol in enumerate(solutions):
            s_train = sol.get('training_results') or []
            s_test = sol.get('testing_results') or []

            s_tpriority, s_toverlap = compute_scores(s_train)
            s_tepriority, s_teoverlap = compute_scores(s_test)

            # Choose the metric for ranking solutions: prefer training priority/overlap,
            # fall back to testing priority/overlap if training has zero entries.
            rank_priority = s_tpriority
            rank_overlap = s_toverlap

            # Compare to current best: priority first, overlap as tiebreaker
            if (rank_priority > best_priority) or (
                rank_priority == best_priority and rank_overlap > best_overlap
            ):
                best_idx = idx
                best_priority = rank_priority
                best_overlap = rank_overlap
                best_testing_priority = s_tepriority
                best_testing_overlap = s_teoverlap

        # Store best solution info on the top-level result for easy printing.
        if best_idx is not None:
            result['highest_solution_index'] = best_idx
            result['highest_training_solution_priority_score'] = best_priority
            result['highest_training_solution_overlap_score'] = best_overlap
            result['highest_testing_solution_priority_score'] = best_testing_priority
            result['highest_testing_solution_overlap_score'] = best_testing_overlap


def summarize_and_print_result(result: Dict[str, Any], task_id: Optional[str] = None, progress: Optional[str] = None) -> None:
    """Compute scores (if not already) and print a concise summary for `result`.

    This consolidates the duplicated printing logic used in single, parallel and
    sequential batch flows. It mutates `result` by ensuring score fields are set
    (via `calculate_average_score`).
    """
    # Ensure computed fields are present
    calculate_average_score(result)

    tid = task_id or result.get('task_id', 'unknown')
    header = "LANGGRAPH MULTI-SOLUTION AGENT RESULTS"
    if progress:
        header = f"{header} {progress}"

    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  Task ID: {tid}")
    print(f"  Workflow Completed: {'✓' if result.get('workflow_completed') else '✗'}")

    # If solutions exist, print training examples for the selected highest solution
    solutions_list = result.get('solutions_list', [])
    hs_idx = result.get('highest_solution_index')
    if solutions_list and hs_idx is not None and 0 <= hs_idx < len(solutions_list):
        training_results = solutions_list[hs_idx].get('training_results', [])
        for i, train_example in enumerate(training_results):
            overlap = train_example.get('overlap_percentage')
            matching_size = train_example.get('matching_size')
            err = train_example.get('error_message')
            ov_str = f"{float(overlap):.1f}%" if overlap is not None else "N/A"
            ms_str = str(matching_size)
            err_str = shorten_error(err)
            line = f"    Train {i+1}: matching_size={ms_str} overlap={ov_str}"
            if err_str:
                line += f"  err='{err_str}'"
            print(line)

        testing_results = solutions_list[hs_idx].get('testing_results', [])
        for i, test_example in enumerate(testing_results):
            overlap = test_example.get('overlap_percentage')
            matching_size = test_example.get('matching_size')
            err = test_example.get('error_message')
            ov_str = f"{float(overlap):.1f}%" if overlap is not None else "N/A"
            ms_str = str(matching_size)
            err_str = shorten_error(err)
            line = f"    Test  {i+1}: matching_size={ms_str} overlap={ov_str}"
            if err_str:
                line += f"  err='{err_str}'"
            print(line)

    # Print computed priority & average scores if present
    highest_solution_index = result.get('highest_solution_index')
    highest_priority_score = result.get('highest_training_solution_priority_score')
    highest_overlap_score = result.get('highest_training_solution_overlap_score')
    highest_testing_priority_score = result.get('highest_testing_solution_priority_score')
    highest_testing_overlap_score = result.get('highest_testing_solution_overlap_score')
    print(f"  Attempts: {result.get('attempts', 0)}")
    print(f"  Execution Time: {result.get('execution_time', 0.0):.2f}s")

    # Print highest-scoring solution (Priority Score primary, Overlap Score tie-breaker)
    print(f"  Selected Solution (Highest training): ")
    print(f"    Index {highest_solution_index}")
    print(f"    Training Priority Score: {(highest_priority_score or 0.0):.1f}%  Overlap Score: {(highest_overlap_score or 0.0):.1f}%")
    print(f"    Testing Priority Score: {(highest_testing_priority_score or 0.0):.1f}%  Testing Overlap Score: {(highest_testing_overlap_score or 0.0):.1f}%")

def print_summary(agent: MultiSolutionARCLangGraphAgent, all_results: List[Dict[str, Any]], task_ids: List[str]) -> Dict[str, Any]:
    """Save summary of all task results to summary.json."""
    total_tasks = len(all_results)
    workflow_completions = sum(1 for r in all_results if r.get('workflow_completed'))
    num_tests_successful = sum(1 for r in all_results if r.get('highest_testing_solution_priority_score', 0) >= 1.0)

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks processed: {total_tasks}")
    print(f"Completed workflows: {workflow_completions}")
    print(f"Workflow completion rate: {workflow_completions / total_tasks * 100:.1f}%")
    print(f"Number of tests fully successful: {num_tests_successful}")
    print(f"Test success rate: {num_tests_successful / total_tasks * 100:.1f}%")
    print(f"{'='*80}\n")



def parse_arguments():
    """Parse command line arguments with defaults from ALL_CAPS variables."""
    parser = argparse.ArgumentParser(description="LangGraph multi-solution agent runner for ARC tasks")
    parser.add_argument("--model", type=str, default=MODEL,
                       help=f"Model to use (e.g., gpt-4o-mini, gemini-2.0-flash) (default: {MODEL})")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default=MODE,
                       help=f"Test mode: single task or batch (default: {MODE})")
    parser.add_argument("--task-id", type=str, default=TASK_ID,
                       help=f"Specific task ID to test (for single mode) (default: {TASK_ID})")
    parser.add_argument("--task-index", type=int, default=TASK_INDEX,
                       help=f"Task index to test (for single mode) (default: {TASK_INDEX})")
    parser.add_argument("--num-tasks", type=int, default=NUM_TASKS,
                       help=f"Number of tasks for batch mode (default: {NUM_TASKS})")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                       help=f"Maximum attempts per task (default: {MAX_ATTEMPTS})")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                       help=f"Random seed for reproducibility (default: {RANDOM_SEED})")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                       help="Number of parallel workers for batch mode (default: 1 = sequential)")
    # Year and file path overrides for ARC data
    parser.add_argument("--year", type=int, default=YEAR, dest="YEAR",
                       help=f"Year for ARC dataset directory (default: {YEAR})")
    parser.add_argument("--training-tasks-json", type=str, default=TRAINING_TASKS_JSON, dest="TRAINING_TASKS_JSON",
                       help=f"Path to training tasks JSON (default: {TRAINING_TASKS_JSON})")
    parser.add_argument("--training-solutions-json", type=str, default=TRAINING_SOLUTIONS_JSON, dest="TRAINING_SOLUTIONS_JSON",
                       help=f"Path to training solutions JSON (default: {TRAINING_SOLUTIONS_JSON})")
    parser.add_argument("--evaluation-tasks-json", type=str, default=EVALUATION_TASKS_JSON, dest="EVALUATION_TASKS_JSON",
                       help=f"Path to evaluation tasks JSON (default: {EVALUATION_TASKS_JSON})")
    parser.add_argument("--evaluation-solutions-json", type=str, default=EVALUATION_SOLUTIONS_JSON, dest="EVALUATION_SOLUTIONS_JSON",
                       help=f"Path to evaluation solutions JSON (default: {EVALUATION_SOLUTIONS_JSON})")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Initialize LLM
    llm = initialize_llm_from_config(args.model)
    if llm is None:
        return 1
    
    # Create output directory first
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Load tasks and solutions
    try:
        training_tasks = load_arc_tasks(args.TRAINING_TASKS_JSON)
        training_solutions = load_arc_solutions(args.TRAINING_SOLUTIONS_JSON)
        evaluation_tasks = load_arc_tasks(args.EVALUATION_TASKS_JSON)
        evaluation_solutions = load_arc_solutions(args.EVALUATION_SOLUTIONS_JSON)
    except FileNotFoundError as e:
        print(f"Error: Could not load ARC data: {e}")
        return 1
    
    # Save parameters
    save_params(output_dir, args, args.model)
    
    # Save task IDs
    save_task_ids(output_dir, training_tasks, evaluation_tasks)

    # Create a MultiSolution agent for this run
    print(f"Initialize the multi-solution LangGraph agent...")
    agent = MultiSolutionARCLangGraphAgent(
        llm=llm,
        code_llm=llm,
        num_initial_solutions=NUM_INITIAL_SOLUTIONS,
        num_loops=NUM_LOOPS,
        num_seed_solutions=NUM_SEED_SOLUTIONS,
        num_refinements=NUM_REFINEMENTS,
        num_solutions_per_refinement=NUM_SOLUTIONS_PER_REFINEMENT,
        num_fusions=NUM_FUSIONS,
        num_solutions_per_fusion=NUM_SOLUTIONS_PER_FUSION,
        enable_visual_cue=ENABLE_VISUAL_CUE,
        enable_code_predict=ENABLE_CODE_PREDICT,
        enable_llm_predict=ENABLE_LLM_PREDICT)

    # Initialize a thread-safe shared lock for consecutive printing
    shared_lock = threading.Lock()  
    all_results = []
    task_ids_processed = []
    
    if args.mode == "single":
        # Single task test
        if args.task_id:
            if args.task_id in training_tasks:
                task_id = args.task_id
                task_data = training_tasks[task_id]
            elif args.task_id in evaluation_tasks:
                task_id = args.task_id
                task_data = evaluation_tasks[task_id]
            else:
                print(f"Error: Task ID '{args.task_id}' not found")
                return 1
        elif args.task_index is not None:
            task_id, task_data, task_solution = get_task_by_index(training_tasks, training_solutions, args.task_index)
        else:
            # Random task
            task_id, task_data, task_solution = get_task_by_index(training_tasks, training_solutions, None)
        
        print(f"Testing single task: {task_id}")
        
        # Run LangGraph agent with max attempts (reuse shared agent)
        langgraph_result = agent.solve_task(task_id, task_data, task_solution, max_attempts=args.max_attempts)

        # Compute training/testing priority & average scores
        try:
            calculate_average_score(langgraph_result)
        except Exception as e:
            print(f"Warning: could not calculate scores for task {task_id}: {e}")
        
        all_results.append(langgraph_result)
        task_ids_processed.append(task_id)

        # Save individual task result
        save_task_result(output_dir, task_id, langgraph_result)

        # Print summary using helper
        summarize_and_print_result(langgraph_result, task_id=task_id)
        
    elif args.mode == "batch":
        # Batch test (supports parallel execution when --workers > 1)
        print(f"Running batch test with {args.num_tasks} tasks (workers={args.workers})")

        random.seed(args.random_seed)
        selected_task_ids = random.sample(list(training_tasks.keys()),
                                        min(args.num_tasks, len(training_tasks)))

        # Helper to run a single task (creates a fresh agent to avoid shared-state issues)
        print_lock = threading.Lock()

        def run_single_task(task_id: str):
            task_data = training_tasks[task_id]
            task_solution = training_solutions.get(task_id)

            # Create a per-task multi-solution agent and attach helpers snapshot
            local_agent = MultiSolutionARCLangGraphAgent(
                llm=llm,
                code_llm=llm,
                enable_visual_cue=ENABLE_VISUAL_CUE,
                enable_code_predict=ENABLE_CODE_PREDICT,
                enable_llm_predict=ENABLE_LLM_PREDICT
            )

            start_time = time.time()
            try:
                result = local_agent.solve_task(task_id, task_data, task_solution, max_attempts=args.max_attempts)
            except Exception as e:
                print(f"Error processing task {task_id}: {e}")
                result = {
                    "task_id": task_id,
                    "error": str(e),
                    "workflow_completed": False,
                    "attempts": 0,
                    "testing_success_rate": 0.0,
                }

            result.setdefault("execution_time", time.time() - start_time)

            # Compute training/testing priority & average scores for this result
            try:
                calculate_average_score(result)
            except Exception as e:
                print(f"Warning: could not calculate scores for task {task_id}: {e}")

            return task_id, result

        if args.workers and args.workers > 1:
            # Parallel execution using threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {executor.submit(run_single_task, tid): tid for tid in selected_task_ids}
                completed = 0
                for future in concurrent.futures.as_completed(future_to_task):
                    tid, langgraph_result = future.result()
                    completed += 1

                    all_results.append(langgraph_result)
                    task_ids_processed.append(tid)

                    # Save result
                    save_task_result(output_dir, tid, langgraph_result)

                    # Print concise result summary (thread-safe)
                    with print_lock:
                        # Use helper to print full concise summary (with progress)
                        summarize_and_print_result(langgraph_result, task_id=tid, progress=f"({completed}/{len(selected_task_ids)})")

        else:
            # Sequential execution (workers==1) — reuse the shared agent for efficiency
            for i, task_id in enumerate(selected_task_ids, 1):
                print(f"\n{'='*80}")
                print(f"PROCESSING TASK {i}/{args.num_tasks}: {task_id}")
                print(f"{'='*80}")

                task_data = training_tasks[task_id]
                task_solution = training_solutions.get(task_id)

                # Run LangGraph agent with max attempts (reuse shared agent)
                langgraph_result = agent.solve_task(task_id, task_data, task_solution, max_attempts=args.max_attempts)

                # Compute training/testing priority & average scores
                try:
                    calculate_average_score(langgraph_result)
                except Exception as e:
                    print(f"Warning: could not calculate scores for task {task_id}: {e}")

                all_results.append(langgraph_result)
                task_ids_processed.append(task_id)

                # Save individual task result
                save_task_result(output_dir, task_id, langgraph_result)

                # Display concise summary via helper
                summarize_and_print_result(langgraph_result, task_id=task_id)
    
    # Save summary
    print_summary(agent, all_results, task_ids_processed)
    return 0


def initialize_llm_from_config(model_name: str):
    """Initialize LLM based on model_configs.py configuration."""
    import os
    
    # Find the model key
    model_key = find_model_key(model_name)
    if not model_key:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return None
    
    # Get model config
    config = MODEL_CONFIGS[model_key]
    
    # Handle aliases
    if "alias_of" in config:
        config = MODEL_CONFIGS[config["alias_of"]]
    
    provider = config.get("provider")
    
    try:
        if provider == "google" or provider == "learnlm":
            # Google Gemini models - suppress ALTS warnings during import
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_key,
                temperature=0.6,
                max_output_tokens=50000
            )
        
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=model_key,
                temperature=0.6,
                max_tokens=2000
            )
        
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=model_key,
                temperature=0.6,
                max_tokens=2000
            )
        
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_key,
                temperature=0.6
            )
        
        else:
            print(f"Unsupported provider: {provider}")
            return None
            
    except ImportError as e:
        print(f"Error importing model provider: {e}")
        print(f"Please install the required package for {provider}")
        return None
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None


if __name__ == "__main__":
    _sys.exit(main())
