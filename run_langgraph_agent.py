#!/usr/bin/env python3
"""
LangGraph agent runner for ARC tasks with comprehensive output structure.
Compatible with run_arc_prompt.py format.
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
import sys
import time
import concurrent.futures
import threading
from typing import Dict, List, Any, Tuple, Optional

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import LangGraph agent
from arc_langgraph_agent import ARCLangGraphAgent

# Import model configurations and utilities
from model_configs import MODEL_CONFIGS, find_model_key


# ============================================================================
# CONFIGURATION VARIABLES - Modify these as needed
# ============================================================================

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


def get_prediction_from_solution(solution: Dict[str, Any], task_data: Dict[str, Any]) -> Optional[List[List[int]]]:
    """Extract prediction from solution for a test case."""
    # Handle different solution formats
    # If solution is a WorkflowOutput, extract the code
    if 'code' in solution:
        code = solution['code']
    elif 'generated_code' in solution:
        code = solution['generated_code']
    else:
        return None
    
    if not code:
        return None
    
    try:
        # Import helper functions
        from arc_langgraph_agent.tools import FUNCTION_MAP
        
        # Create execution environment with test input
        exec_globals = {}
        
        # Add built-in helper functions to execution environment
        for name, func in FUNCTION_MAP.items():
            exec_globals[name] = func
        
        # If code is a list of lines, join them
        if isinstance(code, list):
            code = '\n'.join(code)
            
        exec(code, exec_globals)
        
        # Look for transform function
        if 'transform' in exec_globals:
            transform_func = exec_globals['transform']
            test_input = task_data['test'][0]['input']
            result = transform_func(test_input)
            
            # Ensure result is a valid grid format
            if isinstance(result, list) and all(isinstance(row, list) for row in result):
                return result
    except Exception as e:
        print(f"Warning: Could not execute solution code: {e}")
    
    return None


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
    for key in ("training_results", "testing_results"):
        entries = result.get(key)
        if not entries:
            continue
        for entry in entries:
            # Set list-of-strings fields (override or add)
            entry["expected_output"] = grid_to_string_lines(entry["expected_output"])
            entry["predicted_output"] = grid_to_string_lines(entry["predicted_output"])
            entry["input"] = grid_to_string_lines(entry["input"])

    with open(os.path.join(output_dir, f"{task_id}.json"), 'w') as f:
        json.dump(result, f, indent=2)


def print_summary(agent: ARCLangGraphAgent, all_results: List[Dict[str, Any]], task_ids: List[str]) -> Dict[str, Any]:
    """Save summary of all task results to summary.json."""
    total_tasks = len(all_results)
    workflow_successes = sum(1 for r in all_results if r.get('workflow_completed'))
    num_tests_successful = sum(1 for r in all_results if r.get('testing_success_rate', 0) >= 1.0)
    num_helpers_created = sum(len(r.get('new_helpers', {})) for r in all_results)
    num_unique_helpers_created = len(set(
        helper_name 
        for r in all_results 
        for helper_name in r.get('new_helpers', {}).keys()
    ))

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks processed: {total_tasks}")
    print(f"Successful workflows: {workflow_successes}")
    print(f"Workflow success rate: {workflow_successes / total_tasks * 100:.1f}%")
    print(f"Number of tests fully successful: {num_tests_successful}")
    print(f"Test success rate: {num_tests_successful / total_tasks * 100:.1f}%")
    print(f"Total new helpers created: {num_helpers_created}")
    print(f"Unique new helpers created: {num_unique_helpers_created}")
    print(f"{'='*80}\n")



def parse_arguments():
    """Parse command line arguments with defaults from ALL_CAPS variables."""
    parser = argparse.ArgumentParser(description="LangGraph agent runner for ARC tasks")
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
    
    print("Initializing run-specific toolbox with default helper functions...")
    # Load tasks and solutions
    try:
        training_tasks = load_arc_tasks("data/arc-2024/arc-agi_training_challenges.json")
        training_solutions = load_arc_solutions("data/arc-2024/arc-agi_training_solutions.json")
        evaluation_tasks = load_arc_tasks("data/arc-2024/arc-agi_evaluation_challenges.json")
        evaluation_solutions = load_arc_solutions("data/arc-2024/arc-agi_evaluation_solutions.json")
    except FileNotFoundError as e:
        print(f"Error: Could not load ARC data: {e}")
        return 1
    
    # Save parameters
    save_params(output_dir, args, args.model)
    
    # Save task IDs
    save_task_ids(output_dir, training_tasks, evaluation_tasks)

    # Build a shared agent for this run using suggested helpers from a representative task


    print(f"Initialize the list of helpers for the shared agent...")

    agent = ARCLangGraphAgent(llm=llm, code_llm=llm, max_attempts=args.max_attempts)
    # Initialize a thread-safe shared helpers store from the agent's defaults.
    # This will be copied (snapshot) by each worker when it starts and
    # updated by workers when they finish.
    shared_helpers = dict(agent.available_helpers)
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
        
        all_results.append(langgraph_result)
        task_ids_processed.append(task_id)
        
        # Save individual task result
        save_task_result(output_dir, task_id, langgraph_result)
        
        # Display results
        print(f"\n{'='*60}")  
        print(f"LANGGRAPH AGENT RESULTS")
        print(f"{'='*60}")
        print(f"  Task ID: {task_id}")
        print(f"  Workflow Completed: {'✓' if langgraph_result['workflow_completed'] else '✗'}")
        
        # Display training example details
        if langgraph_result.get('training_results'):
            for i, train_example in enumerate(langgraph_result['training_results']):
                overlap_percentage = train_example.get('overlap_percentage', 0)
                matching_size = train_example.get('matching_size', False)
                success = "✓" if overlap_percentage >= 100 and matching_size else "✗"
                print(f"    Train {i+1}: {success} Overlap: {overlap_percentage:.1f}% Matching Size: {matching_size}")

        # Display new helpers (name and description)
        if langgraph_result.get('new_helpers'):
            print(f"  New Helpers:")
            for helper_name, helper_info in langgraph_result['new_helpers'].items():
                print(f"    {helper_name}: {helper_info.get('description', '')}")
        
        print(f"  Testing Success Rate: {langgraph_result['testing_success_rate']:.1f}%")
        
        # Display testing example details  
        if langgraph_result.get('testing_results'):
            for i, test_example in enumerate(langgraph_result['testing_results']):
                overlap_percentage = test_example.get('overlap_percentage', 0)
                matching_size = test_example.get('matching_size', False)
                success = "✓" if overlap_percentage >= 100 and matching_size else "✗"
                print(f"    Test {i+1}: {success} Overlap: {overlap_percentage:.1f}% Matching Size: {matching_size}")

        print(f"  Attempts: {langgraph_result['attempts']}")
        print(f"  Execution Time: {langgraph_result['execution_time']:.2f}s")
        
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

            # Snapshot the shared helpers under lock so each worker starts with
            # a consistent copy of the available helper toolbox.
            with shared_lock:
                helpers_snapshot = dict(shared_helpers)

            # Create a per-task agent initialized with the snapshot of helpers
            # to avoid cross-task helper/state interference.
            local_agent = ARCLangGraphAgent(
                llm=llm,
                code_llm=llm,
                max_attempts=args.max_attempts,
                available_helpers=helpers_snapshot
            )

            start_time = time.time()
            try:
                result = local_agent.solve_task(task_id, task_data, task_solution, max_attempts=args.max_attempts)
            except Exception as e:
                result = {
                    "task_id": task_id,
                    "error": str(e),
                    "workflow_completed": False,
                    "attempts": 0,
                    "testing_success_rate": 0.0,
                }

            result.setdefault("execution_time", time.time() - start_time)

            # Merge any new helpers produced by this run into the shared helpers dict.
            new_helpers = result.get("new_helpers") or {}
            if new_helpers:
                with shared_lock:
                    # Simple merge: newer helpers override older ones by key.
                    for hname, hdef in new_helpers.items():
                        shared_helpers[hname] = hdef

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
                        print(f"\n{'='*60}")
                        print(f"LANGGRAPH AGENT RESULTS ({completed}/{len(selected_task_ids)})")
                        print(f"{'='*60}")
                        print(f"  Task ID: {tid}")
                        print(f"  Workflow Completed: {'✓' if langgraph_result.get('workflow_completed') else '✗'}")
                        print(f"  Testing Success Rate: {langgraph_result.get('testing_success_rate', 0.0):.1f}%")
                        print(f"  Attempts: {langgraph_result.get('attempts', 'N/A')}")
                        print(f"  Execution Time: {langgraph_result.get('execution_time', 0.0):.2f}s")

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
                agent.update_available_helpers(langgraph_result.get('new_helpers', {}))
                all_results.append(langgraph_result)
                task_ids_processed.append(task_id)

                # Save individual task result
                save_task_result(output_dir, task_id, langgraph_result)

                # Display results (same formatting as before)
                print(f"\n{'='*60}")
                print(f"LANGGRAPH AGENT RESULTS")
                print(f"{'='*60}")
                print(f"  Task ID: {task_id}")
                print(f"  Workflow Completed: {'✓' if langgraph_result['workflow_completed'] else '✗'}")

                if langgraph_result.get('training_results'):
                    for i, train_example in enumerate(langgraph_result['training_results']):
                        overlap_percentage = train_example.get('overlap_percentage', 0)
                        matching_size = train_example.get('matching_size', False)
                        correct = "✓" if train_example.get('correct', False) else "✗"
                        print(f"    Train {i+1}: {correct} Overlap: {overlap_percentage:.1f}% Matching Size: {matching_size}")

                if langgraph_result.get('new_helpers'):
                    print(f"  New Helpers:")
                    for helper_name, helper_info in langgraph_result['new_helpers'].items():
                        print(f"    {helper_name}: {helper_info.get('description', '')}")

                print(f"  Testing Success Rate: {langgraph_result['testing_success_rate'] * 100:.1f}%")

                if langgraph_result.get('testing_results'):
                    for i, test_example in enumerate(langgraph_result['testing_results']):
                        overlap_percentage = test_example.get('overlap_percentage', 0)
                        matching_size = test_example.get('matching_size', False)
                        correct = "✓" if test_example.get('correct', False) else "✗"
                        print(f"    Test {i+1}: {correct} Overlap: {overlap_percentage:.1f}% Matching Size: {matching_size}")

                print(f"  Attempts: {langgraph_result['attempts']}")
                print(f"  Execution Time: {langgraph_result['execution_time']:.2f}s")
    
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
    sys.exit(main())