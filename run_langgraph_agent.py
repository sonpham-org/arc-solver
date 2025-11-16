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

# Task selection for single mode
TASK_ID = None  # Specific task ID to test (for single mode)
TASK_INDEX = None  # Task index to test (for single mode)

# Batch mode configuration
NUM_TASKS = 5  # Number of tasks for batch mode

# Processing configuration
MAX_ATTEMPTS = 1  # Maximum attempts per task
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


def create_task_result_format(
    task_id: str, 
    task_data: Dict[str, Any], 
    langgraph_result: Dict[str, Any],
    training_solutions: Dict[str, List[List[List[int]]]],
    evaluation_solutions: Dict[str, List[List[List[int]]]]
) -> Dict[str, Any]:
    """Create result in run_arc_prompt.py format."""
    
    trains = []
    tests = []
    
    # Get final solution for prediction generation
    final_solution = langgraph_result.get('final_solution')
    
    # Process training examples
    for i, example in enumerate(task_data['train']):
        # Initialize metrics
        success = False
        overlap = 0.0
        iou = 0.0
        predicted_output = None
        
        if final_solution:
            try:
                predicted_output = get_prediction_from_solution(final_solution, {"test": [{"input": example["input"]}]})
                if predicted_output and example["output"]:
                    # Calculate overlap and IOU
                    pred_h, pred_w = len(predicted_output), len(predicted_output[0]) if predicted_output else 0
                    exp_h, exp_w = len(example["output"]), len(example["output"][0]) if example["output"] else 0
                    
                    # Calculate intersection (overlapping area)
                    intersection_h = min(pred_h, exp_h)
                    intersection_w = min(pred_w, exp_w)
                    intersection_area = intersection_h * intersection_w
                    
                    # Calculate matching cells in intersection area
                    matching_cells = 0
                    for i in range(intersection_h):
                        for j in range(intersection_w):
                            if predicted_output[i][j] == example["output"][i][j]:
                                matching_cells += 1
                    
                    # Calculate union (total area covered by both grids)
                    union_area = pred_h * pred_w + exp_h * exp_w - intersection_area
                    
                    # Calculate metrics
                    overlap = (matching_cells / intersection_area * 100.0) if intersection_area > 0 else 0.0
                    iou = (matching_cells / union_area * 100.0) if union_area > 0 else 0.0
                    success = (overlap == 100.0 and pred_h == exp_h and pred_w == exp_w)
            except Exception:
                pass
        
        # Convert grids to string format
        input_lines = [''.join(map(str, row)) for row in example["input"]]
        output_lines = [''.join(map(str, row)) for row in example["output"]]
        predict_lines = [''.join(map(str, row)) for row in predicted_output] if predicted_output else []
        
        # Calculate sizes
        input_size = f"{len(example['input'])}x{len(example['input'][0]) if example['input'] else 0}"
        output_size = f"{len(example['output'])}x{len(example['output'][0]) if example['output'] else 0}"
        predict_size = f"{len(predicted_output)}x{len(predicted_output[0]) if predicted_output else 0}" if predicted_output else "0x0"
        
        train_entry = {
            'input': input_lines,
            'output': output_lines,
            'valid_predict': predicted_output is not None,
            'predict': predict_lines,
            'iou': iou,
            'overlap': overlap,
            'correct': success,
            'transformation_succeed': [predicted_output is not None],
            'input_size': input_size,
            'output_size': output_size,
            'predict_size': predict_size
        }
        trains.append(train_entry)
    
    # Process test examples
    for i, example in enumerate(task_data['test']):
        # Get expected output from solutions
        expected_output = None
        if task_id in training_solutions:
            expected_output = training_solutions[task_id][i] if i < len(training_solutions[task_id]) else None
        elif task_id in evaluation_solutions:
            expected_output = evaluation_solutions[task_id][i] if i < len(evaluation_solutions[task_id]) else None
        
        # Generate prediction
        predicted_output = None
        if final_solution:
            try:
                predicted_output = get_prediction_from_solution(final_solution, {"test": [{"input": example["input"]}]})
            except Exception:
                pass
        
        # Calculate metrics
        success = False
        overlap = 0.0
        iou = 0.0
        if predicted_output and expected_output:
            try:
                # Calculate sizes
                pred_h, pred_w = len(predicted_output), len(predicted_output[0]) if predicted_output else 0
                exp_h, exp_w = len(expected_output), len(expected_output[0]) if expected_output else 0
                
                # Calculate intersection (overlapping area)
                intersection_h = min(pred_h, exp_h)
                intersection_w = min(pred_w, exp_w)
                intersection_area = intersection_h * intersection_w
                
                # Calculate matching cells in intersection area
                matching_cells = 0
                for row_i in range(intersection_h):
                    for col_j in range(intersection_w):
                        if predicted_output[row_i][col_j] == expected_output[row_i][col_j]:
                            matching_cells += 1
                
                # Calculate union (total area covered by both grids)
                union_area = pred_h * pred_w + exp_h * exp_w - intersection_area
                
                # Calculate metrics
                overlap = (matching_cells / intersection_area * 100.0) if intersection_area > 0 else 0.0
                iou = (matching_cells / union_area * 100.0) if union_area > 0 else 0.0
                success = (overlap == 100.0 and pred_h == exp_h and pred_w == exp_w)
            except Exception:
                pass
        
        # Convert to string format
        input_lines = [''.join(map(str, row)) for row in example["input"]]
        output_lines = [''.join(map(str, row)) for row in expected_output] if expected_output else []
        predict_lines = [''.join(map(str, row)) for row in predicted_output] if predicted_output else []
        
        # Calculate sizes
        input_size = f"{len(example['input'])}x{len(example['input'][0]) if example['input'] else 0}"
        output_size = f"{len(expected_output)}x{len(expected_output[0]) if expected_output else 0}" if expected_output else "0x0"
        predict_size = f"{len(predicted_output)}x{len(predicted_output[0]) if predicted_output else 0}" if predicted_output else "0x0"
        
        test_entry = {
            'input': input_lines,
            'output': output_lines,
            'valid_predict': predicted_output is not None,
            'predict': predict_lines,
            'iou': iou,
            'overlap': overlap,
            'correct': success,
            'transformation_succeed': [predicted_output is not None],
            'input_size': input_size,
            'output_size': output_size,
            'predict_size': predict_size
        }
        tests.append(test_entry)
    
    # Calculate summary statistics
    train_correct = sum(1 for t in trains if t['correct'])
    test_correct = sum(1 for t in tests if t['correct'])
    
    result = {
        'task_id': task_id,
        'trains': trains,
        'tests': tests,
        'success': langgraph_result['success'],
        'attempts': langgraph_result['attempts'],
        'execution_time': langgraph_result['execution_time'],
        'train_accuracy': (train_correct / len(trains)) * 100 if trains else 0,
        'test_accuracy': (test_correct / len(tests)) * 100 if tests else 0,
        'json_generations_success_rate': langgraph_result['json_generations_success_rate']
    }
    
    # Add generated code if available
    if langgraph_result.get('generated_code'):
        # Convert string to list of lines for easier review
        result['generated_code'] = langgraph_result['generated_code'].split('\n')
    
    return result


def save_task_result(output_dir: str, task_id: str, result: Dict[str, Any]) -> None:
    """Save individual task result to JSON file."""
    with open(os.path.join(output_dir, f"{task_id}.json"), 'w') as f:
        json.dump(result, f, indent=2)


def save_summary(output_dir: str, all_results: List[Dict[str, Any]], task_ids: List[str]) -> Dict[str, Any]:
    """Save summary statistics."""
    total_tasks = len(all_results)
    successful_json_generations = sum(1 for r in all_results if r.get('success', False))
    
    # Calculate train/test accuracy
    train_correct = sum(sum(1 for t in r['trains'] if t['correct']) for r in all_results)
    train_total = sum(len(r['trains']) for r in all_results)
    test_correct = sum(sum(1 for t in r['tests'] if t['correct']) for r in all_results)
    test_total = sum(len(r['tests']) for r in all_results)
    
    # Calculate average overlap percentages
    train_overlaps = []
    test_overlaps = []
    
    for r in all_results:
        for train_example in r['trains']:
            if train_example.get('overlap') is not None:
                train_overlaps.append(train_example['overlap'])
        for test_example in r['tests']:
            if test_example.get('overlap') is not None:
                test_overlaps.append(test_example['overlap'])
    
    avg_train_overlap = sum(train_overlaps) / len(train_overlaps) if train_overlaps else 0.0
    avg_test_overlap = sum(test_overlaps) / len(test_overlaps) if test_overlaps else 0.0
    
    summary = {
        "total_tasks": total_tasks,
        "successful_json_generations": successful_json_generations,
        "json_generations_success_rate": (successful_json_generations / total_tasks) * 100 if total_tasks > 0 else 0,
        "train_accuracy": (train_correct / train_total) * 100 if train_total > 0 else 0,
        "test_accuracy": (test_correct / test_total) * 100 if test_total > 0 else 0,
        "avg_train_overlap": avg_train_overlap,
        "avg_test_overlap": avg_test_overlap,
        "task_ids": task_ids,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def run_langgraph_agent_on_task(
    agent: ARCLangGraphAgent,
    task_id: str,
    task_data: Dict[str, Any],
    task_solution: Optional[List[List[List[int]]]],
    max_attempts: int = 5
) -> Dict[str, Any]:
    """Test LangGraph agent on a single task with max attempts enforcement.
    """

    print(f"Processing task {task_id} with max {max_attempts} attempts...")
    start_time = time.time()
    
    # Solve the task with max attempts override
    result = agent.solve_task(task_id, task_data, max_attempts=max_attempts)    
    result["execution_time"] = time.time() - start_time
    
    # Add new helper functions if needed, but not here bro.
    return result


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

    agent = ARCLangGraphAgent(llm=llm, max_attempts=args.max_attempts)
    
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
        langgraph_result = run_langgraph_agent_on_task(agent, task_id, task_data, task_solution, args.max_attempts)
        
        # Create result in run_arc_prompt.py format
        result = create_task_result_format(
            task_id, task_data, langgraph_result, 
            training_solutions, evaluation_solutions
        )
        
        all_results.append(result)
        task_ids_processed.append(task_id)
        
        # Save individual task result
        save_task_result(output_dir, task_id, result)
        
        # Display results
        print(f"\n{'='*60}")  
        print(f"LANGGRAPH AGENT RESULTS")
        print(f"{'='*60}")
        print(f"  Task ID: {task_id}")
        print(f"  Success: {langgraph_result['success']}")
        print(f"  JSON Generation Success Rate: {langgraph_result['json_generations_success_rate']:.2%}")
        print(f"  Training Success Rate: {result['train_accuracy']:.1f}%")
        
        # Display training example details
        if result.get('trains'):
            for i, train_example in enumerate(result['trains']):
                overlap = train_example.get('overlap', 0)
                iou = train_example.get('iou', 0)
                correct = "✓" if train_example.get('correct', False) else "✗"
                print(f"    Train {i+1}: {correct} Overlap: {overlap:.1f}% IoU: {iou:.1f}%")
        
        print(f"  Testing Success Rate: {result['test_accuracy']:.1f}%")
        
        # Display testing example details  
        if result.get('tests'):
            for i, test_example in enumerate(result['tests']):
                overlap = test_example.get('overlap', 0)
                iou = test_example.get('iou', 0) 
                correct = "✓" if test_example.get('correct', False) else "✗"
                print(f"    Test {i+1}: {correct} Overlap: {overlap:.1f}% IoU: {iou:.1f}%")
        
        print(f"  Attempts: {langgraph_result['attempts']}")
        print(f"  Execution Time: {langgraph_result['execution_time']:.2f}s")
        
        if langgraph_result.get('generated_code'):
            print(f"\n  Generated Code:")
            if isinstance(langgraph_result['generated_code'], str):
                code_lines = langgraph_result['generated_code'].split('\n')
            else:
                code_lines = langgraph_result['generated_code']
            for line in code_lines:
                print(f"    {line}")
        
    elif args.mode == "batch":
        # Batch test
        print(f"Running batch test with {args.num_tasks} tasks")
        
        random.seed(args.random_seed)
        selected_task_ids = random.sample(list(training_tasks.keys()), 
                                        min(args.num_tasks, len(training_tasks)))
        
        for i, task_id in enumerate(selected_task_ids, 1):
            print(f"\n{'='*80}")  
            print(f"PROCESSING TASK {i}/{args.num_tasks}: {task_id}")
            print(f"{'='*80}")
            
            task_data = training_tasks[task_id]
            task_solution = training_solutions.get(task_id)

            # Run LangGraph agent with max attempts (reuse shared agent)
            langgraph_result = run_langgraph_agent_on_task(agent, task_id, task_data, task_solution, args.max_attempts)
            
            # Create result in run_arc_prompt.py format
            result = create_task_result_format(
                task_id, task_data, langgraph_result, 
                training_solutions, evaluation_solutions
            )
            
            all_results.append(result)
            task_ids_processed.append(task_id)
            
            # Save individual task result
            save_task_result(output_dir, task_id, result)
            
            # Display brief results
            print(f"  Success: {'✓' if langgraph_result['success'] else '✗'}")
            print(f"  JSON Generation Success Rate: {langgraph_result['json_generations_success_rate']:.2%}")
            print(f"  Attempts: {langgraph_result['attempts']}")
            print(f"  Execution Time: {langgraph_result['execution_time']:.2f}s")
    
    # Save summary
    summary = save_summary(output_dir, all_results, task_ids_processed)
    
    # Display final summary
    print(f"\n{'='*80}")  
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks processed: {summary['total_tasks']}")
    print(f"Successful generations: {summary['successful_json_generations']}")
    print(f"JSON generation success rate: {summary['json_generations_success_rate']:.1f}%")
    print(f"Training accuracy: {summary['train_accuracy']:.1f}%")
    print(f"Test accuracy: {summary['test_accuracy']:.1f}%")
    print(f"Average training overlap: {summary['avg_train_overlap']:.1f}%")
    print(f"Average test overlap: {summary['avg_test_overlap']:.1f}%")
    print(f"")
    print(f"Note: 'Accuracy' requires perfect size + content match. 'Overlap' measures content similarity in intersection area.")
    
    # Display updated toolbox statistics
    final_stats = toolbox.get_statistics()
    print(f"\n📚 Final Run Toolbox Status:")
    print(f"  Total functions: {final_stats['total_functions']}")
    print(f"  Total usage records: {final_stats['total_usage_records']}")
    print(f"  Categories: {list(final_stats['categories'].keys())}")
    
    # Export the run toolbox to the output directory
    toolbox_export_path = toolbox.export_toolbox()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Run toolbox database: {toolbox.storage_path}")
    print(f"Run toolbox export: {toolbox_export_path}")
    
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