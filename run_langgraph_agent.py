#!/usr/bin/env python3
"""
LangGraph multi-solution agent runner for ARC tasks.
This script mirrors `run_langgraph_agent.py` but uses the
`ARCLangGraphAgent` to collect multiple solutions per task.
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
import signal
import atexit
import logging
from typing import Dict, List, Any, Tuple, Optional

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the MultiSolution LangGraph agent
from agent.agent import ARCLangGraphAgent

# Import model configurations and utilities
from model_configs import MODEL_CONFIGS, find_model_key, get_pricing_for, estimate_cost

# Optional Qdrant imports - keep at module import time so missing packages
# are detected early. If unavailable, `QDRANT_AVAILABLE` will be False.
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    VectorParams = None
    Distance = None


# ==========================================================================
# CONFIGURATION VARIABLES - Modify these as needed
# ==========================================================================

# Model selection: Choose from available models in model_configs.py
# For fast local debugging prefer an Ollama-hosted local model (free/local).
# Reasoning model is used for reasoning & reflection
# Coding model is used for code generation & execution
REASONING_MODEL = "gemini-2.5-flash-lite"  # e.g., "gpt-4o-mini", "gemini-2.0-flash", "llama3.1", "qwen2.5:32b"
TRANSFORMATION_STEPS_MODEL = "gemini-2.5-flash-lite"  # e.g., "gpt-4o-mini", "gemini-2.0-flash", "llama3.1", "qwen2.5:32b"
CODING_MODEL = "gemini-2.5-flash-lite"  # e.g., "gpt-4o-mini", "gemini-2.0-flash", "llama3.1", "qwen2.5:32b"
USE_VLLM = False

# Test mode configuration
MODE = "batch"  # "single" or "batch"
NUM_WORKERS = 8  # Number of parallel workers for batch mode

# Task selection for single mode
TASK_ID = None  # Specific task ID to test (for single mode)
TASK_INDEX = None  # Task index to test (for single mode)

# Batch mode configuration
NUM_TASKS = 100  # Number of tasks for batch mode
EVALUATE_ONLY = True

# Processing configuration
MAX_ATTEMPTS = 10  # Maximum attempts per task
RANDOM_SEED = 42  # Random seed for reproducibility
DEBUG = False  # Enable debug logging (prompts, responses, code generation)

ENABLE_PARALLEL_EVAL = False  # Whether to enable parallel evaluation of examples
ENABLE_CODE_PREDICT = True  # Whether to enable code-predicted outputs during testing
ENABLE_LLM_PREDICT = False # Whether to enable LLM-predicted outputs during testing
ENABLE_VISUAL_CUE = False  # When True, generate and pass input/output images to the LLM
ENABLE_RAG_HINT = True  # When True, enable retrieval-augmented generation hints from past reasoning traces

NUM_INITIAL_SOLUTIONS = 10
NUM_LOOPS = 5
NUM_SEED_SOLUTIONS = 10
NUM_REFINEMENTS = 2
NUM_SOLUTIONS_PER_REFINEMENT = 5
NUM_FUSIONS = 2
NUM_SOLUTIONS_PER_FUSION = 5
RECURSION_LIMIT = 50  # LangGraph recursion limit (default is 25, increase for long workflows)

# Year selection for ARC dataset directory (change to 2025 if using 2025 data)
YEAR = 2024
 
# Optional resume run identifier used when starting the script. If set to
# 'latest' the runner will resume the most recent folder under
# `output/output_agent`. If set to a specific folder name, the runner will
# attempt to resume that folder. When `None`, a new timestamped output
# folder is created for the run.
RESUME_RUN = None

# Default ARC JSON paths (will be exposed as argparse defaults)
TRAINING_TASKS_JSON = f"data/arc-{YEAR}/arc-agi_training_challenges.json"
TRAINING_SOLUTIONS_JSON = f"data/arc-{YEAR}/arc-agi_training_solutions.json"
EVALUATION_TASKS_JSON = f"data/arc-{YEAR}/arc-agi_evaluation_challenges.json"
EVALUATION_SOLUTIONS_JSON = f"data/arc-{YEAR}/arc-agi_evaluation_solutions.json"
TEST_TASKS_JSON = f"data/arc-{YEAR}/arc-agi_test_challenges.json"

# Qdrant client and collection (populated if RAG hints are initialized)
# These are module-level to make it easy for other helper functions (e.g.
# `store_record`) to access the active client/collection without threading
# the values through many call sites. If you prefer to avoid globals,
# initialize_qdrant_vector_store can return `(client, collection_name)` and
# the caller can hold them explicitly.
QDRANT_CLIENT = None
QDRANT_COLLECTION_NAME = None
QDRANT_AVAILABLE = globals().get('QDRANT_AVAILABLE', False)


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the entire application.
    
    When debug=True, sets logging to DEBUG level and shows detailed LLM interactions.
    When debug=False, sets logging to INFO level for normal operation.
    
    This only needs to be called once at startup. All modules that import logging
    will automatically use this configuration.
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any existing configuration
    )
    
    # Set level for agent modules specifically
    logging.getLogger('agent').setLevel(level)
    logging.getLogger('agent.actions').setLevel(level)
    logging.getLogger('agent.debug').setLevel(level)
    
    if debug:
        logging.info("Debug logging enabled - will show LLM prompts and responses")
    else:
        logging.info("Normal logging enabled - use --debug for detailed output")


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


def create_output_directory(timestamp: str) -> str:
    """Create an output directory using an externally provided timestamp.

    The `timestamp` should be a string produced by `datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")`.
    """
    output_dir = os.path.join("output/output_agent", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_params(output_dir: str, args: argparse.Namespace) -> None:
    """Save run parameters to params.json by serializing the argparse Namespace.

    This function simply converts `args` to a dictionary (using `vars`) and
    writes it to `params.json` inside `output_dir`. It also records a
    `timestamp` for the run.
    """
    # Convert Namespace to dict; fall back to __dict__ if necessary
    try:
        params = dict(vars(args) or {})
    except Exception:
        params = dict(getattr(args, '__dict__', {}) or {})

    # Add/override a timestamp field
    params['timestamp'] = datetime.datetime.now().isoformat()

    # Ensure output directory exists and write the params file
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    with open(os.path.join(output_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=2)


def save_task_ids(output_dir: str, training_tasks: Dict, evaluation_tasks: Dict, test_tasks: Optional[Dict] = None) -> None:
    """Save task ID lists to separate files.

    Writes three files into `output_dir`:
      - `training_task_ids.txt`
      - `evaluation_task_ids.txt`
      - `test_task_ids.txt` (only if `test_tasks` is provided)
    """
    with open(os.path.join(output_dir, "training_task_ids.txt"), 'w') as f:
        for task_id in sorted((training_tasks or {}).keys()):
            f.write(f"{task_id}\n")

    with open(os.path.join(output_dir, "evaluation_task_ids.txt"), 'w') as f:
        for task_id in sorted((evaluation_tasks or {}).keys()):
            f.write(f"{task_id}\n")

    with open(os.path.join(output_dir, "test_task_ids.txt"), 'w') as f:
        for task_id in sorted((test_tasks or {}).keys()):
            f.write(f"{task_id}\n")


def initialize_qdrant_vector_store(database_dir: str, timestamp: Optional[str] = None, resume: bool = False):
    """Initialize a Qdrant vector collection for reasoning traces.

    Uses embedded (file-based) Qdrant client that doesn't require external service.
    Creates a timestamped collection name (so multiple runs can coexist)
    and writes a small `collection_info.json` file under `database_dir`.

    Args:
        database_dir: Directory to store Qdrant database files
        timestamp: Optional timestamp for collection naming
        resume: If True, attempt to load existing collection from database_dir

    Returns a tuple `(client, collection_name)` on success, or `None` on failure.
    """
    global QDRANT_CLIENT, QDRANT_COLLECTION_NAME
    if not QDRANT_AVAILABLE:
        print("Qdrant client not available (missing dependency)")
        return None

    try:
        os.makedirs(database_dir, exist_ok=True)
    except Exception:
        pass

    try:
        # Use embedded (file-based) Qdrant - no external service needed!
        # This persists to disk and starts/stops automatically
        client = QdrantClient(path=database_dir)
        print(f"Initialized embedded Qdrant client at: {database_dir}")

        # Check if resuming from existing database
        collection_name = None
        if resume:
            # Try to load existing collection info
            info_path = os.path.join(database_dir, "collection_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        collection_name = info.get('collection_name')
                        
                    # Verify collection exists
                    if collection_name and client.collection_exists(collection_name):
                        print(f"Resuming existing Qdrant collection: {collection_name}")
                        QDRANT_CLIENT = client
                        QDRANT_COLLECTION_NAME = collection_name
                        return client, collection_name
                    else:
                        print(f"Collection '{collection_name}' not found in database, creating new...")
                        collection_name = None
                except Exception as e:
                    print(f"Could not resume Qdrant collection: {e}")
                    collection_name = None

        # Create new collection if not resuming or resume failed
        if not collection_name:
            # Use provided timestamp (preferred) or generate one for the collection name
            if timestamp:
                # normalize timestamp for collection name (remove separators)
                ts = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%S-%f").strftime("%Y%m%dT%H%M%S%f")
            else:
                ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")

            collection_name = f"arc_reasoning_{ts}"

            # Check if collection exists (shouldn't for new timestamps, but be safe)
            if client.collection_exists(collection_name):
                print(f"Collection {collection_name} already exists, deleting...")
                client.delete_collection(collection_name=collection_name)

            # Create the collection with the desired vector schema
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

            # Persist collection info for later lookup
            info = {"collection_name": collection_name, "timestamp": ts}
            try:
                with open(os.path.join(database_dir, "collection_info.json"), 'w') as f:
                    json.dump(info, f)
            except Exception:
                pass

            print(f"Created new Qdrant collection: {collection_name}")

        QDRANT_CLIENT = client
        QDRANT_COLLECTION_NAME = collection_name
        return client, collection_name

    except Exception as e:
        print(f"Failed to initialize Qdrant client/collection: {e}")
        return None


def cleanup_qdrant():
    """Cleanup function to gracefully close Qdrant client."""
    global QDRANT_CLIENT
    if QDRANT_CLIENT is not None:
        try:
            # Embedded Qdrant client closes automatically, but we can explicitly close
            if hasattr(QDRANT_CLIENT, 'close'):
                QDRANT_CLIENT.close()
            print("\nClosed Qdrant client successfully")
        except Exception as e:
            print(f"\nError closing Qdrant client: {e}")
        finally:
            QDRANT_CLIENT = None


def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals gracefully."""
    print("\n\nReceived interrupt signal, cleaning up...")
    cleanup_qdrant()
    sys.exit(0)


def save_task_result(task_folder: str, result: Dict[str, Any]) -> None:
    """Save individual task result to JSON file.

    `task_folder` must be the full path to the per-task output directory
    (typically `os.path.join(output_dir, task_id)`). The function will
    ensure the folder exists and write `<task_id>.json` inside it. If the
    task_id cannot be inferred from the `result`, the folder basename will
    be used as the task id.
    """
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

                # Ensure provided per-task folder exists
                try:
                    os.makedirs(task_folder, exist_ok=True)
                except Exception:
                    pass

                for vc in trans.get('_visual_cues'):
                    ex_idx = vc.get('example_index')
                    in_b64 = vc.get('input_b64')
                    out_b64 = vc.get('output_b64')
                    # Derive task_id for filenames: prefer result's task_id if present
                    task_id_for_name = result.get('task_id') or os.path.basename(task_folder)
                    if in_b64:
                        try:
                            img_bytes = base64.b64decode(in_b64)
                            fname_in = f"{task_id_for_name}_sol{si}_ex{ex_idx}_input.png"
                            fpath_in = os.path.join(task_folder, fname_in)
                            with open(fpath_in, 'wb') as imgf:
                                imgf.write(img_bytes)
                            visual_files.append(os.path.join(os.path.basename(task_folder), fname_in))
                        except Exception:
                            pass
                    if out_b64:
                        try:
                            img_bytes = base64.b64decode(out_b64)
                            fname_out = f"{task_id_for_name}_sol{si}_ex{ex_idx}_expected.png"
                            fpath_out = os.path.join(task_folder, fname_out)
                            with open(fpath_out, 'wb') as imgf:
                                imgf.write(img_bytes)
                            visual_files.append(os.path.join(os.path.basename(task_folder), fname_out))
                        except Exception:
                            pass
            if visual_files:
                solution['_visual_cue_files'] = visual_files
        except Exception:
            pass

    # Ensure the JSON is saved inside the supplied per-task folder
    try:
        os.makedirs(task_folder, exist_ok=True)
    except Exception:
        pass

    # Try to determine the task_id; fall back to folder basename
    task_id = result.get('task_id') or os.path.basename(task_folder)
    out_path = os.path.join(task_folder, f"{task_id}.json")

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

def print_summary(agent: ARCLangGraphAgent, all_results: List[Dict[str, Any]], task_ids: List[str], output_dir: str, reasoning_model: Optional[str] = None, transformation_model: Optional[str] = None, coding_model: Optional[str] = None) -> Dict[str, Any]:
    """Print and save summary of all task results to summary.json."""
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
    
    # Build summary dictionary
    summary = {
        'total_tasks': total_tasks,
        'workflow_completions': workflow_completions,
        'workflow_completion_rate': workflow_completions / total_tasks * 100 if total_tasks > 0 else 0.0,
        'num_tests_successful': num_tests_successful,
        'test_success_rate': num_tests_successful / total_tasks * 100 if total_tasks > 0 else 0.0,
        'task_ids': task_ids,
        'models': {
            'reasoning': reasoning_model,
            'transformation': transformation_model,
            'code': coding_model
        }
    }
    
    # Token usage & cost summary for all 3 LLMs
    try:
        # Collect token counts from all three LLMs
        llm_counts = {}
        total_input = 0
        total_output = 0
        total_tokens = 0
        
        # Reasoning LLM
        if hasattr(agent, 'llm') and hasattr(agent.llm, 'get_token_counts'):
            llm_counts['reasoning'] = agent.llm.get_token_counts()
            total_input += llm_counts['reasoning'].get('input_tokens', 0)
            total_output += llm_counts['reasoning'].get('output_tokens', 0)
            total_tokens += llm_counts['reasoning'].get('total_tokens', 0)
        
        # Transformation LLM
        if hasattr(agent, 'transformation_llm') and hasattr(agent.transformation_llm, 'get_token_counts'):
            llm_counts['transformation'] = agent.transformation_llm.get_token_counts()
            total_input += llm_counts['transformation'].get('input_tokens', 0)
            total_output += llm_counts['transformation'].get('output_tokens', 0)
            total_tokens += llm_counts['transformation'].get('total_tokens', 0)
        
        # Code LLM
        if hasattr(agent, 'code_llm') and hasattr(agent.code_llm, 'get_token_counts'):
            llm_counts['code'] = agent.code_llm.get_token_counts()
            total_input += llm_counts['code'].get('input_tokens', 0)
            total_output += llm_counts['code'].get('output_tokens', 0)
            total_tokens += llm_counts['code'].get('total_tokens', 0)

        if llm_counts:
            # Add token counts to summary
            summary['token_usage'] = {
                'total_input_tokens': int(total_input),
                'total_output_tokens': int(total_output),
                'total_tokens': int(total_tokens),
                'per_llm': {
                    llm_type: {
                        'input_tokens': int(counts.get('input_tokens', 0)),
                        'output_tokens': int(counts.get('output_tokens', 0)),
                        'total_tokens': int(counts.get('total_tokens', 0))
                    }
                    for llm_type, counts in llm_counts.items()
                }
            }
            
            print(f"Token usage summary (all LLMs combined):")
            print(f"  Total input tokens : {int(total_input)}")
            print(f"  Total output tokens: {int(total_output)}")
            print(f"  Total tokens       : {int(total_tokens)}")
            print()
            
            # Print per-LLM breakdown
            for llm_type, counts in llm_counts.items():
                in_t = int(counts.get('input_tokens', 0))
                out_t = int(counts.get('output_tokens', 0))
                tot_t = int(counts.get('total_tokens', in_t + out_t))
                print(f"  {llm_type.capitalize()} LLM: {in_t} input, {out_t} output, {tot_t} total")
            print()

            # Calculate costs for each LLM if model names are available
            total_cost = 0.0
            cost_breakdown = {}
            
            # Use provided model names
            models = {
                'reasoning': reasoning_model,
                'transformation': transformation_model,
                'code': coding_model
            }
            
            print(f"Cost breakdown by LLM:")
            for llm_type, model in models.items():
                if llm_type in llm_counts and model:
                    counts = llm_counts[llm_type]
                    in_t = int(counts.get('input_tokens', 0))
                    out_t = int(counts.get('output_tokens', 0))
                    
                    try:
                        cost = estimate_cost(model, input_tokens=in_t, output_tokens=out_t)
                        if cost is not None:
                            cost_breakdown[llm_type] = float(cost)
                            total_cost += cost
                            print(f"  {llm_type.capitalize()} LLM ({model}): ${cost:.6f}")
                    except Exception:
                        pass
            
            if total_cost > 0:
                summary['cost_estimate'] = {
                    'total_cost': float(total_cost),
                    'per_llm': cost_breakdown
                }
                print(f"\n  Estimated TOTAL cost: ${total_cost:.6f}")
            else:
                print(f"\n  Cost estimation unavailable (pricing data not found)")

    except Exception as e:
        # Fail silently on token/cost reporting so summary still prints
        pass
    
    # Save summary to JSON file
    try:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    except Exception as e:
        print(f"\nWarning: Could not save summary.json: {e}")
    
    return summary


def parse_arguments():
    """Parse command line arguments with defaults from ALL_CAPS variables."""
    parser = argparse.ArgumentParser(description="LangGraph multi-solution agent runner for ARC tasks")
    parser.add_argument("--reasoning-model", type=str, default=REASONING_MODEL,
                       help=f"Model to use (e.g., gpt-4o-mini, gemini-2.0-flash) (default: {REASONING_MODEL})")
    parser.add_argument("--coding-model", type=str, default=CODING_MODEL,
                       help=f"Coding model to use (e.g., gpt-4o-mini, gemini-2.0-flash) (default: {CODING_MODEL})")
    parser.add_argument("--transformation-steps-model", type=str, default=TRANSFORMATION_STEPS_MODEL,
                        help=f"Model to use for transformation steps (e.g., gpt-4o-mini, gemini-2.0-flash) (default: {TRANSFORMATION_STEPS_MODEL})")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default=MODE,
                       help=f"Test mode: single task or batch (default: {MODE})")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="When set, run ONLY on the evaluation dataset (applies to single or batch mode)")
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
    parser.add_argument("--debug", action="store_true", default=DEBUG,
                       help=f"Enable debug logging (shows LLM prompts/responses) (default: {DEBUG})")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                       help="Number of parallel workers for batch mode (default: 1 = sequential)")
    
    # Year and file path overrides for ARC data
    parser.add_argument("--year", type=int, default=YEAR,
                       help=f"Year for ARC dataset directory (default: {YEAR})")
    parser.add_argument("--training-tasks-json", type=str, default=TRAINING_TASKS_JSON,
                       help=f"Path to training tasks JSON (default: {TRAINING_TASKS_JSON})")
    parser.add_argument("--training-solutions-json", type=str, default=TRAINING_SOLUTIONS_JSON,
                       help=f"Path to training solutions JSON (default: {TRAINING_SOLUTIONS_JSON})")
    parser.add_argument("--evaluation-tasks-json", type=str, default=EVALUATION_TASKS_JSON,
                       help=f"Path to evaluation tasks JSON (default: {EVALUATION_TASKS_JSON})")
    parser.add_argument("--evaluation-solutions-json", type=str, default=EVALUATION_SOLUTIONS_JSON,
                       help=f"Path to evaluation solutions JSON (default: {EVALUATION_SOLUTIONS_JSON})")
    parser.add_argument("--test-tasks-json", type=str, default=TEST_TASKS_JSON,
                       help=f"Path to test tasks JSON (default: {TEST_TASKS_JSON})")
    parser.add_argument("--use-vllm", action="store_true", default=USE_VLLM,
                       help="When set, initialize the LLM using vLLM/langchain_vllm (local) instead of other providers")
    
    # Flags to override behaviour of the multi-solution agent
    parser.add_argument("--enable-parallel-eval", action="store_true",
                       default=ENABLE_PARALLEL_EVAL,
                       help=f"Enable parallel evaluation of examples (default: {ENABLE_PARALLEL_EVAL})")
    parser.add_argument("--enable-code-predict", action="store_true",
                       default=ENABLE_CODE_PREDICT,
                       help=f"Enable code-predicted outputs during testing (default: {ENABLE_CODE_PREDICT})")
    parser.add_argument("--enable-llm-predict", action="store_true",
                       default=ENABLE_LLM_PREDICT,
                       help=f"Enable LLM-predicted outputs during testing (default: {ENABLE_LLM_PREDICT})")
    parser.add_argument("--enable-visual-cue", action="store_true",
                       default=ENABLE_VISUAL_CUE,
                       help=f"Generate and pass input/output images to the LLM (default: {ENABLE_VISUAL_CUE})")
    parser.add_argument("--enable-rag-hint", action="store_true",
                       default=ENABLE_RAG_HINT,
                       help=f"Enable retrieval-augmented generation hints from past traces (default: {ENABLE_RAG_HINT})")
    
    # Numeric workflow configuration exposed as CLI flags
    parser.add_argument("--num-initial-solutions", type=int,
                        default=NUM_INITIAL_SOLUTIONS,
                        help=f"Number of initial solutions to generate (default: {NUM_INITIAL_SOLUTIONS})")
    parser.add_argument("--num-loops", type=int,
                        default=NUM_LOOPS,
                        help=f"Number of main algorithm loops (default: {NUM_LOOPS})")
    parser.add_argument("--num-seed-solutions", type=int,
                        default=NUM_SEED_SOLUTIONS,
                        help=f"Number of seed solutions to start from (default: {NUM_SEED_SOLUTIONS})")
    parser.add_argument("--num-refinements", type=int,
                        default=NUM_REFINEMENTS,
                        help=f"Number of refinement rounds (default: {NUM_REFINEMENTS})")
    parser.add_argument("--num-solutions-per-refinement", type=int,
                        default=NUM_SOLUTIONS_PER_REFINEMENT,
                        help=f"Solutions generated per refinement (default: {NUM_SOLUTIONS_PER_REFINEMENT})")
    parser.add_argument("--num-fusions", type=int,
                        default=NUM_FUSIONS,
                        help=f"Number of fusion rounds (default: {NUM_FUSIONS})")
    parser.add_argument("--num-solutions-per-fusion", type=int,
                        default=NUM_SOLUTIONS_PER_FUSION,
                        help=f"Solutions generated per fusion (default: {NUM_SOLUTIONS_PER_FUSION})")
    parser.add_argument("--recursion-limit", type=int,
                        default=RECURSION_LIMIT,
                        help=f"LangGraph recursion limit (default: {RECURSION_LIMIT})")
    parser.add_argument("--resume-run", type=str, default=RESUME_RUN,
                       help="Optional resume run identifier: 'latest' or exact folder name under output/output_agent")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Setup logging first so all subsequent operations can use it
    setup_logging(debug=args.debug)
    
    # Initialize LLM (optionally using vLLM if requested)
    llm = TokenTrackingLLM(initialize_llm_from_config(args.reasoning_model, use_vllm=args.use_vllm))
    if llm is None:
        return 1
    transformation_llm = TokenTrackingLLM(initialize_llm_from_config(args.transformation_steps_model, use_vllm=args.use_vllm))
    if transformation_llm is None:
        return 1
    code_llm = TokenTrackingLLM(initialize_llm_from_config(args.coding_model, use_vllm=args.use_vllm))
    if code_llm is None:
        return 1
    
    # Create output directory first (generate timestamp here so other
    # components can reuse it, e.g. for Qdrant collection naming)
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    output_dir = create_output_directory(run_timestamp)

    # If requested, allow resuming an existing run instead of using the
    # newly created timestamped folder.
    if getattr(args, 'resume_run', None):
        resume_val = args.resume_run
        base_root = os.path.join("output", "output_agent")
        if resume_val == 'latest':
            # pick latest modified directory under base_root
            try:
                entries = [os.path.join(base_root, d) for d in os.listdir(base_root) if os.path.isdir(os.path.join(base_root, d))]
                if not entries:
                    print("No previous runs found to resume.")
                    return 1
                latest = max(entries, key=os.path.getmtime)
                output_dir = latest
                # Extract timestamp from directory name for Qdrant collection naming
                run_timestamp = os.path.basename(output_dir)
                print(f"Resuming latest run: {output_dir}")
            except Exception as e:
                print(f"Failed to locate latest run to resume: {e}")
                return 1
        else:
            candidate = os.path.join(base_root, resume_val)
            if not os.path.isdir(candidate):
                print(f"Requested resume run '{resume_val}' not found under {base_root}")
                return 1
            output_dir = candidate
            # Extract timestamp from directory name for Qdrant collection naming
            run_timestamp = os.path.basename(output_dir)
            print(f"Resuming specified run: {output_dir}")
    else:
        print(f"Output directory: {output_dir}")
    
    # Load tasks and solutions
    try:
        training_tasks = load_arc_tasks(args.training_tasks_json)
        training_solutions = load_arc_solutions(args.training_solutions_json)
        evaluation_tasks = load_arc_tasks(args.evaluation_tasks_json)
        evaluation_solutions = load_arc_solutions(args.evaluation_solutions_json)
        test_tasks = load_arc_tasks(args.test_tasks_json)
    except FileNotFoundError as e:
        print(f"Error: Could not load ARC data: {e}")
        return 1

    # Determine which dataset to use based on evaluate-only flag.
    # - If `--evaluate-only` is set, use ONLY the evaluation dataset.
    # - Otherwise, combine training and evaluation datasets so runs cover both.
    if getattr(args, 'evaluate_only', False):
        active_tasks = evaluation_tasks
        active_solutions = evaluation_solutions
        print("EVALUATE-ONLY mode: using evaluation dataset for all runs")
    else:
        # Merge training + evaluation tasks + test tasks. Training entries will override
        # identical task IDs from evaluation (if any) by updating with
        # training after evaluation.
        active_tasks = {}
        active_tasks.update(evaluation_tasks or {})
        active_tasks.update(training_tasks or {})
        active_tasks.update(test_tasks or {})

        # Merge solutions similarly, preferring training solutions when
        # a task ID exists in both sets.
        active_solutions = {}
        active_solutions.update(evaluation_solutions or {})
        active_solutions.update(training_solutions or {})

        print("Using combined training + evaluation + test dataset for runs")
    
    # Save parameters (serialize args — includes model selections)
    save_params(output_dir, args)

    # Save task IDs (include test tasks as well)
    save_task_ids(output_dir, training_tasks, evaluation_tasks, test_tasks)

    # Initialize Qdrant Vector Store database if they are available
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_qdrant)
    
    if args.enable_rag_hint:
        # Determine if we're resuming an existing run
        is_resuming = bool(getattr(args, 'resume_run', None))
        initialize_qdrant_vector_store(
            database_dir=os.path.join(output_dir, "qdrant_db"), 
            timestamp=run_timestamp,
            resume=is_resuming
        )

    # Create a MultiSolution agent for this run (use parsed flags to override defaults)
    print(f"Initialize the multi-solution LangGraph agent...")
    agent = ARCLangGraphAgent(
        llm=llm,
        transformation_llm=transformation_llm,
        code_llm=code_llm,
        num_initial_solutions=getattr(args, 'num_initial_solutions', NUM_INITIAL_SOLUTIONS),
        num_loops=getattr(args, 'num_loops', NUM_LOOPS),
        num_seed_solutions=getattr(args, 'num_seed_solutions', NUM_SEED_SOLUTIONS),
        num_refinements=getattr(args, 'num_refinements', NUM_REFINEMENTS),
        num_solutions_per_refinement=getattr(args, 'num_solutions_per_refinement', NUM_SOLUTIONS_PER_REFINEMENT),
        num_fusions=getattr(args, 'num_fusions', NUM_FUSIONS),
        num_solutions_per_fusion=getattr(args, 'num_solutions_per_fusion', NUM_SOLUTIONS_PER_FUSION),
        enable_parallel_eval=args.enable_parallel_eval,
        enable_visual_cue=args.enable_visual_cue,
        enable_rag_hint=args.enable_rag_hint,
        enable_code_predict=args.enable_code_predict,
        enable_llm_predict=args.enable_llm_predict,
        recursion_limit=getattr(args, 'recursion_limit', RECURSION_LIMIT),
        qdrant_client=QDRANT_CLIENT,
        qdrant_collection_name=QDRANT_COLLECTION_NAME)

    all_results = []
    task_ids_processed = []
    
    if args.mode == "single":
        # Single task test
        if args.task_id:
            # Look up the requested task ID in the active dataset (which may be
            # evaluation-only or the merged training+evaluation set).
            if args.task_id in active_tasks:
                task_id = args.task_id
                task_data = active_tasks[task_id]
                task_solution = active_solutions.get(task_id)
            else:
                print(f"Error: Task ID '{args.task_id}' not found in the active dataset")
                return 1
        elif args.task_index is not None:
            task_id, task_data, task_solution = get_task_by_index(active_tasks, active_solutions, args.task_index)
        else:
            # Random task from the active dataset
            task_id, task_data, task_solution = get_task_by_index(active_tasks, active_solutions, None)
        
        print(f"Testing single task: {task_id}")
        
        # Compute per-task folder and run LangGraph agent with max attempts
        task_folder = os.path.join(output_dir, task_id)
        # Run LangGraph agent with max attempts (reuse shared agent)
        langgraph_result = agent.solve_task(task_id, task_data, task_folder, task_solution, max_attempts=args.max_attempts)

        # Compute training/testing priority & average scores
        try:
            calculate_average_score(langgraph_result)
        except Exception as e:
            print(f"Warning: could not calculate scores for task {task_id}: {e}")
        
        all_results.append(langgraph_result)
        task_ids_processed.append(task_id)

        # Save individual task result into the per-task output folder
        save_task_result(task_folder, langgraph_result)

        # Print summary using helper
        summarize_and_print_result(langgraph_result, task_id=task_id)
        
    elif args.mode == "batch":
        # Batch test (supports parallel execution when --workers > 1)
        print(f"Running batch test with {args.num_tasks} tasks (workers={args.workers})")

        random.seed(args.random_seed)
        selected_task_ids = random.sample(list(active_tasks.keys()),
                        min(args.num_tasks, len(active_tasks)))

        # Helper to run a single task (creates a fresh agent to avoid shared-state issues)

        def run_single_task(task_id: str, progress_info: str = ""):
            task_data = active_tasks[task_id]
            task_solution = active_solutions.get(task_id, None)

            # Create a per-task multi-solution agent and attach helpers snapshot
            local_agent = ARCLangGraphAgent(
                llm=llm,
                transformation_llm=transformation_llm,
                code_llm=code_llm,
                num_initial_solutions=getattr(args, 'num_initial_solutions', NUM_INITIAL_SOLUTIONS),
                num_loops=getattr(args, 'num_loops', NUM_LOOPS),
                num_seed_solutions=getattr(args, 'num_seed_solutions', NUM_SEED_SOLUTIONS),
                num_refinements=getattr(args, 'num_refinements', NUM_REFINEMENTS),
                num_solutions_per_refinement=getattr(args, 'num_solutions_per_refinement', NUM_SOLUTIONS_PER_REFINEMENT),
                num_fusions=getattr(args, 'num_fusions', NUM_FUSIONS),
                num_solutions_per_fusion=getattr(args, 'num_solutions_per_fusion', NUM_SOLUTIONS_PER_FUSION),
                enable_parallel_eval=args.enable_parallel_eval,
                enable_visual_cue=args.enable_visual_cue,
                enable_rag_hint=args.enable_rag_hint,
                enable_code_predict=args.enable_code_predict,
                enable_llm_predict=args.enable_llm_predict,
                recursion_limit=getattr(args, 'recursion_limit', RECURSION_LIMIT),
                qdrant_client=QDRANT_CLIENT,
                qdrant_collection_name=QDRANT_COLLECTION_NAME)

            # Compute per-task folder for this task
            task_folder = os.path.join(output_dir, task_id)

            start_time = time.time()
            result = local_agent.solve_task(task_id, task_data, task_folder, task_solution, max_attempts=args.max_attempts)
            result.setdefault("execution_time", time.time() - start_time)

            # Compute training/testing priority & average scores for this result
            try:
                calculate_average_score(result)
            except Exception as e:
                print(f"Warning: could not calculate scores for task {task_id}: {e}")

            # Save result to file
            save_task_result(task_folder, result)

            # Print concise result summary
            summarize_and_print_result(result, task_id=task_id, progress=progress_info)

            return task_id, result

        if args.workers and args.workers > 1:
            # Parallel execution using threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks with progress info
                future_to_task = {
                    executor.submit(run_single_task, tid, f"({i+1}/{len(selected_task_ids)})"): tid 
                    for i, tid in enumerate(selected_task_ids)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_task):
                    tid, langgraph_result = future.result()
                    all_results.append(langgraph_result)
                    task_ids_processed.append(tid)

        else:
            # Sequential execution (workers==1) — reuse the shared agent for efficiency
            for i, task_id in enumerate(selected_task_ids, 1):
                print(f"\n{'='*80}")
                print(f"PROCESSING TASK {i}/{args.num_tasks}: {task_id}")
                print(f"{'='*80}")

                task_data = active_tasks[task_id]
                task_solution = active_solutions.get(task_id)

                # Compute per-task folder and run LangGraph agent with max attempts
                task_folder = os.path.join(output_dir, task_id)
                # Run LangGraph agent with max attempts (reuse shared agent)
                langgraph_result = agent.solve_task(task_id, task_data, task_folder, task_solution, max_attempts=args.max_attempts)

                # Compute training/testing priority & average scores
                try:
                    calculate_average_score(langgraph_result)
                except Exception as e:
                    print(f"Warning: could not calculate scores for task {task_id}: {e}")

                all_results.append(langgraph_result)
                task_ids_processed.append(task_id)

                # Save individual task result in the per-task output folder
                save_task_result(task_folder, langgraph_result)

                # Display concise summary via helper
                summarize_and_print_result(langgraph_result, task_id=task_id)
    
    # Save summary (include model names so we can estimate costs)
    print_summary(agent, all_results, task_ids_processed, output_dir,
                  reasoning_model=args.reasoning_model,
                  transformation_model=args.transformation_steps_model,
                  coding_model=args.coding_model)
    
    # Cleanup Qdrant before exiting
    cleanup_qdrant()
    return 0


class TokenTrackingLLM:
    def __init__(self, llm):
        self.llm = llm
        # Initialize cumulative token counters
        self.input_tokens = 0    # tokens sent as input/prompts
        self.output_tokens = 0   # tokens in model completions/responses
        self.total_tokens = 0    # cumulative total tokens (input + output when available)

    def invoke(self, *args, **kwargs):
        response = self.llm.invoke(*args, **kwargs)
        # Count tokens directly from the prompt/messages passed and the response
        # content. This uses a simple whitespace-based token approximation.

        # Build prompt text from common call patterns: `prompt`, `messages`,
        # or first positional string arg.
        prompt_text = ""
        try:
            if 'prompt' in kwargs and kwargs.get('prompt') is not None:
                prompt_obj = kwargs.get('prompt')
                prompt_text = prompt_obj if isinstance(prompt_obj, str) else str(prompt_obj)
            elif 'messages' in kwargs and kwargs.get('messages'):
                msgs = kwargs.get('messages')
                # messages expected as list[dict] with 'content'
                if isinstance(msgs, (list, tuple)):
                    parts = []
                    for m in msgs:
                        if isinstance(m, dict):
                            parts.append(m.get('content', ''))
                        else:
                            parts.append(str(m))
                    prompt_text = ' '.join(parts)
            elif len(args) > 0 and isinstance(args[0], str):
                prompt_text = args[0]
        except Exception:
            prompt_text = ''

        def _count_tokens_whitespace(s):
            if not s:
                return 0
            try:
                return len(str(s).split())
            except Exception:
                return 0

        prompt_i = _count_tokens_whitespace(prompt_text)

        # Get response text/content
        response_text = None
        try:
            response_text = getattr(response, 'content', None)
        except Exception:
            response_text = None
        if response_text is None:
            # try common dict shapes
            if isinstance(response, dict):
                response_text = response.get('content') or response.get('text') or str(response)
            else:
                response_text = str(response)

        completion_i = _count_tokens_whitespace(response_text)
        total_i = prompt_i + completion_i

        # Update internal cumulative counters
        self.input_tokens += int(prompt_i or 0)
        self.output_tokens += int(completion_i or 0)
        self.total_tokens += int(total_i or 0)

        # Prepare per-call usage report
        return response

    # Accessors for cumulative token counts
    def get_input_token_count(self) -> int:
        """Return cumulative input (prompt) tokens seen so far."""
        return int(self.input_tokens)

    def get_output_token_count(self) -> int:
        """Return cumulative output (completion) tokens seen so far."""
        return int(self.output_tokens)

    def get_total_token_count(self) -> int:
        """Return cumulative total tokens seen so far."""
        return int(self.total_tokens)

    def get_token_counts(self) -> Dict[str, int]:
        """Return all cumulative token counts as a dict with separate keys.

        Keys: `input_tokens`, `output_tokens`, `total_tokens`.
        """
        return {
            "input_tokens": self.get_input_token_count(),
            "output_tokens": self.get_output_token_count(),
            "total_tokens": self.get_total_token_count()
        }


def initialize_llm_from_config(model_name: str, use_vllm: bool = False):
    """Initialize LLM based on model_configs.py configuration.

    If `use_vllm` is True, attempt to initialize a vLLM-backed LLM using
    `langchain_vllm.VLLM`. If that package isn't installed, a helpful message
    will be printed and None returned.
    """
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
        # If the caller explicitly requested vLLM, try to initialize that first.
        if use_vllm:
            try:
                from langchain_community.llms import VLLM
                # Resolve a HuggingFace/vLLM identifier if possible
                try:
                    from model_configs import resolve_hf_model
                    model_for_vllm = resolve_hf_model(model_name) or config.get("hf_model") or model_key
                except Exception:
                    model_for_vllm = config.get("hf_model") or model_key

                # Ensure HuggingFace hub is allowed to perform repo lookups/downloads
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("HUGGINGFACE_HUB_OFFLINE", None)

                return VLLM(model=model_for_vllm, local_files_only=False,
                            temperature=0.6, max_output_tokens=100000)
            except ImportError as e:
                print(f"vLLM requested but package not installed: {e}")
                print("Install with: pip install vllm langchain-vllm")
                return None

        if provider == "google" or provider == "learnlm":
            # Google Gemini models - suppress ALTS warnings during import
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_key,
                temperature=0.6,
                max_output_tokens=100000
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
