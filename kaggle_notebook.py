#!/usr/bin/env python3
"""
Kaggle-ready consolidated version of `run_langgraph_agent.py`.

This single-file script contains simplified, self-contained stubs for:
- model configuration helpers (minimal `MODEL_CONFIGS` + lookup helpers)
- a `DummyLLM` and `TokenTrackingLLM` wrapper
- a lightweight `ARCLangGraphAgent` that simulates solving tasks
- utilities for loading/saving ARC-style JSON files and printing summaries

The goal is to run in a Kaggle notebook or environment without requiring
external model provider packages. Replace the `DummyLLM` with a real
provider for production runs.
"""

import os
import sys
import json
import random
import datetime
import time
import argparse
import threading
import base64
from typing import Dict, List, Any, Optional, Tuple

# -------------------------
# Minimal model config stubs
# -------------------------
MODEL_CONFIGS = {
    "gemini-2.5-flash": {"provider": "google", "hf_model": "google/gemini-2.5-flash"},
    "gpt-4o-mini": {"provider": "openai", "hf_model": "openai/gpt-4o-mini"},
    "local-dummy": {"provider": "local", "hf_model": "local/dummy"},
}


def find_model_key(name: str) -> Optional[str]:
    # Return a known key that matches the provided name, or None
    if not name:
        return None
    if name in MODEL_CONFIGS:
        return name
    # Try simple aliasing: match by prefix
    for k in MODEL_CONFIGS:
        if k.lower().startswith(name.lower()):
            return k
    # As a last resort, try to return any key
    return None


def get_pricing_for(model_name: str) -> Optional[Dict[str, float]]:
    # Very small fake pricing table for summary prints
    return {"input_per_m": 0.02, "output_per_m": 0.02}


def estimate_cost(model_name: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    p = get_pricing_for(model_name) or {"input_per_m": 0.0, "output_per_m": 0.0}
    return (input_tokens / 1_000_000.0) * p["input_per_m"] + (output_tokens / 1_000_000.0) * p["output_per_m"]


def resolve_hf_model(name: str) -> Optional[str]:
    # Identity mapping in this stub
    return name


# -------------------------
# Minimal Dummy LLM + Token Tracker
# -------------------------
class DummyLLM:
    """Very small echoing LLM replacement for offline runs.

    It accepts `prompt` or `messages` and returns an object with `content`.
    """

    def __init__(self, model_name: str = "local-dummy", temperature: float = 0.0, max_tokens: int = 2000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, *args, **kwargs):
        # Build a representative text from arguments
        text = ""
        if "prompt" in kwargs and kwargs["prompt"] is not None:
            text = kwargs["prompt"]
        elif "messages" in kwargs and kwargs["messages"]:
            parts = []
            for m in kwargs["messages"]:
                if isinstance(m, dict):
                    parts.append(m.get("content", ""))
                else:
                    parts.append(str(m))
            text = " ".join(parts)
        elif len(args) > 0 and isinstance(args[0], str):
            text = args[0]

        # Return a simple object with `content` attribute, similar to many LLM wrappers
        class Resp:
            def __init__(self, content):
                self.content = content

        # Echo back the prompt trimmed to max_tokens
        resp_text = (text or "[dummy response]")
        return Resp(resp_text)


class TokenTrackingLLM:
    def __init__(self, llm):
        self.llm = llm
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def invoke(self, *args, **kwargs):
        response = self.llm.invoke(*args, **kwargs)

        def _count_tokens_whitespace(s):
            if not s:
                return 0
            try:
                return len(str(s).split())
            except Exception:
                return 0

        # Extract prompt-like text
        prompt_text = ""
        try:
            if "prompt" in kwargs and kwargs.get("prompt") is not None:
                prompt_obj = kwargs.get("prompt")
                prompt_text = prompt_obj if isinstance(prompt_obj, str) else str(prompt_obj)
            elif "messages" in kwargs and kwargs.get("messages"):
                msgs = kwargs.get("messages")
                if isinstance(msgs, (list, tuple)):
                    parts = []
                    for m in msgs:
                        if isinstance(m, dict):
                            parts.append(m.get("content", ""))
                        else:
                            parts.append(str(m))
                    prompt_text = " ".join(parts)
            elif len(args) > 0 and isinstance(args[0], str):
                prompt_text = args[0]
        except Exception:
            prompt_text = ""

        prompt_i = _count_tokens_whitespace(prompt_text)

        # Get response text
        response_text = None
        try:
            response_text = getattr(response, "content", None)
        except Exception:
            response_text = None
        if response_text is None:
            response_text = str(response)

        completion_i = _count_tokens_whitespace(response_text)
        total_i = prompt_i + completion_i

        self.input_tokens += int(prompt_i or 0)
        self.output_tokens += int(completion_i or 0)
        self.total_tokens += int(total_i or 0)

        return response

    def get_token_counts(self) -> Dict[str, int]:
        return {"input_tokens": int(self.input_tokens), "output_tokens": int(self.output_tokens), "total_tokens": int(self.total_tokens)}


def initialize_llm_from_config(model_name: str, use_vllm: bool = False):
    """Return a local DummyLLM instance for Kaggle runs."""
    model_key = find_model_key(model_name) or "local-dummy"
    # Always return a dummy model for Kaggle-ready script
    return DummyLLM(model_key, temperature=0.6, max_tokens=2000)


# -------------------------
# Simplified ARCLangGraphAgent stub
# -------------------------
class ARCLangGraphAgent:
    """A minimal, deterministic agent stub that "solves" ARC tasks by
    returning trivial transformations: echoing inputs or copying expected outputs.

    Use this as a placeholder for Kaggle demo runs. Replace with the
    real agent implementation for full functionality.
    """

    def __init__(self, llm=None, transformation_llm=None, code_llm=None, **kwargs):
        self.llm = llm
        self.transformation_llm = transformation_llm
        self.code_llm = code_llm

    def solve_task(self, task_id: str, task_data: Dict[str, Any], task_solution: Optional[Any] = None, max_attempts: int = 1) -> Dict[str, Any]:
        start = time.time()
        # Create a synthetic solution structure compatible with downstream helpers
        # Try to pick an expected output if provided (ARC tasks have `output` or `train`/`test` structure)
        training_results = []
        testing_results = []

        # If the dataset uses 'train' examples (ARC-like), mirror them
        examples = []
        if isinstance(task_data, dict):
            # Common ARC keys: "train" is list of {input, output}, "test" is list of {input}
            if "train" in task_data and isinstance(task_data["train"], list):
                examples = task_data["train"]
            elif "input" in task_data and "output" in task_data:
                examples = [{"input": task_data.get("input"), "output": task_data.get("output")}]

        # Fallback: create one dummy example
        if not examples:
            examples = [{"input": [[0]], "output": [[0]]}]

        for ex in examples:
            expected = ex.get("output")
            predicted = expected  # naive perfect prediction
            training_results.append({
                "input": ex.get("input"),
                "expected_output": expected,
                "predicted_output": predicted,
                "code_success": True,
                "matching_size": True,
                "overlap_percentage": 100.0,
            })

        # Build a single candidate solution container
        solution = {
            "training_results": training_results,
            "testing_results": testing_results,
            "step_by_step_transformation": {"steps": ["echo"]},
        }

        result = {
            "task_id": task_id,
            "workflow_completed": True,
            "solutions_list": [solution],
            "attempts": 1,
            "execution_time": time.time() - start,
        }
        return result


# -------------------------
# File I/O and helper functions (adapted from run_langgraph_agent.py)
# -------------------------
def load_arc_tasks(file_path: str) -> Dict[str, Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)


def load_arc_solutions(file_path: str) -> Dict[str, List[Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)


def create_output_directory() -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    output_dir = os.path.join("output_agent", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_params(output_dir: str, args: argparse.Namespace) -> None:
    try:
        params = dict(vars(args) or {})
    except Exception:
        params = dict(getattr(args, '__dict__', {}) or {})
    params['timestamp'] = datetime.datetime.now().isoformat()
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)


def save_task_result(output_dir: str, task_id: str, result: Dict[str, Any]) -> None:
    task_folder = os.path.join(output_dir, task_id)
    os.makedirs(task_folder, exist_ok=True)
    out_path = os.path.join(task_folder, f"{task_id}.json")
    # Normalize some fields for readability
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)


def shorten_error(msg: Optional[str], max_len: int = 120) -> str:
    if not msg:
        return ""
    s = " ".join(str(msg).split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def calculate_average_score(result: Dict[str, Any]) -> None:
    def compute_scores(entries: Optional[List[Dict[str, Any]]]) -> Tuple[float, float]:
        if not entries:
            return 0.0, 0.0
        total = len(entries)
        num_correct = sum(1 for e in entries if bool(e.get('code_success', False)))
        priority = (num_correct / total) * 100.0 if total > 0 else 0.0
        overlaps = [float(e.get('overlap_percentage', 0.0)) for e in entries]
        average = sum(overlaps) / len(overlaps) if overlaps else 0.0
        return priority, average

    solutions = result.get('solutions_list') or []
    if solutions:
        best_idx = None
        best_priority = -1.0
        best_overlap = -1.0
        for idx, sol in enumerate(solutions):
            s_train = sol.get('training_results') or []
            s_tpriority, s_toverlap = compute_scores(s_train)
            if (s_tpriority > best_priority) or (s_tpriority == best_priority and s_toverlap > best_overlap):
                best_idx = idx
                best_priority = s_tpriority
                best_overlap = s_toverlap
        if best_idx is not None:
            result['highest_solution_index'] = best_idx
            result['highest_training_solution_priority_score'] = best_priority
            result['highest_training_solution_overlap_score'] = best_overlap


def summarize_and_print_result(result: Dict[str, Any], task_id: Optional[str] = None, progress: Optional[str] = None) -> None:
    calculate_average_score(result)
    tid = task_id or result.get('task_id', 'unknown')
    header = "KAGGLE AGENT RESULTS"
    if progress:
        header = f"{header} {progress}"

    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  Task ID: {tid}")
    print(f"  Workflow Completed: {'✓' if result.get('workflow_completed') else '✗'}")

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

    print(f"  Attempts: {result.get('attempts', 0)}")
    print(f"  Execution Time: {result.get('execution_time', 0.0):.2f}s")


def print_summary(agent: ARCLangGraphAgent, all_results: List[Dict[str, Any]], task_ids: List[str], model_name: Optional[str] = None) -> None:
    total_tasks = len(all_results)
    workflow_completions = sum(1 for r in all_results if r.get('workflow_completed'))
    num_tests_successful = sum(1 for r in all_results if r.get('highest_training_solution_priority_score', 0) >= 1.0)
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks processed: {total_tasks}")
    print(f"Completed workflows: {workflow_completions}")
    print(f"Workflow completion rate: {workflow_completions / (total_tasks or 1) * 100:.1f}%")
    print(f"Number of tasks with training priority >=1%: {num_tests_successful}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Kaggle-ready LangGraph agent stub runner")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single")
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--reasoning-model", type=str, default="local-dummy")
    parser.add_argument("--training-tasks-json", type=str, default=None)
    parser.add_argument("--training-solutions-json", type=str, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_arguments() if argv is None else parse_arguments()

    # Initialize LLM wrappers (dummy for Kaggle)
    llm = TokenTrackingLLM(initialize_llm_from_config(args.reasoning_model))
    transformation_llm = TokenTrackingLLM(initialize_llm_from_config(args.reasoning_model))
    code_llm = TokenTrackingLLM(initialize_llm_from_config(args.reasoning_model))

    # Create output dir
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")

    # Load tasks if provided; otherwise create small synthetic dataset
    training_tasks = {}
    training_solutions = {}
    if args.training_tasks_json and os.path.exists(args.training_tasks_json):
        try:
            training_tasks = load_arc_tasks(args.training_tasks_json)
        except Exception as e:
            print(f"Failed to load tasks: {e}")

    if not training_tasks:
        # Make a tiny synthetic dataset with two tasks
        training_tasks = {
            "task_dummy_1": {"input": [[0, 1], [1, 0]], "output": [[1, 1], [1, 0]]},
            "task_dummy_2": {"train": [{"input": [[2]], "output": [[2]]}]},
        }

    agent = ARCLangGraphAgent(llm=llm, transformation_llm=transformation_llm, code_llm=code_llm)

    all_results = []
    task_ids_processed = []

    # Run either single or small batch
    if args.mode == "single":
        # pick by id, index, or first
        if args.task_id and args.task_id in training_tasks:
            task_id = args.task_id
        elif args.task_index is not None:
            ids = list(training_tasks.keys())
            task_id = ids[args.task_index % len(ids)]
        else:
            task_id = next(iter(training_tasks.keys()))

        print(f"Running single task: {task_id}")
        task_data = training_tasks[task_id]
        result = agent.solve_task(task_id, task_data, None)
        calculate_average_score(result)
        all_results.append(result)
        task_ids_processed.append(task_id)
        save_task_result(output_dir, task_id, result)
        summarize_and_print_result(result, task_id=task_id)

    else:
        # Batch mode: pick a few tasks
        ids = list(training_tasks.keys())
        selected = ids[: args.num_tasks]
        for i, tid in enumerate(selected, 1):
            print(f"Processing {i}/{len(selected)}: {tid}")
            res = agent.solve_task(tid, training_tasks[tid], None)
            calculate_average_score(res)
            all_results.append(res)
            task_ids_processed.append(tid)
            save_task_result(output_dir, tid, res)
            summarize_and_print_result(res, task_id=tid, progress=f"({i}/{len(selected)})")

    print_summary(agent, all_results, task_ids_processed, model_name=args.reasoning_model)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
