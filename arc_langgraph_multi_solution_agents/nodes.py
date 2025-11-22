"""
Nodes for the ARC LangGraph Agent workflow.

This module should contain only the node functions. All helper and
action-related utilities have been moved to `actions.py`.
"""

import copy
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import schema and node-facing types
from .schema import AgentState, CodeSolution, ExampleResult

# Import actions (helpers, prompts, execution, refinement)
from .actions import (
    generate_solutions_with_reasoning,
    execute_transformation_code,
    generate_llm_predicted_output,
    calculate_grid_results,
    evaluate_example
)


def generate_code_node(state: AgentState, llm, code_llm) -> AgentState:
    """
    Generate Python code to solve the ARC problem using reasoning-first approach.

    This node uses the available helper functions and analyzes the training
    examples to generate a solution using the provided language model.
    """
    task_data = state["task_data"]

    # Analyze the training examples (kept for node-level logging/analysis)
    training_examples = task_data["train"]

    # Generate code using the reasoning-first approach from actions
    python_codes, reasoning_trace, transformation_solutions_list = generate_solutions_with_reasoning(
        llm,
        code_llm,
        training_examples
    )

    solutions_list = []
    for transformation, code in zip(transformation_solutions_list, python_codes):
        solution = {
            "main_code": code,
            "reasoning_trace": reasoning_trace,
            "step_by_step_transformation": transformation,
        }
        solutions_list.append(solution)

    # Update state
    new_state = copy.deepcopy(state)
    new_state["solutions_list"] = solutions_list

    return new_state


def test_code_node(state: AgentState, llm, code_llm) -> AgentState:
    """
    Test the generated code against training examples.

    This node executes the current solution on all training examples
    and calculates the success rate.
    """
    # Deep copy the new state, and shove in the training results to previous_training_results
    # And then reset training results
    new_state = copy.deepcopy(state)
    # Read runtime flags from the state (propagated by the agent)
    enable_code_predict = new_state.get("enable_code_predict", True)
    enable_llm_predict = new_state.get("enable_llm_predict", True)
    solutions_list = new_state.get("solutions_list", [])
    task_data = new_state.get("task_data", {})
    training_examples = task_data.get("train", [])
    testing_examples = task_data.get("test", [])
    enable_parallel = new_state.get("enable_parallel_eval", False)
    for solution in solutions_list:
        main_code = solution.get("main_code", "")
        training_successes = 0
        llm_training_successes = 0
        testing_successes = 0
        llm_testing_successes = 0
        training_results: List[ExampleResult] = []
        testing_results: List[ExampleResult] = []
        # Per-solution local accumulators
        for i, example in enumerate(training_examples):
            input_grid = example["input"]
            expected_output = example["output"]
        # Use centralized helper to evaluate training examples (possibly parallel)
        transformation_steps = solution.get("step_by_step_transformation", []) if solution else []
        if enable_parallel:
            tasks = []
            with ThreadPoolExecutor() as ex:
                for i, example in enumerate(training_examples):
                    tasks.append(ex.submit(
                        lambda idx, ex_in, ex_out: (idx, evaluate_example(
                            llm,
                            main_code,
                            transformation_steps,
                            ex_in,
                            ex_out,
                            enable_code_predict=enable_code_predict,
                            enable_llm_predict=enable_llm_predict,
                        )), i, example["input"], example["output"]))

                for future in tqdm(as_completed([t for t in tasks]), total=len(tasks), desc="Training eval"):
                    try:
                        # futures return a tuple (idx, result)
                        idx, result = future.result()
                    except Exception as e:
                        # create a failure result
                        idx = -1
                        result = {
                            "input": None,
                            "expected_output": None,
                            "predicted_output": None,
                            "matching_size": False,
                            "overlap_percentage": 0.0,
                            "error_message": str(e),
                            "code_success": False,
                            "llm_predicted_output": None,
                            "llm_matching_size": False,
                            "llm_overlap_percentage": 0.0,
                            "llm_error_message": str(e),
                            "llm_success": False,
                        }

                    # Attach example index and store
                    result["example_index"] = idx
                    training_results.append(result)
                    training_successes += 1 if result.get("code_success") else 0
                    llm_training_successes += 1 if result.get("llm_success") else 0
        else:
            for i, example in enumerate(training_examples):
                input_grid = example["input"]
                expected_output = example["output"]
                result = evaluate_example(
                    llm,
                    main_code,
                    transformation_steps,
                    input_grid,
                    expected_output,
                    enable_code_predict=enable_code_predict,
                    enable_llm_predict=enable_llm_predict,
                )
                result["example_index"] = i
                training_results.append(result)
                training_successes += 1 if result.get("code_success") else 0
                llm_training_successes += 1 if result.get("llm_success") else 0
        training_success_rate = training_successes / len(training_examples) if training_examples else 0.0
        llm_training_success_rate = llm_training_successes / len(training_examples) if training_examples else 0.0
        # Evaluate testing examples (possibly parallel)
        if enable_parallel:
            tasks = []
            with ThreadPoolExecutor() as ex:
                for i, example in enumerate(testing_examples):
                    tasks.append(ex.submit(
                        lambda idx, ex_in, ex_out: (idx, evaluate_example(
                            llm,
                            main_code,
                            transformation_steps,
                            ex_in,
                            ex_out,
                            enable_code_predict=enable_code_predict,
                            enable_llm_predict=enable_llm_predict,
                        )), i, example["input"], example.get("output", None)))

                for future in tqdm(as_completed([t for t in tasks]), total=len(tasks), desc="Testing eval"):
                    try:
                        idx, result = future.result()
                    except Exception as e:
                        idx = -1
                        result = {
                            "input": None,
                            "expected_output": None,
                            "predicted_output": None,
                            "matching_size": False,
                            "overlap_percentage": 0.0,
                            "error_message": str(e),
                            "code_success": False,
                            "llm_predicted_output": None,
                            "llm_matching_size": False,
                            "llm_overlap_percentage": 0.0,
                            "llm_error_message": str(e),
                            "llm_success": False,
                        }

                    result["example_index"] = idx
                    testing_results.append(result)
                    testing_successes += 1 if result.get("code_success") else 0
                    llm_testing_successes += 1 if result.get("llm_success") else 0
        else:
            for i, example in enumerate(testing_examples):
                input_grid = example["input"]
                expected_output = example.get("output", None)
                result = evaluate_example(
                    llm,
                    main_code,
                    transformation_steps,
                    input_grid,
                    expected_output,
                    enable_code_predict=enable_code_predict,
                    enable_llm_predict=enable_llm_predict,
                )
                result["example_index"] = i
                testing_results.append(result)
                testing_successes += 1 if result.get("code_success") else 0
                llm_testing_successes += 1 if result.get("llm_success") else 0

        # Update state
        solution["training_results"] = training_results
        solution["training_success_rate"] = training_success_rate
        solution["llm_training_success_rate"] = llm_training_success_rate
        solution["testing_results"] = testing_results
        solution["testing_success_rate"] = testing_successes / len(testing_examples) if testing_examples else 0.0   
        solution["llm_testing_success_rate"] = llm_testing_successes / len(testing_examples) if testing_examples else 0.0
    return new_state

def should_continue_predicate(state: AgentState) -> str:
    """
    Determine if the workflow should continue or end.

    Returns:
        "refine" if we should refine based on failures
        "end" if we should end the workflow
    """
    success_rate = state.get("success_rate", 0.0)
    attempt_number = state.get("attempt_number", 1)
    max_attempts = state.get("max_attempts", 5)
    current_solution = state.get("current_solution")

    # If no solution, end
    if not current_solution:
        return "refine"

    # If perfect success, end
    if success_rate >= 1.0:
        return "end"

    # If max attempts reached, end
    if attempt_number >= max_attempts:
        return "end"

    # If we have failures and haven't reached max attempts, refine
    if success_rate < 1.0:
        # Increment state attempt number
        return "refine"

    return "end"


def finalize_node(state: AgentState) -> AgentState:
    """
    Finalize the workflow and prepare the final output.
    """
    new_state = copy.deepcopy(state)
    return new_state