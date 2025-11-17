"""
Nodes for the ARC LangGraph Agent workflow.

This module should contain only the node functions. All helper and
action-related utilities have been moved to `actions.py`.
"""

import copy
from typing import List, Dict, Optional, Any

# Import schema and node-facing types
from .schema import AgentState, CodeSolution, ExampleResult, HelperFunction

# Import actions (helpers, prompts, execution, refinement)
from .actions import (
    extract_helper_functions,
    generate_python_transformation_code_with_reasoning,
    execute_transformation_code,
    calculate_grid_results,
    refine_solution_based_on_failures,
)


def extract_helpers_node(state: AgentState, llm) -> AgentState:
    """
    Extract potential useful helper functions from the generated solution
    and add them to the growing tool set.
    """
    current_solution = state.get("current_solution")
    if not current_solution:
        return state

    training_examples = state["task_data"]["train"]
    global_library = state.get("global_helper_library", [])

    # Extract new helper functions from the current code using the actions helper
    extracted_helpers = extract_helper_functions(
        llm,
        current_solution["main_code"],
        training_examples,
        global_library,
    )

    # Update state with new helpers
    new_state = copy.deepcopy(state)
    new_state["extracted_helpers"] = extracted_helpers

    # Add to available helpers for this task
    for h in extracted_helpers:
        func_name = h.get("name") if isinstance(h, dict) else str(h)
        # Add to new helpers if not already in available_helpers
        if func_name not in new_state.get("available_helpers", {}):
            new_state.setdefault("available_helpers", {})[func_name] = h
            new_state.setdefault("new_helpers", {})[func_name] = h

    # CRITICAL: Update current solution's helper_functions for execution
    if new_state.get("current_solution"):
        new_state["current_solution"]["helper_functions"] = extracted_helpers

    return new_state


def generate_code_node(state: AgentState, llm) -> AgentState:
    """
    Generate Python code to solve the ARC problem using reasoning-first approach.

    This node uses the available helper functions and analyzes the training
    examples to generate a solution using the provided language model.
    """
    task_data = state["task_data"]
    available_helpers = state.get("available_helpers", {})
    previous_solutions = state.get("previous_solutions", [])

    # Analyze the training examples (kept for node-level logging/analysis)
    training_examples = task_data["train"]

    # Generate code using the reasoning-first approach from actions
    main_code, reasoning_trace, transformation_steps = generate_python_transformation_code_with_reasoning(
        llm,
        training_examples,
        available_helpers,
        previous_solutions,
    )

    # Create the solution with reasoning traces
    solution = {
        "main_code": main_code,
        "helper_functions": [],
        "reasoning_trace": reasoning_trace,
        "step_by_step_transformation": transformation_steps,
    }

    # Update state
    new_state = copy.deepcopy(state)
    new_state["current_solution"] = solution
    new_state["should_continue"] = True
    new_state["attempt_number"] = state["attempt_number"] + 1

    return new_state


def test_code_node(state: AgentState) -> AgentState:
    """
    Test the generated code against training examples.

    This node executes the current solution on all training examples
    and calculates the success rate.
    """
    # Deep copy the new state, and shove in the training results to previous_training_results
    # And then reset training results
    new_state = copy.deepcopy(state)
    if "training_results" in state:
        previous_results = state["training_results"]
        if "previous_run_training_results" not in new_state:
            new_state["previous_run_training_results"] = []
        new_state["previous_run_training_results"].append(previous_results)
        new_state["training_results"] = []
        new_state["training_success_rate"] = 0.0
        new_state["should_continue"] = False

    task_data = state["task_data"]
    current_solution = state.get("current_solution")

    if not current_solution:
        return new_state

    training_examples = task_data["train"]
    training_results = []
    successful_tests = 0

    # Prepare the execution environment
    helper_functions = current_solution.get("helper_functions", [])
    main_code = current_solution.get("main_code", "")

    for i, example in enumerate(training_examples):
        input_grid = example["input"]
        expected_output = example["output"]

        # Execute the code (from actions)
        predicted_output, error = execute_transformation_code(main_code, input_grid, helper_functions)

        if error is None and predicted_output is not None:
            # Calculate overlap percentage
            matching_size, overlap_percentage = calculate_grid_results(predicted_output, expected_output)
            success = matching_size and overlap_percentage >= 100.0  # Perfect match required

            training_result = {
                "example_index": i,
                "success": success,
                "input": input_grid,
                "predicted_output": predicted_output,
                "expected_output": expected_output,
                "matching_size": matching_size,
                "overlap_percentage": overlap_percentage,
                "error_message": None,
            }

            if success:
                successful_tests += 1
        else:
            # Execution failed or returned None
            training_result = {
                "example_index": i,
                "success": False,
                "input": input_grid,
                "predicted_output": None,
                "expected_output": expected_output,
                "matching_size": False,
                "overlap_percentage": 0.0,
                "error_message": error or "Code execution returned None",
            }
        training_results.append(training_result)
    success_rate = successful_tests / len(training_examples) if training_examples else 0.0

    # Update state
    new_state = copy.deepcopy(state)
    new_state["training_results"] = training_results
    new_state["training_success_rate"] = success_rate
    return new_state


def refinement_node(state: AgentState, llm) -> AgentState:
    """
    Refine the solution based on test failures.

    This node analyzes failed test cases and attempts to improve the solution.
    """
    training_results = state.get("training_results", [])
    current_solution = state.get("current_solution")
    task_data = state["task_data"]
    attempt_number = state.get("attempt_number", 1)
    max_attempts = state.get("max_attempts", 5)

    # Analyze failures
    failed_tests = [t for t in training_results if not t.get("success")]

    # Generate refinement based on failures using LLM with reflection (from actions)
    reflection_history = state.get("reflection_history", [])
    refined_solution, reflection_record = refine_solution_based_on_failures(
        llm, current_solution, training_results, task_data["train"], reflection_history
    )

    # Update state for next attempt
    new_state = copy.deepcopy(state)
    new_state.setdefault("previous_solutions", []).append(current_solution)
    new_state["current_solution"] = refined_solution
    new_state["attempt_number"] = attempt_number + 1
    new_state["should_continue"] = True

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
    current_solution = state.get("current_solution")
    task_data = state.get("task_data", {})

    # Generate final predictions for all test cases (may be multiple)
    final_predictions = []
    if current_solution and task_data.get("test"):
        helper_functions = current_solution.get("helper_functions", [])
        main_code = current_solution.get("main_code", "")

        for idx, test_case in enumerate(task_data["test"]):
            test_input = test_case.get("input")
            predicted = None
            try:
                predicted, error = execute_transformation_code(
                    main_code,
                    test_input,
                    helper_functions,
                )
                if error is not None:
                    state.setdefault("errors", []).append(f"Test case {idx}: failed to generate prediction: {error}")
                    predicted = None
            except Exception as e:
                # Log error but continue to next test case
                state.setdefault("errors", []).append(f"Test case {idx}: exception during prediction: {e}")
                predicted = None

            final_predictions.append(predicted)

    new_state = copy.deepcopy(state)
    # Store as plural to match AgentState schema: list of outputs for all test cases
    new_state["final_predictions"] = final_predictions

    # Capture the current list of available tool calls (helpers) so callers
    # can inspect whether the toolset is growing across workflow steps.
    available_helpers = new_state.get("available_helpers", []) or []
    global_helpers = new_state.get("global_helper_library", []) or []

    available_tool_calls = []
    for h in available_helpers:
        if isinstance(h, dict) and "name" in h:
            available_tool_calls.append(h["name"])
        else:
            available_tool_calls.append(str(h))

    # Include global helpers as well (deduplicated)
    for h in global_helpers:
        name = h.get("name") if isinstance(h, dict) else str(h)
        if name not in available_tool_calls:
            available_tool_calls.append(name)

    # Deduplicate available_tool_calls
    deduped_tool_calls = list(dict.fromkeys(available_tool_calls))
    # Store snapshot in state and print for immediate visibility when running
    new_state["available_tool_calls_snapshot"] = deduped_tool_calls
    print("Num available tool calls:", len(deduped_tool_calls))
    print("Num unique tool calls:", len(set(deduped_tool_calls)))

    new_state["should_continue"] = False

    return new_state