"""
Schema definitions for the ARC LangGraph Agent.

This module defines the TypedDict structures used to maintain state 
throughout the LangGraph workflow for ARC problem solving.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union, Annotated
from langchain_core.messages import BaseMessage

# Define a reducer that takes the first value (for immutable fields)
def take_first(left, right):
    """Reducer that takes the first (left) value, ignoring updates."""
    return left if left is not None else right


def immutable_dict(left, right):
    """Reducer for immutable dict fields - preserves first non-empty value."""
    # If left is a non-empty dict, keep it
    if left and isinstance(left, dict) and left != {}:
        return left
    # Otherwise, take right if it's a non-empty dict
    if right and isinstance(right, dict) and right != {}:
        return right
    # Fallback: prefer left over right
    return left if left is not None else right


def take_latest(left, right):
    """Reducer that takes the latest (right) value."""
    return right if right is not None else left


class ARCTask(TypedDict):
    """Represents an ARC task with training and test examples."""
    train: List[Dict[str, List[List[int]]]]  # List of {'input': grid, 'output': grid}
    test: List[Dict[str, List[List[int]]]]   # List of {'input': grid}


class HelperFunction(TypedDict):
    """Represents a helper function with its code and description."""
    name: str
    code: str
    description: str
    parameters: Optional[List[str]]
    usage_count: Optional[int]  # For tracking usage from toolbox
    success_rate: Optional[float]  # For tracking performance from toolbox


class ExampleResult(TypedDict):
    """Result of testing code on a single training example."""
    example_index: int
    success: bool
    input: List[List[int]]
    expected_output: List[List[int]]
    predicted_output: Optional[List[List[int]]]
    matching_size: bool
    overlap_percentage: float
    error_message: Optional[str]
    llm_predicted_output: Optional[List[List[int]]]
    llm_matching_size: Optional[bool]
    llm_overlap_percentage: Optional[float]
    llm_error_message: Optional[str]


class CodeSolution(TypedDict):
    """Represents a complete code solution with helper functions."""
    main_code: str
    helper_functions: Dict[str, HelperFunction]
    reasoning_trace: str  # Full reasoning analysis of patterns. Maybe should just do some summary.
    step_by_step_transformation: List[str]  # Clear transformation steps


class AgentState(TypedDict):
    """Main state for the ARC LangGraph Agent workflow."""
    
    # Task information (immutable)
    task_id: Annotated[str, take_first]
    task_data: Annotated[Dict[str, Any], immutable_dict]

    # Current attempt information
    attempt_number: Annotated[int, take_latest]
    max_attempts: Annotated[int, take_latest]
    
    # Generated solutions
    current_solution: Annotated[Optional[CodeSolution], take_latest]
    previous_solutions: Annotated[List[CodeSolution], lambda x, y: x + y]  # Append new solutions
    
    # Test results from training and testing examples 
    training_results: Annotated[List[ExampleResult], lambda x, y: x + y]  # Append new test results
    training_success_rate: Annotated[float, take_latest]
    final_predictions: Annotated[List[List[List[int]]], take_latest]  # Prediction for the test cases
    previous_run_training_results: Annotated[List[List[ExampleResult]], lambda x, y: x + y]  # Append new test results 
    
    # Available helper functions (accumulated across attempts)
    available_helpers: Annotated[Dict[str, HelperFunction], lambda x, y: {**x, **y}]  # Append new helpers
    new_helpers: Annotated[Dict[str, HelperFunction], lambda x, y: {**x, **y}]  # Append extracted helpers

    # Workflow control
    should_continue: Annotated[bool, take_latest]
    
    # Additional metadata
    metadata: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]  # Merge metadata dicts


class WorkflowOutput(TypedDict):
    """Final output from the ARC agent workflow."""
    task_id: str
    workflow_completed: bool
    attempts: int
    solutions: List[CodeSolution]
    training_results: List[ExampleResult]
    training_success_rate: float
    testing_results: List[ExampleResult]
    testing_success_rate: float
    llm_testing_success_rate: float
    execution_time: float
    new_helpers: Dict[str,HelperFunction]


# Type aliases for common data structures
Grid = List[List[int]]
GridPair = Dict[str, Grid]  # {'input': grid, 'output': grid}