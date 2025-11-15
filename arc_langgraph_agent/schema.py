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


class TestResult(TypedDict):
    """Result of testing code on a single training example."""
    example_index: int
    success: bool
    predicted_output: Optional[List[List[int]]]
    expected_output: List[List[int]]
    overlap_percentage: float
    error_message: Optional[str]


class CodeSolution(TypedDict):
    """Represents a complete code solution with helper functions."""
    main_code: str
    helper_functions: List[HelperFunction]
    reasoning_trace: str  # Full reasoning analysis of patterns
    step_by_step_transformation: List[str]  # Clear transformation steps
    confidence_score: float


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
    
    # Test results from training examples
    test_results: Annotated[List[TestResult], lambda x, y: x + y]  # Append new test results
    success_rate: Annotated[float, take_latest]
    
    # Available helper functions (accumulated across attempts)
    available_helpers: Annotated[List[HelperFunction], lambda x, y: x + y]  # Append new helpers
    extracted_helpers: Annotated[List[HelperFunction], lambda x, y: x + y]  # Append extracted helpers
    global_helper_library: Annotated[List[HelperFunction], lambda x, y: x + y]  # Append to global library
    
    # Messages for LLM interaction
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]  # Append new messages
    
    # Workflow control
    should_continue: Annotated[bool, take_latest]
    final_prediction: Annotated[Optional[List[List[int]]], take_latest]
    
    # Error tracking
    errors: Annotated[List[str], lambda x, y: x + y]  # Append new errors
    reflection_history: Annotated[List[Dict[str, Any]], lambda x, y: x + y]  # Append reflections
    
    # Additional metadata
    metadata: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]  # Merge metadata dicts


class WorkflowOutput(TypedDict):
    """Final output from the ARC agent workflow."""
    task_id: str
    success: bool
    attempts: int
    solutions: List[CodeSolution]
    test_results: List[TestResult]
    execution_time: float
    reflection_history: List[Dict[str, Any]]
    helper_functions: List[HelperFunction]
    
    # Additional fields for toolbox integration
    helper_functions_used: List[str]  # Names of helper functions used
    new_helper_functions: List[Dict[str, Any]]  # New functions discovered
    code: str  # Generated code
    test_success: bool  # Whether test was successful


# Type aliases for common data structures
Grid = List[List[int]]
GridPair = Dict[str, Grid]  # {'input': grid, 'output': grid}