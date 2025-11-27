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

class ExampleResult(TypedDict):
    """Result of testing code on a single training example."""
    example_index: int
    input: List[List[int]]
    expected_output: List[List[int]]
    predicted_output: Optional[List[List[int]]]
    matching_size: bool
    overlap_percentage: float
    error_message: Optional[str]
    code_success: bool
    llm_predicted_output: Optional[List[List[int]]]
    llm_matching_size: Optional[bool]
    llm_overlap_percentage: Optional[float]
    llm_error_message: Optional[str]
    llm_success: bool


class CodeSolution(TypedDict):
    """Represents a complete code solution."""
    main_code: str
    reasoning_trace: str  # Full reasoning analysis of patterns. Maybe should just do some summary.
    step_by_step_transformation: List[str]  # Clear transformation steps

    evaluated: bool
    training_results: List[ExampleResult]
    training_success_rate: float
    training_overlap_average: float
    training_error_rate: float
    llm_training_success_rate: float
    llm_training_overlap_average: float
    testing_results: List[ExampleResult]
    testing_success_rate: float
    testing_overlap_average: float
    testing_error_rate: float
    llm_testing_success_rate: float
    llm_testing_overlap_average: float


class SolutionGeneration(TypedDict):
    """Represents a generation of solutions refined"""    
    generation: int
    solutions_list: List[CodeSolution]
    average_training_success_rate: float
    average_training_overlap_score: float
    average_training_error_rate: float
    average_testing_success_rate: float
    average_testing_overlap_score: float
    average_testing_error_rate: float

class AgentState(TypedDict):
    """Main state for the ARC LangGraph Agent workflow."""
    
    # Task information (immutable)
    task_id: Annotated[str, take_first]
    task_data: Annotated[Dict[str, Any], immutable_dict]

    # Runtime visual flag: prefer the latest provided value (allow agent to set)
    enable_visual_cue: Annotated[bool, take_latest]
    
    # Generated solutions
    seed_solutions_list: Optional[List[CodeSolution]]
    fused_solutions_list: Optional[List[List[CodeSolution]]]
    mutated_solutions_list: Optional[List[List[CodeSolution]]]
    solutions_list: Optional[List[CodeSolution]]

    # Evolulionary test time compute metrics
    current_loop: int
    num_initial_solutions: Annotated[Optional[int], take_latest]
    num_loops: Annotated[Optional[int], take_latest]
    num_seed_solutions: Annotated[Optional[int], take_latest]
    num_refinements: Annotated[Optional[int], take_latest]
    num_solutions_per_refinement: Annotated[Optional[int], take_latest]
    num_fusions: Annotated[Optional[int], take_latest]
    num_solutions_per_fusion: Annotated[Optional[int], take_latest]

    # Generations of mutated solutions
    current_generation: Annotated[Optional[int], take_latest]
    max_generations: Annotated[Optional[int], take_latest]
    generations: Annotated[Optional[List[SolutionGeneration]], take_latest]
    
    # Runtime flags propagated from agent instance
    enable_code_predict: Annotated[bool, take_latest]
    enable_llm_predict: Annotated[bool, take_latest]
    enable_parallel_eval: Annotated[bool, take_latest]
    
    # Additional metadata
    metadata: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]  # Merge metadata dicts


class WorkflowOutput(TypedDict):
    """Final output from the ARC agent workflow."""
    task_id: str
    workflow_completed: bool
    solutions_list: List[CodeSolution]
    highest_solution_index: int
    highest_training_solution_priority_score: float
    highest_training_solution_overlap_score: float
    num_loops: int
    execution_time: float


# Type aliases for common data structures
Grid = List[List[int]]
GridPair = Dict[str, Grid]  # {'input': grid, 'output': grid}
