"""
Main ARC LangGraph Agent class.

This module provides the primary interface for running the LangGraph-based
ARC problem solver.
"""

import time
from typing import Dict, Any, Optional, List
import json

from langgraph.graph import StateGraph, START, END

# Import our nodes (moved here from workflow.py for convenience)
from .nodes import (
    generate_code_node,
    test_code_node,
    refinement_node,
    extract_helpers_node,
    should_continue_predicate,
    finalize_node
)

from .schema import AgentState, WorkflowOutput


def create_arc_workflow(llm) -> StateGraph:
    """
    Create the LangGraph workflow for ARC problem solving.
    """
    workflow = StateGraph(AgentState)

    # Add nodes with LLM access
    workflow.add_node("generate_code", lambda state: generate_code_node(state, llm))
    workflow.add_node("test_code", test_code_node)
    workflow.add_node("extract_helpers", lambda state: extract_helpers_node(state, llm))
    workflow.add_node("refine_code", lambda state: refinement_node(state, llm))
    workflow.add_node("finalize", finalize_node)

    # Add edges
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "extract_helpers")
    workflow.add_edge("extract_helpers", "test_code")

    workflow.add_conditional_edges(
        "test_code",
        should_continue_predicate, 
        # This could be a little bit more sophisticated and human here.
        # But AgentState needs more stuffs in it.
        {
            "refine": "refine_code",
            "end": "finalize"
        }
    )

    workflow.add_edge("refine_code", "test_code")
    workflow.add_edge("finalize", END)

    return workflow


def create_initial_state(task_id: str,
                        task_data: Dict[str, Any],
                        max_attempts: int = 5) -> AgentState:
    """
    Create the initial state for the workflow.
    """
    return {
        "task_id": task_id,
        "task_data": task_data,
        "attempt_number": 0,
        "max_attempts": max_attempts,
        "current_solution": None,
        "previous_solutions": [],
        "training_results": [],
        "training_success_rate": 0.0,
        "testing_results": [],
        "testing_success_rate": 0.0,
        "available_helpers": {},
        "new_helpers": {},
        "should_continue": True,
        "metadata": {}
    }


def compile_workflow(llm) -> Any:
    """Compile the workflow for execution."""
    workflow = create_arc_workflow(llm)
    return workflow.compile()


class ARCLangGraphAgent:
    """
    LangGraph-based agent for solving ARC problems.
    
    This agent uses a workflow-based approach to iteratively generate,
    test, and refine solutions for ARC tasks.
    """

    def __init__(self, llm, max_attempts: int = 5, available_helpers: Dict[str, Dict] = None):
        """
        Initialize the ARC LangGraph Agent.
        
        Args:
            llm: The language model to use for code generation (e.g., ChatOpenAI, ChatAnthropic)
            max_attempts: Maximum number of refinement attempts
            initial_helpers: Pre-loaded helper functions from persistent toolbox
        """
        self.max_attempts = max_attempts
        self.llm = llm
        
        # Store pre-loaded helpers under `available_helpers`
        self.available_helpers = available_helpers or {}

        # Preload and normalize helper functions at initialization so
        # `solve_task` can simply copy them into the workflow initial state.
        self._load_default_helpers()
        self.workflow = compile_workflow(llm)

    def update_available_helpers(self, new_helpers: Dict[str, Dict]) -> None:
        """
        Update the available helper functions with new ones.
        
        Args:
            new_helpers: Dictionary of new helper functions to add
        """
        self.available_helpers.update(new_helpers)

    def _load_default_helpers(self) -> None:
        """
        Load and normalize default helper functions from `.tools.FUNCTION_MAP`
        into `self.available_helpers`. Extracts source, docstring, parameter
        names and initializes usage metrics.
        """
        from inspect import getsource, signature

        # Only load defaults when no helpers were provided at init
        if self.available_helpers:
            return

        try:
            from .tools import FUNCTION_MAP
        except Exception:
            FUNCTION_MAP = {}

        processed = {}
        for name, func in FUNCTION_MAP.items():
            try:
                src = getsource(func)
            except Exception:
                src = ''

            desc = func.__doc__.strip() if getattr(func, '__doc__', None) else ''

            try:
                params = list(signature(func).parameters.keys())
            except Exception:
                params = []

            processed[name] = {
                'name': name,
                'code': src,
                'description': desc,
                'parameters': params,
                'usage_count': 0,
                'success_rate': 0.0
            }

        self.available_helpers = processed
        print(f"  📚 Loaded {len(self.available_helpers)} default helper functions into available_helpers")
        
    def solve_task(self, task_id: str, task_data: Dict[str, Any], task_solution: Optional[List[List[List[int]]]] = None, max_attempts: int = 5) -> WorkflowOutput:
        """
        Solve a single ARC task using the LangGraph workflow.
        
        Args:
            task_id: The task identifier
            task_data: Task data containing train/test examples
            task_solution: Contains the expected outputs for the test examples, if available
            max_attempts: Maximum number of attempts (overrides instance setting)
            
        Returns:
            WorkflowOutput containing the results
        """
        start_time = time.time()
        
        # Create initial state with custom max_attempts and helper functions
        initial_state = create_initial_state(task_id, task_data, max_attempts)
        
        # Copy the preloaded helper definitions into the workflow initial state
        initial_state['available_helpers'] = dict(self.available_helpers)
        
        # Initialize final_state to avoid UnboundLocalError
        workflow_results = {
            "task_id": task_id,
            "workflow_completed": False,
            "attempts": 0,
            "solutions": [],
            "training_results": [],
            "training_success_rate": 0.0,
            "testing_results": [],
            "testing_success_rate": 0.0,
            "execution_time": 0.0,
            "new_helpers": {}
        }

        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        execution_time = time.time() - start_time

        # Build WorkflowOutput-compatible dict (matches schema.WorkflowOutput)
        output: WorkflowOutput = {
            "task_id": task_id,
            "workflow_completed": True,
            "attempts": final_state["attempt_number"],
            "solutions": final_state["previous_solutions"] + ([final_state["current_solution"]] if final_state["current_solution"] else []),
            "training_results": final_state["training_results"],
            "training_success_rate": final_state["training_success_rate"],
            "execution_time": execution_time,
            "new_helpers": workflow_results.get("new_helpers", {})
        }

        # At this point, final_predictions needed to be compared with task_solution to create
        # testing_results and testing_success_rate and add to output.
        if task_solution is not None:
            from .nodes import calculate_grid_results, execute_transformation_code

            testing_results = []
            successful = 0
            total = len(task_solution)

            for i, expected_output in enumerate(task_solution):
                input_grid = task_data["test"][i]["input"]

                predicted_output, error = execute_transformation_code(
                    final_state.get("current_solution", {}).get("main_code", ""),
                    input_grid,
                    final_state.get("current_solution", {}).get("helper_functions", {})
                )

                matching_size = False
                overlap = 0.0
                success_flag = False
                error_message = None

                if error is None and predicted_output is not None:
                    matching_size, overlap = calculate_grid_results(predicted_output, expected_output)
                    success_flag = matching_size and overlap >= 100.0
                else:
                    if error is not None:
                        error_message = str(error)

                if success_flag:
                    successful += 1

                testing_results.append({
                    "example_index": i,
                    "success": success_flag,
                    "input": input_grid,
                    "predicted_output": predicted_output,
                    "expected_output": expected_output,
                    "matching_size": matching_size,
                    "overlap_percentage": float(overlap),
                    "error_message": error_message
                })

            testing_success_rate = successful / total if total > 0 else 0.0

            output["testing_results"] = testing_results
            output["testing_success_rate"] = testing_success_rate

        return output
    
    def solve_multiple_tasks(self, tasks: Dict[str, Dict[str, Any]]) -> Dict[str, WorkflowOutput]:
        """
        Solve multiple ARC tasks. Which bascially calls solve_task in a loop or in parallel.
        
        Args:
            tasks: Dictionary mapping task_id to task_data
            
        Returns:
            Dictionary mapping task_id to WorkflowOutput
        """
        results = {}
        
        for task_id, task_data in tasks.items():
            print(f"Solving task: {task_id}")
            result = self.solve_task(task_id, task_data)
            results[task_id] = result
            
            # Print summary
            success_rate = result.get("training_success_rate", 0.0)
            attempts = result.get("attempts", 0)
            time_taken = result.get("execution_time", 0.0)
            
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Attempts: {attempts}/{self.max_attempts}")
            print(f"  Time: {time_taken:.2f}s")
            print()
        
        return results
    
    def get_solution_code(self, result: WorkflowOutput) -> Optional[str]:
        """
        Extract the main solution code from a workflow result.
        
        Args:
            result: WorkflowOutput from solve_task
            
        Returns:
            The main transformation code, or None if no solution
        """
        sols = result.get("solutions", [])
        if not sols:
            return None
        final_solution = sols[-1]
        return final_solution.get("main_code")
    
    def get_helper_functions(self, result: WorkflowOutput) -> List[Dict[str, Any]]:
        """
        Extract the helper functions from a workflow result.
        
        Args:
            result: WorkflowOutput from solve_task
            
        Returns:
            List of helper function definitions
        """
        sols = result.get("solutions", [])
        if not sols:
            return []
        final_solution = sols[-1]
        return final_solution.get("helper_functions", {})
    
    def export_solution_to_json(self, result: WorkflowOutput) -> Dict[str, Any]:
        """
        Export the solution in a format compatible with the existing ARC solver.
        
        Args:
            result: WorkflowOutput from solve_task
            
        Returns:
            Dictionary in ARC solver JSON format
        """
        sols = result.get("solutions", [])
        final_solution = sols[-1] if sols else None

        if not final_solution:
            return {
                "helper_python_functions": [],
                "step_by_step_transformations": [],
                "python_code": [
                    "def transform(grid):",
                    "    return grid"
                ]
            }
        
        # Convert to existing format
        helper_functions = []
        for helper in final_solution.get("helper_functions", []):
            helper_functions.append(helper["code"])
        
        # Convert main code to list of lines
        main_code_lines = final_solution["main_code"].split('\n')
        
        # Convert step-by-step description
        steps = []
        descriptions = final_solution.get("step_by_step_description", [])
        for i, desc in enumerate(descriptions, 1):
            steps.append({
                "step_number": i,
                "description": [desc],
                "pseudo_code": []
            })
        
        return {
            "helper_python_functions": helper_functions,
            "step_by_step_transformations": steps,
            "python_code": main_code_lines
        }
    
    def test_solution_on_examples(self, result: WorkflowOutput, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test the solution on training examples and return detailed results.
        
        Args:
            result: WorkflowOutput from solve_task
            task_data: Original task data
            
        Returns:
            Dictionary with detailed test results
        """
        # Return example_results (both train and test) matching schema.ExampleResult
        from .nodes import execute_transformation_code, calculate_grid_results

        sols = result.get("solutions", [])
        final_solution = sols[-1] if sols else None
        if not final_solution:
            return {"error": "No solution to test"}

        training_examples = task_data.get("train", [])
        testing_examples = task_data.get("test", [])

        def _single_build(examples, include_expected: bool = True):
            example_results = []
            for i, example in enumerate(examples):
                input_grid = example.get("input")
                expected_output = example.get("output") if include_expected else []

                predicted_output = None
                overlap = 0.0
                success_flag = False
                matching_size = False
                error_message = None

                try:
                    predicted_output, error = execute_transformation_code(
                        final_solution.get("main_code", ""),
                        input_grid,
                        final_solution.get("helper_functions", {})
                    )

                    if error is None and predicted_output is not None and include_expected and expected_output:
                        matching_size, overlap = calculate_grid_results(predicted_output, expected_output)
                        success_flag = matching_size and overlap >= 100.0
                    else:
                        if error is not None:
                            error_message = str(error)
                        overlap = 0.0
                        success_flag = False
                        matching_size = False

                except Exception as e:
                    predicted_output = None
                    overlap = 0.0
                    success_flag = False
                    matching_size = False
                    error_message = str(e)

                example_results.append({
                    "example_index": i,
                    "success": success_flag,
                    "input": input_grid,
                    "predicted_output": predicted_output,
                    "expected_output": expected_output if expected_output is not None else [],
                    "matching_size": matching_size,
                    "overlap_percentage": float(overlap),
                    "error_message": error_message
                })

            successful = sum(1 for r in example_results if r["success"])
            total = len(example_results)
            overall = successful / total if total > 0 else 0.0
            return example_results, overall

        training_results, training_success_rate = _single_build(training_examples, include_expected=True)
        testing_results, testing_success_rate = _single_build(testing_examples, include_expected=False)

        return {
            "example_results": {
                "train": training_results,
                "test": testing_results
            },
            "training_success_rate": training_success_rate,
            "testing_success_rate": testing_success_rate,
            "successful_examples": sum(1 for r in training_results if r["success"]),
            "total_examples": len(training_results)
        }