"""
Main ARC LangGraph Agent class.

This module provides the primary interface for running the LangGraph-based
ARC problem solver.
"""

import time
from typing import Dict, Any, Optional, List
import json

from .workflow import compile_workflow, create_initial_state
from .schema import AgentState, WorkflowOutput


class ARCLangGraphAgent:
    """
    LangGraph-based agent for solving ARC problems.
    
    This agent uses a workflow-based approach to iteratively generate,
    test, and refine solutions for ARC tasks.
    """
    
    def __init__(self, llm, max_attempts: int = 5, initial_helpers: List[Dict] = None):
        """
        Initialize the ARC LangGraph Agent.
        
        Args:
            llm: The language model to use for code generation (e.g., ChatOpenAI, ChatAnthropic)
            max_attempts: Maximum number of refinement attempts
            initial_helpers: Pre-loaded helper functions from persistent toolbox
        """
        self.max_attempts = max_attempts
        self.llm = llm
        self.initial_helpers = initial_helpers or []
        self.workflow = compile_workflow(llm)
        
    def solve_task(self, task_id: str, task_data: Dict[str, Any], max_attempts: int = 5) -> WorkflowOutput:
        """
        Solve a single ARC task using the LangGraph workflow.
        
        Args:
            task_id: The task identifier
            task_data: Task data containing train/test examples
            max_attempts: Maximum number of attempts (overrides instance setting)
            
        Returns:
            WorkflowOutput containing the results
        """
        start_time = time.time()
        
        # Create initial state with custom max_attempts and helper functions
        initial_state = create_initial_state(task_id, task_data, max_attempts)
        
        # Pre-load helper functions from toolbox
        if self.initial_helpers:
            from .schema import HelperFunction
            initial_state.helper_functions = [
                HelperFunction(
                    name=helper['name'],
                    description=helper['description'],
                    code=helper['code'],
                    usage_count=helper.get('usage_count', 0),
                    success_rate=helper.get('success_rate', 0.0)
                )
                for helper in self.initial_helpers
            ]
            print(f"  📚 Loaded {len(self.initial_helpers)} helper functions from toolbox")
        
        # Initialize final_state to avoid UnboundLocalError
        final_state = {
            "success_rate": 0.0,
            "current_solution": None,
            "attempt_number": 1,
            "final_prediction": None,
            "previous_solutions": [],
            "helper_functions": initial_state.helper_functions if hasattr(initial_state, 'helper_functions') else []
        }

        try:
            # Run the workflow
            workflow_result = self.workflow.invoke(initial_state)
            
            # Update final_state with workflow results
            if workflow_result:
                final_state.update(workflow_result)
            
            # Extract results
            success = final_state.get("success_rate", 0.0) >= 1.0
            best_solution = final_state.get("current_solution")
            best_success_rate = final_state.get("success_rate", 0.0)
            attempts_made = final_state.get("attempt_number", 1)
            final_prediction = final_state.get("final_prediction")
            all_solutions = final_state.get("previous_solutions", [])
            if best_solution:
                all_solutions.append(best_solution)
            
        except Exception as e:
            # Handle workflow execution errors
            success = False
            best_solution = None
            best_success_rate = 0.0
            attempts_made = 1
            final_prediction = None
            all_solutions = []
            print(f"Error running workflow for task {task_id}: {e}")
        
        execution_time = time.time() - start_time
        
        # Create output
        # Convert to output format with additional tracking
        output = WorkflowOutput(
            task_id=task_id,
            success=success,
            attempts=attempts_made,
            solutions=all_solutions,
            test_results=[],  # Will be populated by workflow
            execution_time=time.time() - start_time,
            reflection_history=[],  # Will be populated by workflow
            helper_functions=final_state.get("helper_functions", []),
            
            # Additional fields for toolbox integration
            helper_functions_used=[hf.name for hf in final_state.get("helper_functions", []) if getattr(hf, 'usage_count', 0) > 0],
            new_helper_functions=[],  # Will be populated by helper extraction
            code=best_solution.get('main_code', '') if best_solution else '',
            test_success=success
        )
        
        return output
    
    def solve_multiple_tasks(self, tasks: Dict[str, Dict[str, Any]]) -> Dict[str, WorkflowOutput]:
        """
        Solve multiple ARC tasks.
        
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
            success_rate = result["best_success_rate"]
            attempts = result["attempts_made"]
            time_taken = result["execution_time"]
            
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
        final_solution = result.get("final_solution")
        if final_solution:
            return final_solution.get("main_code")
        return None
    
    def get_helper_functions(self, result: WorkflowOutput) -> List[Dict[str, Any]]:
        """
        Extract the helper functions from a workflow result.
        
        Args:
            result: WorkflowOutput from solve_task
            
        Returns:
            List of helper function definitions
        """
        final_solution = result.get("final_solution")
        if final_solution:
            return final_solution.get("helper_functions", [])
        return []
    
    def export_solution_to_json(self, result: WorkflowOutput) -> Dict[str, Any]:
        """
        Export the solution in a format compatible with the existing ARC solver.
        
        Args:
            result: WorkflowOutput from solve_task
            
        Returns:
            Dictionary in ARC solver JSON format
        """
        final_solution = result.get("final_solution")
        
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
        from .nodes import execute_transformation_code, calculate_grid_overlap
        
        final_solution = result.get("final_solution")
        if not final_solution:
            return {"error": "No solution to test"}
        
        training_examples = task_data.get("train", [])
        test_results = []
        
        for i, example in enumerate(training_examples):
            input_grid = example["input"]
            expected_output = example["output"]
            
            try:
                predicted_output = execute_transformation_code(
                    final_solution["main_code"],
                    input_grid,
                    final_solution["helper_functions"]
                )
                
                if predicted_output is not None:
                    overlap = calculate_grid_overlap(predicted_output, expected_output)
                    success = overlap >= 100.0
                else:
                    overlap = 0.0
                    success = False
                    predicted_output = None
                
                test_results.append({
                    "example_index": i,
                    "success": success,
                    "overlap_percentage": overlap,
                    "predicted_output": predicted_output,
                    "expected_output": expected_output
                })
                
            except Exception as e:
                test_results.append({
                    "example_index": i,
                    "success": False,
                    "overlap_percentage": 0.0,
                    "predicted_output": None,
                    "expected_output": expected_output,
                    "error": str(e)
                })
        
        # Calculate overall statistics
        successful_tests = sum(1 for r in test_results if r["success"])
        total_tests = len(test_results)
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        return {
            "test_results": test_results,
            "overall_success_rate": overall_success_rate,
            "successful_examples": successful_tests,
            "total_examples": total_tests
        }