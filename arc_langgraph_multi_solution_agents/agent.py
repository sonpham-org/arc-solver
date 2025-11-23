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
    finalize_node,
    evolve_code_node
)

from .schema import AgentState, WorkflowOutput


def _all_examples_success(results):
    if not results:
        return False
    return all(bool(r.get('code_success', False)) for r in results)


def finalize_pred(state):
    """Predicate: True when we should finalize.

    Finalize when either:
    - Any `CodeSolution` in `solutions_list` has ALL training examples
      with `code_success == True` (we only check training examples), OR
    - We've reached or exceeded `max_generations`.
    """
    sols = state.get('solutions_list') or []

    # Check for a perfect training-only solution
    perfect_training = any(
        _all_examples_success(sol.get('training_results', []))
        for sol in sols
    )

    cur = int(state.get('current_generation', 0))
    mx = int(state.get('max_generations', 0))

    return perfect_training or (cur >= mx)

def out_of_loops(state):
    """Predicate: True when we've exhausted all loops."""
    cur_loop = int(state.get('current_loop', 0))
    max_loops = int(state.get('num_loops', 0))
    return cur_loop >= max_loops

def create_arc_workflow(llm, code_llm) -> StateGraph:
    """
    Create the LangGraph workflow for ARC problem solving.
    """
    workflow = StateGraph(AgentState)

    # Add nodes with LLM access
    workflow.add_node("generate_code", lambda state: generate_code_node(state, llm, code_llm))

    # Add test and refine nodes. Keep `test_code_node` as the node body
    # and use conditional edges (predicates) to decide the next step.
    workflow.add_node("test_code", lambda state: test_code_node(state, llm, code_llm))
    workflow.add_node("evolve_code", lambda state: evolve_code_node(state, llm, code_llm))
    workflow.add_node("finalize", finalize_node)

    # Add edges. Use conditional edges out of `test_code` so the graph can
    # choose the next node based on the current `state`.
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "test_code")

    # Decision node approach: route from `test_code` to a dedicated
    # `decide` node which returns the next node name. This mirrors the
    # `builder.add_conditional_edges` pattern in your example.
    workflow.add_edge("test_code", "decide")
    workflow.add_node("decide", lambda state: state)

    # decide_next returns the literal name of the next node; mapping
    # instructs the graph how to map those strings to nodes.
    def decide_next(state):
        return "finalize" if finalize_pred(state) or out_of_loops(state) else "evolve_code"

    workflow.add_conditional_edges("decide", decide_next, {
        "finalize": "finalize", 
        "evolve_code": "evolve_code"})
    workflow.add_edge("evolve_code", "test_code")
    workflow.add_edge("finalize", END)

    return workflow


def create_initial_state(task_id: str,
                        task_data: Dict[str, Any],
                        agent) -> AgentState:
    """
    Create the initial state for the workflow.
    """
    return {
        "task_id": task_id,
        "task_data": task_data,
        "solutions_list": [],
        "metadata": {},
        "current_loop": 0,
        "num_initial_solutions": agent.num_initial_solutions,
        "num_loops": agent.num_loops,
        "num_seed_solutions": agent.num_seed_solutions,
        "num_refinements": agent.num_refinements,
        "num_solutions_per_refinement": agent.num_solutions_per_refinement,
        "num_fusions": agent.num_fusions,
        "num_solutions_per_fusion": agent.num_solutions_per_fusion,
        "enable_visual_cue": agent.enable_visual_cue,
    }

class MultiSolutionARCLangGraphAgent:
    """
    LangGraph-based agent for solving ARC problems.
    
    This agent generates multiple solution attempts for each task,
    storing all solutions in the workflow state.
    """

    def __init__(self, llm, code_llm,
                 num_initial_solutions: int = 10,
                 num_loops: int = 3,
                 num_seed_solutions: int = 10,
                 num_refinements: int = 5,
                 num_solutions_per_refinement: int = 3,
                 num_fusions: int = 5,
                 num_solutions_per_fusion: int = 3,
                 enable_code_predict: bool = True,
                 enable_llm_predict: bool = False,
                 enable_parallel_eval: bool = False,
                 enable_visual_cue: bool = False,
                 max_generations: int = 3):
        """
        Initialize the ARC LangGraph Agent.
        
        Args:
            llm: The language model to use for reasoning and general tasks (e.g., ChatOpenAI, ChatAnthropic)
            code_llm: The language model to use for code-related tasks (e.g., code generation, extraction)
            enable_code_predict: Whether to enable code-predicted outputs during testing
            enable_llm_predict: Whether to enable LLM-predicted outputs during testing
        """
        # LLM instances
        self.llm = llm
        self.code_llm = code_llm

        # Evolutionary parameters
        self.num_initial_solutions = num_initial_solutions
        self.num_loops = num_loops
        self.num_seed_solutions = num_seed_solutions
        self.num_refinements = num_refinements
        self.num_solutions_per_refinement = num_solutions_per_refinement
        self.num_fusions = num_fusions
        self.num_solutions_per_fusion = num_solutions_per_fusion

        # Code enabling flags
        self.enable_code_predict = enable_code_predict  # Whether to enable code-predicted outputs during testing
        self.enable_llm_predict = enable_llm_predict  # Whether to enable LLM-predicted outputs during testing
        self.enable_parallel_eval = enable_parallel_eval  # Whether to parallelize evaluation with threads and tqdm
        self.enable_visual_cue = enable_visual_cue
        self.max_generations = int(max_generations)

        # Create arc workflow
        self.workflow = create_arc_workflow(llm, code_llm).compile()

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
        
        # Assign solution to test output
        for i, test_example in enumerate(task_data.get("test", [])):
            if task_solution and i < len(task_solution):
                test_example['output'] = task_solution[i]
        
        # Create initial state with custom max_attempts and helper functions

        initial_state = create_initial_state(task_id, task_data, self)
        
        # Propagate runtime flags into the workflow state so nodes can read them
        initial_state['enable_code_predict'] = bool(self.enable_code_predict)
        initial_state['enable_llm_predict'] = bool(self.enable_llm_predict)

        # New: allow nodes to enable parallelized evaluation
        initial_state['enable_parallel_eval'] = bool(self.enable_parallel_eval)

        # Generation counters for refinement loop
        initial_state['current_generation'] = int(initial_state.get('current_generation', 0))
        initial_state['max_generations'] = int(getattr(self, 'max_generations', 0))

        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        execution_time = time.time() - start_time

        # Build WorkflowOutput-compatible dict (matches schema.WorkflowOutput)
        solutions_list = final_state.get("solutions_list", [])
        best_idx = None
        best_priority = -1.0
        best_overlap = -1.0
        best_testing_priority = -1.0
        best_testing_overlap = -1.0
        def compute_scores(entries):
            if not entries:
                return 0.0, 0.0
            total = len(entries)
            num_correct = sum(1 for e in entries if bool(e.get('code_success', False)))
            priority = (num_correct / total) * 100.0 if total > 0 else 0.0
            matching_entries = [e for e in entries if e.get('matching_size')]
            if matching_entries:
                overlaps = [float(e.get('overlap_percentage', 0.0)) for e in matching_entries]
                average = sum(overlaps) / len(overlaps)
            else:
                average = 0.0
            return priority, average
        for idx, solution in enumerate(solutions_list):

            s_train = solution.get('training_results') or []
            s_test = solution.get('testing_results') or []
            s_tpriority, s_toverlap = compute_scores(s_train)
            s_tepriority, s_teoverlap = compute_scores(s_test)

            rank_priority = s_tpriority
            rank_overlap = s_toverlap

            if (rank_priority > best_priority) or (rank_priority == best_priority and rank_overlap > best_overlap):
                best_idx = idx
                best_priority = rank_priority
                best_overlap = rank_overlap
                best_testing_priority = s_tepriority
                best_testing_overlap = s_teoverlap
        output: WorkflowOutput = {
            "task_id": task_id,
            "workflow_completed": True,
            "solutions_list": final_state.get("solutions_list", []),
            "highest_solution_index": best_idx,
            "highest_training_solution_priority_score": best_priority,
            "highest_training_solution_overlap_score": best_overlap,
            "highest_testing_solution_priority_score": best_testing_priority,
            "highest_testing_solution_overlap_score": best_testing_overlap,
            "num_loops": final_state.get("current_loop", 0),
            "execution_time": execution_time,
        }
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