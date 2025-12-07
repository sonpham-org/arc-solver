"""
Main ARC LangGraph Agent class.

This module provides the primary interface for running the LangGraph-based
ARC problem solver.
"""

import time
from typing import Dict, Any, Optional, List, cast
import json
import os

from langgraph.graph import StateGraph, START, END

# Import our nodes (moved here from workflow.py for convenience)
from .nodes import (
    generate_code_node,
    test_code_node,
    finalize_node,
    evolve_code_node,
    save_state_node
)

from .schema import AgentState, WorkflowOutput


def _all_examples_success(results):
    if not results:
        return False
    return all(bool(r.get('code_success', False)) for r in results)


def one_solution_succeeded(state):
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

    return perfect_training

def out_of_loops(state):
    """Predicate: True when we've exhausted all loops."""
    cur_loop = int(state.get('current_loop', 0))
    max_loops = int(state.get('num_loops', 0))
    print("Current loop", cur_loop, "Max loops", max_loops)
    return cur_loop >= max_loops


def decide_setup(state):
    """Top-level setup decision helper (routing only).

    This function no longer performs file I/O. The actual load/merge
    is done in `setup_node`. Here we decide which branch to take based
    on whether the setup step indicated a resume (via
    `state['_resumed_from_latest']`) or — when that flag isn't present
    — by checking whether `latest_state.json` exists under
    `state['task_folder']`.
    """
    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [decide_setup] Routing based on setup outcome...")
    try:
        # If setup_node ran and explicitly set this flag, honor it
        if state.get('_resumed_from_latest'):
            return 'decide'

        # Otherwise, do a cheap existence check (no loading here)
        task_folder = state.get('task_folder')
        if not task_folder:
            return 'generate_code'
        latest_path = os.path.join(task_folder, 'latest_state.json')
        if os.path.exists(latest_path):
            return 'decide'
        return 'generate_code'
    except Exception:
        return 'generate_code'


def load_latest_state(path: Optional[str] = None) -> AgentState:
    """Load `latest_state.json` and return it as an AgentState-like dict.

    This is a small, runtime-checking loader modeled after
    `load_state_demo.py`: it ensures the loaded JSON is a dict, emits
    warnings for missing expected keys, and guarantees common list
    fields exist.
    """
    if path is None:
        raise ValueError("path must be provided to load_latest_state")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"latest_state.json not found at: {path}")

    with open(path, 'r') as fh:
        loaded = json.load(fh)

    if not isinstance(loaded, dict):
        raise TypeError("Loaded state is not a JSON object/dict")

    state = cast(AgentState, loaded)

    # Sanity checks (non-exhaustive)
    required_keys = ["task_id", "task_data", "task_folder", "solutions_list"]
    for k in required_keys:
        if k not in state:
            print(f"Warning: expected key '{k}' missing from loaded state")

    # Ensure lists exist
    for list_key in ("solutions_list", "seed_solutions_list", "fused_solutions_list", "mutated_solutions_list"):
        if state.get(list_key) is None:
            state[list_key] = []  # type: ignore[index]

    return state


def setup_node(state: AgentState) -> AgentState:
    """Node to perform setup: optionally load and merge `latest_state.json`.

    If a `latest_state.json` exists under `state['task_folder']`, this
    function will load it, merge it into the running `state`, and set
    the `_resumed_from_latest` flag so subsequent routing knows we
    resumed. On any error we leave the state alone and mark the flag
    as False.
    """
    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [setup_node] Running setup (may resume from latest_state.json)")
    try:
        task_folder = state.get('task_folder')
        if not task_folder:
            state['_resumed_from_latest'] = False
            return state

        latest_path = os.path.join(task_folder, 'latest_state.json')
        if os.path.exists(latest_path):
            try:
                loaded = load_latest_state(latest_path)
                # Preserve a small set of fields from the caller state
                preserved = {
                    'task_id': state.get('task_id'),
                    'task_folder': state.get('task_folder'),
                    'task_data': state.get('task_data'),
                    "enable_visual_cue": state.get('enable_visual_cue'),
                    "enable_rag_hint": state.get('enable_rag_hint'),
                    "enable_code_predict": state.get('enable_code_predict'),
                    "enable_llm_predict": state.get('enable_llm_predict'),
                    "enable_parallel_eval": state.get('enable_parallel_eval'),
                    "num_loops": state.get('num_loops'),
                    "num_initial_solutions": state.get('num_initial_solutions'),
                    "num_seed_solutions": state.get('num_seed_solutions'),
                    "num_refinements": state.get('num_refinements'),
                    "num_solutions_per_refinement": state.get('num_solutions_per_refinement'),
                    "num_fusions": state.get('num_fusions'),
                    "num_solutions_per_fusion": state.get('num_solutions_per_fusion'),
                    "max_generations": state.get('max_generations'),
                }
                state.clear()
                state.update(loaded)
                state.update(preserved)
                state['_resumed_from_latest'] = True
                print(f"[setup_node] Loaded latest state from {latest_path}")
            except Exception as e:
                print(f"[setup_node] Failed to load latest_state.json: {e}")
                state['_resumed_from_latest'] = False
        else:
            state['_resumed_from_latest'] = False
    except Exception as e:
        print(f"[setup_node] Unexpected error during setup: {e}")
        state['_resumed_from_latest'] = False

    return state


def create_arc_workflow(llm, transformation_llm, code_llm) -> StateGraph:
    """
    Create the LangGraph workflow for ARC problem solving.
    """
    workflow = StateGraph(AgentState)

    # Setup node: decides whether to resume or start fresh (see top-level decide_setup)

    # Add nodes with LLM access
    # `setup` is a lightweight node that performs optional resume/load
    workflow.add_node("setup", setup_node)
    workflow.add_node("generate_code", lambda state: generate_code_node(state, llm, transformation_llm, code_llm))
    workflow.add_node("evolve_code", lambda state: evolve_code_node(state, llm, transformation_llm, code_llm))
    workflow.add_node("test_code", lambda state: test_code_node(state, llm, transformation_llm, code_llm))
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("save_state", save_state_node)

    # Add edges. Use conditional edges out of `test_code` so the graph can
    # choose the next node based on the current `state`.
    # Start by running setup which will route to generate or evolve
    workflow.add_edge(START, "setup")
    workflow.add_edge("generate_code", "test_code")
    # Decision node approach: run `save_state` after `test_code`, then
    # route to a dedicated `decide` node which returns the next node name.
    workflow.add_edge("test_code", "save_state")
    workflow.add_edge("save_state", "decide")
    workflow.add_node("decide", lambda state: state)

    # decide_next returns the literal name of the next node; mapping
    # instructs the graph how to map those strings to nodes.
    def decide_next(state):
        # 1. If out of loops -> finalize
        if out_of_loops(state):
            return "finalize"
        
        # 2. If all_examples_success -> finalize
        if one_solution_succeeded(state):
            return "finalize"
        
        # 3. If solutions_list is empty -> generate_code
        solutions_list = state.get('solutions_list', [])
        if not solutions_list:
            return "generate_code"
        
        # 4. Else -> evolve_code
        return "evolve_code"

    workflow.add_conditional_edges("decide", decide_next, {
        "finalize": "finalize",
        "generate_code": "generate_code",
        "evolve_code": "evolve_code"})

    # Conditional edges out of `setup`: either resume (evolve) or start fresh (generate)
    workflow.add_conditional_edges("setup", decide_setup, {
        "decide": "decide",
        "generate_code": "generate_code",
    })
    workflow.add_edge("evolve_code", "test_code")
    workflow.add_edge("finalize", END)

    return workflow


def create_initial_state(task_id: str,
                        task_data: Dict[str, Any],
                        task_folder: str,
                        agent) -> AgentState:
    """
    Create the initial state for the workflow.
    """
    return {
        "task_id": task_id,
        "task_data": task_data,
        "task_folder": task_folder,
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
        "num_retries": 0,
        "enable_visual_cue": agent.enable_visual_cue,
        "enable_rag_hint": agent.enable_rag_hint,
        "generations": [],
        "current_generation": 0,
        "max_generations": agent.max_generations,
    }

class ARCLangGraphAgent:
    """
    LangGraph-based agent for solving ARC problems.
    
    This agent generates multiple solution attempts for each task,
    storing all solutions in the workflow state.
    """

    def __init__(self, llm, transformation_llm, code_llm,
                 num_initial_solutions: int = 10,
                 num_loops: int = 3,
                 num_seed_solutions: int = 10,
                 num_refinements: int = 5,
                 num_solutions_per_refinement: int = 3,
                 num_fusions: int = 5,
                 num_solutions_per_fusion: int = 3,
                 enable_parallel_eval: bool = False,
                 enable_code_predict: bool = True,
                 enable_llm_predict: bool = False,
                 enable_visual_cue: bool = False,
                 enable_rag_hint: bool = False,
                 max_generations: int = 3,
                 recursion_limit: int = 200,
                 qdrant_client=None,
                 qdrant_collection_name: str = None):
        """
        Initialize the ARC LangGraph Agent.
        
        Args:
            llm: The language model to use for reasoning and general tasks (e.g., ChatOpenAI, ChatAnthropic)
            code_llm: The language model to use for code-related tasks (e.g., code generation, extraction)
            enable_code_predict: Whether to enable code-predicted outputs during testing
            enable_llm_predict: Whether to enable LLM-predicted outputs during testing
            recursion_limit: LangGraph recursion limit (default 200, increase for longer workflows)
        """
        # LLM instances
        self.llm = llm
        self.code_llm = code_llm
        self.transformation_llm = transformation_llm
        
        # Qdrant client for RAG storage (if enabled)
        self.qdrant_client = qdrant_client
        self.qdrant_collection_name = qdrant_collection_name
        
        # Set globals so store_record can find them in worker threads
        if qdrant_client is not None:
            import run_langgraph_agent
            run_langgraph_agent.QDRANT_CLIENT = qdrant_client
            run_langgraph_agent.QDRANT_COLLECTION_NAME = qdrant_collection_name

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
        self.enable_rag_hint = enable_rag_hint
        self.max_generations = int(max_generations)
        self.recursion_limit = int(recursion_limit)

        # Create arc workflow
        self.workflow = create_arc_workflow(llm, transformation_llm, code_llm).compile()

    def solve_task(self, task_id: str, task_data: Dict[str, Any], task_folder: str, task_solution: Optional[List[List[List[int]]]] = None, max_attempts: int = 5) -> WorkflowOutput:
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

        initial_state = create_initial_state(task_id, task_data, task_folder, self)
        
        # Propagate runtime flags into the workflow state so nodes can read them
        initial_state['enable_code_predict'] = bool(self.enable_code_predict)
        initial_state['enable_llm_predict'] = bool(self.enable_llm_predict)

        # New: allow nodes to enable parallelized evaluation
        initial_state['enable_parallel_eval'] = bool(self.enable_parallel_eval)

        # Generation counters for refinement loop
        initial_state['current_generation'] = int(initial_state.get('current_generation', 0))
        initial_state['max_generations'] = int(getattr(self, 'max_generations', 0))

        # Run the workflow with custom recursion limit
        final_state = self.workflow.invoke(
            initial_state,
            config={"recursion_limit": self.recursion_limit}
        )
        execution_time = time.time() - start_time

        # Print per-generation summary (including the current generation)
        gens = final_state.get("generations") or []
        if gens:
            print("Generation summaries:")
            # Use `final_state['solutions_list']` to compute per-generation maxima when possible
            all_solutions = final_state.get("solutions_list", []) or []
            for g in gens:
                gen_idx = g.get("generation")
                avg_train = g.get("average_training_success_rate", 0.0)
                avg_overlap = g.get("average_training_overlap_score", 0.0)

                # Gather solutions that belong to this generation (if solutions track a 'generation' field)
                gen_solutions = [s for s in all_solutions if s.get("generation") == gen_idx]
                if gen_solutions:
                    try:
                        max_train = max(float(s.get("training_success_rate", 0.0)) for s in gen_solutions)
                    except Exception:
                        max_train = 0.0
                    try:
                        max_overlap = max(float(s.get("training_overlap_average", 0.0)) for s in gen_solutions)
                    except Exception:
                        max_overlap = 0.0
                    print(f"  Generation {gen_idx}: avg training success {avg_train:.2%}, avg overlap {avg_overlap:.2f}%, max training success {max_train:.2%}, max overlap {max_overlap:.2f}%")
                else:
                    # Fall back to printing averages only if we can't find per-solution records
                    print(f"  Generation {gen_idx}: avg training success {avg_train:.2%}, avg overlap {avg_overlap:.2f}%")

        # Also include the currently active generation (solutions_list in final_state)
        current_gen = int(final_state.get("current_generation", 0))
        curr_solutions = final_state.get("solutions_list", []) or []
        if curr_solutions:
            # compute averages and maxima for current generation
            total = len(curr_solutions)
            if total > 0:
                avg_train = sum(sol.get("training_success_rate", 0.0) for sol in curr_solutions) / total
                avg_overlap = sum(sol.get("training_overlap_average", 0.0) for sol in curr_solutions) / total
                try:
                    max_train = max(float(sol.get("training_success_rate", 0.0)) for sol in curr_solutions)
                except Exception:
                    max_train = 0.0
                try:
                    max_overlap = max(float(sol.get("training_overlap_average", 0.0)) for sol in curr_solutions)
                except Exception:
                    max_overlap = 0.0
            else:
                avg_train = 0.0
                avg_overlap = 0.0
                max_train = 0.0
                max_overlap = 0.0
            print(f"  Current Generation {current_gen}: avg training success {avg_train:.2%}, avg overlap {avg_overlap:.2f}%, max training success {max_train:.2%}, max overlap {max_overlap:.2f}%")

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