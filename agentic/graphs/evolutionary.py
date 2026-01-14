"""
Evolutionary graph builder for multi-loop refinement and fusion.

This graph builder implements the current ARCLangGraphAgent's evolutionary
workflow with multiple loops of solution generation, refinement, and fusion.
"""

import os
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from .base import BaseGraphBuilder
from ..schema import AgentState
from ..nodes import (
    generate_code_node,
    test_code_node,
    finalize_node,
    evolve_code_node,
    save_state_node,
    verify_solutions_node
)


def _all_examples_success(results):
    """Helper to check if all examples succeeded."""
    if not results:
        return False
    return all(bool(r.get('code_success', False)) for r in results)


def one_solution_succeeded(state):
    """Predicate: True when any solution has all training examples passing."""
    sols = state.get('solutions_list') or []
    return any(
        _all_examples_success(sol.get('training_results', []))
        for sol in sols
    )


def out_of_loops(state):
    """Predicate: True when we've exhausted all loops."""
    cur_loop = int(state.get('current_loop', 0))
    max_loops = int(state.get('num_loops', 0))
    print("Current loop", cur_loop, "Max loops", max_loops)
    return cur_loop >= max_loops


def decide_setup(state):
    """Routing decision after setup node.
    
    Routes to 'decide' if resuming from latest state, otherwise to
    'generate_code' for fresh start.
    """
    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [decide_setup] Routing based on setup outcome...")
    try:
        if state.get('_resumed_from_latest'):
            return 'decide'
        
        task_folder = state.get('task_folder')
        if not task_folder:
            return 'generate_code'
        latest_path = os.path.join(task_folder, 'latest_state.json')
        if os.path.exists(latest_path):
            return 'decide'
        return 'generate_code'
    except Exception:
        return 'generate_code'


def setup_node(state):
    """Setup node that handles optional resume from latest_state.json and creates augmented data."""
    from ..agent import load_latest_state
    from ..augmentation import augment_task_data
    
    task_folder = state.get('task_folder')
    tid = state.get('task_id', 'unknown')
    print(f"Task {tid} [setup_node] Running setup (may resume from latest_state.json)")
    
    try:
        if task_folder:
            latest_path = os.path.join(task_folder, 'latest_state.json')
            if os.path.exists(latest_path):
                print(f"[setup_node] Found latest_state.json at {latest_path}")
                loaded = load_latest_state(latest_path)
                preserved = {
                    "task_id": state.get('task_id'),
                    "task_data": state.get('task_data'),
                    "task_folder": state.get('task_folder'),
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
                    "num_augmentations": state.get('num_augmentations'),
                    "num_inloop_augmentations": state.get('num_inloop_augmentations'),
                    "max_generations": state.get('max_generations'),
                }
                state.clear()
                state.update(loaded)
                state.update(preserved)
                state['_resumed_from_latest'] = True
                print(f"[setup_node] Loaded latest state from {latest_path}")
            else:
                state['_resumed_from_latest'] = False
        else:
            state['_resumed_from_latest'] = False
    except Exception as e:
        print(f"[setup_node] Error during setup: {e}")
        state['_resumed_from_latest'] = False
    
    # Generate augmented data if num_augmentations > 0 and not resuming
    if not state.get('_resumed_from_latest'):
        num_augmentations = state.get('num_augmentations', 0)
        if num_augmentations > 0:
            try:
                task_data = state.get('task_data')
                if task_data and 'train' in task_data:
                    print(f"[setup_node] Generating {num_augmentations} augmented training examples...")
                    augment_data = augment_task_data(task_data, num_augmentations)
                    state['augment_data'] = augment_data
                    print(f"[setup_node] Created {len(augment_data.get('train', []))} augmented examples")
                else:
                    state['augment_data'] = None
            except Exception as e:
                print(f"[setup_node] Error generating augmented data: {e}")
                state['augment_data'] = None
        else:
            state['augment_data'] = None
    
    return state


class EvolutionaryGraphBuilder(BaseGraphBuilder):
    """Graph builder for evolutionary multi-loop ARC solving.
    
    This builder creates a workflow with:
    - Initial solution generation
    - Iterative refinement and fusion loops
    - Automatic resume from saved state
    - Early termination when perfect solution found
    """
    
    def build(self) -> StateGraph:
        """Build the evolutionary LangGraph workflow.
        
        Returns:
            StateGraph: Compiled workflow ready for execution
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("setup", setup_node)
        workflow.add_node("generate_code", 
                         lambda state: generate_code_node(state, self.llm, 
                                                          self.transformation_llm, 
                                                          self.code_llm))
        workflow.add_node("evolve_code", 
                         lambda state: evolve_code_node(state, self.llm,
                                                        self.transformation_llm,
                                                        self.code_llm))
        workflow.add_node("test_code", 
                         lambda state: test_code_node(state, self.llm,
                                                      self.transformation_llm,
                                                      self.code_llm))
        workflow.add_node("verify_solutions",
                         lambda state: verify_solutions_node(state, self.llm,
                                                            self.transformation_llm,
                                                            self.code_llm))
        workflow.add_node("finalize", finalize_node)
        workflow.add_node("save_state", save_state_node)
        workflow.add_node("decide", lambda state: state)
        
        # Add edges
        workflow.add_edge(START, "setup")
        workflow.add_edge("generate_code", "test_code")
        workflow.add_edge("test_code", "save_state")
        workflow.add_edge("save_state", "decide")
        workflow.add_edge("evolve_code", "test_code")
        workflow.add_edge("verify_solutions", "finalize")
        workflow.add_edge("finalize", END)
        
        # Conditional routing from setup
        workflow.add_conditional_edges("setup", decide_setup, {
            "decide": "decide",
            "generate_code": "generate_code",
        })
        
        # Conditional routing from decide
        def decide_next(state):
            # First check if we have a perfect solution
            if one_solution_succeeded(state):
                # Check if we haven't verified yet or verification failed
                if not state.get('verification_passed'):
                    return "verify_solutions"
                else:
                    # Verification passed, finalize
                    return "finalize"
            
            # Check if out of loops
            if out_of_loops(state):
                # Out of loops - select best solution and finalize
                return "finalize"
            
            # Continue evolution
            solutions_list = state.get('solutions_list', [])
            if not solutions_list:
                return "generate_code"
            
            return "evolve_code"
        
        workflow.add_conditional_edges("decide", decide_next, {
            "finalize": "finalize",
            "generate_code": "generate_code",
            "evolve_code": "evolve_code",
            "verify_solutions": "verify_solutions"
        })
        
        return workflow.compile()
    
    def create_initial_state(self,
                           task_id: str,
                           task_data: Dict[str, Any],
                           task_folder: str,
                           task_solution=None,
                           max_attempts: int = 10,
                           **kwargs) -> AgentState:
        """Create initial state for evolutionary workflow.
        
        Args:
            task_id: Task identifier
            task_data: Task specification
            task_folder: Output folder
            task_solution: Optional ground truth
            max_attempts: Maximum attempts (used for max_generations)
            **kwargs: Additional state parameters from agent config
            
        Returns:
            AgentState: Initial state dictionary
        """
        return {
            "task_id": task_id,
            "task_data": task_data,
            "task_folder": task_folder,
            "solutions_list": [],
            "metadata": {},
            "current_loop": 0,
            "num_initial_solutions": self.get_config_param('num_initial_solutions', 10),
            "num_loops": self.get_config_param('num_loops', 3),
            "num_seed_solutions": self.get_config_param('num_seed_solutions', 10),
            "num_refinements": self.get_config_param('num_refinements', 5),
            "num_solutions_per_refinement": self.get_config_param('num_solutions_per_refinement', 3),
            "num_fusions": self.get_config_param('num_fusions', 5),
            "num_solutions_per_fusion": self.get_config_param('num_solutions_per_fusion', 3),
            "num_retries": 0,
            "enable_visual_cue": self.get_config_param('enable_visual_cue', False),
            "enable_rag_hint": self.get_config_param('enable_rag_hint', False),
            "enable_code_predict": self.get_config_param('enable_code_predict', True),
            "enable_llm_predict": self.get_config_param('enable_llm_predict', False),
            "enable_parallel_eval": self.get_config_param('enable_parallel_eval', False),
            "llm_as_judge_verification": self.get_config_param('llm_as_judge_verification', False),
            "cove_verification": self.get_config_param('cove_verification', False),
            "adversarial_verification": self.get_config_param('adversarial_verification', False),
            "verification_confidence_threshold": self.get_config_param('verification_confidence_threshold', 0.75),
            "verification_num_augmentations": self.get_config_param('verification_num_augmentations', 10),
            "verification_passed": False,
            "generations": [],
            "current_generation": 0,
            "max_generations": self.get_config_param('max_generations', max_attempts),
        }
