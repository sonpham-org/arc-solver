"""
Simple graph builder for basic ARC solving.

This graph builder creates a minimal workflow with just:
- Generate initial solutions
- Test solutions
- Finalize results

No evolutionary loops, no refinement, no fusion - just a simple
generate-test-finalize workflow. Useful for quick testing or as a
baseline comparison.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from .base import BaseGraphBuilder
from ..schema import AgentState
from ..nodes import (
    generate_code_node,
    test_code_node,
    finalize_node
)


class SimpleGraphBuilder(BaseGraphBuilder):
    """Simple graph builder for basic ARC solving.
    
    Creates a minimal workflow:
    1. Generate code solutions
    2. Test solutions on training examples
    3. Finalize and return results
    
    No evolutionary loops - just one pass through.
    """
    
    def build(self) -> StateGraph:
        """Build the simple LangGraph workflow.
        
        Returns:
            StateGraph: Compiled workflow ready for execution
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes - only the essentials
        workflow.add_node("generate_code", 
                         lambda state: generate_code_node(state, self.llm, 
                                                          self.transformation_llm, 
                                                          self.code_llm))
        workflow.add_node("test_code", 
                         lambda state: test_code_node(state, self.llm,
                                                      self.transformation_llm,
                                                      self.code_llm))
        workflow.add_node("finalize", finalize_node)
        
        # Add edges - simple linear flow
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "test_code")
        workflow.add_edge("test_code", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def create_initial_state(self,
                           task_id: str,
                           task_data: Dict[str, Any],
                           task_folder: str,
                           task_solution=None,
                           max_attempts: int = 10,
                           **kwargs) -> AgentState:
        """Create initial state for simple workflow.
        
        Args:
            task_id: Task identifier
            task_data: Task specification
            task_folder: Output folder
            task_solution: Optional ground truth
            max_attempts: Maximum attempts (used for num_initial_solutions)
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
            # Simple graph only does one generation
            "num_initial_solutions": self.get_config_param('num_initial_solutions', max_attempts),
            "num_loops": 1,  # Simple graph doesn't loop
            "num_seed_solutions": 0,
            "num_refinements": 0,
            "num_solutions_per_refinement": 0,
            "num_fusions": 0,
            "num_solutions_per_fusion": 0,
            "num_retries": 0,
            "enable_visual_cue": self.get_config_param('enable_visual_cue', False),
            "enable_rag_hint": self.get_config_param('enable_rag_hint', False),
            "enable_code_predict": self.get_config_param('enable_code_predict', True),
            "enable_llm_predict": self.get_config_param('enable_llm_predict', False),
            "enable_parallel_eval": self.get_config_param('enable_parallel_eval', False),
            "generations": [],
            "current_generation": 0,
            "max_generations": 1,  # Simple graph doesn't iterate
        }
