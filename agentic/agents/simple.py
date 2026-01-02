"""
Simple ARC agent for basic problem solving.

This agent uses the SimpleGraphBuilder for a minimal workflow without
evolutionary loops. Useful for quick testing or as a baseline.
"""

from typing import Optional

from .base import BaseARCAgent
from ..graphs.simple import SimpleGraphBuilder


class SimpleARCAgent(BaseARCAgent):
    """Simple ARC agent with minimal workflow.
    
    This agent uses a simple generate-test-finalize workflow without
    evolutionary loops, refinement, or fusion. Useful for:
    - Quick testing and debugging
    - Baseline comparisons
    - Resource-constrained environments
    """
    
    def __init__(self, llm, transformation_llm, code_llm,
                 num_initial_solutions: int = 5,
                 enable_parallel_eval: bool = False,
                 enable_code_predict: bool = True,
                 enable_llm_predict: bool = False,
                 enable_visual_cue: bool = False,
                 enable_rag_hint: bool = False,
                 recursion_limit: int = 50):
        """Initialize the Simple ARC Agent.
        
        Args:
            llm: The language model for reasoning
            transformation_llm: The language model for transformation steps  
            code_llm: The language model for code generation
            num_initial_solutions: Number of solutions to generate
            enable_parallel_eval: Whether to parallelize evaluation
            enable_code_predict: Whether to enable code-predicted outputs
            enable_llm_predict: Whether to enable LLM-predicted outputs
            enable_visual_cue: Whether to pass visual cues to LLM
            enable_rag_hint: Whether to use RAG hints
            recursion_limit: LangGraph recursion limit
        """
        super().__init__(
            llm,
            transformation_llm,
            code_llm,
            num_initial_solutions=num_initial_solutions,
            enable_parallel_eval=enable_parallel_eval,
            enable_code_predict=enable_code_predict,
            enable_llm_predict=enable_llm_predict,
            enable_visual_cue=enable_visual_cue,
            enable_rag_hint=enable_rag_hint,
            recursion_limit=recursion_limit
        )
    
    def _create_default_graph_builder(self) -> SimpleGraphBuilder:
        """Create the simple graph builder for this agent.
        
        Returns:
            SimpleGraphBuilder: Graph builder for minimal workflow
        """
        return SimpleGraphBuilder(
            self.llm,
            self.transformation_llm,
            self.code_llm,
            **self.config
        )
