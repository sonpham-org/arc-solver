"""
Base agent class for ARC problem solving.

This module provides the abstract base class that all ARC agents must
implement. Agents coordinate graph builders and handle the solve_task
interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import os

from ..schema import AgentState, WorkflowOutput
from ..graphs.base import BaseGraphBuilder


class BaseARCAgent(ABC):
    """Abstract base class for ARC problem-solving agents.
    
    Agents use a graph builder to create the workflow and implement the
    solve_task interface to run tasks through the workflow.
    """
    
    def __init__(self, 
                 llm, 
                 transformation_llm, 
                 code_llm,
                 graph_builder: Optional[BaseGraphBuilder] = None,
                 **kwargs):
        """Initialize the ARC agent.
        
        Args:
            llm: Main reasoning LLM
            transformation_llm: LLM for transformation steps
            code_llm: LLM for code generation
            graph_builder: Optional custom graph builder instance
            **kwargs: Additional configuration parameters
        """
        self.llm = llm
        self.transformation_llm = transformation_llm
        self.code_llm = code_llm
        self.config = kwargs
        
        # Use provided graph builder or create default one
        if graph_builder is None:
            graph_builder = self._create_default_graph_builder()
        
        self.graph_builder = graph_builder
        self.workflow = self.graph_builder.build()
    
    @abstractmethod
    def _create_default_graph_builder(self) -> BaseGraphBuilder:
        """Create the default graph builder for this agent type.
        
        Returns:
            BaseGraphBuilder: Graph builder instance
        """
        pass
    
    def solve_task(self,
                  task_id: str,
                  task_data: Dict[str, Any],
                  task_folder: str,
                  task_solution: Optional[List[List[List[int]]]] = None,
                  max_attempts: int = 10) -> WorkflowOutput:
        """Solve an ARC task using the configured workflow.
        
        Args:
            task_id: Unique task identifier
            task_data: Task specification with train/test examples
            task_folder: Output directory for this task
            task_solution: Optional ground truth solutions
            max_attempts: Maximum solution attempts
            
        Returns:
            WorkflowOutput: Results including solutions and metadata
        """
        # Create initial state using the graph builder
        initial_state = self.graph_builder.create_initial_state(
            task_id=task_id,
            task_data=task_data,
            task_folder=task_folder,
            task_solution=task_solution,
            max_attempts=max_attempts,
            **self.config
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(
            initial_state,
            {"recursion_limit": self.config.get('recursion_limit', 50)}
        )
        
        # Extract and return results
        return self._extract_results(final_state)
    
    def _extract_results(self, final_state: AgentState) -> WorkflowOutput:
        """Extract WorkflowOutput from final state.
        
        Args:
            final_state: Final state after workflow execution
            
        Returns:
            WorkflowOutput: Structured output results
        """
        # Get the best solutions
        solutions_list = final_state.get('solutions_list', [])
        
        # Find solutions with all training examples passing
        perfect_solutions = [
            sol for sol in solutions_list
            if all(r.get('code_success', False) 
                  for r in sol.get('training_results', []))
        ]
        
        # Use perfect solutions if available, otherwise use all solutions
        candidate_solutions = perfect_solutions if perfect_solutions else solutions_list
        
        # Sort by number of training successes
        def training_success_count(sol):
            return sum(1 for r in sol.get('training_results', [])
                      if r.get('code_success', False))
        
        candidate_solutions.sort(key=training_success_count, reverse=True)
        
        # Extract test predictions from best solutions
        test_predictions = []
        for sol in candidate_solutions[:2]:  # Top 2 solutions
            test_results = sol.get('test_results', [])
            for test_result in test_results:
                pred = test_result.get('output')
                if pred is not None and pred not in test_predictions:
                    test_predictions.append(pred)
        
        return WorkflowOutput(
            task_id=final_state.get('task_id', ''),
            solutions_list=solutions_list,
            test_predictions=test_predictions,
            metadata={
                'current_loop': final_state.get('current_loop', 0),
                'num_loops': final_state.get('num_loops', 0),
                'perfect_solutions_found': len(perfect_solutions),
                'total_solutions': len(solutions_list)
            }
        )
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration parameter.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
