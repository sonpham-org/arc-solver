"""
Base class for pluggable graph builders.

This module provides the abstract base class that all graph builders must
implement. Graph builders are responsible for constructing the LangGraph
workflow with nodes and edges.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph

from ..schema import AgentState


class BaseGraphBuilder(ABC):
    """Abstract base class for LangGraph workflow builders.
    
    Subclasses implement specific graph architectures (e.g., evolutionary,
    simple, etc.) by defining how nodes are connected and what routing
    logic is used.
    """
    
    def __init__(self, llm, transformation_llm, code_llm, **kwargs):
        """Initialize the graph builder with LLM instances.
        
        Args:
            llm: Main reasoning LLM for reasoning traces and reflections
            transformation_llm: LLM for generating transformation steps
            code_llm: LLM for code generation
            **kwargs: Additional configuration parameters
        """
        self.llm = llm
        self.transformation_llm = transformation_llm
        self.code_llm = code_llm
        self.config = kwargs
    
    @abstractmethod
    def build(self) -> StateGraph:
        """Build and return the LangGraph workflow.
        
        Returns:
            StateGraph: Compiled LangGraph workflow ready for invocation
        """
        pass
    
    @abstractmethod
    def create_initial_state(self,
                           task_id: str,
                           task_data: Dict[str, Any],
                           task_folder: str,
                           **kwargs) -> AgentState:
        """Create the initial state dictionary for the workflow.
        
        Args:
            task_id: Unique identifier for the task
            task_data: Task specification with train/test examples
            task_folder: Output folder for this task
            **kwargs: Additional state initialization parameters
            
        Returns:
            AgentState: Initial state dictionary
        """
        pass
    
    def add_nodes(self, workflow: StateGraph) -> None:
        """Add nodes to the workflow graph.
        
        Subclasses can override this to add custom nodes. Default
        implementation does nothing.
        
        Args:
            workflow: StateGraph instance to add nodes to
        """
        pass
    
    def add_edges(self, workflow: StateGraph) -> None:
        """Add edges (transitions) between nodes.
        
        Subclasses can override this to add custom routing logic.
        Default implementation does nothing.
        
        Args:
            workflow: StateGraph instance to add edges to
        """
        pass
    
    def get_config_param(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration parameter.
        
        Args:
            key: Configuration parameter name
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
