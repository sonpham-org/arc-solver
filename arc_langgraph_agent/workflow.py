"""
Workflow definition for the ARC LangGraph Agent.

This module defines the LangGraph workflow that connects the various nodes
for ARC problem solving, including code generation, testing, and refinement.
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END

# Import our nodes
from .nodes import (
    generate_code_node,
    test_code_node,
    refinement_node,
    extract_helpers_node,
    should_continue_predicate,
    finalize_node
)
from .schema import AgentState


def create_arc_workflow(llm) -> StateGraph:
    """
    Create the LangGraph workflow for ARC problem solving.
    
    The workflow follows this pattern:
    1. Generate code solution
    2. Test the solution against training examples
    3. If not perfect, refine and retry (up to max attempts)
    4. Finalize with prediction for test case
    """
    
    # Create the StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes with LLM access
    workflow.add_node("generate_code", lambda state: generate_code_node(state, llm))
    workflow.add_node("test_code", test_code_node)
    workflow.add_node("extract_helpers", lambda state: extract_helpers_node(state, llm))
    workflow.add_node("refine_code", lambda state: refinement_node(state, llm))
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    # Start with code generation
    workflow.add_edge(START, "generate_code")
    
    # After generating code, extract potential helpers
    workflow.add_edge("generate_code", "extract_helpers")
    
    # After extracting helpers, test the code
    workflow.add_edge("extract_helpers", "test_code")
    
    # After testing, decide whether to refine or end
    workflow.add_conditional_edges(
        "test_code",
        should_continue_predicate,
        {
            "refine": "refine_code",
            "test": "test_code",
            "end": "finalize"
        }
    )
    
    # After refinement, test again directly (skip extract_helpers to avoid re-extraction)
    workflow.add_edge("refine_code", "test_code")
    
    # Finalize ends the workflow
    workflow.add_edge("finalize", END)
    
    return workflow


def create_initial_state(task_id: str, 
                        task_data: Dict[str, Any], 
                        max_attempts: int = 5,
                        global_helpers: Optional[List] = None) -> AgentState:
    """
    Create the initial state for the workflow.
    
    Args:
        task_id: The ARC task ID
        task_data: The task data with train/test examples
        max_attempts: Maximum refinement attempts
        global_helpers: Growing set of helper functions from previous tasks
        
    Returns:
        Initial AgentState for the workflow
    """
    return {
        "task_id": task_id,
        "task_data": task_data,
        "attempt_number": 1,
        "max_attempts": max_attempts,
        "current_solution": None,
        "previous_solutions": [],
        "test_results": [],
        "success_rate": 0.0,
        "available_helpers": global_helpers or [],
        "extracted_helpers": [],
        "global_helper_library": global_helpers or [],
        "reflection_history": [],
        "messages": [],
        "should_continue": True,
        "final_prediction": None,
        "errors": [],
        "metadata": {}
    }


def compile_workflow(llm) -> Any:
    """
    Compile the workflow for execution.
    
    Args:
        llm: The language model to use for code generation
    
    Returns:
        Compiled workflow ready for execution
    """
    workflow = create_arc_workflow(llm)
    return workflow.compile()