#!/usr/bin/env python3
"""
Run LangGraph agent with detailed prompt logging enabled.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and patch the nodes to add logging
import arc_langgraph_agent.nodes as nodes_module
from arc_langgraph_agent.nodes import *
from arc_langgraph_agent.schema import AgentState
import json
from datetime import datetime

# Create a log file for prompts
log_file = f"prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log_prompt(prompt_type, prompt_content):
    """Log prompt to file and optionally print to console."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"{prompt_type} - {datetime.now()}\n")
        f.write(f"{'='*80}\n")
        f.write(prompt_content)
        f.write(f"\n{'='*80}\n\n")

# Store original functions
original_generate_solution_node = nodes_module.generate_solution_node
original_refine_solution_node = nodes_module.refine_solution_node

def logged_generate_solution_node(state: AgentState) -> AgentState:
    """Generate solution node with prompt logging."""
    print(f"\n🔍 GENERATING INITIAL SOLUTION (Attempt {state.attempts + 1})")
    
    # Build the prompt
    prompt = build_python_focused_prompt(
        state.task_data.get("train", []),
        state.helper_functions,
        state.solutions
    )
    
    # Log the prompt
    log_prompt("INITIAL GENERATION PROMPT", prompt)
    print(f"📝 Logged generation prompt to {log_file}")
    
    # Call original function
    result = original_generate_solution_node(state)
    
    # Log the generated code
    if result.solutions:
        latest_solution = result.solutions[-1]
        log_prompt("GENERATED CODE", latest_solution.get('main_code', 'No code generated'))
        print(f"✅ Generated {len(latest_solution.get('main_code', ''))} chars of code")
    
    return result

# Monkey patch the refine node to add logging  
def logged_refine_solution_node(state: AgentState) -> AgentState:
    """Refine solution node with prompt logging."""
    print(f"\n🔄 REFINING SOLUTION (Attempt {state.attempts})")
    
    if not state.solutions or not state.test_results:
        return original_refine_solution_node(state)
    
    # Build reflection prompt
    current_solution = state.solutions[-1]
    failed_tests = [tr for tr in state.test_results if not tr.get("passed", False)]
    
    reflection_prompt = build_arc_style_reflection_prompt(
        current_solution,
        failed_tests,
        state.task_data.get("train", []),
        state.reflection_history
    )
    
    # Log the reflection prompt
    log_prompt("REFLECTION PROMPT", reflection_prompt)
    print(f"📝 Logged reflection prompt to {log_file}")
    print(f"🔍 Analyzing {len(failed_tests)} failed test(s)")
    
    # Call original function
    result = original_refine_solution_node(state)
    
    # Log the refined code
    if len(result.solutions) > len(state.solutions):
        latest_solution = result.solutions[-1]
        log_prompt("REFINED CODE", latest_solution.get('main_code', 'No code generated'))
        print(f"✅ Refined to {len(latest_solution.get('main_code', ''))} chars of code")
    
    return result

# Apply the monkey patches (replace functions in the module)
nodes_module.generate_solution_node = logged_generate_solution_node
nodes_module.refine_solution_node = logged_refine_solution_node

# Now import and run the main script
from run_langgraph_agent import main as run_main

def main():
    """Main function with prompt logging enabled."""
    print(f"🚀 LANGGRAPH AGENT WITH PROMPT LOGGING")
    print(f"📄 Prompts will be logged to: {log_file}")
    print(f"{'='*60}")
    
    # Run the main agent
    result = run_main()
    
    print(f"\n📄 All prompts logged to: {log_file}")
    print(f"📊 View prompts with: cat {log_file}")
    
    return result

if __name__ == "__main__":
    sys.exit(main())